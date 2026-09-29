import torch
from torch import nn
from torch.nn import functional as F

from typing import OrderedDict, Optional, List, Tuple


# =============================================================================
# Core quantize / dequantize / fake-quantize (all operate on plain tensors)
# =============================================================================


def _quantize_tensorwise(
    weight: torch.Tensor, bits: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    """
    Symmetric tensorwise quantization: single scale for the entire weight matrix.
    Returns (int_tensor, scale) where scale is a scalar tensor.
    """

    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    abs_max = weight.abs().max().clamp(min=1e-8)
    scale = abs_max / qmax

    int_w = (weight / scale).round().clamp(qmin, qmax).to(torch.int8)

    return int_w, scale


def _quantize_channelwise(
    weight: torch.Tensor, bits: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    """
    Symmetric channelwise quantization: one scale per output row.
    Returns (int_tensor, scales) where scales has shape [out_features, 1].
    """

    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1

    abs_max = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = abs_max / qmax

    int_w = (weight / scale).round().clamp(qmin, qmax).to(torch.int8)

    return int_w, scale


def quantize_tensor(
    weight: torch.Tensor, bits: int, granularity: str
) -> Tuple[torch.Tensor, torch.Tensor]:

    """Dispatch to the appropriate quantization function based on granularity."""

    if granularity == "tensor":
        return _quantize_tensorwise(weight, bits)

    elif granularity == "channel":
        return _quantize_channelwise(weight, bits)

    else:
        raise ValueError(
            f"granularity expected to be 'tensor' or 'channel', got '{granularity}'"
        )


def dequantize_tensor(int_w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize an integer weight tensor back to float using the stored scale."""

    return int_w.float() * scale


def fake_quantize_tensor(
    weight: torch.Tensor, bits: int, granularity: str
) -> torch.Tensor:

    """Quantize then immediately dequantize, staying in float. Round-trip error only."""

    int_w, scale = quantize_tensor(weight, bits, granularity)

    float_w = dequantize_tensor(int_w, scale)

    return float_w


# =============================================================================
# QAT wrapper (training-time: fake-quantize with STE)
# =============================================================================


class QATLinear(nn.Module):

    """
    Drop-in wrapper around nn.Linear that fake-quantizes weights during forward.
    To be used during Quantization-aware Training.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        bits: int,
        granularity: str,
    ):
        super().__init__()
        self.linear = original_linear
        self.bits = bits
        self.granularity = granularity

        if self.granularity not in ("tensor", "channel"):
            raise ValueError(
                f"granularity expected to be 'tensor' or 'channel', got '{self.granularity}'"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = fake_quantize_tensor(self.linear.weight, self.bits, self.granularity)

        # NOTE: Straight-Through Estimator (STE). The forward pass uses the
        # fake-quantized weights, but .detach() on the difference means
        # gradients flow through to the original fp weights unchanged.
        w = self.linear.weight + (w_q - self.linear.weight).detach()

        return F.linear(x, w, self.linear.bias)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


class QATMultiheadAttention(nn.Module):

    """
    Drop-in wrapper around nn.MultiheadAttention that fake-quantizes
    in_proj_weight and out_proj.weight during forward via STE.

    Designed for open_clip models whose attention layers use
    nn.MultiheadAttention with a fused in_proj_weight [3*dim, dim]
    instead of separate nn.Linear layers for Q, K, V.
    """

    def __init__(
        self,
        original_mha: nn.MultiheadAttention,
        bits: int,
        granularity: str,
    ):
        super().__init__()
        self.mha = original_mha
        self.bits = bits
        self.granularity = granularity

        if self.granularity not in ("tensor", "channel"):
            raise ValueError(
                f"granularity expected to be 'tensor' or 'channel', got '{self.granularity}'"
            )

        if not self.mha._qkv_same_embed_dim:
            raise ValueError(
                "QATMultiheadAttention only supports _qkv_same_embed_dim=True "
                "(fused in_proj_weight). Separate q/k/v projections are not supported."
            )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # STE on in_proj_weight [3*dim, dim]
        w_in = self.mha.in_proj_weight
        w_in_q = fake_quantize_tensor(w_in, self.bits, self.granularity)
        w_in_ste = w_in + (w_in_q - w_in).detach()

        # STE on out_proj.weight [dim, dim]
        w_out = self.mha.out_proj.weight
        w_out_q = fake_quantize_tensor(w_out, self.bits, self.granularity)
        w_out_ste = w_out + (w_out_q - w_out).detach()

        # F.multi_head_attention_forward expects (L, N, E) layout.
        is_batched = query.dim() == 3
        if self.mha.batch_first and is_batched:
            query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        attn_output, attn_weights = F.multi_head_attention_forward(
            query,
            key,
            value,
            self.mha.embed_dim,
            self.mha.num_heads,
            w_in_ste,
            self.mha.in_proj_bias,
            self.mha.bias_k,
            self.mha.bias_v,
            self.mha.add_zero_attn,
            self.mha.dropout,
            w_out_ste,
            self.mha.out_proj.bias,
            training=self.mha.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        if self.mha.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)

        return attn_output, attn_weights

    @property
    def in_proj_weight(self):
        return self.mha.in_proj_weight

    @property
    def in_proj_bias(self):
        return self.mha.in_proj_bias

    @property
    def out_proj(self):
        return self.mha.out_proj


# =============================================================================
# Quantized inference wrapper (stores int weights + scales, dequantizes on forward)
# =============================================================================


class QuantizedLinear(nn.Module):

    """
    Inference-time wrapper that holds weights as intB + scales, where B is a lower-precision integer data type.
    Dequantizes on the fly during forward.
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        bits: int,
        granularity: str,
    ):

        super().__init__()
        self.bits = bits
        self.granularity = granularity
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        # Quantize and store as buffers (not parameters — no gradients)
        int_w, scale = quantize_tensor(original_linear.weight.data, bits, granularity)
        self.register_buffer("int_weight", int_w)
        self.register_buffer("scale", scale)

        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = dequantize_tensor(self.int_weight, self.scale)
        return F.linear(x, w, self.bias)

    @property
    def weight(self):

        """Return the dequantized weight (for inspection / compatibility)."""

        return dequantize_tensor(self.int_weight, self.scale)


# =============================================================================
# Model-level enable / disable / convert helpers (all mutate the model in-place)
# =============================================================================


def enable_qat_(
    model: nn.Module,
    bits: int,
    granularity: str,
    skip_modules: frozenset[str],
) -> nn.Module:

    """
    Replace all nn.Linear layers in the backbone with QATLinear wrappers.
    Skips modules whose names appear in *skip_modules*.
    """

    for name, module in model.named_children():
        if name in skip_modules:
            continue
        if isinstance(module, QATLinear):
            # Idempotent: avoid recursively wrapping module.linear on repeated calls.
            if module.bits != bits or module.granularity != granularity:
                raise ValueError(
                    f"QAT already enabled on submodule {name!r} with "
                    f"{module.bits}-bit/{module.granularity}; "
                    f"requested {bits}-bit/{granularity}"
                )
            continue
        if isinstance(module, nn.Linear):
            setattr(model, name, QATLinear(module, bits, granularity))
        else:
            enable_qat_(module, bits, granularity, skip_modules=skip_modules)
    return model

def disable_qat_(model: nn.Module) -> nn.Module:

    """Remove QATLinear wrappers, restoring original nn.Linear layers."""

    for name, module in model.named_children():

        if isinstance(module, QATLinear):
            setattr(model, name, module.linear)

        else:
            disable_qat_(module)


def quantize_model_(
    model: nn.Module,
    bits: int,
    granularity: str,
    skip_modules: frozenset[str],
) -> nn.Module:
    """
    Replace all nn.Linear layers in the backbone with QuantizedLinear.
    Stores weights as int8 + scales; dequantizes on the fly during forward.
    Skips modules whose names appear in *skip_modules*.

    Works on both plain models (nn.Linear) and QAT models (QATLinear).
    """

    for name, module in model.named_children():
        if name in skip_modules:
            continue
        if isinstance(module, QATLinear):
            # Extract the inner nn.Linear from the QAT wrapper
            setattr(
                model, name,
                QuantizedLinear(module.linear, bits, granularity),
            )
        elif isinstance(module, nn.Linear):
            setattr(
                model, name,
                QuantizedLinear(module, bits, granularity),
            )
        else:
            quantize_model_(module, bits, granularity, skip_modules=skip_modules)


def dequantize_model_(model: nn.Module) -> nn.Module:
    for name, module in model.named_children():
        if isinstance(module, QuantizedLinear):
            device = module.int_weight.device
            linear = nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                device=device,
            )
            linear.weight.data.copy_(dequantize_tensor(module.int_weight, module.scale))
            if module.bias is not None:
                linear.bias.data.copy_(module.bias)
            setattr(model, name, linear)
        else:
            dequantize_model_(module)


# =============================================================================
# PTQ (in-place fake-quantize of all backbone nn.Linear weights)
# =============================================================================


def apply_ptq_(
    model: nn.Module,
    bits: int,
    granularity: str,
    skip_modules: frozenset[str],
    _prefix: str = "",
) -> List[str]:

    """
    Apply post-training quantization in-place: fake-quantize (round-trip) the
    weight of every nn.Linear in the backbone so it only contains values
    representable at the target bit-width. Skips modules whose names appear
    in *skip_modules*.

    Returns the list of fully-qualified names of every nn.Linear that was
    fake-quantized.
    """

    quantized: List[str] = []

    for name, module in model.named_children():
        if name in skip_modules:
            continue
        full = f"{_prefix}{name}"
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                module.weight.copy_(
                    fake_quantize_tensor(module.weight, bits, granularity)
                )
            quantized.append(full)
        else:
            quantized.extend(
                apply_ptq_(
                    module,
                    bits,
                    granularity,
                    skip_modules=skip_modules,
                    _prefix=full + ".",
                )
            )

    return quantized


# =============================================================================
# Open-CLIP variants (handle nn.MultiheadAttention's fused in_proj_weight)
# =============================================================================


def enable_qat_open_clip_(
    model: nn.Module,
    bits: int,
    granularity: str,
    skip_modules: frozenset[str],
) -> nn.Module:

    """
    Replace all nn.Linear layers with QATLinear wrappers **and** all
    nn.MultiheadAttention layers with QATMultiheadAttention wrappers.
    Skips modules whose names appear in *skip_modules*.
    """

    for name, module in model.named_children():
        if name in skip_modules:
            continue
        if isinstance(module, QATMultiheadAttention):
            if module.bits != bits or module.granularity != granularity:
                raise ValueError(
                    f"QAT already enabled on MHA submodule {name!r} with "
                    f"{module.bits}-bit/{module.granularity}; "
                    f"requested {bits}-bit/{granularity}"
                )
            continue
        if isinstance(module, QATLinear):
            if module.bits != bits or module.granularity != granularity:
                raise ValueError(
                    f"QAT already enabled on submodule {name!r} with "
                    f"{module.bits}-bit/{module.granularity}; "
                    f"requested {bits}-bit/{granularity}"
                )
            continue
        if isinstance(module, nn.MultiheadAttention):
            setattr(model, name, QATMultiheadAttention(module, bits, granularity))
        elif isinstance(module, nn.Linear):
            setattr(model, name, QATLinear(module, bits, granularity))
        else:
            enable_qat_open_clip_(module, bits, granularity, skip_modules=skip_modules)
    return model


def disable_qat_open_clip_(model: nn.Module) -> nn.Module:

    """Remove QATMultiheadAttention and QATLinear wrappers, restoring originals."""

    for name, module in model.named_children():
        if isinstance(module, QATMultiheadAttention):
            setattr(model, name, module.mha)
        elif isinstance(module, QATLinear):
            setattr(model, name, module.linear)
        else:
            disable_qat_open_clip_(module)


def apply_ptq_open_clip_(
    model: nn.Module,
    bits: int,
    granularity: str,
    skip_modules: frozenset[str],
    _prefix: str = "",
) -> List[str]:

    """
    Apply post-training quantization in-place: fake-quantize (round-trip) the
    weight of every nn.Linear **and** the in_proj_weight of every
    nn.MultiheadAttention in the backbone. Skips modules whose names appear
    in *skip_modules*.

    Returns the list of fully-qualified names of every layer/parameter that
    was fake-quantized.
    """

    quantized: List[str] = []

    for name, module in model.named_children():
        if name in skip_modules:
            continue
        full = f"{_prefix}{name}"
        if isinstance(module, nn.MultiheadAttention):
            # Fake-quantize the fused in_proj_weight in-place.
            with torch.no_grad():
                module.in_proj_weight.copy_(
                    fake_quantize_tensor(module.in_proj_weight, bits, granularity)
                )
            quantized.append(f"{full}.in_proj_weight")
            # Recurse into MHA children to find out_proj (nn.Linear).
            quantized.extend(
                apply_ptq_open_clip_(
                    module, bits, granularity,
                    skip_modules=skip_modules, _prefix=full + ".",
                )
            )
        elif isinstance(module, nn.Linear):
            with torch.no_grad():
                module.weight.copy_(
                    fake_quantize_tensor(module.weight, bits, granularity)
                )
            quantized.append(full)
        else:
            quantized.extend(
                apply_ptq_open_clip_(
                    module, bits, granularity,
                    skip_modules=skip_modules, _prefix=full + ".",
                )
            )

    return quantized
