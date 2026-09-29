"""AWQ: activation-aware weight quantization for nn.Linear weights.

Why this module exists. Reviewers objected that the paper's only PTQ baseline
is vanilla round-to-nearest (`apply_ptq_`), and named GPTQ, AWQ and SmoothQuant
as the strong methods a 2026 submission should be measured against. GPTQ is
covered by `src/gptq.py`. AWQ (Lin et al., MLSys 2024) is the second, and it is
a genuinely *different* axis of attack: GPTQ compensates rounding error after
the fact using second-order information, while AWQ changes what gets rounded in
the first place, by rescaling each input channel according to how salient its
activations are. Having both means the rebuttal's competitor column is not one
method's idiosyncrasy. This is a from-scratch implementation written against
the official reference at `references/llm-awq/` (MIT-HAN-LAB), used by
`evaluate_fp_awq.py` (the AWQ(FP) baseline, Task 1) and `qv_transfer_awq.py`
(AWQ on FP + alpha*QV, Task 2).

The algorithm. Weight-only quantization error is not uniformly costly: a weight
column multiplying a high-magnitude activation channel contributes far more
output error than one multiplying a near-dead channel. AWQ exploits the fact
that quantization is *not* scale-equivariant — for a per-channel diagonal s,

    Q(W diag(s)) diag(1/s)  !=  Q(W)

because dividing by s after rounding shrinks the effective step on the channels
where s is large. AWQ searches s to minimize the layer's output error, over the
one-parameter family

    s = (mean_tokens |x|)^alpha,   alpha in {0, 1/n_grid, ..., (n_grid-1)/n_grid}

renormalized by sqrt(s.max() * s.min()) so the family is centred on 1. alpha=0
gives s == 1 exactly, i.e. **plain RTN is always in the search space**, so AWQ
can never score worse than RTN on the search objective. A second stage
(`auto_clip`) then shrinks each output channel's clipping threshold over a grid
of factors 1.00, 0.95, ..., 0.55, trading a few saturated outliers for a finer
step on the bulk of the distribution, again by output MSE.

Calibration is sequential, as in `src/gptq.py`: layers are quantized one at a
time in forward execution order, each calibrating on activations produced by an
upstream that has already been quantized. Official AWQ obtains the same
propagation by processing whole transformer blocks in order and feeding each
block's output to the next.

Why no scale folding. Official AWQ never stores diag(1/s) in the weight: it
folds s into the preceding op — exactly, via `scale_ln_fcs` (divide the
LayerNorm's weight and bias) or `scale_fc_fc` (divide the preceding Linear's
rows) — or, where no exact fold exists (a GELU between the two Linears), wraps
the activation in a `ScaledActivation` module computing act(x)/s. We instead
write the fully-simulated weight

    W_awq = Q(W diag(s)) diag(1/s)

in place and fold nothing. This is **accuracy-equivalent to the full official
pipeline, not an approximation of it**: the LN and FC folds are exact identities
by construction, and `ScaledActivation` followed by a layer holding Q(W diag(s))
computes (act(x)/s) Q(W diag(s))^T, which is algebraically the same function as
act(x) W_awq^T. Folding is a *deployment* optimization — it removes a runtime
divide — not an accuracy one. W_awq remains a genuine b-bit weight-only model:
it is exactly int_b (x) per-row scale (x) per-input-channel s, and s can be
folded at export time by the same three primitives. What we buy is the
`apply_ptq_` contract — no module is ever replaced, so pre-registered forward
hooks survive (the property `003_qat_transfer_activ` relies on) — and
architecture-agnosticism: no per-block pair table, no `Catcher`, no
`layer_kwargs` plumbing, and the module works unchanged on BERT.

Deviations from the official implementation, and their impact:

* Quantization grid. Official `pseudo_quantize_tensor` defaults to *asymmetric*
  min/max quantization with a zero-point and group size 128 (its symmetric
  branch is dead code — it reads `min_val` before assignment, and its own
  comment says "we actually never used this"). We plug in the project's grid
  from `src.quantization`: symmetric, true zero, scale = absmax/(2^(b-1) - 1),
  per-tensor or per-output-row. This is the only number-changing deviation and
  it is required: with identical grids the RTN, GPTQ and AWQ columns of the
  results table differ *only* in method, and AWQ at n_grid=1 with clipping off
  degenerates bit-exactly to `apply_ptq_` (see `code/test/awq.py`).
* Search objective. Official minimizes the output error of an enclosing
  submodule (`module2inspect` = the whole `self_attn` or `mlp`) and shares one s
  across all Linears fed by the same tensor, because s must be folded into the
  single preceding op they share. We minimize each Linear's own output error.
  For the models this repo studies the two coincide structurally: timm's
  ViT/DeiT/Swin blocks fuse Q, K and V into one `attn.qkv`, so every Linear
  (`attn.qkv`, `attn.proj`, `mlp.fc1`, `mlp.fc2`) has its own distinct input and
  official AWQ's sharing groups are singletons. What remains is where the error
  is measured — at the Linear's output rather than after the attention softmax
  or the MLP's second layer. Since we do not fold, no correctness constraint
  ties the two; the deviation buys the architecture-agnosticism described above.
* Token subsampling. Official runs the scale search over every cached
  calibration token and subsamples to 512 tokens only for the clip search. We
  cache a deterministic strided subsample of `_N_SCALE_TOKEN` tokens and use the
  first `_N_CLIP_TOKEN` of them for clipping. This bounds the per-layer cache at
  a few MB (fc2's input is 4x the model width; the full cache for a ViT-B at 4
  batches of 128 images would be ~1.2 GB for that layer alone) and makes the
  search cost independent of batch size. The activation statistic itself is NOT
  subsampled: `mean_tokens |x|` is accumulated exactly over every token seen, so
  the AWQ scale family is unaffected; only the ranking among candidates uses the
  subsample.
* Clip search granularity. Official reshapes to groups of `q_group_size` and
  takes the argmin elementwise per (output channel, group). Under the project's
  taxonomy `channel` means one group per output row, so our per-row argmin is
  the same rule at group size = in_features; under `tensor` there is a single
  shared scale, so a per-row threshold would be meaningless and we search one
  global factor instead.
* Not ported: real INT packing (`WQLinear`), the `q_group_size` axis (the
  project's granularity taxonomy is tensor/channel), the LLM `get_calib_dataset`
  pile-val loader and its block-concatenation (calibration here is the
  receiver's own training split), and the per-architecture `auto_scale_block`
  pair table (moot without folding).
* TF32 is disabled during the search and restored afterwards, matching
  `apply_gptq_`: the search ranks 20 candidates whose losses can be close, and
  it must rank them in the same numerics the GPTQ column was calibrated in.
"""

import logging
import time

import torch
from torch import nn

from typing import Any, Callable, Dict, List, Optional

from src.quantization import fake_quantize_tensor
# Traversal and calibration plumbing is shared verbatim with GPTQ rather than
# duplicated: `_find_linear_layers` encodes the same skip_modules contract as
# `apply_ptq_`, and `_discover_execution_order` / `_StopForward` encode the
# sequential-calibration order both methods need. Any drift between the two
# competitor columns' layer sets would invalidate the comparison.
from src.gptq import (
    _StopForward,
    _default_forward_fn,
    _discover_execution_order,
    _find_linear_layers,
)

log = logging.getLogger(__name__)

# Official constants (references/llm-awq/awq/quantize/auto_clip.py). These are
# deliberately NOT config knobs: they are part of "what AWQ is", and every
# result-affecting knob that IS configurable has to appear in the `awq=` path
# fragment. They are recorded in each run's results JSON instead.
_CLIP_GRID_STEPS = 10      # official: range(int(0.5 * n_grid)) with n_grid=20
_CLIP_MIN_SHRINK = 0.55    # official: 1 - (_CLIP_GRID_STEPS - 1) / 20
_N_CLIP_TOKEN = 512        # official n_sample_token
_N_SCALE_TOKEN = 2048      # ours; official uses every cached token

# Official skips clipping on the query and key projections: their outputs enter
# a qk bmm, so a Linear-output MSE is a poor proxy for the error that matters.
# timm fuses Q, K and V into one `attn.qkv`, which therefore matches here too.
_CLIP_SKIP_PATTERNS = ("qkv", "q_proj", "k_proj", "query", "key")


def awq_path_frag(
    bits: int,
    granularity: str,
    skip_modules,
    num_calib_batches: int,
    n_grid: int,
    clip: bool,
) -> str:
    """
    The single source of the `awq=` path fragment.

    Every writer and reader of the AWQ trees — experiment scripts and
    visualizations alike — calls this, following the `pv_path_frag` precedent in
    `src/pv_tuning.py`. The `gptq=` fragment is spelled by hand at eight sites
    and has stayed consistent only by luck.

    Carries every knob that changes the numbers and nothing that does not, the
    same rule that kept GPTQ's result-invariant `block_size` out of its
    fragment. Booleans render as Python `str(bool)` ("True"/"False"), matching
    how `actorder` already appears on disk.
    """
    skip_modules_sorted = sorted(skip_modules)
    skip_tag = "-".join(skip_modules_sorted) if skip_modules_sorted else "none"
    return (
        f"awq=bits={bits}_gran={granularity}_skip={skip_tag}"
        f"_ncal={num_calib_batches}_ngrid={n_grid}_clip={clip}"
    )


# =============================================================================
# Per-layer solver
# =============================================================================


class AWQ:

    """
    AWQ solver for a single nn.Linear layer.

    Usage:
        solver = AWQ(layer, name="blocks.0.mlp.fc2")
        solver.add_batch(inp)   # the *layer's* inputs, typically via a hook
        solver.quantize_(bits=3, granularity="channel")
        solver.free()

    `scale` holds the searched per-input-channel s after `quantize_`, so tests
    can verify that `weight * scale` lands exactly on the integer grid — the
    property that makes the unfolded form a genuine b-bit model.
    """

    def __init__(self, layer: nn.Linear, name: str = "?"):
        if not isinstance(layer, nn.Linear):
            raise TypeError(
                f"AWQ supports nn.Linear only, got {type(layer).__name__}"
            )
        self.layer = layer
        self.name = name
        self.dev = layer.weight.device
        self.rows = layer.out_features
        self.columns = layer.in_features
        # Exact activation statistic, accumulated over every token seen.
        self.abs_sum: Optional[torch.Tensor] = torch.zeros(
            self.columns, device=self.dev, dtype=torch.float32
        )
        self.n_tokens = 0
        # Subsampled inputs, for ranking candidates only.
        self._chunks: List[torch.Tensor] = []
        self.scale: Optional[torch.Tensor] = None

    def add_batch(self, inp: torch.Tensor) -> None:
        """
        Record a batch of layer inputs.

        *inp* is whatever the layer's forward receives: (..., in_features),
        typically (batch, tokens, in_features) for transformers. The mean-|x|
        statistic that defines AWQ's scale family is accumulated exactly; the
        cached tokens used to *rank* candidates are strided-subsampled, which
        is deterministic and independent of batch size.
        """
        if self.abs_sum is None:
            raise RuntimeError(
                f"AWQ {self.name}: add_batch called after free()"
            )
        x = inp.reshape(-1, inp.shape[-1]).float()
        self.abs_sum += x.abs().sum(dim=0)
        self.n_tokens += x.shape[0]

        keep = min(x.shape[0], _N_SCALE_TOKEN)
        stride = max(1, x.shape[0] // keep)
        self._chunks.append(x[::stride][:keep].clone())

    def _tokens(self, budget: int = _N_SCALE_TOKEN) -> torch.Tensor:
        """Concatenated cache, strided down to at most *budget* rows."""
        x = torch.cat(self._chunks, dim=0)
        if x.shape[0] > budget:
            stride = max(1, x.shape[0] // budget)
            x = x[::stride][:budget]
        return x

    def search_scale_(
        self, bits: int, granularity: str, n_grid: int = 20
    ) -> torch.Tensor:
        """
        Grid-search the per-input-channel scale s minimizing output MSE.

        Returns s with shape (in_features,). Mutates nothing — the caller
        applies it. `n_grid=1` evaluates ratio=0 only, i.e. s == 1, which is
        plain RTN; the search space always contains RTN, so the returned s can
        never be worse than RTN on this objective.
        """
        X = self._tokens()
        W = self.layer.weight.data.float()
        ref = X.matmul(W.t())

        x_max = self.abs_sum / max(self.n_tokens, 1)

        best_scale = torch.ones(self.columns, device=self.dev, dtype=torch.float32)
        best_err = float("inf")

        for i in range(n_grid):
            ratio = i / n_grid
            s = x_max.pow(ratio).clamp(min=1e-4)
            s = s / (s.max() * s.min()).sqrt()
            if not torch.isfinite(s).all():
                # A dead layer (x_max all zeros) or an overflow; ratio=0 always
                # yields a finite s == 1, so a usable candidate always exists.
                continue
            w_q = fake_quantize_tensor(W * s, bits, granularity) / s
            err = (X.matmul(w_q.t()) - ref).pow(2).mean().item()
            if err < best_err:
                best_err = err
                best_scale = s

        return best_scale

    def search_clip_(
        self, bits: int, granularity: str, s: torch.Tensor
    ) -> torch.Tensor:
        """
        Grid-search the clipping threshold on the *scaled* weight W diag(s).

        Returns the per-row (granularity="channel") or scalar (granularity=
        "tensor") maximum absolute value to clamp to, shaped to broadcast
        against the weight. Search order matches official AWQ, which clips
        after scaling: `apply_scale` divides the cached activations by s before
        `auto_clip_block` runs.
        """
        # Official subsamples to `n_sample_token` for the clip search only; the
        # scale search above already ran on the wider cache.
        X = self._tokens(budget=_N_CLIP_TOKEN)
        W = self.layer.weight.data.float()
        ref = X.matmul(W.t())
        # x @ (Q(Ws) diag(1/s)).T == (x diag(1/s)) @ Q(Ws).T
        Xs = X / s
        Ws = W * s

        if granularity == "channel":
            org_max = Ws.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
            best_max = org_max.clone()
            best_err = torch.full(
                (self.rows,), float("inf"), device=self.dev, dtype=torch.float32
            )
        else:
            org_max = Ws.abs().max().clamp(min=1e-8)
            best_max = org_max.clone()
            best_err = float("inf")

        for i in range(_CLIP_GRID_STEPS):
            shrink = 1.0 - i * (1.0 - _CLIP_MIN_SHRINK) / (_CLIP_GRID_STEPS - 1)
            cur_max = org_max * shrink
            w_q = fake_quantize_tensor(
                Ws.clamp(-cur_max, cur_max), bits, granularity
            )
            diff = (Xs.matmul(w_q.t()) - ref).pow(2)
            if granularity == "channel":
                err = diff.mean(dim=0)              # per output row
                improved = err < best_err
                best_err = torch.where(improved, err, best_err)
                best_max = torch.where(
                    improved.unsqueeze(1), cur_max, best_max
                )
            else:
                err = diff.mean().item()
                if err < best_err:
                    best_err = err
                    best_max = cur_max

        return best_max

    def quantize_(
        self,
        bits: int,
        granularity: str,
        n_grid: int = 20,
        clip: bool = True,
    ) -> float:
        """
        Run the AWQ search and write the quantized weight back, in-place.

        Mutates `self.layer.weight` (values only; the tensor object survives).
        Returns the final output MSE on the cached calibration tokens.
        """
        if self.abs_sum is None:
            raise RuntimeError(
                f"AWQ {self.name}: quantize_ called after free() or twice"
            )
        if self.n_tokens == 0:
            raise RuntimeError(
                f"AWQ {self.name}: no calibration data (n_tokens=0); "
                "feed add_batch first"
            )
        if granularity not in ("tensor", "channel"):
            raise ValueError(
                f"granularity expected to be 'tensor' or 'channel', "
                f"got '{granularity}'"
            )

        tick = time.time()

        W = self.layer.weight.data.float()
        s = self.search_scale_(bits, granularity, n_grid=n_grid)
        Ws = W * s

        clip_skipped = any(p in self.name for p in _CLIP_SKIP_PATTERNS)
        if clip and not clip_skipped:
            max_val = self.search_clip_(bits, granularity, s)
            Ws = Ws.clamp(-max_val, max_val)

        w_final = fake_quantize_tensor(Ws, bits, granularity) / s

        X = self._tokens()
        final_err = (
            X.matmul(w_final.t()) - X.matmul(W.t())
        ).pow(2).mean().item()

        with torch.no_grad():
            self.layer.weight.copy_(w_final.to(self.layer.weight.dtype))

        self.scale = s
        log.info(
            "AWQ %s: time=%.2fs, out_mse=%.6g, clip=%s",
            self.name, time.time() - tick, final_err,
            "skipped" if clip_skipped else clip,
        )
        return final_err

    def free(self) -> None:
        self.abs_sum = None
        self._chunks = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Model-level AWQ entry point (drop-in alternative to apply_ptq_)
# =============================================================================


def apply_awq_(
    model: nn.Module,
    bits: int,
    granularity: str,
    skip_modules: frozenset[str],
    calib_loader,
    device,
    num_calib_batches: int = 4,
    n_grid: int = 20,
    clip: bool = True,
    forward_fn: Optional[Callable[[nn.Module, Any, Any], None]] = None,
) -> List[str]:
    """
    Apply AWQ post-training quantization in-place, sequentially.

    Mirrors `apply_ptq_`'s contract: fake-quantizes the weight of every
    nn.Linear outside *skip_modules*, replaces no modules (pre-registered
    forward hooks survive), and returns the fully-qualified names of the
    quantized layers, here in processing (execution) order.

    Do NOT call `apply_ptq_` afterwards. AWQ *is* the quantizer, as GPTQ is in
    `qv_transfer_gptq.py`. The written weight is Q(W diag(s)) diag(1/s), whose
    per-row absmax differs from the grid it was quantized on, so a following
    RTN pass would re-round onto a fresh grid and discard both the
    activation-aware scaling and the searched clipping.

    The model must already be on *device*; calibration batches are moved there
    by *forward_fn*. The first `num_calib_batches` batches are materialized
    once and replayed identically for every layer — the project's seeded
    shuffled train loaders yield different batches on re-iteration, so pulling
    fresh batches per layer would silently calibrate each layer on different
    data. Labels are never used.

    Args:
        model: the model to quantize (mutated in-place).
        bits / granularity / skip_modules: quantization config, same meaning
            and same grid as `apply_ptq_`.
        calib_loader: iterable of batches from the receiver's *training* split.
        device: torch.device the model lives on.
        num_calib_batches: batches to materialize for calibration.
        n_grid: scale-search grid points; ratio=0 (plain RTN) is always tried.
        clip: run the official `auto_clip` stage after the scale search.
        forward_fn: callable (model, batch, device) -> None running one
            forward pass; defaults to the vision batch convention.

    Returns:
        Fully-qualified names of every quantized nn.Linear, execution order.
    """
    if forward_fn is None:
        forward_fn = _default_forward_fn

    linear_layers = _find_linear_layers(model, skip_modules)
    if not linear_layers:
        return []

    calib_batches = []
    for batch in calib_loader:
        if len(calib_batches) >= num_calib_batches:
            break
        calib_batches.append(batch)
    if not calib_batches:
        raise ValueError(
            "calib_loader yielded no batches; AWQ needs calibration data"
        )

    tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    tf32_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    was_training = model.training
    model.eval()

    quantized: List[str] = []
    num_fallback = 0
    try:
        with torch.no_grad():
            order, counts, never_fired = _discover_execution_order(
                model, linear_layers, calib_batches[0], device, forward_fn
            )

            for layer_name in order:
                layer = linear_layers[layer_name]
                solver = AWQ(layer, name=layer_name)
                abort_early = counts[layer_name] == 1

                def hook(module, inp, out, _solver=solver, _abort=abort_early):
                    _solver.add_batch(inp[0].data)
                    if _abort:
                        raise _StopForward

                handle = layer.register_forward_hook(hook)
                try:
                    for batch in calib_batches:
                        try:
                            forward_fn(model, batch, device)
                        except _StopForward:
                            pass
                finally:
                    handle.remove()

                solver.quantize_(
                    bits=bits,
                    granularity=granularity,
                    n_grid=n_grid,
                    clip=clip,
                )
                solver.free()
                quantized.append(layer_name)

            for layer_name in never_fired:
                # Unreachable for our ViTs (every Linear fires); robustness for
                # exotic models. With no activations there is no salience to be
                # aware of, and AWQ at s == 1 is exactly RTN.
                log.warning(
                    "AWQ: layer %s never fired during calibration; "
                    "falling back to RTN", layer_name,
                )
                layer = linear_layers[layer_name]
                layer.weight.copy_(
                    fake_quantize_tensor(layer.weight, bits, granularity)
                )
                quantized.append(layer_name)
                num_fallback += 1
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
        torch.backends.cudnn.allow_tf32 = tf32_cudnn
        if was_training:
            model.train()

    log.info(
        "AWQ: quantized %d layers (%d via never-fired RTN fallback)",
        len(quantized), num_fallback,
    )
    return quantized
