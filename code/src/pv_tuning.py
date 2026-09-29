"""PV-Tuning on the project's uniform quantization grid.

Why this module exists. Every quantization vector in this repository is
`QV = QAT_D - FP_D`, and every QAT checkpoint behind it was produced by exactly
one finetuner: straight-through estimation (`QATLinear` in
`src/quantization.py`). The claim under test is that the QV encodes a
task-agnostic *robustness direction*, so it is worth asking whether a
**better quantization-aware finetuner yields a better-transferring QV**, or
whether the transferable content is a property of the quantization grid and is
insensitive to how the QAT optimum was found.

PV-Tuning (Malinovskii et al., NeurIPS'24 oral, arXiv 2405.14852) is the
natural candidate, because it exists precisely to replace STE: STE's gradient
is a biased surrogate that gets worse as the grid gets coarser, which is the
regime this project cares about (3-bit). The reference implementation is
vendored at `references/AQLM/` (`finetune.py`, `src/pv_optimizer.py`).

Verified on hardware: a 4-step ViT-B/16 run at `delta_decay=0,
max_code_change_per_step=1, p_every=1` reproduced `finetune_qat.py`'s trajectory
with a **bitwise identical** straight-through buffer and **identical integer
codes** on all 48 quantized layers. If you repeat that comparison, note that
checking a GPU-trained PV checkpoint against a CPU-recomputed
`fake_quantize_tensor` reference leaves a ~1.2e-7 residual: CUDA's float
division in `scale = absmax / qmax` differs from the CPU's by one ulp. That is
an artifact of the comparison, not of PV — compare the codes, or recompute the
reference on the same device.

PV-Tuning is *not* a PTQ method and is deliberately not implemented like one:
it produces a checkpoint, not an evaluation-time transform. There is no
`apply_pv_` counterpart to `apply_ptq_` / `apply_gptq_`.

The algorithm, ported to a uniform grid. The reference operates on AQLM
codebooks, where P = discrete code assignments and V = the continuous
codebooks and scales. On this project's symmetric, per-output-channel grid the
same split reads:

    B                     straight-through buffer (continuous, an nn.Parameter)
    s = absmax(B) / qmax  derived per channel, not a Parameter    <- V, implicit
    q = clamp(round(B/s), qmin, qmax)                             <- P, the codes
    W_hat = q * s         the weight actually used in the forward pass

* The **V-step** is the ordinary AdamW step already present in the training
  loop. Gradients reach `B` through the STE in `PVLinear.forward`, and reach
  every non-quantized parameter (norms, biases, patch embed, head) directly.
  The scale is derived rather than learned, which is what makes `apply_ptq_`
  a no-op on the saved checkpoint (see "Checkpoint contract" below); a learned
  LSQ-style scale would break that and is deliberately not implemented.
* The **P-step** is `pv_step_`, called after `optimizer.step()`. It re-derives
  the scale from the updated buffer and re-projects `B` onto the grid, subject
  to the three constraints that distinguish a P-step from plain re-rounding:
  `max_code_change_per_step` (only the highest-improvement fraction of entries
  may move), `trust_ratio` (a per-channel cap on how far W_hat may move in one
  step), and `temperature` (sample rather than take the top-k). Afterwards
  `delta_decay` pulls the buffer toward the grid:
  `B <- delta * W_hat + (1 - delta) * B`.

The degenerate corner is the point of the whole construction. At
`delta_decay=0, max_code_change_per_step=1.0, trust_ratio=None,
temperature=0.0`, `pv_step_` re-projects every entry, so `codes * scale` is
exactly `fake_quantize_tensor(B, bits, granularity)` and `PVLinear.forward`
reduces bitwise to `QATLinear.forward`. **The repository's existing QAT is a
corner of the PV knob grid**, which makes the comparison one-knob-at-a-time and
gives an exact regression test (`code/test/pv_tuning.py`). The equivalence is
stated at the point of a forward pass that follows a P-step, which is what the
training loop does when `p_every=1`; between an optimizer step and the next
P-step the codes are by design stale, that staleness being the mechanism.

Checkpoint contract. `settle_pv_` performs one unconstrained P-step with
`delta_decay=1`, after which `B == W_hat` and `disable_pv_` leaves a plain
`nn.Linear` whose weight is the dequantized grid point. Two consequences the
rest of the pipeline depends on:

* The saved state dict has exactly the FP key set (wrappers stripped, same as
  QAT saving), so `QV = PV_D - FP_D` is a well-defined state-dict subtraction.
* `apply_ptq_` is a **no-op** on a settled checkpoint. Because
  `qmax = 2^(b-1) - 1` while `qmin = -2^(b-1)`, the clamp is never binding on
  the negative side, so `|B|/s <= qmax` everywhere and `max|q| = qmax` is
  attained -- re-deriving the scale from W_hat therefore returns the same
  scale and re-rounding returns the same codes. Empirically this holds
  bit-exactly (zero weight delta over 2400 random tensors spanning 2/3/4/8
  bits x tensor/channel), not merely to within rounding. A settled PV
  checkpoint already *is* its own post-training-quantized model. Asserted in
  `code/test/pv_tuning.py` and recorded at evaluation time as
  `ptq_max_abs_weight_delta`.

The grid itself is imported from `src.quantization` and never re-derived, for
the same reason `src/gptq.py` imports it: a PV column and a QAT column of the
results table must differ only in how the optimum was found, never in where the
grid points are.
"""

import math
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

from src.quantization import dequantize_tensor, quantize_tensor


# =============================================================================
# PV wrapper (training-time: fake-quantize through explicitly stored codes)
# =============================================================================


class PVLinear(nn.Module):

    """
    Drop-in wrapper around nn.Linear holding an explicit (codes, scale) pair.

    Unlike QATLinear, which re-derives the fake-quantized weight from the
    latent weight on every forward, PVLinear keeps the discrete part as state:
    `self.linear.weight` is the straight-through buffer B, while `codes` and
    `scale` are buffers that only ever change inside `pv_step_`. That is the
    P/V split -- the continuous half moves every optimizer step, the discrete
    half moves only when a P-step decides it may.
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

        codes, scale = quantize_tensor(self.linear.weight.detach(), bits, granularity)

        self.register_buffer("codes", codes)
        self.register_buffer("scale", scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_q = dequantize_tensor(self.codes, self.scale)

        # NOTE: Straight-Through Estimator (STE), identical to QATLinear's. The
        # forward uses the grid point, .detach() sends the gradient to the
        # buffer B unchanged. The only difference from QATLinear is *which*
        # grid point: PVLinear uses the stored codes, which a constrained
        # P-step may have deliberately left stale.
        w = self.linear.weight + (w_q - self.linear.weight).detach()

        return F.linear(x, w, self.linear.bias)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


# =============================================================================
# Model-level enable / disable (mutate the model in-place, mirroring quantization.py)
# =============================================================================


def enable_pv_(
    model: nn.Module,
    bits: int,
    granularity: str,
    skip_modules: frozenset[str],
) -> nn.Module:

    """
    Replace all nn.Linear layers in the backbone with PVLinear wrappers.
    Skips modules whose names appear in *skip_modules*.
    """

    for name, module in model.named_children():
        if name in skip_modules:
            continue
        if isinstance(module, PVLinear):
            # Idempotent: avoid recursively wrapping module.linear on repeated calls.
            if module.bits != bits or module.granularity != granularity:
                raise ValueError(
                    f"PV already enabled on submodule {name!r} with "
                    f"{module.bits}-bit/{module.granularity}; "
                    f"requested {bits}-bit/{granularity}"
                )
            continue
        if isinstance(module, nn.Linear):
            setattr(model, name, PVLinear(module, bits, granularity))
        else:
            enable_pv_(module, bits, granularity, skip_modules=skip_modules)
    return model


def disable_pv_(model: nn.Module) -> None:

    """
    Remove PVLinear wrappers, restoring original nn.Linear layers.

    Call `settle_pv_` first if the resulting weights are meant to be the
    quantized ones -- this function does not touch weights, it only unwraps.
    """

    for name, module in model.named_children():

        if isinstance(module, PVLinear):
            setattr(model, name, module.linear)

        else:
            disable_pv_(module)


# =============================================================================
# The P-step
# =============================================================================


def _as_groups(tensor: torch.Tensor, granularity: str) -> torch.Tensor:

    """
    View a weight-shaped tensor as [num_groups, group_size], where a group is
    the set of entries sharing a scale: one row per output channel for
    "channel", the whole tensor for "tensor".
    """

    if granularity == "channel":
        return tensor.reshape(tensor.shape[0], -1)

    return tensor.reshape(1, -1)


def _select_by_budget(
    changed: torch.Tensor,
    improvement: torch.Tensor,
    max_code_change_per_step: float,
    temperature: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:

    """
    Pick which changed entries are allowed to move this step.

    Ranking is by squared-error improvement, so the budget is spent where the
    grid is currently worst-fitting. At *temperature* > 0 the top-k is replaced
    by Gumbel-top-k sampling on log-improvement, the reference implementation's
    `code_selection_temperature`: it keeps the same expected ranking but lets
    low-improvement entries move occasionally, which matters when a small
    budget would otherwise repeatedly re-select the same entries.
    """

    if max_code_change_per_step >= 1.0:
        return changed

    num_allowed = max(1, math.ceil(max_code_change_per_step * changed.numel()))
    if num_allowed >= changed.numel():
        return changed

    flat_changed = changed.reshape(-1)
    flat_improvement = improvement.reshape(-1)

    # Unchanged entries must never be selected: rank them below every candidate.
    keys = torch.where(
        flat_changed, flat_improvement, torch.full_like(flat_improvement, -1.0)
    )

    if temperature > 0.0:
        noise = -torch.log(
            -torch.log(
                torch.rand(
                    keys.shape,
                    device=keys.device,
                    dtype=keys.dtype,
                    generator=generator,
                ).clamp_min(1e-30)
            ).clamp_min(1e-30)
        )
        sampled = torch.log(flat_improvement.clamp_min(1e-30)) / temperature + noise
        keys = torch.where(
            flat_changed, sampled, torch.full_like(sampled, float("-inf"))
        )

    selected = torch.zeros_like(flat_changed)
    selected[torch.topk(keys, num_allowed).indices] = True

    return (selected & flat_changed).reshape(changed.shape)


def _apply_trust_ratio(
    accept: torch.Tensor,
    improvement: torch.Tensor,
    delta_w: torch.Tensor,
    w_hat_old: torch.Tensor,
    trust_ratio: float,
    granularity: str,
) -> torch.Tensor:

    """
    Shrink the accepted set until each group's move satisfies
    ||delta W_hat|| <= trust_ratio * ||W_hat||.

    Moves are ranked by improvement *per unit of squared move*, and the longest
    prefix of that order whose running cost fits the budget is kept. Because a
    cumulative sum is monotone, that prefix is exact -- no search needed.

    Ranking by the ratio rather than by raw improvement matters only in the
    pathological case: a single entry whose code jumps several levels can cost
    more than the whole budget, and under raw-improvement ordering it would sit
    at the front of the cumulative sum and veto every cheaper move behind it.
    In the regime the training loop actually runs in, one optimizer step moves
    a weight far less than one scale step, so every code that moves at all
    moves by exactly one level, every cost is exactly `scale**2`, and the two
    orderings coincide.
    """

    delta_sq = delta_w**2

    # improvement/cost, with non-accepted entries ranked below every candidate.
    ratio = torch.where(
        accept,
        improvement / delta_sq.clamp_min(torch.finfo(delta_sq.dtype).tiny),
        torch.full_like(improvement, -1.0),
    )

    group_ratio = _as_groups(ratio, granularity)
    group_delta_sq = _as_groups(delta_sq, granularity)
    group_budget_sq = (trust_ratio**2) * _as_groups(w_hat_old**2, granularity).sum(
        dim=1, keepdim=True
    )

    order = torch.argsort(group_ratio, dim=1, descending=True)
    cumulative = torch.cumsum(torch.gather(group_delta_sq, 1, order), dim=1)
    keep_sorted = (cumulative <= group_budget_sq) & (
        torch.gather(group_ratio, 1, order) >= 0.0
    )

    keep = torch.zeros_like(keep_sorted).scatter_(1, order, keep_sorted)

    return keep.reshape(accept.shape) & accept


def pv_step_(
    model: nn.Module,
    delta_decay: float,
    max_code_change_per_step: float,
    trust_ratio: Optional[float] = None,
    temperature: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, float]:

    """
    Perform one P-step over every PVLinear in *model*, in-place.

    Call after `optimizer.step()`. The scale is re-derived from the updated
    buffer first (that is the V half landing), then the codes are re-projected
    under the budget and trust constraints, then `delta_decay` pulls the buffer
    toward the grid.

    Returns aggregate statistics for logging: how much of the code budget was
    actually spent and how much squared error the step bought.
    """

    if not 0.0 <= delta_decay <= 1.0:
        raise ValueError(f"delta_decay expected in [0, 1], got {delta_decay}")
    if not 0.0 < max_code_change_per_step <= 1.0:
        raise ValueError(
            f"max_code_change_per_step expected in (0, 1], got {max_code_change_per_step}"
        )
    if trust_ratio is not None and trust_ratio <= 0.0:
        raise ValueError(f"trust_ratio expected positive or None, got {trust_ratio}")
    if temperature < 0.0:
        raise ValueError(f"temperature expected non-negative, got {temperature}")

    num_codes = 0
    num_changed = 0
    num_layers = 0
    total_improvement = 0.0

    with torch.no_grad():
        for module in model.modules():

            if not isinstance(module, PVLinear):
                continue

            buffer = module.linear.weight
            codes_old = module.codes.to(buffer.dtype)

            # The unconstrained projection: both the re-derived scale (V) and
            # the target codes (P) come from the same call the RTN path uses.
            codes_target, scale = quantize_tensor(
                buffer, module.bits, module.granularity
            )
            codes_target = codes_target.to(buffer.dtype)

            # Improvement is measured under the NEW scale: the scale moves
            # every step regardless (it is continuous), only the codes are
            # rationed. Non-negative by construction -- codes_target is the
            # per-entry minimizer of the rounding error.
            w_hat_old = codes_old * scale
            improvement = ((buffer - w_hat_old) ** 2 - (buffer - codes_target * scale) ** 2).clamp_min(0.0)

            changed = codes_target != codes_old

            accept = _select_by_budget(
                changed,
                improvement,
                max_code_change_per_step,
                temperature,
                generator,
            )

            if trust_ratio is not None:
                accept = _apply_trust_ratio(
                    accept,
                    improvement,
                    (torch.where(accept, codes_target, codes_old) - codes_old) * scale,
                    w_hat_old,
                    trust_ratio,
                    module.granularity,
                )

            codes_new = torch.where(accept, codes_target, codes_old)

            module.codes.copy_(codes_new.round().to(module.codes.dtype))
            module.scale.copy_(scale)

            if delta_decay > 0.0:
                buffer.copy_(
                    delta_decay * dequantize_tensor(module.codes, module.scale)
                    + (1.0 - delta_decay) * buffer
                )

            num_codes += accept.numel()
            num_changed += int(accept.sum().item())
            total_improvement += float(improvement[accept].sum().item())
            num_layers += 1

    return {
        "pv_layers": float(num_layers),
        "pv_codes": float(num_codes),
        "pv_codes_changed": float(num_changed),
        "pv_code_change_fraction": (num_changed / num_codes) if num_codes else 0.0,
        "pv_squared_error_reduction": total_improvement,
    }


def settle_pv_(model: nn.Module) -> Dict[str, float]:

    """
    Collapse every PVLinear onto the grid: one unconstrained P-step with
    `delta_decay=1`, after which the straight-through buffer equals the
    dequantized grid point.

    This is what makes a saved PV checkpoint self-consistent -- it is called
    before every evaluation and before every save, so what is written to disk
    is the quantized model rather than a latent that merely rounds to it.
    """

    return pv_step_(
        model,
        delta_decay=1.0,
        max_code_change_per_step=1.0,
        trust_ratio=None,
        temperature=0.0,
    )


# =============================================================================
# Sidecar state (the part of PV that the FP-shaped checkpoint cannot hold)
# =============================================================================


def pv_sidecar_state(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:

    """
    Snapshot `{codes, scale, latent}` for every PVLinear, keyed by module name.

    Call this *before* `settle_pv_`. The codes and scale are recoverable from a
    settled checkpoint by re-quantizing it, but the pre-settle straight-through
    buffer is not, and it is the object a latent-QV ablation (`QV = B - FP`,
    the exact analogue of what QAT checkpoints store) would need. Saving it
    costs one extra full-precision copy on disk and makes that ablation
    possible without re-running any training.
    """

    state: Dict[str, Dict[str, torch.Tensor]] = {}

    for name, module in model.named_modules():
        if not isinstance(module, PVLinear):
            continue
        state[name] = {
            "codes": module.codes.detach().cpu().clone(),
            "scale": module.scale.detach().cpu().clone(),
            "latent": module.linear.weight.detach().cpu().clone(),
        }

    return state


def pv_module_names(model: nn.Module) -> List[str]:

    """Fully-qualified names of every PVLinear, for the enable-time audit."""

    return [
        name for name, module in model.named_modules() if isinstance(module, PVLinear)
    ]


# =============================================================================
# Path fragment
# =============================================================================


def pv_path_frag(
    bits: int,
    granularity: str,
    skip_modules,
    delta_decay: float,
    max_code_change_per_step: float,
    trust_ratio: Optional[float],
    p_every: int,
    temperature: float,
) -> str:

    """
    Build the `pv=...` path fragment, occupying the slot QAT's paths give to
    `qat=...`.

    Every knob here is result-defining, so every knob is in the fragment (there
    is no PV analogue of GPTQ's `block_size`, which is excluded from its
    fragment precisely because it changes only wall-clock). This lives in the
    library rather than being re-spelled per script so that the finetuning
    script, both baseline evaluators, the transfer phase, and the plotting
    scripts cannot drift apart -- a fragment disagreement silently splits one
    sweep across two directory trees, which is the classic failure mode in this
    repository's path grammar.

    Floats are normalized through `float()` so that `0` and `0.0` cannot
    produce two different directories for the same run.
    """

    skip_modules_sorted = sorted(skip_modules)
    skip_tag = "-".join(skip_modules_sorted) if skip_modules_sorted else "none"
    trust_tag = "none" if trust_ratio is None else f"{float(trust_ratio)}"

    return (
        f"pv=bits={int(bits)}"
        f"_gran={granularity}"
        f"_skip={skip_tag}"
        f"_delta={float(delta_decay)}"
        f"_tau={float(max_code_change_per_step)}"
        f"_trust={trust_tag}"
        f"_pevery={int(p_every)}"
        f"_temp={float(temperature)}"
    )
