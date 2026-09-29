"""GPTQ: Hessian-compensated post-training quantization for nn.Linear weights.

Why this module exists. Reviewers objected that the paper's only PTQ baseline
is vanilla round-to-nearest (`apply_ptq_`). GPTQ (Frantar et al., ICLR 2023)
is the one reviewer-named strong-PTQ method that ports cleanly to ViT/BERT,
because it is architecture-agnostic: it quantizes each nn.Linear weight matrix
independently, using second-order information about that layer's inputs. This
module is a from-scratch implementation written against the official reference
at `references/gptq/` (IST-DASLab), used by `evaluate_fp_gptq.py` (the GPTQ(FP)
baseline, Task 1) and `qv_transfer_gptq.py` (GPTQ on FP + alpha*QV, Task 2).

The algorithm. For a layer with weight W and calibration inputs X, RTN rounds
each weight independently, minimizing weight-space error ||W - What||. GPTQ
instead minimizes the layer-output error ||WX - What X||, whose Hessian with
respect to any weight row is H = 2 X X^T (shared across rows). It quantizes W
column by column on a *fixed* grid; after quantizing column j it updates the
not-yet-quantized columns by the error feedback e = (w_j - q_j) / [C]_jj,
W -= e * C_j, where C is the upper Cholesky factor of H^-1 — absorbing each
column's rounding error into the remaining columns instead of accumulating it.
Columns are processed in lazy blocks (`block_size`) so the rank-1 updates
batch into one matmul per block: block size changes wall-clock only, never the
result. `actorder` (official `--act-order`) processes columns by decreasing
diag(H) so high-salience columns are quantized while the most freedom to
compensate remains.

Calibration is sequential, matching the official pipeline at its
`--true-sequential` granularity: layers are quantized one at a time in forward
execution order, and each layer's Hessian is accumulated by replaying the
calibration batches through the full model *after* its predecessors were
quantized in place. Each layer therefore calibrates against the activation
distribution it will actually see at inference, compensating upstream
quantization drift — the propagation the official implementation performs via
cached per-block replays. For a ViT block (qkv -> proj -> fc1 -> fc2, one
Linear per official subset) per-layer sequencing is exactly the official
`--true-sequential` subset structure.

Deviations from the official implementation, and their impact:

* Quantization grid. The official `Quantizer` is a pluggable component (the
  solver only ever calls `quantize(w, scale, zero, maxq)`); its symmetric mode
  uses scale = 2*absmax/(2^b - 1) with an offset zero-point, and the official
  *default* is asymmetric (`--sym` is opt-in). We plug in the project's grid
  from `src.quantization` instead: scale = absmax/(2^(b-1) - 1), true zero —
  the exact grid `apply_ptq_` uses. This is the only number-changing deviation
  (sub-percent, direction-neutral: finer step vs exact-extreme trade-off, 8
  levels either way) and it is required: with identical grids the RTN and GPTQ
  columns of the results table differ *only* in error compensation, and GPTQ
  with H proportional to I degenerates bit-exactly to `apply_ptq_` (see
  `code/test/gptq.py`).
* Replay strategy. Official caches each transformer block's inputs and replays
  only the block; we replay the full model with an early-abort hook. The
  activations reaching the target layer are computed by the same ops in the
  same order, so results are identical; the cost is redundant prefix compute.
  Bought: model-agnosticism — no per-family block-list assumptions (Swin's
  hierarchical stages would otherwise need special-casing).
* Not ported: `groupsize`/`static_groups` (official default is -1 anyway; the
  project's granularity taxonomy is tensor/channel), Conv2d/Conv1D paths
  (`apply_ptq_` quantizes only nn.Linear, and both columns must quantize the
  identical layer set), and `add_batch`'s `out` argument (DEBUG-only).
* Write-back via `weight.copy_()` under no_grad instead of `.data =`:
  identical values, but preserves tensor identity and validates shape —
  `apply_ptq_` parity.
* TF32 is disabled during the solve and restored afterwards; official disables
  it globally at import, which a library module must not do. Evaluation
  numerics are shared by all baseline columns either way.
* Cholesky robustness: official crashes on a non-positive-definite H; we retry
  with 10x damping (up to `_MAX_DAMP_RETRIES`) and finally fall back to RTN
  for that layer, loudly logged. Bit-identical whenever the first attempt
  succeeds (the normal case); activates only where official has no defined
  behavior, and pulls the layer toward RTN — it can never inflate GPTQ.
"""

import logging
import math
import time

import torch
from torch import nn

from typing import Any, Callable, Dict, List, Optional, Tuple

from src.quantization import fake_quantize_tensor, quantize_tensor

log = logging.getLogger(__name__)

_MAX_DAMP_RETRIES = 3


def gptq_path_frag(
    bits: int,
    granularity: str,
    skip_modules,
    num_calib_batches: int,
    percdamp: float,
    actorder: bool,
) -> str:
    """
    The `gptq=` path fragment, in one place.

    Historical note: this fragment predates the helper and is spelled by hand at
    eight sites (both `evaluate_fp_gptq.py` twins, `qv_transfer_gptq.py`,
    `compute_gptq_checkpoints.py`, `qv_transfer_gptqv.py`, and three
    visualizations). Those are left alone — they work, and rewriting them is a
    separate cleanup — but new callers use this, following `awq_path_frag` in
    `src/awq.py` and `pv_path_frag` in `src/pv_tuning.py`.

    Carries every knob that changes the numbers and nothing that does not:
    `block_size` is deliberately excluded, being result-invariant solver
    batching that would otherwise split identical results across paths. Booleans
    render as Python `str(bool)` ("True"/"False"), matching what is on disk.
    """
    skip_modules_sorted = sorted(skip_modules)
    skip_tag = "-".join(skip_modules_sorted) if skip_modules_sorted else "none"
    return (
        f"gptq=bits={bits}_gran={granularity}_skip={skip_tag}"
        f"_ncal={num_calib_batches}_percdamp={percdamp}_actorder={actorder}"
    )


# =============================================================================
# Per-layer solver
# =============================================================================


class GPTQ:

    """
    GPTQ solver for a single nn.Linear layer.

    Usage:
        solver = GPTQ(layer, name="blocks.0.attn.qkv")
        solver.add_batch(inp)   # the *layer's* inputs, typically via a hook
        solver.quantize_(bits=3, granularity="channel")
        solver.free()

    `H` and `nsamples` are public so tests can inject a synthetic Hessian
    (e.g. H = I, which must reproduce plain RTN exactly).
    """

    def __init__(self, layer: nn.Linear, name: str = "?"):
        if not isinstance(layer, nn.Linear):
            raise TypeError(
                f"GPTQ supports nn.Linear only, got {type(layer).__name__}"
            )
        self.layer = layer
        self.name = name
        self.dev = layer.weight.device
        self.rows = layer.out_features
        self.columns = layer.in_features
        self.H: Optional[torch.Tensor] = torch.zeros(
            (self.columns, self.columns), device=self.dev
        )
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor) -> None:
        """
        Accumulate the layer Hessian H = 2 E[x x^T] from a batch of layer inputs.

        *inp* is whatever the layer's forward receives: (..., in_features),
        typically (batch, tokens, in_features) for transformers. The streaming
        renormalization matches the official `add_batch` verbatim — `nsamples`
        counts leading-dim elements (sequences/images, not tokens), and H is
        kept as a running average so its overall scale is independent of how
        many batches were fed. The absolute scale of H cancels inside
        `quantize_` (damping is relative to mean(diag(H))); only conditioning
        matters.
        """
        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        inp = inp.reshape((-1, inp.shape[-1])).t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def _cholesky_inverse_factor(
        self, H: torch.Tensor, percdamp: float
    ) -> Optional[torch.Tensor]:
        """
        Upper Cholesky factor of H^-1, with escalating-damp retries.

        Returns None when every attempt fails (caller falls back to RTN).
        """
        diag_idx = torch.arange(self.columns, device=H.device)
        damp = percdamp
        for attempt in range(1 + _MAX_DAMP_RETRIES):
            H_work = H.clone()
            H_work[diag_idx, diag_idx] += damp * torch.mean(torch.diag(H))
            try:
                H_work = torch.linalg.cholesky(H_work)
                H_work = torch.cholesky_inverse(H_work)
                H_work = torch.linalg.cholesky(H_work, upper=True)
                return H_work
            except torch.linalg.LinAlgError:
                log.warning(
                    "GPTQ %s: Cholesky failed at percdamp=%g "
                    "(attempt %d/%d); retrying with 10x damping",
                    self.name, damp, attempt + 1, 1 + _MAX_DAMP_RETRIES,
                )
                damp *= 10.0
        return None

    def quantize_(
        self,
        bits: int,
        granularity: str,
        block_size: int = 128,
        percdamp: float = 0.01,
        actorder: bool = False,
    ) -> float:
        """
        Run GPTQ column-wise quantization with error feedback, in-place.

        Mutates `self.layer.weight` (values only; the tensor object survives).
        Returns the accumulated proxy loss sum((w - q)^2 / d^2) / 2 (NaN when
        the Cholesky fallback demoted the layer to RTN). Operation order
        replicates the official `fasterquant` exactly; the quantization grid
        is the project's own (see module docstring).
        """
        if self.H is None:
            raise RuntimeError(
                f"GPTQ {self.name}: quantize_ called after free() or twice"
            )
        if self.nsamples == 0:
            # Refuse rather than proceed: an all-zero H marks *every* column
            # dead, and the dead-column rule would zero the entire weight.
            raise RuntimeError(
                f"GPTQ {self.name}: no calibration data (nsamples=0); "
                "feed add_batch or inject H for testing"
            )

        tick = time.time()

        W = self.layer.weight.data.clone().float()

        # Grid from the project's own primitives — the exact scales apply_ptq_
        # would use — computed once from the original W (before dead-column
        # zeroing, matching official find_params placement) and held fixed
        # while error feedback mutates columns. Never re-derive a scale from a
        # single column: that would silently change the grid.
        _, scale = quantize_tensor(W, bits, granularity)
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
        col_scale = scale.squeeze(1) if granularity == "channel" else scale

        H = self.H
        self.H = None

        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        hinv = self._cholesky_inverse_factor(H, percdamp)
        if hinv is None:
            log.error(
                "GPTQ %s: Cholesky failed after %d damping escalations; "
                "falling back to RTN for this layer",
                self.name, _MAX_DAMP_RETRIES,
            )
            with torch.no_grad():
                self.layer.weight.copy_(
                    fake_quantize_tensor(self.layer.weight, bits, granularity)
                )
            return float("nan")

        losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        for i1 in range(0, self.columns, block_size):
            i2 = min(i1 + block_size, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            losses1 = torch.zeros_like(W1)
            hinv1 = hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = hinv1[i, i]

                q = (w / col_scale).round().clamp(qmin, qmax) * col_scale

                Q1[:, i] = q
                losses1[:, i] = (w - q) ** 2 / d ** 2

                err = (w - q) / d
                W1[:, i:] -= err.unsqueeze(1).matmul(hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err

            Q[:, i1:i2] = Q1
            losses[:, i1:i2] = losses1 / 2

            W[:, i2:] -= Err1.matmul(hinv[i1:i2, i2:])

        if actorder:
            Q = Q[:, invperm]

        with torch.no_grad():
            self.layer.weight.copy_(Q.to(self.layer.weight.dtype))

        total_loss = losses.sum().item()
        log.info(
            "GPTQ %s: time=%.2fs, proxy_loss=%.6f",
            self.name, time.time() - tick, total_loss,
        )
        return total_loss

    def free(self) -> None:
        self.H = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# Model traversal, calibration plumbing
# =============================================================================


class _StopForward(Exception):
    """Raised by the capture hook to abort the forward pass early once the
    target layer's input has been recorded (official `Catcher` trick)."""


def _find_linear_layers(
    model: nn.Module,
    skip_modules: frozenset[str],
    _prefix: str = "",
) -> Dict[str, nn.Linear]:
    """
    Recursively find all nn.Linear layers, skipping children whose attribute
    name appears in *skip_modules* at any depth — the same traversal contract
    as `apply_ptq_`.
    """
    result: Dict[str, nn.Linear] = {}
    for name, module in model.named_children():
        if name in skip_modules:
            continue
        full = f"{_prefix}{name}"
        if isinstance(module, nn.Linear):
            result[full] = module
        else:
            result.update(
                _find_linear_layers(module, skip_modules, _prefix=full + ".")
            )
    return result


def _default_forward_fn(model: nn.Module, batch: Any, device) -> None:
    """
    Vision batch convention: dict batches carry 'images', tuple/list batches
    put inputs first, bare tensors are the inputs. Text callers pass their own
    `forward_fn` (e.g. model(input_ids=..., attention_mask=...)).
    """
    if isinstance(batch, dict):
        inputs = batch["images"]
    elif isinstance(batch, (list, tuple)):
        inputs = batch[0]
    else:
        inputs = batch
    model(inputs.to(device))


def _discover_execution_order(
    model: nn.Module,
    layers: Dict[str, nn.Linear],
    batch: Any,
    device,
    forward_fn: Callable[[nn.Module, Any, Any], None],
) -> Tuple[List[str], Dict[str, int], List[str]]:
    """
    Run one probe batch and record the order in which the target Linears first
    fire. Sequential quantization must follow *execution* order, not
    named_children order — attribute declaration order is not a dependency
    order. Also counts fires per layer: a layer firing more than once per
    forward (a shared module) cannot use the early-abort hook, since aborting
    at its first call would drop the later call sites' inputs from H.

    Returns (execution_order, fire_counts, never_fired).
    """
    order: List[str] = []
    counts: Dict[str, int] = {name: 0 for name in layers}
    handles = []

    def make_hook(name: str):
        def hook(module, inp, out):
            if counts[name] == 0:
                order.append(name)
            counts[name] += 1
        return hook

    for name, module in layers.items():
        handles.append(module.register_forward_hook(make_hook(name)))
    try:
        forward_fn(model, batch, device)
    finally:
        for h in handles:
            h.remove()

    never_fired = [name for name in layers if counts[name] == 0]
    return order, counts, never_fired


# =============================================================================
# Model-level GPTQ entry point (drop-in alternative to apply_ptq_)
# =============================================================================


def apply_gptq_(
    model: nn.Module,
    bits: int,
    granularity: str,
    skip_modules: frozenset[str],
    calib_loader,
    device,
    num_calib_batches: int = 4,
    percdamp: float = 0.01,
    actorder: bool = False,
    block_size: int = 128,
    forward_fn: Optional[Callable[[nn.Module, Any, Any], None]] = None,
) -> List[str]:
    """
    Apply GPTQ post-training quantization in-place, sequentially.

    Mirrors `apply_ptq_`'s contract: fake-quantizes the weight of every
    nn.Linear outside *skip_modules*, replaces no modules (pre-registered
    forward hooks survive — the property 003_qat_transfer_activ relies on),
    and returns the fully-qualified names of the quantized layers, here in
    processing (execution) order.

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
        percdamp: Hessian damping as a fraction of mean(diag(H)).
        actorder: official `--act-order` (quantize columns by decreasing
            diag(H)).
        block_size: lazy-update batching of the solver loop; result-invariant.
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
            "calib_loader yielded no batches; GPTQ needs calibration data"
        )

    # Official disables TF32 globally at import for Hessian precision; a
    # library function scopes it instead.
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
                solver = GPTQ(layer, name=layer_name)
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
                    block_size=block_size,
                    percdamp=percdamp,
                    actorder=actorder,
                )
                solver.free()
                quantized.append(layer_name)

            for layer_name in never_fired:
                # Unreachable for our ViTs (every Linear fires); robustness
                # for exotic models. RTN equals GPTQ with H proportional to I.
                log.warning(
                    "GPTQ: layer %s never fired during calibration; "
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
        "GPTQ: quantized %d layers (%d via never-fired RTN fallback)",
        len(quantized), num_fallback,
    )
    return quantized
