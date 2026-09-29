"""CPU-only tests for PV-Tuning on the project's uniform quantization grid.

These tests require no checkpoint, dataset, or GPU. Two of them are contracts
the rest of the pipeline rests on, and nothing downstream should be run if
either fails:

* `test_degenerates_to_qat` -- at delta_decay=0, max_code_change=1, PVLinear is
  bitwise QATLinear. This is what makes the PV column and the QAT column of the
  results table differ in one knob rather than in an unknown number of them.
* `test_settled_checkpoint_survives_ptq` -- `apply_ptq_` recovers a settled PV
  checkpoint's integer codes exactly. This is what makes a settled checkpoint
  already-quantized, hence what makes `QV = PV_D - FP_D` mean what the
  transfer phase assumes it means.

Run: uv run --active python code/test/pv_tuning.py
"""

import copy
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from src.pv_tuning import (
    PVLinear,
    disable_pv_,
    enable_pv_,
    pv_module_names,
    pv_sidecar_state,
    pv_step_,
    settle_pv_,
)
from src.quantization import (
    QATLinear,
    apply_ptq_,
    fake_quantize_tensor,
    quantize_tensor,
)

BITS = 3
GRANULARITY = "channel"


class TinyNet(nn.Module):

    """Two-layer backbone plus a head, mirroring the skip_modules=[head] setup."""

    def __init__(self, in_features=12, hidden=16, num_classes=5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x):
        return self.head(self.norm(self.block(x)))


def _perturb_(model, scale=0.05, seed=0):
    """Stand in for a V-step: move every buffer the way AdamW would."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(
                torch.randn(
                    parameter.shape, generator=generator, dtype=parameter.dtype
                )
                * scale
            )


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_degenerates_to_qat() -> None:
    torch.manual_seed(0)

    original = nn.Linear(12, 16)
    pv = PVLinear(copy.deepcopy(original), BITS, GRANULARITY)
    qat = QATLinear(copy.deepcopy(original), BITS, GRANULARITY)

    x = torch.randn(4, 12)

    # Freshly wrapped: PVLinear's codes come from the same quantize_tensor call.
    assert torch.equal(pv(x), qat(x))

    # After a V-step, a full-budget P-step must restore the equality -- this is
    # the state the training loop is in whenever p_every=1.
    for step in range(3):
        with torch.no_grad():
            update = torch.randn(original.weight.shape) * 0.1
            pv.linear.weight.add_(update)
            qat.linear.weight.add_(update)

        pv_step_(pv, delta_decay=0.0, max_code_change_per_step=1.0)

        assert torch.equal(pv.linear.weight, qat.linear.weight), step
        assert torch.equal(pv(x), qat(x)), step

    # And the STE gradient is the same one QAT sees.
    pv.zero_grad()
    qat.zero_grad()
    pv(x).sum().backward()
    qat(x).sum().backward()
    assert torch.equal(pv.linear.weight.grad, qat.linear.weight.grad)

    print("PASS  delta_decay=0, max_code_change=1 is bitwise QATLinear")


def test_settled_checkpoint_survives_ptq() -> None:
    torch.manual_seed(1)

    model = TinyNet()
    enable_pv_(model, BITS, GRANULARITY, skip_modules=frozenset({"head"}))

    # Drive it through a few constrained steps so the codes are genuinely
    # stale before settling -- settling a fresh model would prove nothing.
    for step in range(4):
        _perturb_(model, scale=0.1, seed=step)
        pv_step_(
            model,
            delta_decay=0.9,
            max_code_change_per_step=0.05,
            trust_ratio=0.5,
        )

    settle_pv_(model)

    codes_before = {
        name: module.codes.clone()
        for name, module in model.named_modules()
        if isinstance(module, PVLinear)
    }

    disable_pv_(model)
    weights_before = {
        name: module.weight.detach().clone()
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }

    quantized = apply_ptq_(
        model, bits=BITS, granularity=GRANULARITY, skip_modules=frozenset({"head"})
    )
    assert quantized == ["block.0", "block.2"], quantized

    max_abs_delta = 0.0
    for name in quantized:
        module = dict(model.named_modules())[name]

        # The integer codes must round-trip exactly. This is the load-bearing
        # half: it is what makes the checkpoint a grid point rather than
        # something that merely rounds to one.
        codes_after, _ = quantize_tensor(module.weight.detach(), BITS, GRANULARITY)
        assert torch.equal(codes_after, codes_before[name]), name

        max_abs_delta = max(
            max_abs_delta,
            float((module.weight.detach() - weights_before[name]).abs().max()),
        )

    # The weights themselves round-trip up to the floating-point scale
    # round-trip fl(fl(qmax*s)/qmax), i.e. a couple of ulps at most.
    reference = max(float(w.abs().max()) for w in weights_before.values())
    assert max_abs_delta <= 1e-6 * reference, (max_abs_delta, reference)

    print(
        f"PASS  apply_ptq_ recovers a settled PV checkpoint "
        f"(codes exact, max |dw| = {max_abs_delta:.3e})"
    )


# ---------------------------------------------------------------------------
# Constraint tests
# ---------------------------------------------------------------------------


def test_code_change_budget_is_respected() -> None:
    torch.manual_seed(2)

    layer = PVLinear(nn.Linear(32, 24), BITS, GRANULARITY)
    total = layer.codes.numel()

    for tau in (0.01, 0.1, 0.5):
        fresh = PVLinear(copy.deepcopy(layer.linear), BITS, GRANULARITY)
        with torch.no_grad():
            fresh.linear.weight.add_(torch.randn(fresh.linear.weight.shape) * 0.5)

        before = fresh.codes.clone()
        pv_step_(fresh, delta_decay=0.0, max_code_change_per_step=tau)
        changed = int((fresh.codes != before).sum())

        budget = math.ceil(tau * total)
        assert changed <= budget, (tau, changed, budget)
        assert changed > 0, tau

    print("PASS  max_code_change_per_step caps the number of moved codes")


def test_trust_ratio_is_respected() -> None:
    torch.manual_seed(3)

    layer = PVLinear(nn.Linear(32, 24), BITS, GRANULARITY)
    with torch.no_grad():
        layer.linear.weight.add_(torch.randn(layer.linear.weight.shape) * 0.8)

    codes_old = layer.codes.clone()

    # One code move costs exactly one scale step, and a channel's ||W_hat|| is
    # roughly sqrt(fan_in) * |q|_avg * scale, so the ratio has to be loose
    # enough to buy at least one move or the test would pass vacuously.
    trust_ratio = 0.25

    pv_step_(
        layer,
        delta_decay=0.0,
        max_code_change_per_step=1.0,
        trust_ratio=trust_ratio,
    )

    scale = layer.scale
    w_hat_old = codes_old.float() * scale
    w_hat_new = layer.codes.float() * scale

    move = (w_hat_new - w_hat_old).norm(dim=1)
    budget = trust_ratio * w_hat_old.norm(dim=1)
    assert torch.all(move <= budget + 1e-9), (move.max(), budget.min())

    # The constraint must actually bind, or this test proves nothing.
    assert not torch.equal(layer.codes, codes_old)
    unconstrained = PVLinear(copy.deepcopy(layer.linear), BITS, GRANULARITY)
    assert int((unconstrained.codes != codes_old).sum()) > int(
        (layer.codes != codes_old).sum()
    )

    print("PASS  trust_ratio caps the per-channel move of W_hat")


def test_delta_decay_endpoints() -> None:
    torch.manual_seed(4)

    layer = PVLinear(nn.Linear(20, 16), BITS, GRANULARITY)
    with torch.no_grad():
        layer.linear.weight.add_(torch.randn(layer.linear.weight.shape) * 0.3)
    buffer_before = layer.linear.weight.detach().clone()

    # delta_decay=0: the buffer is untouched (pure straight-through).
    zero = copy.deepcopy(layer)
    pv_step_(zero, delta_decay=0.0, max_code_change_per_step=1.0)
    assert torch.equal(zero.linear.weight.detach(), buffer_before)

    # delta_decay=1: the buffer collapses onto the grid point.
    one = copy.deepcopy(layer)
    pv_step_(one, delta_decay=1.0, max_code_change_per_step=1.0)
    w_hat = one.codes.float() * one.scale
    assert torch.equal(one.linear.weight.detach(), w_hat)

    # ... which is exactly what settle_pv_ does, and it is the RTN projection.
    settled = copy.deepcopy(layer)
    settle_pv_(settled)
    assert torch.equal(
        settled.linear.weight.detach(),
        fake_quantize_tensor(buffer_before, BITS, GRANULARITY),
    )

    print("PASS  delta_decay endpoints, and settle_pv_ == RTN projection")


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_key_set_matches_fp() -> None:
    torch.manual_seed(5)

    model = TinyNet()
    fp_keys = list(model.state_dict().keys())

    enable_pv_(model, BITS, GRANULARITY, skip_modules=frozenset({"head"}))
    assert list(model.state_dict().keys()) != fp_keys

    sidecar = pv_sidecar_state(model)
    assert set(sidecar) == {"block.0", "block.2"}
    assert set(sidecar["block.0"]) == {"codes", "scale", "latent"}

    settle_pv_(model)
    disable_pv_(model)

    assert list(model.state_dict().keys()) == fp_keys

    print("PASS  settled + unwrapped PV state dict has exactly the FP key set")


def test_skip_modules_and_idempotence() -> None:
    torch.manual_seed(6)

    model = TinyNet()
    enable_pv_(model, BITS, GRANULARITY, skip_modules=frozenset({"head"}))

    assert pv_module_names(model) == ["block.0", "block.2"]
    assert isinstance(model.head, nn.Linear) and not isinstance(model.head, PVLinear)

    # Re-enabling with the same config is a no-op, not a double wrap.
    enable_pv_(model, BITS, GRANULARITY, skip_modules=frozenset({"head"}))
    assert pv_module_names(model) == ["block.0", "block.2"]

    try:
        enable_pv_(model, BITS + 1, GRANULARITY, skip_modules=frozenset({"head"}))
    except ValueError:
        pass
    else:
        raise AssertionError("expected a ValueError on a config mismatch")

    print("PASS  skip_modules prunes the head; enable_pv_ is idempotent")


def test_gradients_and_error_reduction() -> None:
    torch.manual_seed(7)

    model = TinyNet()
    enable_pv_(model, BITS, GRANULARITY, skip_modules=frozenset({"head"}))

    logits = model(torch.randn(8, 12))
    logits.sum().backward()

    # The V-step reaches the buffers and every non-quantized parameter.
    assert model.block[0].linear.weight.grad is not None
    assert model.block[0].linear.weight.grad.abs().sum() > 0
    assert model.norm.weight.grad is not None
    assert model.head.weight.grad is not None

    _perturb_(model, scale=0.4, seed=11)

    def buffer_to_grid_distance():
        return sum(
            float(
                (module.linear.weight.detach() - module.codes.float() * module.scale)
                .pow(2)
                .sum()
            )
            for module in model.modules()
            if isinstance(module, PVLinear)
        )

    before = buffer_to_grid_distance()
    stats = pv_step_(model, delta_decay=0.0, max_code_change_per_step=1.0)
    after = buffer_to_grid_distance()

    assert after <= before, (before, after)
    assert stats["pv_layers"] == 2.0
    assert stats["pv_squared_error_reduction"] >= 0.0
    assert 0.0 <= stats["pv_code_change_fraction"] <= 1.0

    print("PASS  gradients reach buffers and non-quantized params; P-step reduces error")


def main() -> None:
    test_degenerates_to_qat()
    test_settled_checkpoint_survives_ptq()
    test_code_change_budget_is_respected()
    test_trust_ratio_is_respected()
    test_delta_decay_endpoints()
    test_key_set_matches_fp()
    test_skip_modules_and_idempotence()
    test_gradients_and_error_reduction()
    print("\nALL PV-TUNING TESTS PASSED")


if __name__ == "__main__":
    main()
