"""CPU unit tests for src/gptq.py (no checkpoints, no GPU, no datasets).

Validates the two GPTQ acceptance criteria plus the apply_gptq_ contract:
  * H = I (and damping -> inf) degenerates bit-exactly to RTN, i.e. to
    fake_quantize_tensor / apply_ptq_ — this identity holds only because GPTQ
    shares the project's quantization grid,
  * on correlated inputs GPTQ's layer-*output* error is strictly below RTN's
    (the objective GPTQ minimizes; weight-space error would be the wrong
    metric — GPTQ typically makes it larger),
  * block_size is result-invariant (lazy solver batching, not a quantization
    choice),
  * apply_gptq_: skip_modules honored, module identity and pre-registered
    forward hooks survive (the 003_qat_transfer_activ property), quantized
    weights lie on the *original* per-row grid, layers are processed in
    execution order (not attribute-declaration order), never-fired layers fall
    back to RTN, and shared (multi-fire) layers take the no-early-abort path.

Run: uv run --active python code/test/gptq.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from src.gptq import (
    GPTQ,
    apply_gptq_,
    _discover_execution_order,
    _find_linear_layers,
)
from src.quantization import fake_quantize_tensor, quantize_tensor

BITS = 3
GRANULARITIES = ("tensor", "channel")


def _correlated_inputs(n, d, seed):
    # I.i.d. Gaussians give H ~ I, where GPTQ provably equals RTN and the
    # margin vanishes; a random mixing matrix makes H anisotropic, which is
    # where error compensation actually has room to work.
    g = torch.Generator().manual_seed(seed)
    mix = torch.randn(d, d, generator=g)
    z = torch.randn(n, d, generator=g)
    return z @ mix


def _output_error(x, w_ref, w_quant):
    return ((x @ w_quant.t()) - (x @ w_ref.t())).pow(2).sum().item()


def _assert_on_grid(w_new, w_orig, bits, granularity, name):
    # The grid is defined by the *original* weight's scales; GPTQ output is
    # not an RTN fixed point (error feedback shifts row maxima), so
    # re-applying fake_quantize_tensor would be the wrong check.
    _, scale = quantize_tensor(w_orig, bits, granularity)
    levels = w_new / scale
    assert torch.allclose(levels, levels.round(), atol=1e-3), name
    qmin = -(2 ** (bits - 1))
    qmax = 2 ** (bits - 1) - 1
    r = levels.round()
    assert qmin <= r.min().item() and r.max().item() <= qmax, name


def test_identity_hessian_matches_rtn():
    torch.manual_seed(0)
    for granularity in GRANULARITIES:
        layer = nn.Linear(32, 24)
        w_orig = layer.weight.data.clone()
        solver = GPTQ(layer, name=f"identity/{granularity}")
        solver.H = torch.eye(32)
        solver.nsamples = 1
        solver.quantize_(bits=BITS, granularity=granularity)
        expected = fake_quantize_tensor(w_orig, BITS, granularity)
        assert torch.equal(layer.weight.data, expected), granularity
    print("PASS  identity Hessian degenerates bit-exactly to RTN")


def test_huge_percdamp_matches_rtn():
    torch.manual_seed(1)
    for granularity in GRANULARITIES:
        layer = nn.Linear(16, 12)
        w_orig = layer.weight.data.clone()
        solver = GPTQ(layer, name=f"damp/{granularity}")
        solver.add_batch(_correlated_inputs(256, 16, seed=11))
        solver.quantize_(bits=BITS, granularity=granularity, percdamp=1e8)
        expected = fake_quantize_tensor(w_orig, BITS, granularity)
        assert torch.allclose(layer.weight.data, expected, atol=1e-6), granularity
    print("PASS  damping -> inf degenerates to RTN")


def test_gptq_beats_rtn_output_error():
    x = _correlated_inputs(4096, 64, seed=22)
    for granularity in GRANULARITIES:
        for actorder in (False, True):
            torch.manual_seed(2)
            layer = nn.Linear(64, 48)
            w_orig = layer.weight.data.clone()
            solver = GPTQ(layer, name=f"beats_rtn/{granularity}/act={actorder}")
            solver.add_batch(x)
            solver.quantize_(bits=BITS, granularity=granularity, actorder=actorder)

            gptq_err = _output_error(x, w_orig, layer.weight.data)
            rtn_err = _output_error(
                x, w_orig, fake_quantize_tensor(w_orig, BITS, granularity)
            )
            assert gptq_err < rtn_err, (granularity, actorder, gptq_err, rtn_err)
            assert gptq_err <= 0.99 * rtn_err, (granularity, actorder, gptq_err, rtn_err)
    print("PASS  GPTQ output error strictly below RTN (both granularities, +/- actorder)")


def test_block_size_invariance():
    torch.manual_seed(3)
    x = _correlated_inputs(1024, 48, seed=33)
    layer_a = nn.Linear(48, 32)
    layer_b = nn.Linear(48, 32)
    layer_b.weight.data.copy_(layer_a.weight.data)
    layer_b.bias.data.copy_(layer_a.bias.data)

    for layer, block_size in ((layer_a, 128), (layer_b, 8)):
        solver = GPTQ(layer, name=f"block={block_size}")
        solver.add_batch(x)
        solver.quantize_(bits=BITS, granularity="channel", block_size=block_size)

    assert torch.allclose(layer_a.weight.data, layer_b.weight.data, atol=1e-5)
    print("PASS  block_size is result-invariant (128 vs 8, cross-block path)")


def test_cholesky_fallback_to_rtn():
    # A negative-definite H cannot arise from real data (H = X X^T is PSD) but
    # exercises the escalating-damp retries and the final RTN fallback, which
    # must never crash a dispatched run. The warning/error logs it emits are
    # the point (loud, not silent).
    torch.manual_seed(7)
    layer = nn.Linear(16, 12)
    w_orig = layer.weight.data.clone()
    solver = GPTQ(layer, name="fallback")
    solver.H = -torch.eye(16)
    solver.nsamples = 1
    loss = solver.quantize_(bits=BITS, granularity="channel")
    assert loss != loss, "fallback must return NaN"
    assert torch.equal(
        layer.weight.data, fake_quantize_tensor(w_orig, BITS, "channel")
    )
    print("PASS  indefinite H -> damp retries -> RTN fallback (returns NaN)")


class TinyBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 2 * dim)
        self.fc2 = nn.Linear(2 * dim, dim)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class TinyNet(nn.Module):
    def __init__(self, dim=16, num_classes=4):
        super().__init__()
        self.embed = nn.Linear(8, dim)
        self.block = TinyBlock(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.head(self.block(torch.relu(self.embed(x))))


def test_apply_gptq_end_to_end():
    torch.manual_seed(4)
    net = TinyNet()
    orig_weights = {
        name: mod.weight.data.clone()
        for name, mod in net.named_modules()
        if isinstance(mod, nn.Linear)
    }
    embed_id = id(net.embed)

    # Pre-registered hook must survive apply_gptq_ (no module replacement).
    fire_count = {"n": 0}
    net.block.fc1.register_forward_hook(
        lambda m, i, o: fire_count.__setitem__("n", fire_count["n"] + 1)
    )

    calib = [{"images": torch.randn(32, 8)} for _ in range(6)]
    quantized = apply_gptq_(
        model=net,
        bits=BITS,
        granularity="channel",
        skip_modules=frozenset({"head"}),
        calib_loader=calib,
        device=torch.device("cpu"),
        num_calib_batches=4,
    )

    assert quantized == ["embed", "block.fc1", "block.fc2"], quantized
    assert torch.equal(net.head.weight.data, orig_weights["head"])
    assert id(net.embed) == embed_id and isinstance(net.embed, nn.Linear)
    for name in quantized:
        w_new = dict(net.named_modules())[name].weight.data
        assert not torch.equal(w_new, orig_weights[name]), name
        _assert_on_grid(w_new, orig_weights[name], BITS, "channel", name)

    before = fire_count["n"]
    with torch.no_grad():
        net(torch.randn(5, 8))
    assert fire_count["n"] == before + 1, "pre-registered hook lost"
    print("PASS  apply_gptq_ end-to-end (order, skip, identity, hook survival, grid)")


class OutOfOrderNet(nn.Module):
    # Attribute declaration order deliberately contradicts execution order,
    # and one Linear never fires.
    def __init__(self):
        super().__init__()
        self.second = nn.Linear(8, 4)
        self.first = nn.Linear(8, 8)
        self.unused = nn.Linear(8, 8)

    def forward(self, x):
        return self.second(self.first(x))


def test_execution_order_and_fallback():
    torch.manual_seed(5)
    net = OutOfOrderNet()
    layers = _find_linear_layers(net, frozenset())
    batch = {"images": torch.randn(16, 8)}

    from src.gptq import _default_forward_fn
    order, counts, never_fired = _discover_execution_order(
        net, layers, batch, torch.device("cpu"), _default_forward_fn
    )
    assert order == ["first", "second"], order
    assert counts == {"first": 1, "second": 1, "unused": 0}, counts
    assert never_fired == ["unused"], never_fired

    unused_orig = net.unused.weight.data.clone()
    quantized = apply_gptq_(
        model=net,
        bits=BITS,
        granularity="channel",
        skip_modules=frozenset(),
        calib_loader=[batch, {"images": torch.randn(16, 8)}],
        device=torch.device("cpu"),
        num_calib_batches=2,
    )
    assert quantized == ["first", "second", "unused"], quantized
    # Never-fired layers must land exactly on RTN (== GPTQ with H prop. to I).
    assert torch.equal(
        net.unused.weight.data,
        fake_quantize_tensor(unused_orig, BITS, "channel"),
    )
    print("PASS  execution-order discovery + never-fired RTN fallback")


class SharedNet(nn.Module):
    # `shared` fires twice per forward: the early-abort hook must be disabled
    # for it, else the second call site's inputs would be dropped from H.
    def __init__(self, dim=8):
        super().__init__()
        self.shared = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, 4)

    def forward(self, x):
        return self.out(torch.relu(self.shared(torch.relu(self.shared(x)))))


def test_multi_fire_layer():
    torch.manual_seed(6)
    net = SharedNet()
    orig = {n: m.weight.data.clone() for n, m in net.named_modules() if isinstance(m, nn.Linear)}
    quantized = apply_gptq_(
        model=net,
        bits=BITS,
        granularity="channel",
        skip_modules=frozenset(),
        calib_loader=[{"images": torch.randn(32, 8)} for _ in range(3)],
        device=torch.device("cpu"),
        num_calib_batches=3,
    )
    assert quantized == ["shared", "out"], quantized
    for name in quantized:
        _assert_on_grid(
            dict(net.named_modules())[name].weight.data,
            orig[name], BITS, "channel", name,
        )
    print("PASS  shared (multi-fire) layer takes the no-early-abort path")


def main():
    test_identity_hessian_matches_rtn()
    test_huge_percdamp_matches_rtn()
    test_gptq_beats_rtn_output_error()
    test_block_size_invariance()
    test_cholesky_fallback_to_rtn()
    test_apply_gptq_end_to_end()
    test_execution_order_and_fallback()
    test_multi_fire_layer()
    print("\nALL GPTQ TESTS PASSED")


if __name__ == "__main__":
    main()
