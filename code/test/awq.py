"""CPU unit tests for src/awq.py (no checkpoints, no GPU, no datasets).

Validates that our AWQ is AWQ, and that it honours the `apply_ptq_` contract:

  * n_grid=1 with clipping off degenerates **bit-exactly** to RTN, i.e. to
    fake_quantize_tensor / apply_ptq_. This identity holds only because ratio=0
    yields s == 1 exactly *and* because AWQ shares the project's quantization
    grid; it is the same "degenerates to the simpler method under a knob"
    contract as GPTQ's H = I and PV-tuning's delta_decay = 0.
  * the searched scale is normalized so that s.max() * s.min() == 1 — the
    property that keeps the AWQ family centred on 1 and hence keeps RTN inside
    the search space rather than merely at its boundary.
  * with salient (high-magnitude) input channels, AWQ's layer-*output* error is
    strictly below RTN's. Output error is the objective; weight-space error is
    the wrong metric, since AWQ deliberately makes some weights rounder-off.
  * the clip search never hurts: its grid contains shrink = 1.0, so it is an
    argmin over a set containing the unclipped candidate.
  * **grid representability**: weight * scale lands on the integer grid. This is
    the contract that justifies not folding s — it proves the stored weight is
    a genuine b-bit model (int_b (x) per-row scale (x) per-input-channel s) and
    not an arbitrary float tensor that merely scored well.
  * apply_awq_: skip_modules honored, module identity and pre-registered
    forward hooks survive (the 003_qat_transfer_activ property), execution
    order, never-fired fallback, and the fused-qkv clip skip.

`test_degenerates_to_rtn` and `test_grid_representability` are **contracts**:
if either fails, nothing downstream may be dispatched.

Run: uv run --active python code/test/awq.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from src.awq import (
    AWQ,
    apply_awq_,
    awq_path_frag,
    _CLIP_SKIP_PATTERNS,
)
from src.quantization import fake_quantize_tensor, quantize_tensor

BITS = 3
GRANULARITIES = ("tensor", "channel")


def _salient_inputs(n, d, seed):
    # AWQ's premise is that activation magnitude is wildly non-uniform across
    # input channels. I.i.d. Gaussians make every channel equally salient, so
    # the optimal s is ~1 and AWQ has nothing to exploit; a heavy-tailed
    # per-channel gain is the regime the method was designed for.
    g = torch.Generator().manual_seed(seed)
    gain = torch.exp(3.0 * torch.randn(d, generator=g))
    return torch.randn(n, d, generator=g) * gain


def _output_error(x, w_ref, w_quant):
    return ((x @ w_quant.t()) - (x @ w_ref.t())).pow(2).sum().item()


def _feed(solver, x, chunk=64):
    for i in range(0, x.shape[0], chunk):
        solver.add_batch(x[i:i + chunk])


def test_degenerates_to_rtn():
    torch.manual_seed(0)
    for granularity in GRANULARITIES:
        layer = nn.Linear(32, 24)
        w_orig = layer.weight.data.clone()
        solver = AWQ(layer, name=f"degenerate/{granularity}")
        _feed(solver, _salient_inputs(128, 32, seed=1))
        solver.quantize_(bits=BITS, granularity=granularity, n_grid=1, clip=False)

        expected = fake_quantize_tensor(w_orig, BITS, granularity)
        assert torch.equal(layer.weight.data, expected), (
            f"n_grid=1/clip=False must be bit-exact RTN ({granularity})"
        )
        assert torch.equal(
            solver.scale, torch.ones_like(solver.scale)
        ), f"ratio=0 must give s == 1 exactly ({granularity})"
    print("PASS  n_grid=1 with clipping off is bit-exact RTN")


def test_scale_normalization():
    torch.manual_seed(1)
    layer = nn.Linear(48, 16)
    solver = AWQ(layer, name="norm")
    _feed(solver, _salient_inputs(256, 48, seed=2))
    for n_grid in (1, 5, 20):
        s = solver.search_scale_(BITS, "channel", n_grid=n_grid)
        prod = (s.max() * s.min()).item()
        assert abs(prod - 1.0) < 1e-4, f"s.max()*s.min()={prod} at n_grid={n_grid}"
        assert torch.isfinite(s).all()
        assert (s > 0).all()
    print("PASS  searched scale is normalized to s.max() * s.min() == 1")


def test_awq_beats_rtn_output_error():
    torch.manual_seed(2)
    x = _salient_inputs(512, 64, seed=3)
    for granularity in GRANULARITIES:
        layer = nn.Linear(64, 32)
        w_orig = layer.weight.data.clone()

        solver = AWQ(layer, name=f"beats/{granularity}")
        _feed(solver, x)
        solver.quantize_(bits=BITS, granularity=granularity, n_grid=20, clip=True)

        awq_err = _output_error(x, w_orig, layer.weight.data)
        rtn_err = _output_error(
            x, w_orig, fake_quantize_tensor(w_orig, BITS, granularity)
        )
        assert awq_err < rtn_err, (
            f"AWQ output error {awq_err:.6g} not below RTN {rtn_err:.6g} "
            f"({granularity})"
        )
    print("PASS  AWQ output error is strictly below RTN on salient channels")


def test_clip_never_hurts():
    torch.manual_seed(3)
    x = _salient_inputs(512, 64, seed=4)
    for granularity in GRANULARITIES:
        errs = {}
        for clip in (False, True):
            # Same seed for both arms: the two runs must differ only in `clip`.
            torch.manual_seed(3)
            layer = nn.Linear(64, 32)
            w_orig = layer.weight.data.clone()
            solver = AWQ(layer, name=f"clip/{granularity}")
            _feed(solver, x)
            solver.quantize_(
                bits=BITS, granularity=granularity, n_grid=20, clip=clip
            )
            errs[clip] = _output_error(x, w_orig, layer.weight.data)
        assert errs[True] <= errs[False] * (1.0 + 1e-6), (
            f"clip=True error {errs[True]:.6g} exceeds clip=False "
            f"{errs[False]:.6g} ({granularity}) — the clip grid contains "
            "shrink=1.0, so this cannot happen"
        )
    print("PASS  the clip search never increases output error")


def test_grid_representability():
    torch.manual_seed(4)
    for granularity in GRANULARITIES:
        layer = nn.Linear(64, 32)
        solver = AWQ(layer, name=f"grid/{granularity}")
        _feed(solver, _salient_inputs(256, 64, seed=5))
        solver.quantize_(bits=BITS, granularity=granularity, n_grid=20, clip=True)

        # W_awq diag(s) is what the deployed model would store as integers.
        scaled = layer.weight.data * solver.scale
        _, scale = quantize_tensor(scaled, BITS, granularity)
        levels = scaled / scale
        assert torch.allclose(levels, levels.round(), atol=1e-3), (
            f"weight * scale is not on the integer grid ({granularity})"
        )
        qmin = -(2 ** (BITS - 1))
        qmax = 2 ** (BITS - 1) - 1
        r = levels.round()
        assert qmin <= r.min().item() and r.max().item() <= qmax, (
            f"levels outside [{qmin}, {qmax}] ({granularity})"
        )
    print("PASS  weight * scale lies on the b-bit integer grid")


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


def test_apply_awq_end_to_end():
    torch.manual_seed(5)
    net = TinyNet()
    orig_weights = {
        name: mod.weight.data.clone()
        for name, mod in net.named_modules()
        if isinstance(mod, nn.Linear)
    }
    embed_id = id(net.embed)

    fired = []
    net.block.fc1.register_forward_hook(lambda m, i, o: fired.append(1))

    batches = [{"images": torch.randn(32, 8)} for _ in range(3)]
    quantized = apply_awq_(
        model=net,
        bits=BITS,
        granularity="channel",
        skip_modules=frozenset({"head"}),
        calib_loader=batches,
        device=torch.device("cpu"),
        num_calib_batches=3,
        n_grid=8,
        clip=True,
    )

    assert quantized == ["embed", "block.fc1", "block.fc2"], quantized
    assert "head" not in quantized
    assert torch.equal(net.head.weight.data, orig_weights["head"]), (
        "skip_modules must leave the head untouched"
    )
    assert id(net.embed) == embed_id, "apply_awq_ must not replace modules"
    for name in ("embed", "block.fc1", "block.fc2"):
        mod = dict(net.named_modules())[name]
        assert not torch.equal(mod.weight.data, orig_weights[name]), name

    fired.clear()
    net(torch.randn(4, 8))
    assert fired, "a pre-registered forward hook must survive apply_awq_"
    print("PASS  apply_awq_ end-to-end: order, skips, identity, hooks")


def test_qkv_clip_skip():
    assert any(p in "attn.qkv" for p in _CLIP_SKIP_PATTERNS)
    assert not any(p in "attn.proj" for p in _CLIP_SKIP_PATTERNS)

    x = _salient_inputs(256, 32, seed=6)

    def run(name, clip):
        torch.manual_seed(6)
        layer = nn.Linear(32, 32)
        solver = AWQ(layer, name=name)
        _feed(solver, x)
        solver.quantize_(bits=BITS, granularity="channel", n_grid=8, clip=clip)
        return layer.weight.data.clone()

    # For a qkv-named layer, clip=True must be a no-op (the stage is skipped).
    assert torch.equal(run("attn.qkv", True), run("attn.qkv", False)), (
        "clipping must be skipped on fused qkv"
    )
    # For a non-qkv layer, the stage must actually be able to change something.
    assert not torch.equal(run("attn.proj", True), run("attn.proj", False)), (
        "clipping must run on attn.proj"
    )
    print("PASS  clipping is skipped on fused qkv and runs elsewhere")


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
    torch.manual_seed(7)
    net = OutOfOrderNet()
    unused_orig = net.unused.weight.data.clone()

    quantized = apply_awq_(
        model=net,
        bits=BITS,
        granularity="channel",
        skip_modules=frozenset(),
        calib_loader=[torch.randn(16, 8) for _ in range(2)],
        device=torch.device("cpu"),
        num_calib_batches=2,
        n_grid=4,
        clip=False,
    )
    assert quantized[:2] == ["first", "second"], quantized
    assert quantized[2] == "unused", quantized
    assert torch.equal(
        net.unused.weight.data,
        fake_quantize_tensor(unused_orig, BITS, "channel"),
    ), "a never-fired layer must fall back to RTN"
    print("PASS  execution order respected; never-fired layer falls back to RTN")


def test_batch_conventions_and_forward_fn():
    torch.manual_seed(8)

    # dict / tuple / bare-tensor batches all go through _default_forward_fn.
    for batches in (
        [{"images": torch.randn(16, 8)}],
        [(torch.randn(16, 8), torch.zeros(16, dtype=torch.long))],
        [torch.randn(16, 8)],
    ):
        net = TinyNet()
        quantized = apply_awq_(
            model=net, bits=BITS, granularity="channel",
            skip_modules=frozenset({"head"}), calib_loader=batches,
            device=torch.device("cpu"), num_calib_batches=1, n_grid=4, clip=False,
        )
        assert quantized == ["embed", "block.fc1", "block.fc2"]

    # A custom forward_fn (the shape the text family needs for its tokenizer).
    seen = []

    def custom_forward(model, batch, device):
        seen.append(batch["payload"].shape)
        model(batch["payload"].to(device))

    net = TinyNet()
    apply_awq_(
        model=net, bits=BITS, granularity="channel",
        skip_modules=frozenset({"head"}),
        calib_loader=[{"payload": torch.randn(16, 8)}],
        device=torch.device("cpu"), num_calib_batches=1, n_grid=4, clip=False,
        forward_fn=custom_forward,
    )
    assert seen, "forward_fn must be used"
    print("PASS  dict/tuple/tensor batches and a custom forward_fn all work")


def test_path_fragment():
    frag = awq_path_frag(
        bits=3, granularity="channel", skip_modules=["head"],
        num_calib_batches=4, n_grid=20, clip=True,
    )
    assert frag == (
        "awq=bits=3_gran=channel_skip=head_ncal=4_ngrid=20_clip=True"
    ), frag
    # Sorted and dash-joined, "none" when empty — the project's skip_tag rule.
    assert "skip=a-b" in awq_path_frag(3, "tensor", ["b", "a"], 4, 20, False)
    assert "skip=none" in awq_path_frag(3, "tensor", [], 4, 20, False)
    print("PASS  awq_path_frag renders the documented fragment")


def main():
    test_degenerates_to_rtn()
    test_scale_normalization()
    test_awq_beats_rtn_output_error()
    test_clip_never_hurts()
    test_grid_representability()
    test_apply_awq_end_to_end()
    test_qkv_clip_skip()
    test_execution_order_and_fallback()
    test_batch_conventions_and_forward_fn()
    test_path_fragment()
    print("\nALL AWQ TESTS PASSED")


if __name__ == "__main__":
    main()
