"""Smoke tests for the training-duration multiplier axis.

The migration to an explicit ``mult=`` path component rests on a single
invariant: **at ``epoch_mult = 1.0`` nothing changes**.  Every step count, every
scheduler length and every checkpoint filename must be exactly what the repo
produces today, for every dataset in both epoch tables and across every
(num_batches, accum_steps) shape the five model families produce.  If that holds,
the 140,474 evaluations already on disk remain correct and the migration is a
pure textual insertion; if it does not, every downstream number silently shifts.

The second thing pinned here is the *formatting* of the fragment.  That sounds
cosmetic and is not: Hydra hands overrides through as ``4`` or ``4.0`` or
``"4.0"`` depending on how they were typed, and if those produced different
strings a single budget would fork into two half-populated directory trees.
This is the sharpest failure mode in the whole change, so it is tested first.

Run:  uv run --active python code/test/duration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

import math

from src.duration import (
    UNIT_MULT,
    Duration,
    clamped_warmup,
    mult_path_frag,
    mult_tag,
    parse_role_frag,
    resolve_duration,
    role_path_frag,
    run_meta,
    training_budget,
    unit_steps,
)
from src.text.data.common import DATASET_NAME_TO_EPOCHS as TEXT_EPOCHS
from src.vision.data.common import DATASET_NAME_TO_EPOCHS as VISION_EPOCHS


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
_failures = []


def check(label, condition, detail=""):
    if condition:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}{(' -- ' + detail) if detail else ''}")
        _failures.append(label)


def check_raises(label, exc_type, fn, *args, **kwargs):
    try:
        result = fn(*args, **kwargs)
    except exc_type:
        print(f"  ok   {label}")
        return
    except Exception as exc:  # noqa: BLE001 - we want the wrong-type report
        print(f"  FAIL {label} -- raised {type(exc).__name__}, expected {exc_type.__name__}")
        _failures.append(label)
        return
    print(f"  FAIL {label} -- returned {result!r}, expected {exc_type.__name__}")
    _failures.append(label)


# ---------------------------------------------------------------------------
# 1. Fragment canonicalisation
# ---------------------------------------------------------------------------
def test_fragment_canonicalisation():
    print("\n[1] mult_path_frag canonicalisation")

    # The exact literals. Asserted against hard-coded strings rather than
    # against each other, so a change of convention cannot pass by being
    # self-consistent.
    for value, expected in [
        (1, "mult=1"), (1.0, "mult=1"), ("1", "mult=1"), ("1.0", "mult=1"),
        (0.25, "mult=0.25"), ("0.25", "mult=0.25"),
        (4, "mult=4"), (4.0, "mult=4"), ("4.0", "mult=4"),
        (0.2, "mult=0.2"), (4.8, "mult=4.8"), (5, "mult=5"), (0.5, "mult=0.5"),
        (10, "mult=10"), (0.125, "mult=0.125"),
    ]:
        got = mult_path_frag(value)
        check(f"mult_path_frag({value!r}) == {expected!r}", got == expected, f"got {got!r}")

    # Hydra int-vs-float coercion must not fork the tree.
    check("int 4 and float 4.0 agree", mult_path_frag(4) == mult_path_frag(4.0))
    check("str '4.0' and float 4.0 agree", mult_path_frag("4.0") == mult_path_frag(4.0))
    check("UNIT_MULT is spelled 'mult=1'", mult_path_frag(UNIT_MULT) == "mult=1")

    # Distinct multipliers must never collapse onto one fragment.
    values = [0.125, 0.2, 0.25, 0.5, 1.0, 2.0, 4.0, 4.8, 5.0, 10.0]
    frags = [mult_path_frag(v) for v in values]
    check("distinct multipliers give distinct fragments", len(set(frags)) == len(values),
          f"{frags}")

    # Rejections: a value that cannot be named exactly must fail loudly rather
    # than be silently truncated into another multiplier's directory.
    check_raises("1/3 is rejected (not representable)", ValueError, mult_path_frag, 1 / 3)
    check_raises("zero is rejected", ValueError, mult_path_frag, 0)
    check_raises("negative is rejected", ValueError, mult_path_frag, -1.0)
    check_raises("nan is rejected", ValueError, mult_path_frag, float("nan"))
    check_raises("inf is rejected", ValueError, mult_path_frag, float("inf"))
    check_raises("non-numeric is rejected", ValueError, mult_path_frag, "wat")

    # No fragment may contain a path separator or a second '=' that would break
    # the key=value grammar every reader splits on.
    for value in values:
        frag = mult_path_frag(value)
        check(f"{frag!r} is a well-formed single component",
              "/" not in frag and frag.count("=") == 1 and frag.startswith("mult="))


# ---------------------------------------------------------------------------
# 2. The 1x invariant
# ---------------------------------------------------------------------------
def _today_scheduler_steps(base_epochs, num_batches, accum_steps):
    """Verbatim copy of the formula in all ten finetuners.

    Deliberately duplicated rather than imported: the point is to compare
    resolve_duration against an independent transcription of the existing code,
    not against itself.
    """
    return base_epochs * num_batches // accum_steps


def test_unit_invariant():
    print("\n[2] epoch_mult=1.0 reproduces today's behaviour exactly")

    # Shapes covering every family: accum_steps is 1 on the canonical grid
    # (bs == REFERENCE_BATCH_SIZE) and 2 for the bs=64 large-backbone runs;
    # text uses REFERENCE_BATCH_SIZE=32. num_batches spans Flowers102 (8) to
    # ImageNet (9971).
    shapes = [(8, 1), (26, 1), (58, 1), (188, 1), (352, 1), (743, 1), (9971, 1),
              (16, 2), (116, 2), (376, 2), (704, 2), (19942, 2),
              (37, 1), (137, 1), (1000, 4)]

    tables = [("vision", VISION_EPOCHS), ("text", TEXT_EPOCHS)]
    mismatches = 0
    checked = 0
    for table_name, table in tables:
        for dataset in sorted(table):
            for num_batches, accum_steps in shapes:
                d = resolve_duration(dataset, 1.0, num_batches, accum_steps, table)
                expected_steps = _today_scheduler_steps(table[dataset], num_batches, accum_steps)
                checked += 1
                if d.max_steps != expected_steps:
                    mismatches += 1
                    print(f"  FAIL {table_name}/{dataset} nb={num_batches} accum={accum_steps}: "
                          f"max_steps={d.max_steps}, today={expected_steps}")
                if d.loop_epochs != table[dataset]:
                    mismatches += 1
                    print(f"  FAIL {table_name}/{dataset} nb={num_batches} accum={accum_steps}: "
                          f"loop_epochs={d.loop_epochs}, today={table[dataset]}")
                if d.base_epochs != table[dataset]:
                    mismatches += 1
                    print(f"  FAIL {table_name}/{dataset}: base_epochs != table value")

    check(f"1x matches today across {checked} (dataset, shape) combinations",
          mismatches == 0, f"{mismatches} mismatches")

    # The filename must stay derivable without touching a dataset.
    check("base_epochs is loader-independent",
          resolve_duration("Cars", 1.0, 58, 1, VISION_EPOCHS).base_epochs
          == resolve_duration("Cars", 1.0, 9999, 3, VISION_EPOCHS).base_epochs)

    # int 1 and float 1.0 must resolve identically.
    check("epoch_mult=1 and 1.0 resolve identically",
          resolve_duration("DTD", 1, 27, 1, VISION_EPOCHS)
          == resolve_duration("DTD", 1.0, 27, 1, VISION_EPOCHS))


# ---------------------------------------------------------------------------
# 3. Non-unit multipliers
# ---------------------------------------------------------------------------
def test_scaling():
    print("\n[3] non-unit multipliers")

    # The two experiment arms, against the real measured step counts.
    imagenet = resolve_duration("ImageNet", 0.25, 9971, 1, VISION_EPOCHS)
    check("ImageNet 1x is 9971 steps",
          resolve_duration("ImageNet", 1.0, 9971, 1, VISION_EPOCHS).max_steps == 9971)
    check("ImageNet at mult=0.25 is 2493 steps", imagenet.max_steps == 2493,
          f"got {imagenet.max_steps}")
    check("ImageNet at mult=0.25 still loops 1 epoch", imagenet.loop_epochs == 1,
          f"got {imagenet.loop_epochs}")
    check("ImageNet at mult=0.25 keeps base_epochs=1 for the filename",
          imagenet.base_epochs == 1)

    cars_4x = resolve_duration("Cars", 4.0, 58, 1, VISION_EPOCHS)
    check("Cars 1x is 2030 steps",
          resolve_duration("Cars", 1.0, 58, 1, VISION_EPOCHS).max_steps == 2030)
    check("Cars at mult=4 is 8120 steps", cars_4x.max_steps == 8120, f"got {cars_4x.max_steps}")
    check("Cars at mult=4 loops 140 epochs", cars_4x.loop_epochs == 140,
          f"got {cars_4x.loop_epochs}")
    check("Cars at mult=4 keeps base_epochs=35 for the filename", cars_4x.base_epochs == 35)

    # Monotonicity and proportionality across a range.
    for dataset, num_batches in [("Cars", 58), ("DTD", 27), ("CIFAR10", 352)]:
        steps = [resolve_duration(dataset, m, num_batches, 1, VISION_EPOCHS).max_steps
                 for m in (0.25, 0.5, 1.0, 2.0, 4.0)]
        check(f"{dataset}: budget is strictly increasing in mult",
              all(a < b for a, b in zip(steps, steps[1:])), f"{steps}")
        base = resolve_duration(dataset, 1.0, num_batches, 1, VISION_EPOCHS).max_steps
        check(f"{dataset}: mult=2 is within 1 step of 2x the 1x budget",
              abs(resolve_duration(dataset, 2.0, num_batches, 1, VISION_EPOCHS).max_steps
                  - 2 * base) <= 1)

    # loop_epochs must always be enough to reach max_steps.
    ok = True
    for dataset in sorted(VISION_EPOCHS):
        for mult in (0.2, 0.25, 0.5, 2.0, 4.0, 4.8, 5.0):
            for num_batches, accum in [(8, 1), (58, 1), (352, 1), (9971, 1), (116, 2)]:
                d = resolve_duration(dataset, mult, num_batches, accum, VISION_EPOCHS)
                available = d.loop_epochs * num_batches // accum
                if available < d.max_steps:
                    ok = False
                    print(f"  FAIL {dataset} mult={mult} nb={num_batches} accum={accum}: "
                          f"loop provides {available} steps, need {d.max_steps}")
    check("loop_epochs always provides at least max_steps", ok)

    # A budget can never round down to zero.
    check("tiny multiplier still yields at least one step",
          resolve_duration("Flowers102", 0.001, 8, 1, VISION_EPOCHS).max_steps >= 1)

    check_raises("unknown dataset is rejected", KeyError,
                 resolve_duration, "NotADataset", 1.0, 10, 1, VISION_EPOCHS)
    check_raises("non-representable multiplier is rejected at resolve time",
                 ValueError, resolve_duration, "Cars", 1 / 3, 58, 1, VISION_EPOCHS)
    check_raises("zero num_batches is rejected", ValueError,
                 resolve_duration, "Cars", 1.0, 0, 1, VISION_EPOCHS)


# ---------------------------------------------------------------------------
# 4. Warmup clamping
# ---------------------------------------------------------------------------
def test_warmup():
    print("\n[4] clamped_warmup guards cosine_lr")

    check("long run keeps wl unchanged", clamped_warmup(500, 9971) == 500)
    check("1x Flowers102 (1176 steps) keeps wl=500", clamped_warmup(500, 1176) == 500)

    # The regime a free multiplier makes reachable: Flowers102 at mult=0.25.
    short = resolve_duration("Flowers102", 0.25, 8, 1, VISION_EPOCHS)
    check("Flowers102 at mult=0.25 is 294 steps", short.max_steps == 294,
          f"got {short.max_steps}")
    wl = clamped_warmup(500, short.max_steps)
    check("warmup is clamped below the budget", wl < short.max_steps, f"wl={wl}")
    check("cosine denominator stays positive", short.max_steps - wl > 0)

    check("wl == steps is clamped", clamped_warmup(500, 500) == 499)
    check("degenerate 1-step budget yields zero warmup", clamped_warmup(500, 1) == 0)


# ---------------------------------------------------------------------------
# 5. training_budget and its interaction with limit_num_epochs
# ---------------------------------------------------------------------------
def test_training_budget():
    print("\n[5] training_budget composes the multiplier with the dryrun knob")

    # Without a limit, it is just resolve_duration destructured.
    b = training_budget("Cars", 1.0, 58, 1, VISION_EPOCHS, None)
    check("1x Cars: loop_epochs == 35", b.loop_epochs == 35)
    check("1x Cars: max_steps == 2030", b.max_steps == 2030)
    check("1x Cars: ckpt_epochs == 35", b.ckpt_epochs == 35)

    # limit_num_epochs truncates and never extends -- the property that keeps it
    # orthogonal to the multiplier.
    lim = training_budget("Cars", 1.0, 58, 1, VISION_EPOCHS, 2)
    check("limit=2 truncates loop_epochs", lim.loop_epochs == 2)
    check("limit=2 truncates max_steps to 116", lim.max_steps == 116, f"got {lim.max_steps}")
    check("limit=2 truncates the checkpoint name", lim.ckpt_epochs == 2)

    big = training_budget("Cars", 1.0, 58, 1, VISION_EPOCHS, 999)
    check("limit above the schedule never extends it",
          (big.loop_epochs, big.max_steps, big.ckpt_epochs) == (35, 2030, 35),
          f"got {big}")

    # A limit applied on top of a multiplier truncates the *scaled* budget.
    scaled = training_budget("Cars", 4.0, 58, 1, VISION_EPOCHS, None)
    scaled_lim = training_budget("Cars", 4.0, 58, 1, VISION_EPOCHS, 10)
    check("mult=4 Cars is 140 epochs / 8120 steps",
          (scaled.loop_epochs, scaled.max_steps) == (140, 8120), f"got {scaled}")
    check("mult=4 with limit=10 truncates to 10 epochs / 580 steps",
          (scaled_lim.loop_epochs, scaled_lim.max_steps) == (10, 580), f"got {scaled_lim}")
    check("mult=4 with a limit keeps ckpt_epochs at min(base, limit)",
          scaled_lim.ckpt_epochs == 10)

    # Truncation must never produce a zero-step budget.
    tiny = training_budget("Flowers102", 0.25, 8, 1, VISION_EPOCHS, 1)
    check("truncation never yields zero steps", tiny.max_steps >= 1, f"got {tiny.max_steps}")

    # run_meta carries what no path component records.
    meta = run_meta(scaled, 58, 1, 500)
    check("run_meta records the multiplier", meta["epoch_mult"] == 4.0)
    check("run_meta records the realized step count", meta["max_steps"] == 8120)
    check("run_meta records the reference schedule", meta["base_epochs"] == 35)
    check("run_meta records the clamped warmup", meta["warmup_length"] == 500)
    check("run_meta tolerates a family with no warmup (text)",
          run_meta(scaled, 58, 1, None)["warmup_length"] is None)


# ---------------------------------------------------------------------------
# 6. Transfer-path role fragments
# ---------------------------------------------------------------------------
def test_role_fragments():
    print("\n[6] role_path_frag / parse_role_frag round-trip")

    check("src fragment literal",
          role_path_frag("src", "Cars", 2038, 1.0) == "src=Cars_seed=2038_mult=1")
    check("tgt fragment literal",
          role_path_frag("tgt", "DTD", 2038, 4.0) == "tgt=DTD_seed=2038_mult=4")
    check("fractional multiplier",
          role_path_frag("src", "ImageNet", 2038, 0.25) == "src=ImageNet_seed=2038_mult=0.25")
    check_raises("an invalid role is rejected", ValueError,
                 role_path_frag, "donor", "Cars", 2038, 1.0)

    # Round-trip over every real dataset name, including the awkward ones.
    names = sorted(VISION_EPOCHS) + sorted(TEXT_EPOCHS) + ["Oxford-IIIT_Pet", "a_b-c_seed=x"]
    ok = True
    for name in names:
        for mult in (1.0, 0.2, 0.25, 4.0, 4.8, 10.0):
            for role in ("src", "tgt"):
                frag = role_path_frag(role, name, 2038, mult)
                got = parse_role_frag(frag)
                if got is None or got.role != role or got.dataset != name \
                        or got.seed != 2038 or got.epoch_mult != mult:
                    ok = False
                    print(f"  FAIL round-trip {frag!r} -> {got}")
    check(f"round-trips for {len(names)} dataset names x 6 multipliers x 2 roles", ok)

    # Donor and receiver must be independently expressible -- the whole point.
    s = role_path_frag("src", "CIFAR10", 2038, 1.0)
    t = role_path_frag("tgt", "DTD", 2038, 4.0)
    check("donor and receiver budgets are independent",
          parse_role_frag(s).epoch_mult == 1.0 and parse_role_frag(t).epoch_mult == 4.0)

    # A legacy (pre-axis) fragment must NOT silently parse.
    check("legacy fragment does not parse", parse_role_frag("src=Cars_seed=2038") is None)
    check("garbage does not parse", parse_role_frag("nonsense") is None)

    # mult_tag is the same canonicalisation without the prefix.
    check("mult_tag drops the prefix", mult_tag(4.0) == "4" and mult_tag(0.25) == "0.25")
    check("mult_tag agrees with mult_path_frag",
          all(mult_path_frag(m) == "mult=" + mult_tag(m) for m in (1, 0.25, 4, 4.8, 10)))


# ---------------------------------------------------------------------------
def main():
    print(__doc__.strip().splitlines()[0])
    test_fragment_canonicalisation()
    test_unit_invariant()
    test_scaling()
    test_warmup()
    test_training_budget()
    test_role_fragments()

    print()
    if _failures:
        print(f"FAILED: {len(_failures)} check(s)")
        for label in _failures:
            print(f"  - {label}")
        sys.exit(1)
    print("All duration checks passed.")


if __name__ == "__main__":
    main()
