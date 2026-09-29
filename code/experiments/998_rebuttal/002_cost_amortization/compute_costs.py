"""998 — Cost amortization of QV transfer against per-task QAT

Quantifies the reviewer's N-receiver question: Figure 1 compares the receiver
side of the two pipelines but omits the one-off donor cost, so the cost claim
holds only once that cost is amortized over enough receivers.  This script
computes the amortization factor explicitly.

Cost model.  Both pipelines start from the same set of already fine-tuned FP
checkpoints and quantize them with the same backbone and the same QAT
procedure, so the cost per training sample is identical on both sides and
cancels.  What remains is the number of training samples each pipeline must
process:

    per-task QAT   :  sum over the N receivers of (epochs * train_size)
    QV transfer    :  one donor QAT run, then N vector additions

The receiver side of QV transfer needs no data and no gradients, so its cost is
zero in this model (override with --patch-samples to charge it something).  The
donor's FP checkpoint is not charged either: the timm backbones are already
ImageNet-trained, so the default ImageNet donor needs no extra fine-tuning run.
Because the sum over receivers depends on which N of them are deployed, the
break-even point is reported three ways: for the mean receiver, and for the two
extreme orderings (cheapest-first and costliest-first), which bracket every
possible deployment order.

Pass --flops-per-sample to additionally express every quantity in FLOPs; the
constant rescales the axes but cannot move the break-even point.

Pass --step-time-json (a step_time.json written by measure_step_time.py) or an
explicit --seconds-per-sample to additionally express the headline costs in
GPU-hours on the measured hardware: the donor QAT run, the mean receiver QAT
run, and the full-suite baseline and QV-transfer totals.  Like the FLOPs
rescaling, this is a linear change of units: it annotates the figure with a
concrete "= N GPU-hours on <GPU>" statement but cannot move the break-even
point, and it is entirely optional — with no timing supplied the script behaves
exactly as before.

Reads dataset_sizes.json and writes cost_amortization.json, both inside
evaluations/998_rebuttal/002_cost_amortization/.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is the working directory and on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import os

os.chdir(_PROJECT_ROOT)

import argparse
import json
import statistics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT = "evaluations/998_rebuttal/002_cost_amortization"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--donor",
        default="ImageNet",
        help="Donor task whose QV is reused by every receiver.",
    )
    parser.add_argument(
        "--receivers",
        nargs="+",
        default=None,
        help="Receiver tasks (default: every resolved dataset except the donor).",
    )
    parser.add_argument(
        "--patch-samples",
        type=float,
        default=0.0,
        help="Training samples charged to one receiver-side QV patch (default 0: "
             "the patch is a vector addition, with no data and no gradients).",
    )
    parser.add_argument(
        "--flops-per-sample",
        type=float,
        default=None,
        help="Training FLOPs per sample for the backbone, e.g. 3x the forward "
             "FLOPs from fvcore. Rescales the reported cost into FLOPs.",
    )
    parser.add_argument(
        "--step-time-json",
        default=None,
        help="step_time.json written by measure_step_time.py. Its measured "
             "seconds-per-training-sample turns the sample counts into "
             "GPU-hours on that hardware (optional enrichment).",
    )
    parser.add_argument(
        "--seconds-per-sample",
        type=float,
        default=None,
        help="Measured wall-clock seconds per training sample, given directly "
             "instead of --step-time-json.",
    )
    parser.add_argument(
        "--step-time-key",
        default="seconds_per_sample",
        choices=["seconds_per_sample", "seconds_per_sample_step"],
        help="Which field of step_time.json to read: the wall-clock number "
             "(default, includes dataloader stalls) or the compute-only one.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def _qat_samples(record):
    """Training samples one QAT run on this dataset processes."""
    assert record["epochs"] is not None, "dataset has no entry in DATASET_NAME_TO_EPOCHS"
    return record["epochs"] * record["train_size"]


def _cumulative(costs):
    """Running total of a cost sequence, as a list of length len(costs)."""
    total = 0.0
    out = []
    for c in costs:
        total += c
        out.append(total)
    return out


def _resolve_timing(args):
    """Measured seconds per training sample, plus its provenance.

    Returns (seconds_per_sample, provenance_dict), or (None, None) when the
    caller supplied no timing at all — GPU-hours are optional enrichment.
    """
    if args.seconds_per_sample is not None:
        assert args.step_time_json is None, (
            "Pass either --seconds-per-sample or --step-time-json, not both."
        )
        assert args.seconds_per_sample > 0, "--seconds-per-sample must be positive"
        return args.seconds_per_sample, {"source": "--seconds-per-sample"}

    if args.step_time_json is None:
        return None, None

    assert os.path.exists(args.step_time_json), (
        f"{args.step_time_json} not found. Run measure_step_time.py first."
    )
    with open(args.step_time_json) as f:
        timing = json.load(f)

    assert args.step_time_key in timing, (
        f"{args.step_time_json} has no field {args.step_time_key!r}"
    )
    seconds_per_sample = timing[args.step_time_key]
    assert seconds_per_sample > 0, (
        f"{args.step_time_key} in {args.step_time_json} must be positive"
    )

    provenance = {
        "source": args.step_time_json,
        "field": args.step_time_key,
        "device_name": timing.get("device_name"),
        "model_name": timing.get("model_name"),
        "dataset_name": timing.get("dataset_name"),
        "batch_size": timing.get("batch_size"),
        "num_workers": timing.get("num_workers"),
        "qat": timing.get("qat"),
        "torch_version": timing.get("torch_version"),
        "warmup_batches": timing.get("warmup_batches"),
        "timed_batches": timing.get("timed_batches"),
    }
    return seconds_per_sample, provenance


def _break_even(baseline_cumulative, donor_cost, patch_cost):
    """Smallest N at which QV transfer is no more expensive than per-task QAT.

    Returns None when the crossing lies beyond the available receivers.
    """
    for n, baseline in enumerate(baseline_cumulative, start=1):
        if donor_cost + n * patch_cost <= baseline:
            return n
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    sizes_path = os.path.join(EVAL_ROOT, "dataset_sizes.json")
    assert os.path.exists(sizes_path), (
        f"{sizes_path} not found. Run collect_dataset_sizes.py first."
    )
    with open(sizes_path) as f:
        sizes = json.load(f)["datasets"]

    assert args.donor in sizes, (
        f"Donor {args.donor} missing from dataset_sizes.json (resolved: {sorted(sizes)})"
    )

    receiver_names = args.receivers
    if receiver_names is None:
        receiver_names = sorted(d for d in sizes if d != args.donor)

    missing = [d for d in receiver_names if d not in sizes]
    assert not missing, f"Receivers missing from dataset_sizes.json: {missing}"

    donor_cost = _qat_samples(sizes[args.donor])
    receiver_costs = {d: _qat_samples(sizes[d]) for d in receiver_names}

    costs = list(receiver_costs.values())
    mean_cost = statistics.mean(costs)
    median_cost = statistics.median(costs)

    n_receivers = len(receiver_names)
    ns = list(range(1, n_receivers + 1))

    # Per-task QAT, under the mean receiver and under the two extreme orderings.
    baseline_mean = [mean_cost * n for n in ns]
    cheapest_first = sorted(receiver_names, key=lambda d: receiver_costs[d])
    costliest_first = list(reversed(cheapest_first))
    baseline_cheapest = _cumulative([receiver_costs[d] for d in cheapest_first])
    baseline_costliest = _cumulative([receiver_costs[d] for d in costliest_first])

    # QV transfer: the donor QAT once, then one patch per receiver.
    ours = [donor_cost + n * args.patch_samples for n in ns]

    marginal = mean_cost - args.patch_samples
    assert marginal > 0, (
        "The QV patch is not cheaper than a QAT run under this cost model; "
        "the amortization argument does not apply."
    )
    n_star_mean_exact = donor_cost / marginal

    results = {
        "cost_unit": "training samples seen",
        "cost_model": (
            "Both pipelines run the same QAT procedure on the same backbone, so "
            "the per-sample cost cancels and only samples seen remain. Per-task "
            "QAT pays one QAT run per receiver; QV transfer pays one donor QAT "
            "run in total plus one data-free vector addition per receiver."
        ),
        "donor": args.donor,
        "donor_epochs": sizes[args.donor]["epochs"],
        "donor_train_size": sizes[args.donor]["train_size"],
        "donor_cost": donor_cost,
        "patch_samples": args.patch_samples,
        "n_receivers": n_receivers,
        "receivers": receiver_names,
        "receiver_costs": receiver_costs,
        "receiver_cost_mean": mean_cost,
        "receiver_cost_median": median_cost,
        "receiver_cost_min": min(costs),
        "receiver_cost_max": max(costs),
        "break_even": {
            # The headline number: how many receivers the donor run must serve.
            "n_star_mean_exact": n_star_mean_exact,
            "n_star_mean": _break_even(baseline_mean, donor_cost, args.patch_samples),
            # Ordering extremes, which bracket every deployment order.
            "n_star_cheapest_first": _break_even(
                baseline_cheapest, donor_cost, args.patch_samples
            ),
            "n_star_costliest_first": _break_even(
                baseline_costliest, donor_cost, args.patch_samples
            ),
        },
        "curves": {
            "n": ns,
            "baseline_cumulative_mean": baseline_mean,
            "baseline_cumulative_cheapest_first": baseline_cheapest,
            "baseline_cumulative_costliest_first": baseline_costliest,
            "ours_cumulative": ours,
            "baseline_per_deployment": [mean_cost] * n_receivers,
            "ours_per_deployment": [o / n for o, n in zip(ours, ns)],
        },
        "at_full_suite": {
            "n": n_receivers,
            "baseline_cost": baseline_mean[-1],
            "ours_cost": ours[-1],
            "speedup": baseline_mean[-1] / ours[-1],
        },
    }

    if args.flops_per_sample is not None:
        results["flops_per_sample"] = args.flops_per_sample
        results["flops"] = {
            "donor_cost": donor_cost * args.flops_per_sample,
            "receiver_cost_mean": mean_cost * args.flops_per_sample,
            "baseline_cost_full_suite": baseline_mean[-1] * args.flops_per_sample,
            "ours_cost_full_suite": ours[-1] * args.flops_per_sample,
        }

    # Optional enrichment: turn the sample counts into GPU-hours on measured
    # hardware. Purely a change of units; nothing above depends on it.
    seconds_per_sample, timing_provenance = _resolve_timing(args)
    if seconds_per_sample is not None:
        per_hour = seconds_per_sample / 3600.0
        results["seconds_per_sample"] = seconds_per_sample
        results["step_time_source"] = timing_provenance
        results["gpu_hours"] = {
            "donor_cost": donor_cost * per_hour,
            "receiver_cost_mean": mean_cost * per_hour,
            "baseline_cost_full_suite": baseline_mean[-1] * per_hour,
            "ours_cost_full_suite": ours[-1] * per_hour,
        }

    os.makedirs(EVAL_ROOT, exist_ok=True)
    out_path = os.path.join(EVAL_ROOT, "cost_amortization.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"donor                  {args.donor}: {donor_cost:,} samples "
          f"({sizes[args.donor]['epochs']} epoch(s))")
    print(f"receiver QAT (mean)    {mean_cost:,.0f} samples over {n_receivers} tasks")
    print(f"receiver QAT (median)  {median_cost:,.0f} samples")
    print(f"break-even N*          {n_star_mean_exact:.1f} (mean receiver)")
    print(f"                       {results['break_even']['n_star_costliest_first']}"
          f"-{results['break_even']['n_star_cheapest_first']} across deployment orders")
    print(f"full suite (N={n_receivers})       "
          f"{results['at_full_suite']['speedup']:.1f}x less compute")
    if seconds_per_sample is not None:
        gh = results["gpu_hours"]
        device_name = (timing_provenance or {}).get("device_name") or "the measured GPU"
        print(f"\nGPU-hours on {device_name} "
              f"({seconds_per_sample * 1e3:.3f} ms/sample)")
        print(f"  donor QAT            {gh['donor_cost']:.2f} h")
        print(f"  receiver QAT (mean)  {gh['receiver_cost_mean']:.2f} h")
        print(f"  full suite baseline  {gh['baseline_cost_full_suite']:.2f} h")
        print(f"  full suite ours      {gh['ours_cost_full_suite']:.2f} h")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
