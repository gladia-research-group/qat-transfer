"""998 — Donor benefit against donor cost

For every donor dataset, pairs up the two quantities the cost-amortization
question needs:

    cost      the compute spent producing that donor's quantization vector,
              i.e. one QAT run over the donor's training set,
              epochs * train_size training samples, read from dataset_sizes.json.
    benefit   how many *other* tasks that single QV actually helps, i.e. the
              number of receivers whose Top-1 improves over vanilla PTQ,
              averaged over the backbones of the selected family.

The ratio cost / mean_wins is the amortized price of one successful deployment,
which is what the reviewer is asking for: a donor is only worth its QAT run if
that run buys enough downstream wins.

Defaults follow the paper's data-free regime: cross-task pairs only (the
diagonal is trivially a win, since it is just the receiver's own QAT vector) and
unit scaling lambda=1 (the `delta` field).  --use-best-alpha switches to the
validation-selected lambda and --include-same-task keeps the diagonal.

Writes evaluations/998_rebuttal/002_cost_amortization/donor_benefit.json and
prints a summary table sorted by cost per win.
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

from dotenv import load_dotenv

load_dotenv()

from src.duration import mult_tag

import argparse
import json
import os
import statistics

os.chdir(_PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WIN_LOSS_ROOT = "evaluations/998_rebuttal/001_zero_shot_reframing"
EVAL_ROOT = "evaluations/998_rebuttal/002_cost_amortization"

FAMILIES = [
    "ilharco_timm_supervised",
    "ilharco_open_clip",
    "ilharco_automodelforsequenceclassification",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="ilharco_timm_supervised", choices=FAMILIES)
    parser.add_argument("--seed", type=int, default=2038)
    parser.add_argument("--source-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the DONOR checkpoints "
                             "the aggregated results were produced at.")
    parser.add_argument("--target-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the RECEIVER checkpoints.")
    parser.add_argument("--qat-bits", type=int, default=3)
    parser.add_argument("--ptq-bits", type=int, default=3)
    parser.add_argument("--granularity", default="channel", choices=["tensor", "channel"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument(
        "--use-best-alpha",
        action="store_true",
        help="Score wins at the validation-selected lambda (delta_best) instead of lambda=1.",
    )
    parser.add_argument(
        "--include-same-task",
        action="store_true",
        help="Keep the donor==receiver diagonal (excluded by default, it is trivially a win).",
    )
    parser.add_argument(
        "--out-name",
        default="donor_benefit.json",
        help="File name written under evaluations/998_rebuttal/002_cost_amortization/.",
    )
    return parser.parse_args()


def _run_dir(args):
    return os.path.join(
        f"seed={args.seed}",
        f"smult={mult_tag(args.source_epoch_mult)}",
        f"tmult={mult_tag(args.target_epoch_mult)}",
        f"qat=bits={args.qat_bits}_gran={args.granularity}",
        f"ptq=bits={args.ptq_bits}_gran={args.granularity}",
        f"split={args.split}",
    )


############################################################################
# BEGIN donor cost
############################################################################


def donor_costs(sizes):
    """One QAT run over the donor's train split, in training samples seen."""
    costs = {}
    for name, rec in sizes["datasets"].items():
        costs[name] = {
            "train_size": rec["train_size"],
            "epochs": rec["epochs"],
            "cost_samples": rec["epochs"] * rec["train_size"],
        }
    return costs


############################################################################
# END donor cost
############################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

############################################################################
# BEGIN donor benefit
############################################################################


def donor_benefit(data, delta_key, include_same_task):
    """Per donor, per model: how many receivers the donor's QV helps."""
    per_donor = {d: {"per_model": {}, "deltas": []} for d in data["datasets"]}

    for model_name, model in data["models"].items():
        counts = {d: {"n_wins": 0, "n_receivers": 0} for d in data["datasets"]}
        for pair in model["pairs"]:
            if pair["same_task"] and not include_same_task:
                continue
            donor = pair["donor"]
            delta = pair[delta_key]
            counts[donor]["n_receivers"] += 1
            counts[donor]["n_wins"] += int(delta > 0.0)
            per_donor[donor]["deltas"].append(delta)
        for donor, c in counts.items():
            per_donor[donor]["per_model"][model_name] = {
                "display_name": model["display_name"],
                "n_wins": c["n_wins"],
                "n_receivers": c["n_receivers"],
                "win_rate": c["n_wins"] / c["n_receivers"] if c["n_receivers"] else None,
            }

    return per_donor


def aggregate_donor(donor, entry, costs):
    wins = [m["n_wins"] for m in entry["per_model"].values()]
    n_receivers = [m["n_receivers"] for m in entry["per_model"].values()]
    assert len(set(n_receivers)) == 1, f"{donor}: ragged receiver counts {n_receivers}"

    mean_wins = statistics.fmean(wins)
    cost = costs.get(donor)

    rec = {
        "donor": donor,
        "n_models": len(wins),
        "n_receivers": n_receivers[0],
        "mean_wins": mean_wins,
        "min_wins": min(wins),
        "max_wins": max(wins),
        "stdev_wins": statistics.stdev(wins) if len(wins) > 1 else 0.0,
        "win_rate": mean_wins / n_receivers[0] if n_receivers[0] else None,
        "mean_delta": statistics.fmean(entry["deltas"]) if entry["deltas"] else None,
        "median_delta": statistics.median(entry["deltas"]) if entry["deltas"] else None,
        "train_size": cost["train_size"] if cost else None,
        "epochs": cost["epochs"] if cost else None,
        "cost_samples": cost["cost_samples"] if cost else None,
        "cost_per_win": None,
        "per_model": entry["per_model"],
    }
    if cost is not None and mean_wins > 0:
        rec["cost_per_win"] = cost["cost_samples"] / mean_wins
    return rec


def mark_pareto(donors):
    """A donor is Pareto-optimal if no other donor is both cheaper and better.

    Ties are broken so that a strictly dominated duplicate is dropped: another
    donor dominates if it is no worse on both axes and strictly better on one.
    """
    scored = [d for d in donors if d["cost_samples"] is not None]
    for d in donors:
        d["pareto_optimal"] = False
    for d in scored:
        dominated = any(
            o is not d
            and o["cost_samples"] <= d["cost_samples"]
            and o["mean_wins"] >= d["mean_wins"]
            and (o["cost_samples"] < d["cost_samples"] or o["mean_wins"] > d["mean_wins"])
            for o in scored
        )
        d["pareto_optimal"] = not dominated
    return donors


############################################################################
# END donor benefit
############################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

############################################################################
# BEGIN reporting
############################################################################


def print_summary(results):
    donors = results["donors"]
    s = results["settings"]
    print(
        f"family={results['family']}  models={results['n_models']}  "
        f"lambda={s['alpha_regime']}  same_task="
        f"{'in' if s['include_same_task'] else 'ex'}cluded  win = {s['win_criterion']}"
    )
    header = (
        f"{'donor':<22}{'cost (M)':>10}{'epochs':>8}{'train':>10}"
        f"{'wins':>7}{'spread':>12}{'win%':>8}{'mean d':>9}{'cost/win (M)':>14}  pareto"
    )
    print(header)
    print("-" * len(header))
    for d in donors:
        cost_m = "--" if d["cost_samples"] is None else f"{d['cost_samples'] / 1e6:.3f}"
        cpw = "--" if d["cost_per_win"] is None else f"{d['cost_per_win'] / 1e6:.3f}"
        epochs = "--" if d["epochs"] is None else str(d["epochs"])
        train = "--" if d["train_size"] is None else str(d["train_size"])
        mean_d = "--" if d["mean_delta"] is None else f"{d['mean_delta'] * 100:+.2f}"
        spread = f"[{d['min_wins']}-{d['max_wins']}]"
        print(
            f"{d['donor']:<22}{cost_m:>10}{epochs:>8}{train:>10}"
            f"{d['mean_wins']:>7.1f}{spread:>12}"
            f"{(d['win_rate'] or 0) * 100:>7.1f}%{mean_d:>9}{cpw:>14}"
            f"  {'*' if d['pareto_optimal'] else ''}"
        )


############################################################################
# END reporting
############################################################################


def main():
    args = parse_args()

    delta_key = "delta_best" if args.use_best_alpha else "delta"

    sizes_path = os.path.join(EVAL_ROOT, "dataset_sizes.json")
    assert os.path.exists(sizes_path), (
        f"{sizes_path} not found. Run collect_dataset_sizes.py first."
    )
    with open(sizes_path) as f:
        sizes = json.load(f)

    win_loss_path = os.path.join(
        WIN_LOSS_ROOT, _run_dir(args), f"win_loss_{args.family}.json"
    )
    assert os.path.exists(win_loss_path), (
        f"{win_loss_path} not found. Run aggregate_win_loss.py first."
    )
    with open(win_loss_path) as f:
        data = json.load(f)

    costs = donor_costs(sizes)
    per_donor = donor_benefit(data, delta_key, args.include_same_task)
    donors = [aggregate_donor(d, e, costs) for d, e in per_donor.items()]
    donors = mark_pareto(donors)

    missing_cost = sorted(d["donor"] for d in donors if d["cost_samples"] is None)
    if missing_cost:
        print(f"[WARN] no cost entry in dataset_sizes.json for: {', '.join(missing_cost)}")

    donors.sort(
        key=lambda d: (
            d["cost_per_win"] if d["cost_per_win"] is not None else float("inf")
        )
    )

    results = {
        "family": args.family,
        "display_family": data.get("display_name", args.family),
        "modality": data["modality"],
        "config": data["config"],
        "settings": {
            "seed": args.seed,
            "qat_bits": args.qat_bits,
            "ptq_bits": args.ptq_bits,
            "granularity": args.granularity,
            "split": args.split,
            "delta_key": delta_key,
            "alpha_regime": "best" if args.use_best_alpha else "unit",
            "include_same_task": args.include_same_task,
            "win_criterion": f"{delta_key} > 0",
            "cost_definition": "epochs * train_size (training samples seen by one QAT run)",
        },
        "win_loss_path": win_loss_path,
        "dataset_sizes_path": sizes_path,
        "n_models": len(data["models"]),
        "models": {k: v["display_name"] for k, v in data["models"].items()},
        "datasets": data["datasets"],
        "missing_cost": missing_cost,
        "donors": donors,
    }

    os.makedirs(EVAL_ROOT, exist_ok=True)
    out_path = os.path.join(EVAL_ROOT, args.out_name)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print_summary(results)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
