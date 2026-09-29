"""998 — Cross-family aggregation of zero-shot reframing statistics

Merges the per-family JSONs written by the compute_win_loss_*.py scripts into a
single win_loss.json holding the numbers the zero-shot reframing rests on:

    - per-model cross-task and same-task win rates, at lambda=1 (data-free) and
      at the validation-selected lambda_best (needs receiver data)
    - recovery ratios: the fraction of the receiver's own QAT gain that transfer
      recovers, i.e. the observable side of Proposition 1's cos^2 law
    - lambda_best distribution, overshoot fraction, and how many pairs that fail
      at unit scaling are merely overshooting a positively aligned direction
    - per-family and overall aggregates, both macro (mean over models) and
      micro (pooled over donor-receiver pairs)

Reads and writes inside evaluations/998_rebuttal/001_zero_shot_reframing/.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is the working directory and on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from src.duration import mult_tag

os.chdir(_PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT_OUT = "evaluations/998_rebuttal/001_zero_shot_reframing"

FAMILIES = [
    "ilharco_timm_supervised",
    "ilharco_open_clip",
    "ilharco_automodelforsequenceclassification",
]

FAMILY_DISPLAY_NAMES = {
    "ilharco_timm_supervised":                    "timm (supervised)",
    "ilharco_open_clip":                          "OpenCLIP",
    "ilharco_automodelforsequenceclassification": "Text",
}

# Per-model blocks copied verbatim into the merged record for the table/figure.
MODEL_STAT_BLOCKS = [
    "cross_task", "same_task", "cross_task_best", "same_task_best",
    "recovery", "recovery_best", "alpha_best", "alignment", "tuning_gain",
    "universal_donor",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",        required=True, type=int)
    parser.add_argument("--source-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the DONOR checkpoints "
                             "the aggregated results were produced at.")
    parser.add_argument("--target-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the RECEIVER checkpoints.")
    parser.add_argument("--qat-bits",    required=True, type=int)
    parser.add_argument("--ptq-bits",    required=True, type=int)
    parser.add_argument("--granularity", required=True, choices=["tensor", "channel"])
    parser.add_argument("--families",    default=FAMILIES, nargs="+", choices=FAMILIES,
                        help="Families to merge (default: all three).")
    parser.add_argument("--drop-incomplete", action="store_true",
                        help="Drop models whose donor-receiver matrix is not fully "
                             "populated instead of aggregating over partial matrices.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
def _out_dir(args):
    return os.path.join(
        EVAL_ROOT_OUT,
        f"seed={args.seed}",
        f"smult={mult_tag(args.source_epoch_mult)}",
        f"tmult={mult_tag(args.target_epoch_mult)}",
        f"qat=bits={args.qat_bits}_gran={args.granularity}",
        f"ptq=bits={args.ptq_bits}_gran={args.granularity}",
        "split=test",
    )


# ---------------------------------------------------------------------------
# Statistics — pooled over pairs (micro)
# ---------------------------------------------------------------------------
def _delta_stats(deltas):
    if not deltas:
        return {
            "n": 0, "n_win": 0, "n_loss": 0, "win_rate": None,
            "mean": None, "median": None, "min": None, "max": None,
        }
    n_win = sum(1 for v in deltas if v > 0)
    return {
        "n":        len(deltas),
        "n_win":    n_win,
        "n_loss":   len(deltas) - n_win,
        "win_rate": n_win / len(deltas),
        "mean":     statistics.fmean(deltas),
        "median":   statistics.median(deltas),
        "min":      min(deltas),
        "max":      max(deltas),
    }


def _ratio_stats(ratios):
    vals = [r for r in ratios if r is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None,
                "frac_above_ceiling": None, "p25": None, "p75": None}
    svals = sorted(vals)
    return {
        "n":                  len(vals),
        "mean":               statistics.fmean(vals),
        "median":             statistics.median(vals),
        "p25":                svals[len(svals) // 4],
        "p75":                svals[(3 * len(svals)) // 4],
        "frac_above_ceiling": sum(1 for v in vals if v > 1.0) / len(vals),
    }


def _alpha_best_stats(cross_pairs):
    alphas = [p["alpha_best"] for p in cross_pairs if p["alpha_best"] is not None]
    if not alphas:
        return {"n": 0}

    histogram = {}
    for a in alphas:
        histogram[str(a)] = histogram.get(str(a), 0) + 1

    return {
        "n":            len(alphas),
        "mean":         statistics.fmean(alphas),
        "median":       statistics.median(alphas),
        "frac_below_1": sum(1 for a in alphas if a < 1.0) / len(alphas),
        "frac_at_1":    sum(1 for a in alphas if a == 1.0) / len(alphas),
        "frac_above_1": sum(1 for a in alphas if a > 1.0) / len(alphas),
        "histogram":    dict(sorted(histogram.items(), key=lambda kv: float(kv[0]))),
    }


def _alignment_stats(cross_pairs):
    failing = [p for p in cross_pairs
               if p["delta"] <= 0 and p["delta_best"] is not None]
    if not failing:
        return {"n_failing_at_unit": 0}
    aligned = [p for p in failing if p["delta_best"] > 0]
    return {
        "n_failing_at_unit":      len(failing),
        "n_recovered_at_best":    len(aligned),
        "frac_recovered_at_best": len(aligned) / len(failing),
        "median_alpha_best_of_recovered": (
            statistics.median([p["alpha_best"] for p in aligned]) if aligned else None
        ),
    }


def _tuning_gain(cross_pairs):
    gains = [p["delta_best"] - p["delta"] for p in cross_pairs
             if p["delta_best"] is not None]
    if not gains:
        return {"n": 0}
    return {
        "n":                    len(gains),
        "mean":                 statistics.fmean(gains),
        "median":               statistics.median(gains),
        "frac_worse_than_unit": sum(1 for g in gains if g < 0) / len(gains),
    }


# ---------------------------------------------------------------------------
# Statistics — averaged over models (macro)
# ---------------------------------------------------------------------------
def _macro_stats(entries, block):
    """Mean over models of each model's own statistics for one block.

    Gives every backbone equal weight, so a 22x22 vision matrix does not drown
    out an 11x11 text one.
    """
    win_rates = [e[block]["win_rate"] for e in entries
                 if e.get(block) and e[block]["win_rate"] is not None]
    means     = [e[block]["mean"] for e in entries
                 if e.get(block) and e[block]["mean"] is not None]
    if not win_rates:
        return {"n_models": 0, "win_rate": None, "mean": None,
                "n_models_majority_positive": 0, "n_models_mean_positive": 0}
    return {
        "n_models":                    len(win_rates),
        "win_rate":                    statistics.fmean(win_rates),
        "mean":                        statistics.fmean(means) if means else None,
        "n_models_majority_positive":  sum(1 for v in win_rates if v > 0.5),
        "n_models_mean_positive":      sum(1 for v in means if v > 0),
    }


def summarize(entries, pairs):
    cross = [p for p in pairs if not p["same_task"]]
    same  = [p for p in pairs if p["same_task"]]
    return {
        "cross_task": {
            "macro": _macro_stats(entries, "cross_task"),
            "micro": _delta_stats([p["delta"] for p in cross]),
        },
        "same_task": {
            "macro": _macro_stats(entries, "same_task"),
            "micro": _delta_stats([p["delta"] for p in same]),
        },
        "cross_task_best": {
            "macro": _macro_stats(entries, "cross_task_best"),
            "micro": _delta_stats([p["delta_best"] for p in cross
                                   if p["delta_best"] is not None]),
        },
        "recovery":      _ratio_stats([p["recovery"] for p in cross]),
        "recovery_best": _ratio_stats([p["recovery_best"] for p in cross]),
        "alpha_best":    _alpha_best_stats(cross),
        "alignment":     _alignment_stats(cross),
        "tuning_gain":   _tuning_gain(cross),
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_family(out_dir, family):
    path = os.path.join(out_dir, f"win_loss_{family}.json")
    if not os.path.exists(path):
        print(f"  [MISSING] {path}", file=sys.stderr)
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  [READ ERROR] {path}: {e}", file=sys.stderr)
        return None


def _is_complete(entry):
    return len(entry["pairs"]) == entry["n_cells_expected"]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = _out_dir(args)

    families   = {}
    all_models = []
    all_pairs  = []
    dropped    = []

    for family in args.families:
        data = load_family(out_dir, family)
        if data is None:
            continue

        entries = []
        family_pairs = []
        for model_name, entry in data["models"].items():
            complete = _is_complete(entry)
            if not complete:
                dropped.append({
                    "family":     family,
                    "model_name": model_name,
                    "n_pairs":    len(entry["pairs"]),
                    "n_expected": entry["n_cells_expected"],
                    "dropped":    args.drop_incomplete,
                })
                if args.drop_incomplete:
                    continue

            record = {
                "family":       family,
                "modality":     data["modality"],
                "model_name":   model_name,
                "display_name": entry["display_name"],
                "complete":     complete,
                "n_receivers_with_a_positive_donor": sum(
                    1 for r in entry["per_receiver"].values()
                    if r["n_donors_positive"] > 0
                ),
                "n_receivers": len(entry["per_receiver"]),
            }
            for block in MODEL_STAT_BLOCKS:
                record[block] = entry.get(block)

            entries.append(record)
            all_models.append(record)
            family_pairs.extend(entry["pairs"])

        all_pairs.extend(family_pairs)
        families[family] = {
            "display_name": FAMILY_DISPLAY_NAMES.get(family, family),
            "modality":     data["modality"],
            "datasets":     data["datasets"],
            "config":       data["config"],
            "models":       entries,
            "summary":      summarize(entries, family_pairs),
        }

        summary = families[family]["summary"]
        cross = summary["cross_task"]
        best  = summary["cross_task_best"]
        print(f"{FAMILY_DISPLAY_NAMES.get(family, family)}: "
              f"{cross['macro']['n_models']} models, "
              f"micro win_rate={cross['micro']['win_rate'] * 100:.1f}% at lambda=1 "
              f"-> {best['micro']['win_rate'] * 100:.1f}% at lambda_best, "
              f"median recovery={summary['recovery']['median']:.2f}")

    if not all_models:
        print("No family JSONs found — run the compute_win_loss_*.py scripts first.",
              file=sys.stderr)
        sys.exit(1)

    overall = summarize(all_models, all_pairs)

    results = {
        "seed":        args.seed,
        "qat_bits":    args.qat_bits,
        "ptq_bits":    args.ptq_bits,
        "granularity": args.granularity,
        "eval_split":  "test",
        "baseline":    "fp_ptq",
        "families":    families,
        "models":      all_models,
        "overall":     overall,
        "incomplete_models": dropped,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "win_loss.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    cross = overall["cross_task"]
    best  = overall["cross_task_best"]
    same  = overall["same_task"]
    align = overall["alignment"]
    alpha = overall["alpha_best"]
    gain  = overall["tuning_gain"]

    print(f"\nOverall ({cross['macro']['n_models']} backbones):")
    print(f"  lambda=1     micro: {cross['micro']['n_win']}/{cross['micro']['n']} "
          f"({cross['micro']['win_rate'] * 100:.1f}%), "
          f"mean={cross['micro']['mean'] * 100:+.1f}pp")
    print(f"  lambda=1     macro: win_rate={cross['macro']['win_rate'] * 100:.1f}%, "
          f"{cross['macro']['n_models_majority_positive']}/"
          f"{cross['macro']['n_models']} models majority-positive")
    print(f"  lambda_best  micro: {best['micro']['n_win']}/{best['micro']['n']} "
          f"({best['micro']['win_rate'] * 100:.1f}%), "
          f"mean={best['micro']['mean'] * 100:+.1f}pp")
    print(f"  lambda_best  macro: win_rate={best['macro']['win_rate'] * 100:.1f}%, "
          f"{best['macro']['n_models_majority_positive']}/"
          f"{best['macro']['n_models']} models majority-positive")
    print(f"  tuning gain: mean={gain['mean'] * 100:+.1f}pp, "
          f"{gain['frac_worse_than_unit'] * 100:.1f}% of pairs worse than lambda=1")
    print(f"  recovery of QAT ceiling: median={overall['recovery']['median']:.2f} "
          f"at lambda=1, {overall['recovery_best']['median']:.2f} at lambda_best "
          f"(IQR {overall['recovery']['p25']:.2f}-{overall['recovery']['p75']:.2f})")
    print(f"  lambda_best: median={alpha['median']}, "
          f"{alpha['frac_below_1'] * 100:.1f}% below 1.0 (overshoot at unit scaling)")
    print(f"  of {align['n_failing_at_unit']} pairs failing at lambda=1, "
          f"{align['n_recovered_at_best']} "
          f"({align['frac_recovered_at_best'] * 100:.1f}%) win at some positive lambda "
          f"=> positively aligned, overshoot not anti-alignment")
    print(f"  same-task (QAT ceiling) micro: "
          f"{same['micro']['n_win']}/{same['micro']['n']} positive, "
          f"mean={same['micro']['mean'] * 100:+.1f}pp")
    if dropped:
        print(f"  {len(dropped)} model(s) with an incomplete matrix "
              f"({'dropped' if args.drop_incomplete else 'kept'})", file=sys.stderr)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
