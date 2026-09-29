"""998 — Cross-family aggregation of lambda sensitivity statistics

Merges the per-family JSONs written by the compute_lambda_curves_*.py scripts
into a single lambda_curves.json holding the numbers the answer to "how
sensitive is the method to lambda" rests on:

    - per-model width of the region around the optimum that still pays off, and
      of the plateau within tolerance of the peak: how much precision in lambda
      actually buys anything
    - what the data-free default lambda = 1 costs, as a retention fraction
      against vanilla PTQ or as a regret in accuracy points against the oracle
      lambda, depending on the baseline mode of the inputs
    - how often the default is itself optimal, and how often the curve is
      unimodal as the local quadratic picture behind Proposition 1 predicts
    - the lambda* distribution, and the curve pooled over donor-receiver cells
    - per-family and overall aggregates, both macro (mean over models) and
      micro (pooled over cells)

One subtlety governs the pooled curves.  Models were not all swept on the same
lambda grid: some carry the 11-point shared grid, some a 40-point 0.05..2.0 one.
Pooling a mean over a lambda that only half the models measured would silently
compare different model sets at different lambdas, so every pooled curve is
restricted to the intersection of the grids of the models entering it, and that
intersection is recorded in the output.  Per-model statistics keep each model's
own full grid; only the pooled traces are restricted.

Reads and writes inside evaluations/998_rebuttal/003_lambda_sensitivity/.
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

from lambda_curves_common import (
    BASELINE_FP_PTQ,
    BASELINE_UNIT,
    UNIT_ALPHA,
    curve_key,
    dist_stats,
    pooled_curve,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT_OUT = "evaluations/998_rebuttal/003_lambda_sensitivity"

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
MODEL_STAT_BLOCKS = ["cross_task", "same_task", "universal_donor"]

# Per-model scalars the macro aggregate averages over models.  Each is a
# (block key, path into that block) pair; equal weight per backbone, so a 22x22
# vision matrix does not drown out an 11x11 text one.
MACRO_FIELDS = [
    ("interval_width",  ["median"]),
    ("plateau_width",   ["median"]),
    ("unit_retention",  ["median"]),
    ("unit_regret",     ["median"]),
    ("lambda_star",     ["median"]),
    ("frac_unimodal",   []),
    ("frac_unit_optimal", []),
    ("frac_wins_at_unit", []),
    ("frac_right_censored", []),
    ("frac_left_censored", []),
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
    parser.add_argument("--split",       required=True, choices=["val", "test"])
    parser.add_argument("--baseline",    required=True,
                        choices=[BASELINE_FP_PTQ, BASELINE_UNIT])
    parser.add_argument("--grid",        required=True, choices=["shared", "full"])
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
        f"split={args.split}",
        f"baseline={args.baseline}",
        f"grid={args.grid}",
    )


# ---------------------------------------------------------------------------
# Statistics — pooled over cells (micro)
# ---------------------------------------------------------------------------
def _pooled_grid(entries):
    """Intersection of the per-model grids, as a sorted list of floats.

    Restricting to the intersection is what keeps a pooled mean at a given
    lambda a statement about the same set of models at every lambda.
    """
    grids = [set(e["grid"]) for e in entries if e.get("grid")]
    if not grids:
        return []
    common = set.intersection(*grids)
    return sorted(common)


def _lambda_star_histogram(cells):
    histogram = {}
    for c in cells:
        stats = c.get("stats")
        if not stats or stats.get("lambda_star") is None:
            continue
        key = str(stats["lambda_star"])
        histogram[key] = histogram.get(key, 0) + 1
    return dict(sorted(histogram.items(), key=lambda kv: float(kv[0])))


def _micro_stats(cells, grid):
    """Distributions and the pooled curve over a flat list of cells."""
    stats = [c["stats"] for c in cells if c.get("stats")]
    if not stats:
        return {"n": 0}

    intervals = [s["interval"] for s in stats if s["interval"] is not None]
    plateaus  = [s["plateau"]  for s in stats if s["plateau"]  is not None]
    flags     = lambda key: [s[key] for s in stats if s.get(key) is not None]

    def _frac(values):
        return sum(1 for v in values if v) / len(values) if values else None

    return {
        "n":              len(stats),
        "curve":          pooled_curve([c["curve"] for c in cells if c.get("curve")],
                                       grid),
        "interval_width": dist_stats([i["width"] for i in intervals]),
        "interval_lo":    dist_stats([i["lo"]    for i in intervals]),
        "interval_hi":    dist_stats([i["hi"]    for i in intervals]),
        "plateau_width":  dist_stats([p["width"] for p in plateaus]),
        "lambda_star":    dist_stats([s["lambda_star"] for s in stats]),
        "unit_retention": dist_stats([s["unit_retention"] for s in stats]),
        "unit_regret":    dist_stats([s["unit_regret"] for s in stats]),
        "frac_unimodal":       _frac(flags("unimodal")),
        "frac_unit_optimal":   _frac(flags("unit_is_optimal")),
        "frac_wins_at_unit":   _frac(flags("wins_at_unit")),
        "frac_left_censored":  _frac([i["left_censored"]  for i in intervals]),
        "frac_right_censored": _frac([i["right_censored"] for i in intervals]),
        "n_no_interval":  sum(1 for s in stats if s["interval"] is None),
        "lambda_star_histogram": _lambda_star_histogram(cells),
    }


# ---------------------------------------------------------------------------
# Statistics — averaged over models (macro)
# ---------------------------------------------------------------------------
def _dig(block, path):
    value = block
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _macro_stats(entries, block_name):
    """Mean over models of each model's own summary scalars."""
    out = {"n_models": 0}
    per_field = {}
    for field, path in MACRO_FIELDS:
        values = []
        for e in entries:
            block = e.get(block_name)
            if not block:
                continue
            value = _dig(block.get(field), path) if path else block.get(field)
            if value is not None:
                values.append(value)
        per_field[field] = {
            "n_models": len(values),
            "mean":     statistics.fmean(values) if values else None,
            "median":   statistics.median(values) if values else None,
            "min":      min(values) if values else None,
            "max":      max(values) if values else None,
        }
    out["n_models"] = max((v["n_models"] for v in per_field.values()), default=0)
    out["fields"] = per_field
    return out


def summarize(entries, cells, grid):
    cross = [c for c in cells if not c["same_task"]]
    same  = [c for c in cells if c["same_task"]]
    return {
        "pooled_grid": grid,
        "cross_task": {
            "macro": _macro_stats(entries, "cross_task"),
            "micro": _micro_stats(cross, grid),
        },
        "same_task": {
            "macro": _macro_stats(entries, "same_task"),
            "micro": _micro_stats(same, grid),
        },
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_family(out_dir, family):
    path = os.path.join(out_dir, f"lambda_curves_{family}.json")
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
# Reporting
# ---------------------------------------------------------------------------
def _print_summary(label, summary, baseline):
    cross = summary["cross_task"]
    micro = cross["micro"]
    if micro["n"] == 0:
        print(f"{label}: no cells")
        return

    bits = [f"{cross['macro']['n_models']} models",
            f"{micro['n']} cross-task cells",
            f"median interval width={micro['interval_width']['median']:.2f}",
            f"median plateau width={micro['plateau_width']['median']:.2f}",
            f"median lambda*={micro['lambda_star']['median']}"]
    if baseline == BASELINE_FP_PTQ:
        if micro["unit_retention"]["median"] is not None:
            bits.append(f"median unit retention={micro['unit_retention']['median']:.2f}")
        if micro["frac_wins_at_unit"] is not None:
            bits.append(f"wins at lambda=1 in {micro['frac_wins_at_unit'] * 100:.1f}%")
    else:
        if micro["unit_regret"]["median"] is not None:
            bits.append(f"median unit regret={micro['unit_regret']['median'] * 100:.2f}pp")
        if micro["frac_unit_optimal"] is not None:
            bits.append(f"lambda=1 optimal in {micro['frac_unit_optimal'] * 100:.1f}%")
    if micro["frac_unimodal"] is not None:
        bits.append(f"unimodal in {micro['frac_unimodal'] * 100:.1f}%")
    print(f"{label}: " + ", ".join(bits))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    out_dir = _out_dir(args)

    families   = {}
    all_models = []
    all_cells  = []
    dropped    = []

    for family in args.families:
        data = load_family(out_dir, family)
        if data is None:
            continue

        if data["config"]["baseline"] != args.baseline:
            print(f"  [SKIP] {family}: baseline={data['config']['baseline']} "
                  f"in the file, --baseline {args.baseline} requested",
                  file=sys.stderr)
            continue

        entries      = []
        family_cells = []
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
                "grid":         entry["grid"],
                "n_grid_points": entry["n_grid_points"],
            }
            for block in MODEL_STAT_BLOCKS:
                record[block] = entry.get(block)

            entries.append(record)
            all_models.append(record)
            family_cells.extend(entry["pairs"])

        if not entries:
            continue

        all_cells.extend(family_cells)
        family_grid = _pooled_grid(entries)
        families[family] = {
            "display_name": FAMILY_DISPLAY_NAMES.get(family, family),
            "modality":     data["modality"],
            "datasets":     data["datasets"],
            "config":       data["config"],
            "models":       entries,
            "summary":      summarize(entries, family_cells, family_grid),
        }

        _print_summary(FAMILY_DISPLAY_NAMES.get(family, family),
                       families[family]["summary"], args.baseline)

    if not all_models:
        print("No family JSONs found — run the compute_lambda_curves_*.py scripts "
              "first, with matching --split/--baseline/--grid.", file=sys.stderr)
        sys.exit(1)

    overall_grid = _pooled_grid(all_models)
    overall = summarize(all_models, all_cells, overall_grid)

    results = {
        "seed":        args.seed,
        "qat_bits":    args.qat_bits,
        "ptq_bits":    args.ptq_bits,
        "granularity": args.granularity,
        "eval_split":  args.split,
        "baseline":    args.baseline,
        "grid_mode":   args.grid,
        "pooled_grid": overall_grid,
        "families":    families,
        "models":      all_models,
        "overall":     overall,
        "incomplete_models": dropped,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "lambda_curves.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    _print_summary(f"Overall ({len(all_models)} backbones)", overall, args.baseline)

    micro = overall["cross_task"]["micro"]
    iqr = micro["interval_width"]
    print(f"  interval width IQR: {iqr['p25']:.2f}-{iqr['p75']:.2f} "
          f"(right-censored in {micro['frac_right_censored'] * 100:.1f}% of cells)")
    unit_key = curve_key(UNIT_ALPHA)
    if unit_key in micro["curve"] and micro["curve"][unit_key]["n"]:
        point = micro["curve"][unit_key]
        print(f"  pooled curve at lambda=1: n={point['n']}, "
              f"median={point['median'] * 100:+.2f}pp")
    print(f"  pooled grid: {len(overall_grid)} lambdas "
          f"[{overall_grid[0]}..{overall_grid[-1]}]" if overall_grid else "  pooled grid: empty")
    if dropped:
        print(f"  {len(dropped)} model(s) with an incomplete matrix "
              f"({'dropped' if args.drop_incomplete else 'kept'})", file=sys.stderr)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
