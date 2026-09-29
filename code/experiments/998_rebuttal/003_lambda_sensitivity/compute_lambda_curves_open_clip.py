"""998 — Lambda sensitivity curves (OpenCLIP)

OpenCLIP counterpart of compute_lambda_curves_timm_supervised.py; see that
script's docstring for what the curve statistics mean, why lambda_star is kept
distinct from the validation-selected alpha_best, and why the dense grids force
the analysis onto the validation split with the lambda = 1 default as the
reference rather than vanilla PTQ.

ViT-B-16 / laion2b_s34b_b88k is the one model in this family swept on the finer
0.05..2.0 grid, so `--grid full` gives it 40 points per cell against the 11 the
other backbones have.  Pass `--grid shared` when the curves must be compared
across backbones.

Writes a per-family JSON consumed by aggregate_lambda_curves.py.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is the working directory and on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

os.chdir(_PROJECT_ROOT)

from src.duration import mult_path_frag, mult_tag, role_path_frag
from src.vision.data.common import DATASET_NAME_TO_EPOCHS
from src.vision.utils import sanitize_open_clip_model_name

from lambda_curves_common import (
    BASELINE_FP_PTQ,
    BASELINE_UNIT,
    UNIT_ALPHA,
    curve_key,
    curve_stats,
    discover_grid,
    resolve_grid,
    summarize_cells,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FAMILY   = "ilharco_open_clip"
MODALITY = "vision"

EVAL_ROOT_BASELINES = "evaluations/vision/ilharco_open_clip/000_baselines/vision"
EVAL_ROOT_QV        = "evaluations/vision/ilharco_open_clip/001_qat_transfer/vision/qv_transfer"

EVAL_ROOT_OUT = "evaluations/998_rebuttal/003_lambda_sensitivity"

# Validation-selected best scaling, written by pick_best_alpha.py.  This family
# writes a single file rather than one per head variant.  Kept for reference
# only: the curve statistics use the grid argmax on the evaluated split.
BEST_ALPHA_FILE = "best_alpha.json"
BEST_ALPHA_KEY  = "val_accuracy_patched_qat_ptq"

MODEL_DISPLAY_NAMES = {
    ("ViT-B-16", "laion2b_s34b_b88k"): "ViT-B/16 (LAION)",
    ("ViT-B-16", "openai"):            "ViT-B/16 (OpenAI)",
    ("ViT-L-14", "laion2b_s32b_b82k"): "ViT-L/14 (LAION)",
    ("ViT-H-14", "laion2b_s32b_b79k"): "ViT-H/14 (LAION)",
}


def _metric_key(split):
    """Patched-model accuracy key in a transfer eval_results.json."""
    return f"{split}_accuracy_patched_qat_ptq"


def _baseline_key(split):
    """Vanilla-PTQ accuracy key in a 000_baselines eval_results.json."""
    return f"{split}_accuracy"


def _frac_positive_meaning(baseline):
    if baseline == BASELINE_FP_PTQ:
        return "Delta > 0: the patched model beats vanilla PTQ"
    return "r > 0: the patched model beats the lambda = 1 default"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-names",     required=True, nargs="+",
                        help="open_clip model names, e.g. ViT-B-16 ViT-L-14")
    parser.add_argument("--pretrained-tags", required=True, nargs="+",
                        help="open_clip pretrained tags (parallel to --model-names)")
    parser.add_argument("--seed",            required=True, type=int)
    parser.add_argument("--source-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the DONOR checkpoints.")
    parser.add_argument("--target-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the RECEIVER checkpoints, "
                             "and of the baselines it is compared against.")

    parser.add_argument("--optim",           required=True, choices=["adamw", "sgd"])
    parser.add_argument("--lr",              required=True, type=float)
    parser.add_argument("--wd",              required=True, type=float)
    parser.add_argument("--ls",              required=True, type=float)
    parser.add_argument("--wl",              required=True, type=int)
    parser.add_argument("--max-grad-norm",   required=True, type=float)
    parser.add_argument("--batch-sizes",     required=True, type=int, nargs="+",
                        help="Batch sizes (one per model, parallel to --model-names)")

    parser.add_argument("--qat-bits",        required=True, type=int)
    parser.add_argument("--ptq-bits",        required=True, type=int)
    parser.add_argument("--granularity",     required=True, choices=["tensor", "channel"])
    parser.add_argument("--skip-modules",    required=True, nargs="+",
                        help="One or more module names to skip during quantization "
                             "(no default: must be specified explicitly).")

    parser.add_argument("--split",           required=True, choices=["val", "test"],
                        help="Evaluation split the curves are read from. The dense "
                             "lambda grids exist only on val; test has two or three "
                             "points per cell.")
    parser.add_argument("--baseline",        required=True,
                        choices=[BASELINE_FP_PTQ, BASELINE_UNIT],
                        help="What the curve is measured against: 'fp_ptq' for "
                             "Delta(lambda) against vanilla PTQ (needs a "
                             "vanilla-PTQ evaluation on the same split), 'unit' "
                             "for r(lambda) against the lambda = 1 default.")
    parser.add_argument("--grid",            required=True, choices=["shared", "full"],
                        help="'shared' restricts to the lambdas swept for every "
                             "model, for cross-backbone comparability; 'full' uses "
                             "every lambda present on disk.")

    parser.add_argument("--universal-donor", default=None,
                        help="Donor to report separately as the data-free default "
                             "(e.g. ImageNet). Omit to skip.")

    args = parser.parse_args()
    if len(args.model_names) != len(args.pretrained_tags):
        parser.error("--model-names and --pretrained-tags must have the same length")
    if len(args.model_names) != len(args.batch_sizes):
        parser.error("--model-names and --batch-sizes must have the same length")
    return args


# ---------------------------------------------------------------------------
# Path-fragment helpers
# ---------------------------------------------------------------------------
def _skip_tag(skip_modules):
    return "-".join(sorted(skip_modules)) if len(skip_modules) > 0 else "none"


def _optim_frag(args, batch_size):
    return (f"optim=adamw_lr={args.lr}_wd={args.wd}_ls={args.ls}"
            f"_wl={args.wl}_mgn={args.max_grad_norm}_bs={batch_size}")


def _qat_frag(args):
    return f"qat=bits={args.qat_bits}_gran={args.granularity}_skip={_skip_tag(args.skip_modules)}"


def _ptq_frag(args):
    return f"ptq=bits={args.ptq_bits}_gran={args.granularity}_skip={_skip_tag(args.skip_modules)}"


# ---------------------------------------------------------------------------
# Path builders
# ---------------------------------------------------------------------------
def _fp_ptq_path(model_dir, dataset, seed, target_epoch_mult, optim_frag, ptq_frag, split):
    """Vanilla-PTQ baseline for one receiver.

    The test-split baselines predate any notion of a split in this path and are
    read by 001, 002 and every visualization, so their location is left alone;
    val-split baselines get a split= leaf below seed=.
    """
    parts = [
        EVAL_ROOT_BASELINES, "fp_ptq", model_dir, dataset,
        optim_frag, ptq_frag, f"seed={seed}",
    ]
    if split != "test":
        parts.append(f"split={split}")
    parts.append("eval_results.json")
    return os.path.join(*parts)


def _qv_cell_prefix(model_dir, donor, receiver, seed, source_epoch_mult, target_epoch_mult, optim_frag, qat_frag, ptq_frag):
    """Everything above the qv=alpha=... level for one donor-receiver cell."""
    return os.path.join(
        EVAL_ROOT_QV, model_dir,
        role_path_frag("src", donor, seed, source_epoch_mult),
        role_path_frag("tgt", receiver, seed, target_epoch_mult),
        optim_frag, qat_frag, ptq_frag,
    )


def _qv_eval_path(cell_prefix, alpha, split):
    return os.path.join(cell_prefix, f"qv=alpha={alpha}", f"split={split}",
                        "eval_results.json")


def _load_best_alpha(cell_prefix):
    """Validation-selected scaling for one cell, or None if not picked yet."""
    path = os.path.join(cell_prefix, BEST_ALPHA_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            info = json.load(f).get(BEST_ALPHA_KEY)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  [READ ERROR] {path}: {e}", file=sys.stderr)
        return None
    return info["alpha"] if info is not None else None


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------
def _load_value(path, key):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f).get(key)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  [READ ERROR] {path}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def cell_prefixes(model_dir, datasets, args, optim_frag, qat_frag, ptq_frag):
    """Every donor-receiver cell directory for one model."""
    return [
        _qv_cell_prefix(model_dir, donor, receiver, args.seed,
                        args.source_epoch_mult, args.target_epoch_mult,
                        optim_frag, qat_frag, ptq_frag)
        for receiver in datasets
        for donor in datasets
    ]


def load_pairs(model_dir, datasets, args, grid, optim_frag, qat_frag, ptq_frag):
    """The curve over `grid` for every (donor, receiver) cell."""
    pairs = []
    missing = []

    metric_key   = _metric_key(args.split)
    baseline_key = _baseline_key(args.split)
    unit_key     = curve_key(UNIT_ALPHA)

    for receiver in datasets:
        baseline = None
        if args.baseline == BASELINE_FP_PTQ:
            baseline_path = _fp_ptq_path(model_dir, receiver, args.seed,
                                         optim_frag, ptq_frag, args.split)
            baseline = _load_value(baseline_path, baseline_key)
            if baseline is None:
                missing.append(baseline_path)
                continue

        for donor in datasets:
            cell_prefix = _qv_cell_prefix(model_dir, donor, receiver, args.seed,
                                          args.source_epoch_mult, args.target_epoch_mult,
                                          optim_frag, qat_frag, ptq_frag)

            accs = {}
            for alpha in grid:
                path = _qv_eval_path(cell_prefix, alpha, args.split)
                acc = _load_value(path, metric_key)
                if acc is None:
                    missing.append(path)
                    continue
                accs[curve_key(alpha)] = acc

            if not accs:
                continue

            if args.baseline == BASELINE_FP_PTQ:
                reference = baseline
            else:
                # r(lambda) is undefined without the default it is measured
                # against, so such a cell is dropped rather than silently
                # rebased on some other lambda.
                if unit_key not in accs:
                    missing.append(_qv_eval_path(cell_prefix, UNIT_ALPHA, args.split))
                    continue
                reference = accs[unit_key]

            curve = {k: v - reference for k, v in accs.items()}

            pairs.append({
                "donor":        donor,
                "receiver":     receiver,
                "reference_acc": reference,
                "same_task":    donor == receiver,
                "curve":        curve,
                "stats":        curve_stats(curve, grid, args.baseline),
                # Validation-selected scaling, for reference against 001 only.
                "alpha_best_val": _load_best_alpha(cell_prefix),
            })

    return pairs, missing


# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------
def _out_dir(args):
    return os.path.join(
        EVAL_ROOT_OUT,
        f"seed={args.seed}",
        # The aggregate spans a whole donor x receiver matrix at one pair of
        # budgets. Without these two components an aggregate computed at a
        # different multiplier would silently overwrite this one.
        f"smult={mult_tag(args.source_epoch_mult)}",
        f"tmult={mult_tag(args.target_epoch_mult)}",
        f"qat=bits={args.qat_bits}_gran={args.granularity}",
        f"ptq=bits={args.ptq_bits}_gran={args.granularity}",
        f"split={args.split}",
        f"baseline={args.baseline}",
        f"grid={args.grid}",
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _print_model_summary(entry, baseline):
    summary = entry["cross_task"]
    if summary["n"] == 0:
        print(f"  {entry['display_name']}: no cross-task pairs found")
        return

    unit = summary["curve"].get(curve_key(UNIT_ALPHA))
    bits = [f"{summary['n']} cross-task pairs",
            f"grid={entry['n_grid_points']} pts"]
    if unit and unit["frac_positive"] is not None:
        bits.append(f"frac_positive@1={unit['frac_positive'] * 100:.1f}%")
    if summary["interval_width"]["median"] is not None:
        bits.append(f"median interval width={summary['interval_width']['median']:.2f}")
    if summary["plateau_width"]["median"] is not None:
        bits.append(f"median plateau width={summary['plateau_width']['median']:.2f}")
    if baseline == BASELINE_FP_PTQ:
        if summary["unit_retention"]["median"] is not None:
            bits.append(f"median unit retention={summary['unit_retention']['median']:.2f}")
    else:
        if summary["unit_regret"]["median"] is not None:
            bits.append(f"median unit regret={summary['unit_regret']['median'] * 100:.2f}pp")
        if summary["frac_unit_optimal"] is not None:
            bits.append(f"unit optimal in {summary['frac_unit_optimal'] * 100:.1f}%")
    print(f"  {entry['display_name']}: " + ", ".join(bits))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    datasets = sorted(DATASET_NAME_TO_EPOCHS.keys())
    qat_frag = _qat_frag(args)
    ptq_frag = _ptq_frag(args)

    models = {}
    for model_name, pretrained_tag, batch_size in zip(
        args.model_names, args.pretrained_tags, args.batch_sizes
    ):
        model_dir  = sanitize_open_clip_model_name(model_name, pretrained_tag)
        optim_frag = _optim_frag(args, batch_size)
        model_key  = f"{model_name}__{pretrained_tag}"

        print(f"Loading {model_key} ...")
        prefixes   = cell_prefixes(model_dir, datasets, args,
                                   optim_frag, qat_frag, ptq_frag)
        discovered = discover_grid(prefixes, args.split)
        grid       = resolve_grid(args.grid, discovered)
        if not grid:
            print(f"  {model_key}: no lambdas found on split={args.split}",
                  file=sys.stderr)
            continue

        pairs, missing = load_pairs(model_dir, datasets, args, grid,
                                    optim_frag, qat_frag, ptq_frag)

        cross = [p for p in pairs if not p["same_task"]]
        same  = [p for p in pairs if p["same_task"]]

        entry = {
            "display_name":     MODEL_DISPLAY_NAMES.get((model_name, pretrained_tag), model_key),
            "model_dir":        model_dir,
            "pretrained_tag":   pretrained_tag,
            "batch_size":       batch_size,
            "skip_modules":     list(args.skip_modules),
            "n_datasets":       len(datasets),
            "n_cells_expected": len(datasets) ** 2,
            "grid":             grid,
            "n_grid_points":    len(grid),
            "pairs":            pairs,
            "cross_task":       summarize_cells(cross, grid, args.baseline),
            "same_task":        summarize_cells(same, grid, args.baseline),
            "universal_donor":  summarize_cells(
                [p for p in cross if p["donor"] == args.universal_donor],
                grid, args.baseline,
            ) if args.universal_donor else None,
            "missing":          missing,
        }
        models[model_key] = entry

        _print_model_summary(entry, args.baseline)
        if missing:
            print(f"  {entry['display_name']}: {len(missing)} missing files",
                  file=sys.stderr)

    results = {
        "family":   FAMILY,
        "modality": MODALITY,
        "config": {
            "seed":            args.seed,
            "optim":           args.optim,
            "lr":              args.lr,
            "wd":              args.wd,
            "ls":              args.ls,
            "wl":              args.wl,
            "max_grad_norm":   args.max_grad_norm,
            "qat_bits":        args.qat_bits,
            "ptq_bits":        args.ptq_bits,
            "granularity":     args.granularity,
            "skip_modules":    list(args.skip_modules),
            "grid_mode":       args.grid,
            "eval_split":      args.split,
            "metric_key":      _metric_key(args.split),
            "baseline":        args.baseline,
            "frac_positive_meaning": _frac_positive_meaning(args.baseline),
            "universal_donor": args.universal_donor,
        },
        "datasets": datasets,
        "models":   models,
    }

    # Refuse to overwrite a good aggregate with an empty one. A model that
    # yields zero pairs means the constructed path grammar does not match the
    # tree -- a wrong seed, a wrong optim fragment, a wrong multiplier -- and
    # writing that result out would replace real numbers with silence.
    empty = [name for name, entry in models.items() if not entry.get("pairs")]
    if empty or not models:
        print(
            f"[ERROR] {len(empty)} of {len(models)} model(s) produced zero pairs: "
            f"{', '.join(sorted(empty)) or '(no models at all)'}.\n"
            "        Refusing to write the aggregate; nothing was overwritten.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir = _out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"lambda_curves_{FAMILY}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
