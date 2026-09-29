"""998 — Zero-shot reframing statistics (text, automodel)

For every donor-receiver pair, at unit scaling (lambda=1, data-free) and at the
validation-selected best scaling (lambda_best, needs receiver data), reports how
often QV patching beats vanilla PTQ.  Pairs are split into:

    same-task  : donor == receiver.  At lambda=1 this is algebraically the
                 receiver's own QAT checkpoint, so it is the QAT ceiling
                 rather than a transfer result.
    cross-task : donor != receiver.  This is the genuine zero-shot,
                 data-free transfer setting.

Delta = transfer_acc - baseline_fp_ptq_acc.  A "win" is Delta > 0.

Beyond win rates, three swept-lambda quantities are computed:

    recovery ratio  : Delta / Delta_ceiling(receiver), the fraction of the
                      receiver's own QAT gain that transfer recovers.  This is
                      the observable side of Proposition 1's cos^2 law.
    lambda_best     : its distribution, and the fraction of pairs below 1.0
                      (unit scaling overshoots).
    alignment       : among pairs that lose at lambda=1, how many win at some
                      positive lambda.  By Proposition 1 (Delta > 0 iff
                      0 < lambda < 2 lambda*), winning at any positive lambda
                      implies lambda* > 0, i.e. the donor direction is
                      positively aligned and the failure is overshoot rather
                      than anti-alignment.

Writes a per-family JSON consumed by aggregate_win_loss.py.
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

os.chdir(_PROJECT_ROOT)

from src.duration import mult_path_frag, mult_tag, role_path_frag
from src.text.data.common import DATASET_NAME_TO_EPOCHS
from src.vision.utils import sanitize_hf_model_name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FAMILY   = "ilharco_automodelforsequenceclassification"
MODALITY = "text"

EVAL_ROOT_BASELINES = "evaluations/text/ilharco_automodelforsequenceclassification/000_baselines/text"
EVAL_ROOT_QV        = "evaluations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/text/qv_transfer"

EVAL_ROOT_OUT = "evaluations/998_rebuttal/001_zero_shot_reframing"

TEST_METRIC_KEY = "test_accuracy_fp_head_ptq"
TEST_ACC_KEY    = "test_accuracy"

# Validation-selected best scaling, written by pick_best_alpha.py.
BEST_ALPHA_FILE = "best_alpha_fp_head_ptq.json"
BEST_ALPHA_KEY  = "val_accuracy_fp_head_ptq"

# On the diagonal the patched backbone is the QAT backbone, so this key should
# reproduce the qat_ptq baseline.  Used only by --check-diagonal-against-qat-ptq.
DIAGONAL_CHECK_METRIC_KEY = "test_accuracy_qat_head_ptq"

MODEL_DISPLAY_NAMES = {
    "google-bert/bert-base-uncased":   "BERT-Base",
    "google-bert/bert-large-uncased":  "BERT-Large",
    "google/embeddinggemma-300m":      "EmbeddingGemma",
    "Qwen/Qwen3-Embedding-0.6B":       "Qwen3-Embedding",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-names",     required=True, nargs="+",
                        help="HF model names, e.g. google-bert/bert-base-uncased")
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
    parser.add_argument("--max-grad-norm",   required=True, type=float)
    parser.add_argument("--max-length",      required=True, type=int)
    parser.add_argument("--batch-sizes",     required=True, type=int, nargs="+",
                        help="Batch sizes (one per model, parallel to --model-names)")

    parser.add_argument("--qat-bits",        required=True, type=int)
    parser.add_argument("--ptq-bits",        required=True, type=int)
    parser.add_argument("--granularity",     required=True, choices=["tensor", "channel"])
    parser.add_argument("--skip-modules",    required=True, nargs="+",
                        help="One skip-module name per model, parallel to --model-names "
                             "(e.g. classifier classifier score score).")

    parser.add_argument("--universal-donor", default=None,
                        help="Donor to report separately as the data-free default. "
                             "Omit to skip.")
    parser.add_argument("--check-diagonal-against-qat-ptq", action="store_true",
                        help="Report the discrepancy between the diagonal cells and "
                             "the qat_ptq baseline (they should coincide at lambda=1).")

    args = parser.parse_args()
    if len(args.model_names) != len(args.batch_sizes):
        parser.error("--model-names and --batch-sizes must have the same length")
    if len(args.model_names) != len(args.skip_modules):
        parser.error("--skip-modules must have the same length as --model-names")
    return args


# ---------------------------------------------------------------------------
# Path-fragment helpers
# ---------------------------------------------------------------------------
def _optim_frag(args, batch_size):
    return (f"optim=adamw_lr={args.lr}_wd={args.wd}_ls={args.ls}"
            f"_mgn={args.max_grad_norm}_bs={batch_size}_ml={args.max_length}")


def _qat_frag(args, skip_module):
    return f"qat=bits={args.qat_bits}_gran={args.granularity}_skip={skip_module}"


def _ptq_frag(args, skip_module):
    return f"ptq=bits={args.ptq_bits}_gran={args.granularity}_skip={skip_module}"


# ---------------------------------------------------------------------------
# Per-baseline path builders
# ---------------------------------------------------------------------------
def _fp_ptq_path(model_dir, dataset, seed, target_epoch_mult, optim_frag, ptq_frag):
    return os.path.join(
        EVAL_ROOT_BASELINES, "fp_ptq", model_dir, dataset,
        optim_frag, mult_path_frag(target_epoch_mult), ptq_frag, f"seed={seed}", "eval_results.json",
    )


def _qat_ptq_path(model_dir, dataset, seed, target_epoch_mult, optim_frag, qat_frag, ptq_frag):
    return os.path.join(
        EVAL_ROOT_BASELINES, "qat_ptq", model_dir, dataset,
        optim_frag, mult_path_frag(target_epoch_mult), qat_frag, ptq_frag, f"seed={seed}", "eval_results.json",
    )


# ---------------------------------------------------------------------------
# QV transfer path builders
# ---------------------------------------------------------------------------
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
        print(f"  [NO BEST ALPHA] {path}", file=sys.stderr)
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
        print(f"  [MISSING] {path}", file=sys.stderr)
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
def load_pairs(model_dir, datasets, args, optim_frag, qat_frag, ptq_frag):
    """Every (donor, receiver) cell at alpha=1.0 and at alpha_best, diagonal included."""
    pairs = []
    missing = []

    for receiver in datasets:
        baseline_path = _fp_ptq_path(model_dir, receiver, args.seed, args.target_epoch_mult, optim_frag, ptq_frag)
        baseline = _load_value(baseline_path, TEST_ACC_KEY)
        if baseline is None:
            missing.append(baseline_path)
            continue

        for donor in datasets:
            cell_prefix = _qv_cell_prefix(model_dir, donor, receiver, args.seed,
                                          args.source_epoch_mult, args.target_epoch_mult,
                                          optim_frag, qat_frag, ptq_frag)

            unit_path = _qv_eval_path(cell_prefix, 1.0, "test")
            transfer = _load_value(unit_path, TEST_METRIC_KEY)
            if transfer is None:
                missing.append(unit_path)
                continue

            record = {
                "donor":        donor,
                "receiver":     receiver,
                "baseline_acc": baseline,
                "transfer_acc": transfer,
                "delta":        transfer - baseline,
                "same_task":    donor == receiver,
                "alpha_best":        None,
                "transfer_acc_best": None,
                "delta_best":        None,
            }

            alpha_best = _load_best_alpha(cell_prefix)
            if alpha_best is not None:
                best_path = _qv_eval_path(cell_prefix, alpha_best, "test")
                transfer_best = _load_value(best_path, TEST_METRIC_KEY)
                if transfer_best is None:
                    missing.append(best_path)
                else:
                    record["alpha_best"]        = alpha_best
                    record["transfer_acc_best"] = transfer_best
                    record["delta_best"]        = transfer_best - baseline

            pairs.append(record)

    _attach_recovery(pairs)
    return pairs, missing


def _attach_recovery(pairs):
    """Delta as a fraction of the receiver's own QAT gain (the diagonal cell).

    Proposition 1 predicts this ratio equals cos^2_H(rho_D, rho_R) at the
    optimal scaling, so it is the directly observable side of the theory.
    Left as None where the ceiling is not positive, since the ratio carries no
    meaning when receiver-side QAT does not help either.
    """
    ceiling = {p["receiver"]: p["delta"] for p in pairs if p["same_task"]}
    for p in pairs:
        c = ceiling.get(p["receiver"])
        usable = c is not None and c > 0
        p["ceiling_delta"] = c
        p["recovery"] = p["delta"] / c if usable else None
        p["recovery_best"] = (
            p["delta_best"] / c if usable and p["delta_best"] is not None else None
        )


# ---------------------------------------------------------------------------
# Statistics
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


def _per_receiver_stats(pairs):
    """For each receiver, how many donors help and which one helps most.

    Backs the claim that a receiver can be improved with no receiver-side data,
    provided a donor is picked offline.
    """
    by_receiver = {}
    for p in pairs:
        if p["same_task"]:
            continue
        by_receiver.setdefault(p["receiver"], []).append(p)

    out = {}
    for receiver, ps in by_receiver.items():
        best = max(ps, key=lambda p: p["delta"])
        deltas = [p["delta"] for p in ps]
        out[receiver] = {
            "n_donors":          len(ps),
            "n_donors_positive": sum(1 for v in deltas if v > 0),
            "best_donor":        best["donor"],
            "best_delta":        best["delta"],
            "mean_delta":        statistics.fmean(deltas),
        }
    return out


def _universal_donor_stats(pairs, donor):
    """Cross-task stats restricted to a single fixed donor."""
    if donor is None:
        return None
    deltas = [p["delta"] for p in pairs if p["donor"] == donor and not p["same_task"]]
    stats = _delta_stats(deltas)
    stats["donor"] = donor
    return stats


def _ratio_stats(ratios):
    """Summary of recovery ratios, ignoring cells where the ceiling is unusable."""
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


def _alpha_best_stats(pairs):
    """Distribution of the validation-selected scaling over cross-task pairs."""
    alphas = [p["alpha_best"] for p in pairs
              if not p["same_task"] and p["alpha_best"] is not None]
    if not alphas:
        return {"n": 0}

    histogram = {}
    for a in alphas:
        histogram[str(a)] = histogram.get(str(a), 0) + 1

    return {
        "n":              len(alphas),
        "mean":           statistics.fmean(alphas),
        "median":         statistics.median(alphas),
        "frac_below_1":   sum(1 for a in alphas if a < 1.0) / len(alphas),
        "frac_at_1":      sum(1 for a in alphas if a == 1.0) / len(alphas),
        "frac_above_1":   sum(1 for a in alphas if a > 1.0) / len(alphas),
        "histogram":      dict(sorted(histogram.items(), key=lambda kv: float(kv[0]))),
    }


def _alignment_stats(pairs):
    """Overshoot vs anti-alignment among the pairs that fail at unit scaling.

    Proposition 1 gives Delta > 0 exactly when 0 < lambda < 2 lambda*, so a pair
    that wins at some positive lambda must have lambda* > 0: the donor direction
    is positively aligned and unit scaling simply overshot it.
    """
    failing = [p for p in pairs
               if not p["same_task"] and p["delta"] <= 0 and p["delta_best"] is not None]
    if not failing:
        return {"n_failing_at_unit": 0}

    aligned = [p for p in failing if p["delta_best"] > 0]
    return {
        "n_failing_at_unit":     len(failing),
        "n_recovered_at_best":   len(aligned),
        "frac_recovered_at_best": len(aligned) / len(failing),
        "median_alpha_best_of_recovered": (
            statistics.median([p["alpha_best"] for p in aligned]) if aligned else None
        ),
    }


def _tuning_gain(pairs):
    """Per-pair improvement bought by validation tuning, on the test split."""
    gains = [p["delta_best"] - p["delta"] for p in pairs
             if not p["same_task"] and p["delta_best"] is not None]
    if not gains:
        return {"n": 0}
    return {
        "n":            len(gains),
        "mean":         statistics.fmean(gains),
        "median":       statistics.median(gains),
        "frac_worse_than_unit": sum(1 for g in gains if g < 0) / len(gains),
    }


def compute_model_stats(pairs):
    cross = [p for p in pairs if not p["same_task"]]
    same  = [p for p in pairs if p["same_task"]]
    cross_best = [p["delta_best"] for p in cross if p["delta_best"] is not None]
    same_best  = [p["delta_best"] for p in same  if p["delta_best"] is not None]
    return {
        "cross_task":      _delta_stats([p["delta"] for p in cross]),
        "same_task":       _delta_stats([p["delta"] for p in same]),
        "cross_task_best": _delta_stats(cross_best),
        "same_task_best":  _delta_stats(same_best),
        "recovery":        _ratio_stats([p["recovery"] for p in cross]),
        "recovery_best":   _ratio_stats([p["recovery_best"] for p in cross]),
        "alpha_best":      _alpha_best_stats(pairs),
        "alignment":       _alignment_stats(pairs),
        "tuning_gain":     _tuning_gain(pairs),
    }


# ---------------------------------------------------------------------------
# Diagonal sanity check
# ---------------------------------------------------------------------------
def check_diagonal(model_dir, datasets, args, optim_frag, qat_frag, ptq_frag):
    """At lambda=1 the diagonal is the receiver's own QAT checkpoint.

    Compares it against the qat_ptq baseline and returns the discrepancies.
    Reported, never asserted: a mismatch means the path conventions drifted.
    """
    discrepancies = {}
    for dataset in datasets:
        cell_prefix = _qv_cell_prefix(model_dir, dataset, dataset, args.seed,
                                      args.source_epoch_mult, args.target_epoch_mult,
                                      optim_frag, qat_frag, ptq_frag)
        diag = _load_value(
            _qv_eval_path(cell_prefix, 1.0, "test"),
            DIAGONAL_CHECK_METRIC_KEY,
        )
        qat_ptq = _load_value(
            _qat_ptq_path(model_dir, dataset, args.seed, args.target_epoch_mult, optim_frag, qat_frag, ptq_frag),
            TEST_ACC_KEY,
        )
        if diag is not None and qat_ptq is not None:
            discrepancies[dataset] = diag - qat_ptq

    if not discrepancies:
        return None

    worst = max(discrepancies.items(), key=lambda kv: abs(kv[1]))
    return {
        "n_compared":        len(discrepancies),
        "max_abs_deviation": abs(worst[1]),
        "worst_dataset":     worst[0],
        "per_dataset":       discrepancies,
    }


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
        "split=test",
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    datasets = sorted(DATASET_NAME_TO_EPOCHS.keys())

    models = {}
    for model_name, batch_size, skip_module in zip(
        args.model_names, args.batch_sizes, args.skip_modules
    ):
        model_dir  = sanitize_hf_model_name(model_name)
        optim_frag = _optim_frag(args, batch_size)
        qat_frag   = _qat_frag(args, skip_module)
        ptq_frag   = _ptq_frag(args, skip_module)

        print(f"Loading {model_name} ...")
        pairs, missing = load_pairs(model_dir, datasets, args,
                                    optim_frag, qat_frag, ptq_frag)
        stats = compute_model_stats(pairs)

        entry = {
            "display_name":    MODEL_DISPLAY_NAMES.get(model_name, model_name),
            "model_dir":       model_dir,
            "batch_size":      batch_size,
            "skip_modules":    [skip_module],
            "n_datasets":      len(datasets),
            "n_cells_expected": len(datasets) ** 2,
            "pairs":           pairs,
            "cross_task":      stats["cross_task"],
            "same_task":       stats["same_task"],
            "cross_task_best": stats["cross_task_best"],
            "same_task_best":  stats["same_task_best"],
            "recovery":        stats["recovery"],
            "recovery_best":   stats["recovery_best"],
            "alpha_best":      stats["alpha_best"],
            "alignment":       stats["alignment"],
            "tuning_gain":     stats["tuning_gain"],
            "per_receiver":    _per_receiver_stats(pairs),
            "universal_donor": _universal_donor_stats(pairs, args.universal_donor),
            "missing":         missing,
        }

        if args.check_diagonal_against_qat_ptq:
            entry["diagonal_check"] = check_diagonal(
                model_dir, datasets, args, optim_frag, qat_frag, ptq_frag
            )

        models[model_name] = entry

        cross = stats["cross_task"]
        best  = stats["cross_task_best"]
        if cross["n"] == 0:
            print(f"  {entry['display_name']}: no cross-task pairs found")
        else:
            line = (f"  {entry['display_name']}: {cross['n']} cross-task pairs, "
                    f"win_rate={cross['win_rate'] * 100:.1f}% at lambda=1")
            if best["n"] > 0:
                line += (f", {best['win_rate'] * 100:.1f}% at lambda_best "
                         f"({best['n']} pairs)")
            if stats["recovery"]["median"] is not None:
                line += f", median recovery={stats['recovery']['median']:.2f}"
            print(line)
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
            "max_grad_norm":   args.max_grad_norm,
            "max_length":      args.max_length,
            "qat_bits":        args.qat_bits,
            "ptq_bits":        args.ptq_bits,
            "granularity":     args.granularity,
            "skip_modules":    list(args.skip_modules),
            "alpha":           1.0,
            "eval_split":      "test",
            "metric_key":      TEST_METRIC_KEY,
            "best_alpha_file": BEST_ALPHA_FILE,
            "best_alpha_key":  BEST_ALPHA_KEY,
            "baseline":        "fp_ptq",
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
    out_path = os.path.join(out_dir, f"win_loss_{FAMILY}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
