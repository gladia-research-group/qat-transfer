"""999 — Donor/Receiver LaTeX Table: QV transfer at unit scaling (text, automodel)

Produces a LaTeX table with donor- and receiver-wise transfer statistics
at alpha=1.0 for each text model.  Columns:

    Model | Donor Best/Worst/Mean/Pos.% | Receiver Best/Worst/Mean/Pos.%

Best/Worst/Mean are computed over per-donor (or per-receiver) mean deltas.
Delta = transfer_acc(alpha=1.0) - baseline_fp_ptq_acc.
Self-transfer pairs (donor == receiver) are excluded.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is the working directory and on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[6]
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
EVAL_ROOT_BASELINES = "evaluations/text/ilharco_automodelforsequenceclassification/000_baselines/text"
EVAL_ROOT_QV        = "evaluations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/text/qv_transfer"

TEST_METRIC_KEY = "test_accuracy_fp_head_ptq"
TEST_ACC_KEY    = "test_accuracy"

MODEL_DISPLAY_NAMES = {
    "google-bert/bert-base-uncased":   "BERT-Base",
    "google-bert/bert-large-uncased":  "BERT-Large",
    "google/embeddinggemma-300m":      "EmbeddingGemma",
    "Qwen/Qwen3-Embedding-0.6B":      "Qwen3-Embedding",
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
                        help="Training-budget multiplier of the RECEIVER checkpoints.")

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
def _fp_ptq_path(model_dir, dataset, seed, optim_frag, ptq_frag, *, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_BASELINES, "fp_ptq", model_dir, dataset,
        optim_frag, mult_path_frag(target_epoch_mult), ptq_frag, f"seed={seed}", "eval_results.json",
    )


# ---------------------------------------------------------------------------
# QV transfer path builder (fixed alpha=1.0, split=test)
# ---------------------------------------------------------------------------
def _qv_transfer_path(model_dir, donor, receiver, seed, optim_frag, qat_frag, ptq_frag, *, source_epoch_mult, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_QV, model_dir,
        role_path_frag("src", donor, seed, source_epoch_mult),
        role_path_frag("tgt", receiver, seed, target_epoch_mult),
        optim_frag, qat_frag, ptq_frag,
        "qv=alpha=1.0", "split=test",
        "eval_results.json",
    )


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
def load_deltas(model_dir, datasets, seed, optim_frag, qat_frag, ptq_frag):
    deltas = {}
    for receiver in datasets:
        baseline = _load_value(
            _fp_ptq_path(model_dir, receiver, seed, optim_frag, ptq_frag, target_epoch_mult=args.target_epoch_mult),
            TEST_ACC_KEY,
        )
        for donor in datasets:
            if donor == receiver:
                continue
            transfer = _load_value(
                _qv_transfer_path(model_dir, donor, receiver, seed,
                                  optim_frag, qat_frag, ptq_frag,
                source_epoch_mult=args.source_epoch_mult, target_epoch_mult=args.target_epoch_mult,
            ),
                TEST_METRIC_KEY,
            )
            if transfer is not None and baseline is not None:
                deltas[(donor, receiver)] = transfer - baseline
    return deltas


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def compute_stats(deltas, datasets):
    per_donor = {}
    for d in datasets:
        vals = [deltas[(d, r)] for r in datasets if r != d and (d, r) in deltas]
        if vals:
            per_donor[d] = sum(vals) / len(vals)

    per_receiver = {}
    for r in datasets:
        vals = [deltas[(d, r)] for d in datasets if d != r and (d, r) in deltas]
        if vals:
            per_receiver[r] = sum(vals) / len(vals)

    all_deltas = list(deltas.values())
    n_pos = sum(1 for v in all_deltas if v > 0)
    pos_pct = (n_pos / len(all_deltas) * 100) if all_deltas else 0.0

    donor_means = list(per_donor.values())
    receiver_means = list(per_receiver.values())

    return {
        "donor_best":     max(donor_means) if donor_means else float("nan"),
        "donor_worst":    min(donor_means) if donor_means else float("nan"),
        "donor_mean":     sum(donor_means) / len(donor_means) if donor_means else float("nan"),
        "donor_pos_pct":  pos_pct,
        "receiver_best":  max(receiver_means) if receiver_means else float("nan"),
        "receiver_worst": min(receiver_means) if receiver_means else float("nan"),
        "receiver_mean":  sum(receiver_means) / len(receiver_means) if receiver_means else float("nan"),
        "receiver_pos_pct": pos_pct,
    }


# ---------------------------------------------------------------------------
# LaTeX formatting
# ---------------------------------------------------------------------------
def _fmt_delta(val):
    pp = val * 100
    if pp >= 0:
        return f"+{pp:.1f}"
    return f"$-${abs(pp):.1f}"


def _fmt_pct(val):
    return f"{val:.1f}"


def format_latex_table(rows):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Donor- and receiver-wise transfer statistics at unit quantization vector scaling ($\lambda=1$). Best, Worst, and Mean are Top-1 accuracy change (p.p.) relative to vanilla PTQ. Pos.\% measures the proportion of positive-transfer pairs.}")
    lines.append(r"\label{tab:donor_receiver_text_unit_alpha}")
    lines.append(r"\begin{tabular}{l cccc cccc}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{4}{c}{Donor} & \multicolumn{4}{c}{Receiver} \\")
    lines.append(r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}")
    lines.append(r"Model & Best & Worst & Mean & Pos.\% & Best & Worst & Mean & Pos.\% \\")
    lines.append(r"\midrule")
    for display_name, stats in rows:
        cols = [
            _fmt_delta(stats["donor_best"]),
            _fmt_delta(stats["donor_worst"]),
            _fmt_delta(stats["donor_mean"]),
            _fmt_pct(stats["donor_pos_pct"]),
            _fmt_delta(stats["receiver_best"]),
            _fmt_delta(stats["receiver_worst"]),
            _fmt_delta(stats["receiver_mean"]),
            _fmt_pct(stats["receiver_pos_pct"]),
        ]
        lines.append(f"{display_name} & " + " & ".join(cols) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    datasets = sorted(DATASET_NAME_TO_EPOCHS.keys())

    rows = []
    for model_name, batch_size, skip_module in zip(
        args.model_names, args.batch_sizes, args.skip_modules
    ):
        model_dir = sanitize_hf_model_name(model_name)
        optim_frag = _optim_frag(args, batch_size)
        qat_frag = _qat_frag(args, skip_module)
        ptq_frag = _ptq_frag(args, skip_module)

        print(f"Loading {model_name} ...")
        deltas = load_deltas(model_dir, datasets, args.seed,
                             optim_frag, qat_frag, ptq_frag)
        stats = compute_stats(deltas, datasets)

        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        rows.append((display_name, stats))

        print(f"  {display_name}: {len(deltas)} pairs, "
              f"donor_mean={stats['donor_mean']*100:+.1f}pp, "
              f"pos={stats['donor_pos_pct']:.1f}%")

    tex = format_latex_table(rows)

    qat_frag_out = _qat_frag(args, args.skip_modules[0])
    out_dir = os.path.join(
        "plots", "text", "ilharco_automodelforsequenceclassification",
        "999_paper_stuff", "001_qat_transfer", "donor_receiver_table_unit_alpha",
        f"seed={args.seed}", f"smult={mult_tag(args.source_epoch_mult)}", f"tmult={mult_tag(args.target_epoch_mult)}", qat_frag_out,
        _ptq_frag(args, args.skip_modules[0]),
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "donor_receiver_table_unit_alpha.tex")

    with open(out_path, "w") as f:
        f.write(tex)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
