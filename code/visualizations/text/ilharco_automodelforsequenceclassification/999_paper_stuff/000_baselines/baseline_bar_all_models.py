"""Combined baseline bar chart for all text (automodel) models.

Two subplots per row, shared x-axis (datasets) and a single top legend.
Auto-discovers models and optim frags from the evaluation directory.

Output: single PDF with LaTeX fonts (paper-ready).
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[6]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from src.duration import mult_path_frag

os.chdir(_PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
import numpy as np

from src.text.data.common import DATASET_NAME_TO_EPOCHS
from src.vision.utils import sanitize_hf_model_name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT_BASELINES = "evaluations/text/ilharco_automodelforsequenceclassification/000_baselines/text"

TEST_ACC_KEY = "test_accuracy"

DATASET_ORDER_SWAPS = [("AmazonCounterfactual", "ToxicConversations")]

LABEL_RENAMES = {
    "AmazonCounterfactual": "Counterfactual",
    "TweetSentimentExtraction": "Sentiment",
    "AmazonReviewsClassification": "Reviews",
    "ToxicConversations": "Toxic",
    "MTOPDomain": "MTOP D",
    "MTOPIntent": "MTOP I",
    "MassiveIntent": "Intent",
    "MassiveScenario": "Scenario",
}

# sanitized_dir -> display name
MODEL_DISPLAY_NAMES = {
    "google_bert_bert_base_uncased": "BERT-Base",
    "google_bert_bert_large_uncased": "BERT-Large",
    "google_embeddinggemma_300m": "EmbeddingGemma",
    "Qwen_Qwen3_Embedding_0.6B": "Qwen3-Embedding",
}

MODEL_ORDER = [
    "google_bert_bert_base_uncased",
    "google_bert_bert_large_uncased",
    "google_embeddinggemma_300m",
    "Qwen_Qwen3_Embedding_0.6B",
]


def _swapped_dataset_order(datasets_dict):
    ds = sorted(datasets_dict.keys())
    for a, b in DATASET_ORDER_SWAPS:
        if a in ds and b in ds:
            ia, ib = ds.index(a), ds.index(b)
            ds[ia], ds[ib] = ds[ib], ds[ia]
    return ds


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)

    parser.add_argument("--model-names", required=True, nargs="+",
                        help="HF model names (e.g. google-bert/bert-base-uncased), "
                             "in panel order.")
    parser.add_argument("--skip-modules", required=True, nargs="+",
                        help="Classification-head module skipped during quantization, one "
                             "per model, ordering must match --model-names. This differs "
                             "across this family ('classifier' for BERT, 'score' for "
                             "gemma/Qwen) and must be stated, not discovered.")
    parser.add_argument("--batch-sizes", required=True, nargs="+", type=int,
                        help="Per-model batch size, ordering must match --model-names.")
    parser.add_argument("--max-lengths", required=True, nargs="+", type=int,
                        help="Per-model max sequence length, ordering must match "
                             "--model-names.")

    parser.add_argument("--optim", required=True, choices=["adamw", "sgd"])
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--wd", required=True, type=float)
    parser.add_argument("--ls", required=True, type=float)
    parser.add_argument("--max-grad-norm", required=True, type=float)

    parser.add_argument("--qat-bits", required=True, type=int)
    parser.add_argument("--ptq-bits", required=True, type=int)
    parser.add_argument("--granularity", required=True, choices=["tensor", "channel"])
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if any expected eval_results.json is missing. "
                             "Without it, misses are still counted and reported on stderr.")
    args = parser.parse_args()
    n = len(args.model_names)
    if not (len(args.skip_modules) == len(args.batch_sizes) == len(args.max_lengths) == n):
        parser.error(f"--model-names ({n}), --skip-modules ({len(args.skip_modules)}), "
                     f"--batch-sizes ({len(args.batch_sizes)}) and --max-lengths "
                     f"({len(args.max_lengths)}) must have the same length")
    return args


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _optim_frag(args, batch_size, max_length):
    return (f"optim={args.optim}_lr={args.lr}_wd={args.wd}_ls={args.ls}"
            f"_mgn={args.max_grad_norm}_bs={batch_size}_ml={max_length}")


def _qat_frag(args, skip_module):
    return f"qat=bits={args.qat_bits}_gran={args.granularity}_skip={skip_module}"


def _ptq_frag(args, skip_module):
    return f"ptq=bits={args.ptq_bits}_gran={args.granularity}_skip={skip_module}"


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------
def resolve_models(args):
    """Return [(model_dir, optim_frag, qat_frag, ptq_frag), ...] in --model-names order.

    Nothing is discovered.  The previous version auto-discovered the optim
    fragment by prefix and recovered the skip tag by slicing the qat directory
    name apart, so a mismatched configuration surfaced as an empty panel rather
    than an error.
    """
    fp_root = os.path.join(EVAL_ROOT_BASELINES, "fp")
    if not os.path.isdir(fp_root):
        print(f"[ERROR] fp root not found: {fp_root}", file=sys.stderr)
        sys.exit(1)

    resolved = []
    for model_name, skip_module, batch_size, max_length in zip(
            args.model_names, args.skip_modules, args.batch_sizes, args.max_lengths):
        model_dir = sanitize_hf_model_name(model_name)
        optim_frag = _optim_frag(args, batch_size, max_length)
        model_path = os.path.join(fp_root, model_dir)
        if not os.path.isdir(model_path):
            print(f"[ERROR] no evaluations for {model_name} at {model_path}", file=sys.stderr)
            sys.exit(1)
        datasets = [d for d in os.listdir(model_path)
                    if os.path.isdir(os.path.join(model_path, d))]
        if not any(os.path.isdir(os.path.join(model_path, d, optim_frag, f"seed={args.seed}"))
                   for d in datasets):
            print(f"[ERROR] {model_dir}: no dataset has "
                  f"{optim_frag}/seed={args.seed}", file=sys.stderr)
            sys.exit(1)
        resolved.append((model_dir, optim_frag,
                         _qat_frag(args, skip_module), _ptq_frag(args, skip_module)))
    return resolved


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------
def _load_value(path, key, misses):
    """Read `key` from a results JSON, recording a miss rather than swallowing it.

    A silently-absent file renders as a zero-height bar, which is
    indistinguishable from a genuine zero.
    """
    if not os.path.exists(path):
        misses.append((path, "missing"))
        return None
    try:
        with open(path) as f:
            value = json.load(f).get(key)
    except (OSError, json.JSONDecodeError) as exc:
        misses.append((path, f"unreadable: {exc}"))
        return None
    if value is None:
        misses.append((path, f"no key {key!r}"))
    return value


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(model_dir, optim_frag, seed, qat_frag, ptq_frag, misses, *, target_epoch_mult):
    datasets = _swapped_dataset_order(DATASET_NAME_TO_EPOCHS)
    data = {}
    for dataset in datasets:
        ft_acc = _load_value(os.path.join(
            EVAL_ROOT_BASELINES, "fp", model_dir, dataset,
            optim_frag, mult_path_frag(target_epoch_mult), f"seed={seed}", "eval_results.json"), TEST_ACC_KEY, misses)
        qat_acc = _load_value(os.path.join(
            EVAL_ROOT_BASELINES, "qat", model_dir, dataset,
            optim_frag, mult_path_frag(target_epoch_mult), qat_frag, f"seed={seed}", "eval_results.json"), TEST_ACC_KEY, misses)
        ptq_acc = _load_value(os.path.join(
            EVAL_ROOT_BASELINES, "fp_ptq", model_dir, dataset,
            optim_frag, mult_path_frag(target_epoch_mult), ptq_frag, f"seed={seed}", "eval_results.json"), TEST_ACC_KEY, misses)
        data[dataset] = {"ft": ft_acc, "qat": qat_acc, "ptq": ptq_acc}
    return data


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_all(all_model_data, args):
    datasets = _swapped_dataset_order(DATASET_NAME_TO_EPOCHS)
    n_ds = len(datasets)
    n_models = len(all_model_data)

    n_cols = 2
    n_rows = math.ceil(n_models / n_cols)
    bar_width = 0.25
    x_pos = np.arange(n_ds)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3.2 * n_rows),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    display_labels = [LABEL_RENAMES.get(ds, ds) for ds in datasets]

    for idx, (model_dir, data) in enumerate(all_model_data):
        ax = axes_flat[idx]
        ft_vals = [data[ds]["ft"] if data[ds]["ft"] is not None else 0 for ds in datasets]
        qat_vals = [data[ds]["qat"] if data[ds]["qat"] is not None else 0 for ds in datasets]
        ptq_vals = [data[ds]["ptq"] if data[ds]["ptq"] is not None else 0 for ds in datasets]

        ax.bar(x_pos - bar_width, ft_vals, bar_width,
               label=r"\textbf{FT}" if idx == 0 else None, color="#e09f3e")
        ax.bar(x_pos, qat_vals, bar_width,
               label=r"\textbf{QAT}" if idx == 0 else None, color="#540b0e")
        ax.bar(x_pos + bar_width, ptq_vals, bar_width,
               label=r"\textbf{PTQ}" if idx == 0 else None, color="#9e2a2b")

        ax.set_ylim(0, 1.0)
        ax.set_ylabel(r"Test Accuracy", fontsize=11)
        display_title = MODEL_DISPLAY_NAMES.get(model_dir, model_dir)
        ax.set_title(r"\textbf{" + display_title + "}", fontsize=14)

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xticks(x_pos)
            ax.set_xticklabels(display_labels, fontsize=10, rotation=45, ha="right")
    if n_models % n_cols != 0:
        axes[n_rows - 2][n_cols - 1].set_xticks(x_pos)
        axes[n_rows - 2][n_cols - 1].set_xticklabels(display_labels, fontsize=10, rotation=45, ha="right")
        axes[n_rows - 2][n_cols - 1].tick_params(labelbottom=True)

    axes_flat[0].legend(fontsize=12, frameon=False, loc="upper center",
                        bbox_to_anchor=(1.0, 1.35), ncol=3)

    fig.tight_layout()
    fig.subplots_adjust(top=1 - 0.4 / (3.2 * n_rows))

    out_dir = os.path.join(
        "plots", "text", "ilharco_automodelforsequenceclassification",
        "999_paper_stuff", "000_baselines", "baseline_bar_all_models",
        f"seed={args.seed}",
        f"ptq=bits={args.ptq_bits}_gran={args.granularity}",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "baseline_bar_all_models.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    models = resolve_models(args)

    print(f"Plotting {len(models)} models:")
    all_model_data = []
    misses = []
    for (model_dir, optim_frag, qat_frag, ptq_frag), skip_module in zip(
            models, args.skip_modules):
        display = MODEL_DISPLAY_NAMES.get(model_dir, model_dir)
        print(f"  Loading {display} (skip: {skip_module}) ...")
        before = len(misses)
        data = load_data(model_dir, optim_frag, args.seed, qat_frag, ptq_frag, misses, target_epoch_mult=args.target_epoch_mult)
        model_misses = len(misses) - before
        expected = 3 * len(DATASET_NAME_TO_EPOCHS)
        if model_misses:
            print(f"    [WARN] {model_misses}/{expected} lookups missing", file=sys.stderr)
        if model_misses == expected:
            print(f"[ERROR] {display}: every lookup missed -- the constructed path "
                  f"grammar does not match this tree.", file=sys.stderr)
            sys.exit(1)
        all_model_data.append((model_dir, data))

    if misses:
        print(f"\n[SUMMARY] {len(misses)} missing lookups across "
              f"{len(models)} models:", file=sys.stderr)
        for path, why in misses:
            print(f"  {why}: {path}", file=sys.stderr)
        if args.strict:
            print("[ERROR] --strict: refusing to plot on incomplete data.", file=sys.stderr)
            sys.exit(1)

    plot_all(all_model_data, args)


if __name__ == "__main__":
    main()
