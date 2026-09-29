"""999 — QV Transfer Side-by-Side Heatmaps: Fixed Alpha vs Best Alpha — text

Produces two difference heatmaps side-by-side for each head variant
(FT head / QAT head):

    Left  : cell = qv_transfer_ptq(fixed alpha) accuracy - FT+PTQ accuracy
    Right : cell = qv_transfer_ptq(best alpha) accuracy  - FT+PTQ accuracy

Best alpha is read from pre-computed best_alpha_*.json files
(produced by pick_best_alpha.py).

Layout:
    Row 0 :  [Left heatmap]           [Right heatmap]
    Row 1 :  [3 legend cells]         [Horizontal colorbar]

Output: PDF, full-width, shared color scale, no cell annotations.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np

from src.text.data.common import DATASET_NAME_TO_EPOCHS
from src.vision.utils import sanitize_hf_model_name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT_BASELINES = "evaluations/text/ilharco_automodelforsequenceclassification/000_baselines/text"
EVAL_ROOT_QV        = "evaluations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/text/qv_transfer"

TEST_METRIC_KEYS = {
    "fp_head_ptq":  "test_accuracy_fp_head_ptq",
    "qat_head_ptq": "test_accuracy_qat_head_ptq",
}

QV_METRIC_LABELS = {
    "fp_head_ptq":  "FT Head",
    "qat_head_ptq": "QAT Head",
}

TEST_ACC_KEY = "test_accuracy"

BEST_ALPHA_FILES = {
    "fp_head_ptq":  "best_alpha_fp_head_ptq.json",
    "qat_head_ptq": "best_alpha_qat_head_ptq.json",
}

BEST_ALPHA_KEYS = {
    "fp_head_ptq":  "val_accuracy_fp_head_ptq",
    "qat_head_ptq": "val_accuracy_qat_head_ptq",
}

DATASET_LABEL_RENAMES = {
    "AmazonCounterfactual": "Counterfactual",
    "TweetSentimentExtraction": "Sentiment",
    "AmazonReviewsClassification": "Reviews",
    "ToxicConversations": "Toxic",
    "MTOPDomain": "MTOP D",
    "MTOPIntent": "MTOP I",
    "MassiveIntent": "Intent",
    "MassiveScenario": "Scenario",
}

MODEL_DISPLAY_NAMES = {
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-Embedding",
    "google-bert/bert-base-uncased": "BERT-Base",
    "google-bert/bert-large-uncased": "BERT-Large",
    "google/embeddinggemma-300m": "EmbeddingGemma",
}

DATASET_ORDER_SWAPS = [("AmazonCounterfactual", "ToxicConversations")]


def _swapped_dataset_order(datasets_dict):
    """Sorted dataset list with aesthetic swaps applied."""
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
    parser.add_argument("--model-name",     required=True,
                        help="HF model id, e.g. google-bert/bert-base-uncased")
    parser.add_argument("--seed",           required=True, type=int)
    parser.add_argument("--source-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the DONOR checkpoints.")
    parser.add_argument("--target-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the RECEIVER checkpoints.")

    parser.add_argument("--optim",          required=True, choices=["adamw", "sgd"])
    parser.add_argument("--lr",             required=True, type=float)
    parser.add_argument("--wd",             required=True, type=float)
    parser.add_argument("--ls",             required=True, type=float)
    parser.add_argument("--max-grad-norm",  required=True, type=float)
    parser.add_argument("--batch-size",     required=True, type=int)
    parser.add_argument("--max-length",     required=True, type=int)

    parser.add_argument("--qat-bits",       required=True, type=int)
    parser.add_argument("--ptq-bits",       required=True, type=int)
    parser.add_argument("--granularity",    required=True, choices=["tensor", "channel"])
    parser.add_argument("--skip-modules",   required=True, nargs="+",
                        help="One or more module names to skip during quantization "
                             "(no default: must be specified explicitly).")

    parser.add_argument("--qv-alpha",       required=True, type=float,
                        help="Fixed alpha for the left heatmap.")

    parser.add_argument("--eval-split",     default="test", choices=["val", "test"],
                        help="Which split the qv_transfer results were evaluated on.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path-fragment helpers
# ---------------------------------------------------------------------------
def _skip_tag(skip_modules):
    return "-".join(sorted(skip_modules)) if len(skip_modules) > 0 else "none"


def _optim_frag(optim, lr, wd, ls, mgn, bs, ml):
    del optim
    return f"optim=adamw_lr={lr}_wd={wd}_ls={ls}_mgn={mgn}_bs={bs}_ml={ml}"


def _qat_frag(bits, gran, skip_modules):
    return f"qat=bits={bits}_gran={gran}_skip={_skip_tag(skip_modules)}"


def _ptq_frag(bits, gran, skip_modules):
    return f"ptq=bits={bits}_gran={gran}_skip={_skip_tag(skip_modules)}"


def _qv_frag(alpha):
    return f"qv=alpha={alpha}"


def _split_frag(eval_split):
    return f"split={eval_split}"


# ---------------------------------------------------------------------------
# Per-baseline path builders
# ---------------------------------------------------------------------------
def _fp_ptq_path(model_dir, dataset, seed, optim_frag, ptq_frag, *, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_BASELINES, "fp_ptq", model_dir, dataset,
        optim_frag, mult_path_frag(target_epoch_mult), ptq_frag, f"seed={seed}", "eval_results.json",
    )


# ---------------------------------------------------------------------------
# QV transfer path builders
# ---------------------------------------------------------------------------
def _qv_transfer_path(model_dir, qv_dataset, target_dataset, seed,
                       optim_frag, qat_frag, ptq_frag, qv_frag, eval_split, *, source_epoch_mult, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_QV, model_dir,
        role_path_frag("src", qv_dataset, seed, source_epoch_mult),
        role_path_frag("tgt", target_dataset, seed, target_epoch_mult),
        optim_frag, qat_frag, ptq_frag, qv_frag,
        _split_frag(eval_split),
        "eval_results.json",
    )


def _qv_transfer_cell_prefix(model_dir, qv_dataset, target_dataset, seed,
                             optim_frag, qat_frag, ptq_frag, *, source_epoch_mult, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_QV, model_dir,
        role_path_frag("src", qv_dataset, seed, source_epoch_mult),
        role_path_frag("tgt", target_dataset, seed, target_epoch_mult),
        optim_frag, qat_frag, ptq_frag,
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
# Data loading — fixed alpha
# ---------------------------------------------------------------------------
def load_data_fixed_alpha(args, model_dir, optim_frag, qat_frag, ptq_frag, metric_tag):
    qv_frag     = _qv_frag(args.qv_alpha)
    metric_key  = f"{args.eval_split}_accuracy_{metric_tag}"
    datasets    = sorted(DATASET_NAME_TO_EPOCHS.keys())

    data = {}
    for target_dataset in datasets:
        fp_ptq_acc = _load_value(
            _fp_ptq_path(model_dir, target_dataset, args.seed, optim_frag, ptq_frag, target_epoch_mult=args.target_epoch_mult),
            TEST_ACC_KEY,
        )
        qv_transfer = {}
        for qv_dataset in datasets:
            qv_path = _qv_transfer_path(
                model_dir, qv_dataset, target_dataset, args.seed,
                optim_frag, qat_frag, ptq_frag, qv_frag, args.eval_split,
            
                source_epoch_mult=args.source_epoch_mult, target_epoch_mult=args.target_epoch_mult,
            )
            qv_transfer[qv_dataset] = _load_value(qv_path, metric_key)

        data[target_dataset] = {
            "fp_ptq": fp_ptq_acc,
            "qv_transfer": qv_transfer,
        }

    return data


# ---------------------------------------------------------------------------
# Data loading — best alpha (from pre-computed best_alpha_*.json files)
# ---------------------------------------------------------------------------
def load_data_best_alpha(args, model_dir, optim_frag, qat_frag, ptq_frag, metric_tag):
    datasets = _swapped_dataset_order(DATASET_NAME_TO_EPOCHS)

    best_alpha_file = BEST_ALPHA_FILES[metric_tag]
    best_alpha_key  = BEST_ALPHA_KEYS[metric_tag]
    test_metric_key = TEST_METRIC_KEYS[metric_tag]

    data = {}
    for target_dataset in datasets:
        fp_ptq_acc = _load_value(
            _fp_ptq_path(model_dir, target_dataset, args.seed, optim_frag, ptq_frag, target_epoch_mult=args.target_epoch_mult),
            TEST_ACC_KEY,
        )
        qv_transfer = {}
        for qv_dataset in datasets:
            cell_prefix = _qv_transfer_cell_prefix(
                model_dir, qv_dataset, target_dataset, args.seed,
                optim_frag, qat_frag, ptq_frag,
            
                source_epoch_mult=args.source_epoch_mult, target_epoch_mult=args.target_epoch_mult,
            )

            best_alpha_val = None
            best_alpha_path = os.path.join(cell_prefix, best_alpha_file)
            if os.path.exists(best_alpha_path):
                with open(best_alpha_path) as f:
                    info = json.load(f).get(best_alpha_key)
                    if info is not None:
                        best_alpha_val = info["alpha"]

            best_alpha_acc = None
            if best_alpha_val is not None:
                test_path = os.path.join(
                    cell_prefix, f"qv=alpha={best_alpha_val}",
                    "split=test", "eval_results.json",
                )
                best_alpha_acc = _load_value(test_path, test_metric_key)
            else:
                print(f"  [NO BEST ALPHA] {best_alpha_path}", file=sys.stderr)

            qv_transfer[qv_dataset] = best_alpha_acc

        data[target_dataset] = {
            "fp_ptq": fp_ptq_acc,
            "qv_transfer": qv_transfer,
        }

    return data


# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------
def _build_diff_matrix(data):
    datasets = _swapped_dataset_order(data)
    n = len(datasets)
    z = np.full((n, n), np.nan)
    for i, target_dataset in enumerate(datasets):
        fp_ptq_acc = data[target_dataset]["fp_ptq"]
        for j, qv_dataset in enumerate(datasets):
            qv_acc = data[target_dataset]["qv_transfer"][qv_dataset]
            if qv_acc is not None and fp_ptq_acc is not None:
                z[i, j] = qv_acc - fp_ptq_acc
    return z


# ---------------------------------------------------------------------------
# Robust symmetric bounds
# ---------------------------------------------------------------------------
def _robust_symmetric_bounds(values, center, min_span=0.05, q_low=0.05, q_high=0.95):
    if not values:
        return center - min_span, center + min_span
    svals = sorted(values)
    n = len(svals)

    def _q(q):
        if n == 1:
            return svals[0]
        idx = (n - 1) * q
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return svals[lo] * (1.0 - frac) + svals[hi] * frac

    ql, qh = _q(q_low), _q(q_high)
    span = max(abs(center - ql), abs(qh - center), min_span)
    return center - span, center + span


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_sidebyside(data_fixed, data_best, args, model_dir, optim_frag, qat_frag, metric_tag):
    datasets = _swapped_dataset_order(data_fixed)
    display_datasets = [DATASET_LABEL_RENAMES.get(ds, ds) for ds in datasets]
    n = len(datasets)

    # -- build matrices -------------------------------------------------------
    z_fixed = _build_diff_matrix(data_fixed)
    z_best  = _build_diff_matrix(data_best)

    # -- shared color bounds --------------------------------------------------
    finite_all = (
        z_fixed[np.isfinite(z_fixed)].tolist()
        + z_best[np.isfinite(z_best)].tolist()
    )
    cmin, cmax = _robust_symmetric_bounds(finite_all, center=0.0, min_span=0.02)

    # -- colormap & norm ------------------------------------------------------
    cmap_div = plt.get_cmap("RdYlGn")
    cmap_div.set_bad(color="#d9d9d9")
    norm_div = mcolors.TwoSlopeNorm(vmin=cmin, vcenter=0.0, vmax=cmax)

    # -- figure setup ---------------------------------------------------------
    fig_width = 9.0
    fig = plt.figure(figsize=(fig_width, 6.25))
    display_name = MODEL_DISPLAY_NAMES.get(args.model_name, args.model_name)
    fig.suptitle(r"\textbf{" + display_name + "}", fontsize=12, y=0.87)

    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[1, 0.04],
        width_ratios=[1, 1],
        hspace=0.35,
        wspace=0.08,
    )

    ax_left  = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    # -- left heatmap (fixed alpha) -------------------------------------------
    masked_fixed = np.ma.masked_invalid(z_fixed)
    ax_left.imshow(masked_fixed, cmap=cmap_div, norm=norm_div, aspect="equal")

    ax_left.set_xticks(range(n))
    ax_left.set_xticklabels(display_datasets, rotation=45, ha="right", fontsize=9)
    ax_left.set_yticks(range(n))
    ax_left.set_yticklabels(display_datasets, fontsize=9)
    ax_left.set_xlabel("Donor Dataset", fontsize=9, labelpad=4)
    ax_left.set_ylabel("Receiver Dataset", fontsize=9, labelpad=4)
    ax_left.tick_params(axis="both", length=2, pad=2)
    ax_left.set_title(r"\textbf{constant scaling ($\lambda=1$)}", fontsize=9, pad=6)

    # -- right heatmap (best alpha) -------------------------------------------
    masked_best = np.ma.masked_invalid(z_best)
    im_right = ax_right.imshow(masked_best, cmap=cmap_div, norm=norm_div, aspect="equal")

    ax_right.set_xticks(range(n))
    ax_right.set_xticklabels(display_datasets, rotation=45, ha="right", fontsize=9)
    ax_right.set_yticks(range(n))
    ax_right.set_yticklabels([], fontsize=9)
    ax_right.set_xlabel("Donor Dataset", fontsize=9, labelpad=4)
    ax_right.tick_params(axis="both", length=2, pad=2)
    ax_right.set_title(r"\textbf{best $\lambda$ scaling}", fontsize=9, pad=6)

    # -- legend cells (below left heatmap) ------------------------------------
    ax_legend = fig.add_subplot(gs[1, 0])
    ax_legend.set_axis_off()

    legend_items = [
        ("Harmful",     cmin),
        ("Helpful",     cmax * 0.3),
        ("Strong gain", cmax),
    ]

    cell_width  = 0.08
    cell_height = 0.6
    spacing     = 0.12
    total_width = len(legend_items) * cell_width + (len(legend_items) - 1) * spacing
    x_start     = 0.5 - total_width / 2.0

    for idx, (label, value) in enumerate(legend_items):
        x = x_start + idx * (cell_width + spacing)
        color = cmap_div(norm_div(value))
        rect = plt.Rectangle(
            (x, 0.3), cell_width, cell_height,
            facecolor=color, edgecolor="black", linewidth=0.5,
            transform=ax_legend.transAxes, clip_on=False,
        )
        ax_legend.add_patch(rect)
        ax_legend.text(
            x + cell_width / 2.0, 0.1, label,
            transform=ax_legend.transAxes, ha="center", va="top",
            fontsize=9,
        )

    ax_legend.text(
        0.5, -1.5, "Transfer outcome",
        transform=ax_legend.transAxes, ha="center", va="top",
        fontsize=9,
    )

    # -- horizontal colorbar (below right heatmap) ----------------------------
    cbar_ax = fig.add_subplot(gs[1, 1])
    cbar = fig.colorbar(im_right, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Top-1 improvement of QV patching against naive PTQ", fontsize=9)
    cbar.ax.tick_params(labelsize=6)

    # -- export ---------------------------------------------------------------
    out_dir = os.path.join(
        "plots", "text", "ilharco_automodelforsequenceclassification",
        "999_paper_stuff", "001_qat_transfer",
        "qv_transfer_heatmap_alpha_fixed_vs_best",
        model_dir, f"seed={args.seed}", f"smult={mult_tag(args.source_epoch_mult)}", f"tmult={mult_tag(args.target_epoch_mult)}", optim_frag, qat_frag,
        _ptq_frag(args.ptq_bits, args.granularity, args.skip_modules),
        _qv_frag(args.qv_alpha), _split_frag(args.eval_split),
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"heatmap_alpha_fixed_vs_best_{metric_tag}.pdf")

    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    model_dir  = sanitize_hf_model_name(args.model_name)
    optim_frag = _optim_frag(args.optim, args.lr, args.wd, args.ls,
                             args.max_grad_norm, args.batch_size, args.max_length)
    qat_frag   = _qat_frag(args.qat_bits, args.granularity, args.skip_modules)
    ptq_frag   = _ptq_frag(args.ptq_bits, args.granularity, args.skip_modules)

    for metric_tag in TEST_METRIC_KEYS:
        data_fixed = load_data_fixed_alpha(args, model_dir, optim_frag, qat_frag, ptq_frag, metric_tag)
        data_best  = load_data_best_alpha(args, model_dir, optim_frag, qat_frag, ptq_frag, metric_tag)

        plot_sidebyside(data_fixed, data_best, args, model_dir, optim_frag, qat_frag, metric_tag)


if __name__ == "__main__":
    main()
