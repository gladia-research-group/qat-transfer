"""Mixed-modality QV Transfer Heatmap (best alpha, shared color scale)

Two heatmaps side-by-side: one vision model (timm supervised) and one
text model (automodel for sequence classification).  Both use best-alpha
results from pre-computed best_alpha_*.json files.

    Left  : vision heatmap — cell = qv_transfer_ptq(best alpha) - FT+PTQ
    Right : text heatmap   — cell = qv_transfer_ptq(best alpha) - FT+PTQ

Color scale is shared across both modalities.

Output: one PDF per metric tag (fp_head_ptq, qat_head_ptq).
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

from src.vision.data.common import DATASET_NAME_TO_EPOCHS as VISION_DATASET_NAME_TO_EPOCHS
from src.text.data.common import DATASET_NAME_TO_EPOCHS as TEXT_DATASET_NAME_TO_EPOCHS
from src.vision.utils import sanitize_timm_model_name, sanitize_hf_model_name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VISION_EVAL_ROOT_BASELINES = "evaluations/vision/ilharco_timm_supervised/000_baselines/vision"
VISION_EVAL_ROOT_QV        = "evaluations/vision/ilharco_timm_supervised/001_qat_transfer/vision/qv_transfer"

TEXT_EVAL_ROOT_BASELINES = "evaluations/text/ilharco_automodelforsequenceclassification/000_baselines/text"
TEXT_EVAL_ROOT_QV        = "evaluations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/text/qv_transfer"

METRIC_TAGS = {
    "fp_head_ptq": {
        "best_alpha_file": "best_alpha_fp_head_ptq.json",
        "best_alpha_key":  "val_accuracy_fp_head_ptq",
        "test_metric_key": "test_accuracy_fp_head_ptq",
    },
    "qat_head_ptq": {
        "best_alpha_file": "best_alpha_qat_head_ptq.json",
        "best_alpha_key":  "val_accuracy_qat_head_ptq",
        "test_metric_key": "test_accuracy_qat_head_ptq",
    },
}

TEST_ACC_KEY = "test_accuracy"

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
    "vit_base_patch16_224.orig_in21k": "ViT-B/16",
    "vit_large_patch16_224.orig_in21k": "ViT-L/16",
    "vit_huge_patch14_224.orig_in21k": "ViT-H/14",
    "deit3_base_patch16_224.fb_in1k": "DeiT3-B/16",
    "deit3_large_patch16_224.fb_in1k": "DeiT3-L/16",
    "swin_base_patch4_window7_224.ms_in22k_ft_in1k": "Swin-B",
    "swin_large_patch4_window7_224.ms_in22k_ft_in1k": "Swin-L",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-Embedding",
    "google-bert/bert-base-uncased": "BERT-Base",
    "google-bert/bert-large-uncased": "BERT-Large",
    "google/embeddinggemma-300m": "EmbeddingGemma",
}

VISION_DATASET_ORDER_SWAPS = [("DTD", "TinyImageNet"), ("RenderedSST2", "PCAM")]
TEXT_DATASET_ORDER_SWAPS = [("AmazonCounterfactual", "ToxicConversations")]


def _swapped_dataset_order(datasets_dict, swaps):
    """Sorted dataset list with aesthetic swaps applied."""
    ds = sorted(datasets_dict.keys())
    for a, b in swaps:
        if a in ds and b in ds:
            ia, ib = ds.index(a), ds.index(b)
            ds[ia], ds[ib] = ds[ib], ds[ia]
    return ds


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    # -- shared ---------------------------------------------------------------
    parser.add_argument("--seed",            required=True, type=int)
    parser.add_argument("--source-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the DONOR checkpoints.")
    parser.add_argument("--target-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the RECEIVER checkpoints.")
    parser.add_argument("--optim",           required=True, choices=["adamw", "sgd"])
    parser.add_argument("--qat-bits",        required=True, type=int)
    parser.add_argument("--ptq-bits",        required=True, type=int)
    parser.add_argument("--granularity",     required=True, choices=["tensor", "channel"])
    parser.add_argument("--eval-split",      default="test", choices=["val", "test"],
                        help="Which split the qv_transfer results were evaluated on.")

    # -- vision ---------------------------------------------------------------
    parser.add_argument("--vision-model-name",   required=True,
                        help="timm model name (e.g. vit_base_patch16_224.orig_in21k)")
    parser.add_argument("--vision-lr",           required=True, type=float)
    parser.add_argument("--vision-wd",           required=True, type=float)
    parser.add_argument("--vision-ls",           required=True, type=float)
    parser.add_argument("--vision-wl",           required=True, type=int,
                        help="Warmup length (vision)")
    parser.add_argument("--vision-max-grad-norm", required=True, type=float)
    parser.add_argument("--vision-batch-size",   required=True, type=int)
    parser.add_argument("--vision-skip-modules", required=True, nargs="+",
                        help="Module names to skip during quantization (vision).")

    # -- text -----------------------------------------------------------------
    parser.add_argument("--text-model-name",     required=True,
                        help="HF model name (e.g. google-bert/bert-base-uncased)")
    parser.add_argument("--text-lr",             required=True, type=float)
    parser.add_argument("--text-wd",             required=True, type=float)
    parser.add_argument("--text-ls",             required=True, type=float)
    parser.add_argument("--text-max-grad-norm",  required=True, type=float)
    parser.add_argument("--text-max-length",     required=True, type=int)
    parser.add_argument("--text-batch-size",     required=True, type=int)
    parser.add_argument("--text-skip-module",    required=True,
                        help="Module name to skip during quantization (text).")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path-fragment helpers — vision
# ---------------------------------------------------------------------------
def _vision_skip_tag(skip_modules):
    return "-".join(sorted(skip_modules)) if len(skip_modules) > 0 else "none"


def _vision_optim_frag(args):
    return (f"optim=adamw_lr={args.vision_lr}_wd={args.vision_wd}_ls={args.vision_ls}"
            f"_wl={args.vision_wl}_mgn={args.vision_max_grad_norm}_bs={args.vision_batch_size}")


def _vision_qat_frag(args):
    return f"qat=bits={args.qat_bits}_gran={args.granularity}_skip={_vision_skip_tag(args.vision_skip_modules)}"


def _vision_ptq_frag(args):
    return f"ptq=bits={args.ptq_bits}_gran={args.granularity}_skip={_vision_skip_tag(args.vision_skip_modules)}"


# ---------------------------------------------------------------------------
# Path-fragment helpers — text
# ---------------------------------------------------------------------------
def _text_optim_frag(args):
    return (f"optim=adamw_lr={args.text_lr}_wd={args.text_wd}_ls={args.text_ls}"
            f"_mgn={args.text_max_grad_norm}_bs={args.text_batch_size}_ml={args.text_max_length}")


def _text_qat_frag(args):
    return f"qat=bits={args.qat_bits}_gran={args.granularity}_skip={args.text_skip_module}"


def _text_ptq_frag(args):
    return f"ptq=bits={args.ptq_bits}_gran={args.granularity}_skip={args.text_skip_module}"


# ---------------------------------------------------------------------------
# Per-baseline path builders
# ---------------------------------------------------------------------------
def _fp_ptq_path(eval_root, model_dir, dataset, seed, optim_frag, ptq_frag, *, target_epoch_mult):
    return os.path.join(
        eval_root, "fp_ptq", model_dir, dataset,
        optim_frag, ptq_frag, f"seed={seed}", "eval_results.json",
    )


# ---------------------------------------------------------------------------
# QV transfer path builders
# ---------------------------------------------------------------------------
def _qv_transfer_cell_prefix(eval_root_qv, model_dir, qv_dataset, target_dataset,
                              seed, optim_frag, qat_frag, ptq_frag, *, source_epoch_mult, target_epoch_mult):
    return os.path.join(
        eval_root_qv, model_dir,
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
# Data loading — best alpha
# ---------------------------------------------------------------------------
def load_data_best_alpha(datasets, eval_root_baselines, eval_root_qv,
                         model_dir, seed, optim_frag, qat_frag, ptq_frag,
                         best_alpha_file, best_alpha_key, test_metric_key):
    data = {}
    for target_dataset in datasets:
        fp_ptq_acc = _load_value(
            _fp_ptq_path(eval_root_baselines, model_dir, target_dataset, seed,
                         optim_frag, ptq_frag, target_epoch_mult=args.target_epoch_mult),
            TEST_ACC_KEY,
        )
        qv_transfer = {}
        for qv_dataset in datasets:
            cell_prefix = _qv_transfer_cell_prefix(
                eval_root_qv, model_dir, qv_dataset, target_dataset, seed,
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
def _build_diff_matrix(data, datasets):
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
def plot_sidebyside(vision_data, text_data, args,
                    vision_qat_frag, text_qat_frag, metric_tag):
    vision_datasets = _swapped_dataset_order(vision_data, VISION_DATASET_ORDER_SWAPS)
    text_datasets = _swapped_dataset_order(text_data, TEXT_DATASET_ORDER_SWAPS)
    text_display = [DATASET_LABEL_RENAMES.get(ds, ds) for ds in text_datasets]
    n_vis = len(vision_datasets)
    n_txt = len(text_datasets)

    # -- build matrices -------------------------------------------------------
    z_vis = _build_diff_matrix(vision_data, vision_datasets)
    z_txt = _build_diff_matrix(text_data, text_datasets)

    # -- shared color bounds --------------------------------------------------
    finite_all = (
        z_vis[np.isfinite(z_vis)].tolist()
        + z_txt[np.isfinite(z_txt)].tolist()
    )
    cmin, cmax = _robust_symmetric_bounds(finite_all, center=0.0, min_span=0.02)

    # -- colormap & norm ------------------------------------------------------
    cmap_div = plt.get_cmap("RdYlGn")
    cmap_div.set_bad(color="#d9d9d9")
    norm_div = mcolors.TwoSlopeNorm(vmin=cmin, vcenter=0.0, vmax=cmax)

    # -- figure setup (manual positioning for top-alignment) ------------------
    # Layout: vision heatmap (left, full height), text heatmap (right, top-
    # aligned, shorter), legend cells + colorbar below text heatmap.
    margin_l, margin_r, margin_t, margin_b = 0.12, 0.12, 0.06, 0.12
    gap_h = 0.25                          # horizontal gap between heatmaps
    cell_size_in = 0.28                   # inches per matrix cell

    vis_w_in = n_vis * cell_size_in
    vis_h_in = n_vis * cell_size_in
    txt_w_in = n_txt * cell_size_in
    txt_h_in = n_txt * cell_size_in

    fig_w = margin_l + vis_w_in + gap_h + txt_w_in + margin_r
    fig_h = margin_t + vis_h_in + margin_b

    fig = plt.figure(figsize=(fig_w, fig_h))

    # normalised coordinates
    x_vis = margin_l / fig_w
    w_vis = vis_w_in / fig_w
    h_vis = vis_h_in / fig_h
    y_vis = 1.0 - margin_t / fig_h - h_vis

    x_txt = (margin_l + vis_w_in + gap_h) / fig_w
    w_txt = txt_w_in / fig_w
    h_txt = txt_h_in / fig_h
    y_txt = 1.0 - margin_t / fig_h - h_txt          # top-aligned with vision

    ax_left  = fig.add_axes([x_vis, y_vis, w_vis, h_vis])
    ax_right = fig.add_axes([x_txt, y_txt, w_txt, h_txt])

    # -- left heatmap (vision) ------------------------------------------------
    masked_vis = np.ma.masked_invalid(z_vis)
    ax_left.imshow(masked_vis, cmap=cmap_div, norm=norm_div, aspect="equal")

    ax_left.set_xticks(range(n_vis))
    ax_left.set_xticklabels(vision_datasets, rotation=45, ha="right", fontsize=9)
    ax_left.set_yticks(range(n_vis))
    ax_left.set_yticklabels(vision_datasets, fontsize=9)
    ax_left.set_xlabel("Donor Dataset", fontsize=12, labelpad=4)
    ax_left.set_ylabel("Receiver Dataset", fontsize=12, labelpad=4)
    ax_left.tick_params(axis="both", length=2, pad=2)
    vision_display = MODEL_DISPLAY_NAMES.get(args.vision_model_name, args.vision_model_name)
    ax_left.set_title(r"\textbf{" + f"Vision: {vision_display}" + "}", fontsize=14, pad=6)

    # -- right heatmap (text, labels on the right) ----------------------------
    masked_txt = np.ma.masked_invalid(z_txt)
    im_right = ax_right.imshow(masked_txt, cmap=cmap_div, norm=norm_div, aspect="equal")

    ax_right.set_xticks(range(n_txt))
    ax_right.set_xticklabels(text_display, rotation=-45, ha="left", fontsize=9)
    ax_right.set_yticks(range(n_txt))
    ax_right.set_yticklabels(text_display, fontsize=9)
    ax_right.yaxis.tick_right()
    ax_right.yaxis.set_label_position("right")
    ax_right.set_xlabel("Donor Dataset", fontsize=12, labelpad=4)
    ax_right.tick_params(axis="both", length=2, pad=2)
    text_display_name = MODEL_DISPLAY_NAMES.get(args.text_model_name, args.text_model_name)
    ax_right.set_title(r"\textbf{" + f"Text: {text_display_name}" + "}", fontsize=14, pad=6)

    # -- legend cells + colorbar (below text heatmap) -------------------------
    # Available vertical space below the text heatmap
    below_top = y_txt                     # top of the empty region
    below_bot = y_vis                     # bottom-align with vision bottom
    below_mid = (below_top + below_bot) / 2.0

    # legend cells — pushed down to avoid overlapping text x-tick labels
    legend_h = (below_top - below_mid) * 0.8
    legend_y = below_mid - legend_h * 0.8
    ax_legend = fig.add_axes([x_txt, legend_y, w_txt, legend_h])
    ax_legend.set_axis_off()

    legend_items = [
        ("Harmful",     cmin),
        ("Helpful",     cmax * 0.3),
        ("Strong gain", cmax),
    ]

    # Square cells sized like heatmap cells (cell_size_in inches)
    leg_axes_w = w_txt * fig_w            # legend axes width in inches
    leg_axes_h = legend_h * fig_h         # legend axes height in inches
    sq_w = cell_size_in / leg_axes_w      # square width in axes fraction
    sq_h = cell_size_in / leg_axes_h      # square height in axes fraction
    spacing     = sq_w * 1.8
    total_width = len(legend_items) * sq_w + (len(legend_items) - 1) * spacing
    x_start     = 0.5 - total_width / 2.0

    for idx, (label, value) in enumerate(legend_items):
        x = x_start + idx * (sq_w + spacing)
        color = cmap_div(norm_div(value))
        rect = plt.Rectangle(
            (x, 0.5 - sq_h / 2.0 + 0.1), sq_w, sq_h,
            facecolor=color, edgecolor="black", linewidth=0.5,
            transform=ax_legend.transAxes, clip_on=False,
        )
        ax_legend.add_patch(rect)
        ax_legend.text(
            x + sq_w / 2.0, 0.5 - sq_h / 2.0 + 0.1 - 0.05, label,
            transform=ax_legend.transAxes, ha="center", va="top",
            fontsize=12,
        )

    ax_legend.text(
        0.5, -0.1, "Transfer outcome",
        transform=ax_legend.transAxes, ha="center", va="top",
        fontsize=12,
    )

    # horizontal colorbar
    cbar_h = 0.02
    cbar_y = below_bot + 0.02
    cbar_ax = fig.add_axes([x_txt, cbar_y, w_txt, cbar_h])
    cbar = fig.colorbar(im_right, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Top-1 improvement of QV patching against naive PTQ", fontsize=9)
    cbar.ax.tick_params(labelsize=9)

    # -- export ---------------------------------------------------------------
    vision_model_tag = sanitize_timm_model_name(args.vision_model_name)
    text_model_tag = sanitize_hf_model_name(args.text_model_name)
    combined_qat_frag = f"vision={vision_qat_frag}__text={text_qat_frag}"
    out_dir = os.path.join(
        "plots", "mixed_modalities",
        "ilharco_timm_supervised_ilharco_automodelforsequenceclassification",
        "999_paper_stuff", "001_qat_transfer",
        "qv_transfer_heatmap_best_alpha_shared_cscale",
        f"{vision_model_tag}__{text_model_tag}",
        f"seed={args.seed}", f"smult={mult_tag(args.source_epoch_mult)}", f"tmult={mult_tag(args.target_epoch_mult)}", combined_qat_frag,
        f"vision={_vision_ptq_frag(args)}__text={_text_ptq_frag(args)}",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"heatmap_best_alpha_{metric_tag}.pdf")

    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    vision_model_dir  = sanitize_timm_model_name(args.vision_model_name)
    vision_optim_frag = _vision_optim_frag(args)
    vision_qat_frag   = _vision_qat_frag(args)
    vision_ptq_frag   = _vision_ptq_frag(args)

    text_model_dir  = sanitize_hf_model_name(args.text_model_name)
    text_optim_frag = _text_optim_frag(args)
    text_qat_frag   = _text_qat_frag(args)
    text_ptq_frag   = _text_ptq_frag(args)

    vision_datasets = _swapped_dataset_order(VISION_DATASET_NAME_TO_EPOCHS, VISION_DATASET_ORDER_SWAPS)
    text_datasets   = _swapped_dataset_order(TEXT_DATASET_NAME_TO_EPOCHS, TEXT_DATASET_ORDER_SWAPS)

    for metric_tag, tag_info in METRIC_TAGS.items():
        print(f"Loading vision data ({metric_tag}) for {args.vision_model_name} ...")
        vision_data = load_data_best_alpha(
            vision_datasets,
            VISION_EVAL_ROOT_BASELINES, VISION_EVAL_ROOT_QV,
            vision_model_dir, args.seed,
            vision_optim_frag, vision_qat_frag, vision_ptq_frag,
            tag_info["best_alpha_file"], tag_info["best_alpha_key"],
            tag_info["test_metric_key"],
        )

        print(f"Loading text data ({metric_tag}) for {args.text_model_name} ...")
        text_data = load_data_best_alpha(
            text_datasets,
            TEXT_EVAL_ROOT_BASELINES, TEXT_EVAL_ROOT_QV,
            text_model_dir, args.seed,
            text_optim_frag, text_qat_frag, text_ptq_frag,
            tag_info["best_alpha_file"], tag_info["best_alpha_key"],
            tag_info["test_metric_key"],
        )

        plot_sidebyside(vision_data, text_data, args,
                        vision_qat_frag, text_qat_frag, metric_tag)


if __name__ == "__main__":
    main()
