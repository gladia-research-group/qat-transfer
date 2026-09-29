"""Radar plot: FT+PTQ vs QV Patching+PTQ (λ=1) vs QV Patching+PTQ (best λ) vs QAT+PTQ (mixed modalities)

Two radar plots side-by-side: one vision model (timm supervised) and one
text model (automodel for sequence classification).  Each radar has four webs:

    1. FT+PTQ baseline          (000_baselines/fp_ptq)
    2. QV Patching+PTQ (λ=1)   (001_qat_transfer, best donor at λ=1, unit scaling)
    3. QV Patching+PTQ (best λ) (001_qat_transfer, same donor, best alpha)
    4. QAT+PTQ                  (000_baselines/qat_ptq)

Donor selection: the donor with highest accuracy at λ=1 is selected;
both the λ=1 and best-λ webs use that same donor.

Output: PDF with LaTeX fonts (paper-ready).
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
TEST_ACC_KEY    = "test_accuracy"

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


def _qat_ptq_path(eval_root, model_dir, dataset, seed, optim_frag, qat_frag, ptq_frag, *, target_epoch_mult):
    return os.path.join(
        eval_root, "qat_ptq", model_dir, dataset,
        optim_frag, qat_frag, ptq_frag, f"seed={seed}", "eval_results.json",
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
# Data loading — generic (works for both vision and text)
# ---------------------------------------------------------------------------
def _load_radar_data(datasets, eval_root_baselines, eval_root_qv,
                     model_dir, seed, optim_frag, qat_frag, ptq_frag, tag_info):
    """Return {dataset: {"fp_ptq", "qat_ptq", "qat_transfer_ptq_unit", "qat_transfer_ptq_best"}}."""
    best_alpha_file = tag_info["best_alpha_file"]
    best_alpha_key  = tag_info["best_alpha_key"]
    test_metric_key = tag_info["test_metric_key"]

    data = {}
    for target_dataset in datasets:
        fp_ptq_acc = _load_value(
            _fp_ptq_path(eval_root_baselines, model_dir, target_dataset, seed,
                         optim_frag, ptq_frag, target_epoch_mult=args.target_epoch_mult),
            TEST_ACC_KEY,
        )
        qat_ptq_acc = _load_value(
            _qat_ptq_path(eval_root_baselines, model_dir, target_dataset, seed,
                          optim_frag, qat_frag, ptq_frag, target_epoch_mult=args.target_epoch_mult),
            TEST_ACC_KEY,
        )

        # Step 1: find the best donor at λ=1
        best_unit_acc = None
        best_unit_donor = None
        for qv_dataset in datasets:
            if qv_dataset == target_dataset:
                continue

            cell_prefix = _qv_transfer_cell_prefix(
                eval_root_qv, model_dir, qv_dataset, target_dataset, seed,
                optim_frag, qat_frag, ptq_frag,
            
                source_epoch_mult=args.source_epoch_mult, target_epoch_mult=args.target_epoch_mult,
            )
            unit_path = os.path.join(
                cell_prefix, "qv=alpha=1.0",
                "split=test", "eval_results.json",
            )
            acc = _load_value(unit_path, test_metric_key)
            if acc is not None and (best_unit_acc is None or acc > best_unit_acc):
                best_unit_acc = acc
                best_unit_donor = qv_dataset

        # Step 2: get that donor's best-λ accuracy
        best_transfer_acc = None
        if best_unit_donor is not None:
            cell_prefix = _qv_transfer_cell_prefix(
                eval_root_qv, model_dir, best_unit_donor, target_dataset, seed,
                optim_frag, qat_frag, ptq_frag,
            
                source_epoch_mult=args.source_epoch_mult, target_epoch_mult=args.target_epoch_mult,
            )
            best_alpha_path = os.path.join(cell_prefix, best_alpha_file)
            if os.path.exists(best_alpha_path):
                with open(best_alpha_path) as f:
                    info = json.load(f).get(best_alpha_key)
                if info is not None:
                    best_alpha_val = info["alpha"]
                    test_path = os.path.join(
                        cell_prefix, f"qv=alpha={best_alpha_val}",
                        "split=test", "eval_results.json",
                    )
                    best_transfer_acc = _load_value(test_path, test_metric_key)

        data[target_dataset] = {
            "fp_ptq": fp_ptq_acc,
            "qat_ptq": qat_ptq_acc,
            "qat_transfer_ptq_unit": best_unit_acc,
            "qat_transfer_ptq_best": best_transfer_acc,
        }

    return data


# ---------------------------------------------------------------------------
# Radar plot
# ---------------------------------------------------------------------------
def plot_radar(vision_data, text_data, args,
               vision_qat_frag, text_qat_frag, metric_tag):
    vision_datasets = _swapped_dataset_order(VISION_DATASET_NAME_TO_EPOCHS, VISION_DATASET_ORDER_SWAPS)
    text_datasets = _swapped_dataset_order(TEXT_DATASET_NAME_TO_EPOCHS, TEXT_DATASET_ORDER_SWAPS)

    web_keys = ["fp_ptq", "qat_transfer_ptq_unit", "qat_transfer_ptq_best", "qat_ptq"]
    web_labels = [
        r"\textbf{FT$+$PTQ}",
        r"\textbf{QV Patching$+$PTQ ($\lambda=1$)}",
        r"\textbf{QV Patching$+$PTQ (best $\lambda$)}",
        r"\textbf{QAT$+$PTQ}",
    ]
    web_colors = ["#e07a5f", "#f2cc8f", "#81b29a", "#3d405b"]
    web_markers = ["o", "D", "s", "^"]
    web_linestyles = ["-", "--", "-", "-"]

    LABEL_RENAMES = {
        "AmazonCounterfactual": "Counterfactual",
        "TweetSentimentExtraction": "Sentiment",
        "AmazonReviewsClassification": "Reviews",
        "ToxicConversations": "Toxic",
        "MTOPDomain": "MTOP-D",
        "MTOPIntent": "MTOP-I",
        "MassiveIntent": "Intent",
        "MassiveScenario": "Scenario",
    }

    fig, axes = plt.subplots(1, 2, figsize=(18, 10), subplot_kw=dict(polar=True))

    for ax, data, datasets, model_name, prefix in [
        (axes[0], vision_data, vision_datasets, args.vision_model_name, "Vision"),
        (axes[1], text_data, text_datasets, args.text_model_name, "Text"),
    ]:
        n = len(datasets)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles.append(angles[0])

        for key, label, color, marker, ls in zip(
            web_keys, web_labels, web_colors, web_markers, web_linestyles,
        ):
            values = []
            for ds in datasets:
                v = data[ds].get(key)
                values.append(v if v is not None else np.nan)
            values.append(values[0])
            ax.plot(angles, values, color=color, linewidth=1.5, marker=marker,
                    markersize=4, label=label, linestyle=ls)
            ax.fill(angles, values, color=color, alpha=0.08)

        ax.set_xticks(angles[:-1])
        display_labels = [LABEL_RENAMES.get(ds, ds) for ds in datasets]
        ax.set_xticklabels(display_labels, fontsize=16)
        ax.tick_params(axis="x", pad=20)
        ax.set_ylim(top=1.0)
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        ax.set_title(r"\textbf{" + f"{prefix}: {display_name}" + "}", fontsize=24, pad=35)
        ax.tick_params(axis="y", labelsize=14)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=18,
               frameon=False, bbox_to_anchor=(0.5, 0.02))

    fig.tight_layout(rect=[0, 0.02, 1, 1], w_pad=5)

    # -- export ---------------------------------------------------------------
    vision_model_tag = sanitize_timm_model_name(args.vision_model_name)
    text_model_tag = sanitize_hf_model_name(args.text_model_name)
    combined_qat_frag = f"vision={vision_qat_frag}__text={text_qat_frag}"
    out_dir = os.path.join(
        "plots", "mixed_modalities",
        "ilharco_timm_supervised_ilharco_automodelforsequenceclassification",
        "999_paper_stuff", "001_qat_transfer", "radar_plot_unit_sf_best_sf",
        f"{vision_model_tag}__{text_model_tag}",
        f"seed={args.seed}", f"smult={mult_tag(args.source_epoch_mult)}", f"tmult={mult_tag(args.target_epoch_mult)}", combined_qat_frag,
        f"vision={_vision_ptq_frag(args)}__text={_text_ptq_frag(args)}",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"radar_plot_unit_sf_best_sf_{metric_tag}.pdf")

    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # -- vision ---------------------------------------------------------------
    vision_model_dir = sanitize_timm_model_name(args.vision_model_name)
    vision_optim_frag = _vision_optim_frag(args)
    vision_qat_frag = _vision_qat_frag(args)
    vision_ptq_frag = _vision_ptq_frag(args)

    # -- text -----------------------------------------------------------------
    text_model_dir = sanitize_hf_model_name(args.text_model_name)
    text_optim_frag = _text_optim_frag(args)
    text_qat_frag = _text_qat_frag(args)
    text_ptq_frag = _text_ptq_frag(args)

    for metric_tag, tag_info in METRIC_TAGS.items():
        print(f"Loading vision data ({metric_tag}) for {args.vision_model_name} ...")
        vision_data = _load_radar_data(
            sorted(VISION_DATASET_NAME_TO_EPOCHS.keys()),
            VISION_EVAL_ROOT_BASELINES, VISION_EVAL_ROOT_QV,
            vision_model_dir, args.seed,
            vision_optim_frag, vision_qat_frag, vision_ptq_frag, tag_info,
        )

        print(f"Loading text data ({metric_tag}) for {args.text_model_name} ...")
        text_data = _load_radar_data(
            sorted(TEXT_DATASET_NAME_TO_EPOCHS.keys()),
            TEXT_EVAL_ROOT_BASELINES, TEXT_EVAL_ROOT_QV,
            text_model_dir, args.seed,
            text_optim_frag, text_qat_frag, text_ptq_frag, tag_info,
        )

        # -- plot -------------------------------------------------------------
        plot_radar(vision_data, text_data, args, vision_qat_frag, text_qat_frag, metric_tag)


if __name__ == "__main__":
    main()
