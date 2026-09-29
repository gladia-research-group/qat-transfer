"""Vertical grouped bar chart: FT vs QAT vs PTQ baselines (open_clip).

Per-dataset vertical bars showing test accuracy for three baseline methods:
    1. FT   — floating-point finetuned          (000_baselines/fp)
    2. QAT  — quantization-aware trained         (000_baselines/qat)
    3. PTQ  — FP model + post-training quant     (000_baselines/fp_ptq)

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

from src.duration import mult_path_frag

os.chdir(_PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
import numpy as np

from src.vision.data.common import DATASET_NAME_TO_EPOCHS
from src.vision.utils import sanitize_open_clip_model_name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT_BASELINES = "evaluations/vision/ilharco_open_clip/000_baselines/vision"

TEST_ACC_KEY = "test_accuracy"

DATASET_ORDER_SWAPS = [("DTD", "TinyImageNet"), ("RenderedSST2", "PCAM")]

MODEL_DISPLAY_NAMES = {
    "ViT/B/16": "ViT-B/16",
    "ViT/L/14": "ViT-L/14",
    "ViT/H/14": "ViT-H/14",
    "ViT/B/32": "ViT-B/32",
    "ViT/L/16": "ViT-L/16",
}


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
    parser.add_argument("--model-name",      required=True,
                        help="open_clip model name (e.g. ViT-B-16)")
    parser.add_argument("--pretrained-tag",  required=True,
                        help="open_clip pretrained tag (e.g. openai, laion2b_s34b_b88k)")
    parser.add_argument("--seed",            required=True, type=int)

    parser.add_argument("--optim",           required=True, choices=["adamw", "sgd"])
    parser.add_argument("--lr",              required=True, type=float)
    parser.add_argument("--wd",              required=True, type=float)
    parser.add_argument("--ls",              required=True, type=float)
    parser.add_argument("--wl",              required=True, type=int)
    parser.add_argument("--max-grad-norm",   required=True, type=float)
    parser.add_argument("--batch-size",      required=True, type=int)

    parser.add_argument("--qat-bits",        required=True, type=int)
    parser.add_argument("--ptq-bits",        required=True, type=int)
    parser.add_argument("--granularity",     required=True, choices=["tensor", "channel"])
    parser.add_argument("--skip-modules",    required=True, nargs="+",
                        help="One or more module names to skip during quantization "
                             "(no default: must be specified explicitly).")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path-fragment helpers
# ---------------------------------------------------------------------------
def _skip_tag(skip_modules):
    return "-".join(sorted(skip_modules)) if len(skip_modules) > 0 else "none"


def _optim_frag(args):
    return (f"optim=adamw_lr={args.lr}_wd={args.wd}_ls={args.ls}"
            f"_wl={args.wl}_mgn={args.max_grad_norm}_bs={args.batch_size}")


def _qat_frag(args):
    return f"qat=bits={args.qat_bits}_gran={args.granularity}_skip={_skip_tag(args.skip_modules)}"


def _ptq_frag(args):
    return f"ptq=bits={args.ptq_bits}_gran={args.granularity}_skip={_skip_tag(args.skip_modules)}"


# ---------------------------------------------------------------------------
# Per-baseline path builders
# ---------------------------------------------------------------------------
def _fp_path(model_dir, dataset, seed, optim_frag, *, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_BASELINES, "fp", model_dir, dataset,
        optim_frag, mult_path_frag(target_epoch_mult), f"seed={seed}", "eval_results.json",
    )


def _qat_path(model_dir, dataset, seed, optim_frag, qat_frag, *, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_BASELINES, "qat", model_dir, dataset,
        optim_frag, mult_path_frag(target_epoch_mult), qat_frag, f"seed={seed}", "eval_results.json",
    )


def _fp_ptq_path(model_dir, dataset, seed, optim_frag, ptq_frag, *, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_BASELINES, "fp_ptq", model_dir, dataset,
        optim_frag, mult_path_frag(target_epoch_mult), ptq_frag, f"seed={seed}", "eval_results.json",
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
def load_data(model_dir, args, optim_frag, qat_frag, ptq_frag):
    """Return {dataset: {"ft": float|None, "qat": float|None, "ptq": float|None}}."""
    datasets = _swapped_dataset_order(DATASET_NAME_TO_EPOCHS)

    data = {}
    for dataset in datasets:
        ft_acc = _load_value(
            _fp_path(model_dir, dataset, args.seed, optim_frag, target_epoch_mult=args.target_epoch_mult),
            TEST_ACC_KEY,
        )
        qat_acc = _load_value(
            _qat_path(model_dir, dataset, args.seed, optim_frag, qat_frag, target_epoch_mult=args.target_epoch_mult),
            TEST_ACC_KEY,
        )
        ptq_acc = _load_value(
            _fp_ptq_path(model_dir, dataset, args.seed, optim_frag, ptq_frag, target_epoch_mult=args.target_epoch_mult),
            TEST_ACC_KEY,
        )
        data[dataset] = {"ft": ft_acc, "qat": qat_acc, "ptq": ptq_acc}

    return data


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_bar(data, model_dir, model_name, args):
    datasets = _swapped_dataset_order(DATASET_NAME_TO_EPOCHS)
    n = len(datasets)

    bar_width = 0.25
    x_pos = np.arange(n)

    ft_vals = [data[ds]["ft"] if data[ds]["ft"] is not None else 0 for ds in datasets]
    qat_vals = [data[ds]["qat"] if data[ds]["qat"] is not None else 0 for ds in datasets]
    ptq_vals = [data[ds]["ptq"] if data[ds]["ptq"] is not None else 0 for ds in datasets]

    fig, ax = plt.subplots(figsize=(14, 3.5))

    ax.bar(x_pos - bar_width, ft_vals, bar_width,
           label=r"\textbf{FT}", color="#e09f3e")
    ax.bar(x_pos, qat_vals, bar_width,
           label=r"\textbf{QAT}", color="#540b0e")
    ax.bar(x_pos + bar_width, ptq_vals, bar_width,
           label=r"\textbf{PTQ}", color="#9e2a2b")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, fontsize=11, rotation=45, ha="right")
    ax.set_ylabel(r"Test Accuracy", fontsize=14)
    ax.set_ylim(0, 1.0)

    display_name = model_name.replace("-", "/")
    display_title = MODEL_DISPLAY_NAMES.get(display_name, display_name)
    ax.set_title(r"\textbf{" + display_title + "}", fontsize=18)

    ax.legend(fontsize=12, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    fig.tight_layout()

    # -- export ---------------------------------------------------------------
    out_dir = os.path.join(
        "plots", "vision", "ilharco_open_clip", "999_paper_stuff", "000_baselines",
        "baseline_bar", model_dir, f"seed={args.seed}",
        _ptq_frag(args),
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "baseline_bar.pdf")

    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    model_dir = sanitize_open_clip_model_name(args.model_name, args.pretrained_tag)
    optim_frag = _optim_frag(args)
    qat_frag = _qat_frag(args)
    ptq_frag = _ptq_frag(args)

    display_name = args.model_name.replace("-", "/")
    print(f"Loading data for {display_name} ({args.pretrained_tag}) ...")
    data = load_data(model_dir, args, optim_frag, qat_frag, ptq_frag)
    plot_bar(data, model_dir, args.model_name, args)


if __name__ == "__main__":
    main()
