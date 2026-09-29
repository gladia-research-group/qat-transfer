"""Combined baseline bar chart for all timm supervised models (curated list).

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

from src.vision.data.common import DATASET_NAME_TO_EPOCHS
from src.vision.utils import sanitize_timm_model_name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT_BASELINES = "evaluations/vision/ilharco_timm_supervised/000_baselines/vision"

TEST_ACC_KEY = "test_accuracy"

DATASET_ORDER_SWAPS = [("DTD", "TinyImageNet"), ("RenderedSST2", "PCAM")]

MODEL_DISPLAY_NAMES = {
    "vit_base_patch16_224.orig_in21k": "ViT-B/16",
    "vit_large_patch16_224.orig_in21k": "ViT-L/16",
    "vit_huge_patch14_224.orig_in21k": "ViT-H/14",
    "deit3_base_patch16_224.fb_in1k": "DeiT3-B/16",
    "deit3_large_patch16_224.fb_in1k": "DeiT3-L/16",
    "swin_base_patch4_window7_224.ms_in22k_ft_in1k": "Swin-B",
    "swin_large_patch4_window7_224.ms_in22k_ft_in1k": "Swin-L",
}

MODEL_ORDER = [
    "vit_base_patch16_224.orig_in21k",
    "vit_large_patch16_224.orig_in21k",
    "vit_huge_patch14_224.orig_in21k",
    "deit3_base_patch16_224.fb_in1k",
    "deit3_large_patch16_224.fb_in1k",
    "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    "swin_large_patch4_window7_224.ms_in22k_ft_in1k",
]


def _swapped_dataset_order(datasets_dict):
    ds = sorted(datasets_dict.keys())
    for a, b in DATASET_ORDER_SWAPS:
        if a in ds and b in ds:
            ia, ib = ds.index(a), ds.index(b)
            ds[ia], ds[ib] = ds[ib], ds[ia]
    return ds


def _unsanitize_model_name(sanitized):
    """Best-effort reverse of sanitize_timm_model_name for known models."""
    for orig in MODEL_ORDER:
        if sanitize_timm_model_name(orig) == sanitized:
            return orig
    return sanitized


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int)

    parser.add_argument("--model-names", required=True, nargs="+",
                        help="Models to plot, in panel order.")
    parser.add_argument("--batch-sizes", required=True, nargs="+", type=int,
                        help="Per-model batch size, ordering must match --model-names. "
                             "Large backbones were finetuned at bs=64 with accum_steps=2, "
                             "so their effective batch matches the bs=128 runs -- but the "
                             "path fragment differs and must be stated, not discovered.")

    parser.add_argument("--optim", required=True, choices=["adamw", "sgd"])
    parser.add_argument("--lr", required=True, type=float)
    parser.add_argument("--wd", required=True, type=float)
    parser.add_argument("--ls", required=True, type=float)
    parser.add_argument("--wl", required=True, type=int)
    parser.add_argument("--max-grad-norm", required=True, type=float)

    parser.add_argument("--qat-bits", required=True, type=int)
    parser.add_argument("--ptq-bits", required=True, type=int)
    parser.add_argument("--granularity", required=True, choices=["tensor", "channel"])
    parser.add_argument("--skip-modules", required=True, nargs="+",
                        help="One or more module names to skip during quantization "
                             "(no default: must be specified explicitly).")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if any expected eval_results.json is missing. "
                             "Without it, misses are still counted and reported on stderr.")
    args = parser.parse_args()
    if len(args.model_names) != len(args.batch_sizes):
        parser.error(f"--model-names ({len(args.model_names)}) and --batch-sizes "
                     f"({len(args.batch_sizes)}) must have the same length")
    return args


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _skip_tag(skip_modules):
    return "-".join(sorted(skip_modules)) if len(skip_modules) > 0 else "none"


def _optim_frag(args, batch_size):
    return (f"optim={args.optim}_lr={args.lr}_wd={args.wd}_ls={args.ls}"
            f"_wl={args.wl}_mgn={args.max_grad_norm}_bs={batch_size}")


def _qat_frag(args):
    return f"qat=bits={args.qat_bits}_gran={args.granularity}_skip={_skip_tag(args.skip_modules)}"


def _ptq_frag(args):
    return f"ptq=bits={args.ptq_bits}_gran={args.granularity}_skip={_skip_tag(args.skip_modules)}"


# ---------------------------------------------------------------------------
# Auto-discover models and their optim frags
# ---------------------------------------------------------------------------
def resolve_models(args):
    """Return [(model_name, model_dir, optim_frag), ...] in --model-names order.

    Nothing is discovered.  The caller states which models to plot and the batch
    size each was trained at; this only verifies that the resulting path exists.
    The previous version auto-discovered the optim fragment by taking the first
    directory starting with "optim=", which silently mixed bs=64 and bs=128 runs
    into one figure without saying so.
    """
    fp_root = os.path.join(EVAL_ROOT_BASELINES, "fp")
    if not os.path.isdir(fp_root):
        print(f"[ERROR] fp root not found: {fp_root}", file=sys.stderr)
        sys.exit(1)

    resolved = []
    for model_name, batch_size in zip(args.model_names, args.batch_sizes):
        model_dir = sanitize_timm_model_name(model_name)
        optim_frag = _optim_frag(args, batch_size)
        model_path = os.path.join(fp_root, model_dir)
        if not os.path.isdir(model_path):
            print(f"[ERROR] no evaluations for {model_name} at {model_path}", file=sys.stderr)
            sys.exit(1)
        datasets = [d for d in os.listdir(model_path)
                    if os.path.isdir(os.path.join(model_path, d))]
        if not any(os.path.isdir(os.path.join(model_path, d, optim_frag, f"seed={args.seed}"))
                   for d in datasets):
            print(f"[ERROR] {model_name}: no dataset has "
                  f"{optim_frag}/seed={args.seed}", file=sys.stderr)
            sys.exit(1)
        resolved.append((model_name, model_dir, optim_frag))
    return resolved


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------
def _load_value(path, key, misses):
    """Read `key` from a results JSON, recording a miss rather than swallowing it.

    A silently-absent file renders as a zero-height bar, which is
    indistinguishable from a genuine zero.  Every failure is therefore appended
    to `misses` so the caller can report -- and under --strict, refuse -- a plot
    built on incomplete data.
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

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 3.2 * n_rows),
                             sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, (model_name, data) in enumerate(all_model_data):
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
        display_title = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        ax.set_title(r"\textbf{" + display_title + "}", fontsize=14)

    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xticks(x_pos)
            ax.set_xticklabels(datasets, fontsize=10, rotation=45, ha="right")
    if n_models % n_cols != 0:
        axes[n_rows - 2][n_cols - 1].set_xticks(x_pos)
        axes[n_rows - 2][n_cols - 1].set_xticklabels(datasets, fontsize=10, rotation=45, ha="right")
        axes[n_rows - 2][n_cols - 1].tick_params(labelbottom=True)

    axes_flat[0].legend(fontsize=12, frameon=False, loc="upper center",
                        bbox_to_anchor=(1.0, 1.35), ncol=3)

    fig.tight_layout()
    fig.subplots_adjust(top=1 - 0.4 / (3.2 * n_rows))

    out_dir = os.path.join(
        "plots", "vision", "ilharco_timm_supervised", "999_paper_stuff",
        "000_baselines", "baseline_bar_all_models", f"seed={args.seed}",
        _ptq_frag(args),
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
    qf = _qat_frag(args)
    pf = _ptq_frag(args)

    models = resolve_models(args)

    print(f"Plotting {len(models)} models:")
    all_model_data = []
    misses = []
    for model_name, model_dir, of in models:
        display = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        print(f"  Loading {display} ({of.rsplit('_', 1)[-1]}) ...")
        before = len(misses)
        data = load_data(model_dir, of, args.seed, qf, pf, misses, target_epoch_mult=args.target_epoch_mult)
        model_misses = len(misses) - before
        expected = 3 * len(DATASET_NAME_TO_EPOCHS)
        if model_misses:
            print(f"    [WARN] {model_misses}/{expected} lookups missing", file=sys.stderr)
        if model_misses == expected:
            print(f"[ERROR] {display}: every lookup missed -- the constructed path "
                  f"grammar does not match this tree.", file=sys.stderr)
            sys.exit(1)
        all_model_data.append((model_name, data))

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
