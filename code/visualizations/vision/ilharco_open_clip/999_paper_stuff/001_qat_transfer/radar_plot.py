"""Radar plot: FT+PTQ vs QV Patching+PTQ vs QAT+PTQ

Three radar plots side-by-side (one per model scale: ViT-B, ViT-L, ViT-H).
Each radar has 22 axes (one per vision dataset) and three webs:

    1. FT+PTQ baseline          (000_baselines/fp_ptq)
    2. QV Patching+PTQ         (001_qat_transfer, best donor != target, best alpha)
    3. QAT+PTQ                  (000_baselines/qat_ptq)

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

from src.vision.data.common import DATASET_NAME_TO_EPOCHS
from src.vision.utils import sanitize_open_clip_model_name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT_BASELINES = "evaluations/vision/ilharco_open_clip/000_baselines/vision"
EVAL_ROOT_QV        = "evaluations/vision/ilharco_open_clip/001_qat_transfer/vision/qv_transfer"

DATASET_ORDER_SWAPS = [("DTD", "TinyImageNet"), ("RenderedSST2", "PCAM")]


def _swapped_dataset_order(datasets_dict):
    """Sorted dataset list with aesthetic swaps applied."""
    ds = sorted(datasets_dict.keys())
    for a, b in DATASET_ORDER_SWAPS:
        if a in ds and b in ds:
            ia, ib = ds.index(a), ds.index(b)
            ds[ia], ds[ib] = ds[ib], ds[ia]
    return ds


BEST_ALPHA_FILE = "best_alpha.json"
BEST_ALPHA_KEY  = "val_accuracy_patched_qat_ptq"
TEST_METRIC_KEY = "test_accuracy_patched_qat_ptq"
TEST_ACC_KEY    = "test_accuracy"

MODEL_DISPLAY_NAMES = {
    "ViT/B/16": "ViT-B/16",
    "ViT/L/14": "ViT-L/14",
    "ViT/H/14": "ViT-H/14",
    "ViT/B/32": "ViT-B/32",
    "ViT/L/16": "ViT-L/16",
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-names",     required=True, nargs=3,
                        help="Three open_clip model names, e.g. ViT-B-16 ViT-L-14 ViT-H-14")
    parser.add_argument("--pretrained-tags", required=True, nargs=3,
                        help="Three open_clip pretrained tags (parallel to --model-names)")
    parser.add_argument("--seed",            required=True, type=int)
    parser.add_argument("--source-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the DONOR checkpoints.")
    parser.add_argument("--target-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the RECEIVER checkpoints.")

    parser.add_argument("--optim",           required=True, choices=["adamw", "sgd"])
    parser.add_argument("--lr",              required=True, type=float)
    parser.add_argument("--wd",              required=True, type=float)
    parser.add_argument("--ls",              required=True, type=float)
    parser.add_argument("--wl",              required=True, type=int)
    parser.add_argument("--max-grad-norm",   required=True, type=float)
    parser.add_argument("--batch-sizes",     required=True, type=int, nargs=3,
                        help="Three batch sizes (one per model, parallel to --model-names)")

    parser.add_argument("--qat-bits",        required=True, type=int)
    parser.add_argument("--ptq-bits",        required=True, type=int)
    parser.add_argument("--granularity",     required=True, choices=["tensor", "channel"])
    parser.add_argument("--skip-modules",    required=True, nargs="+",
                        help="One or more module names to skip during quantization "
                             "(no default: must be specified explicitly).")

    parser.add_argument("--eval-split",      default="test", choices=["val", "test"],
                        help="Which split the qv_transfer results were evaluated on.")

    return parser.parse_args()


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
# Per-baseline path builders
# ---------------------------------------------------------------------------
def _fp_ptq_path(model_dir, dataset, seed, optim_frag, ptq_frag, *, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_BASELINES, "fp_ptq", model_dir, dataset,
        optim_frag, mult_path_frag(target_epoch_mult), ptq_frag, f"seed={seed}", "eval_results.json",
    )


def _qat_ptq_path(model_dir, dataset, seed, optim_frag, qat_frag, ptq_frag, *, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_BASELINES, "qat_ptq", model_dir, dataset,
        optim_frag, mult_path_frag(target_epoch_mult), qat_frag, ptq_frag, f"seed={seed}", "eval_results.json",
    )


# ---------------------------------------------------------------------------
# QV transfer path builders
# ---------------------------------------------------------------------------
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
# Data loading — one model
# ---------------------------------------------------------------------------
def load_radar_data(model_dir, args, optim_frag, qat_frag, ptq_frag):
    """Return {dataset: {"fp_ptq": float, "qat_ptq": float, "qat_transfer_ptq": float}}."""
    datasets = _swapped_dataset_order(DATASET_NAME_TO_EPOCHS)

    data = {}
    for target_dataset in datasets:
        fp_ptq_acc = _load_value(
            _fp_ptq_path(model_dir, target_dataset, args.seed, optim_frag, ptq_frag, target_epoch_mult=args.target_epoch_mult),
            TEST_ACC_KEY,
        )
        qat_ptq_acc = _load_value(
            _qat_ptq_path(model_dir, target_dataset, args.seed,
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
                model_dir, qv_dataset, target_dataset, args.seed,
                optim_frag, qat_frag, ptq_frag,
            
                source_epoch_mult=args.source_epoch_mult, target_epoch_mult=args.target_epoch_mult,
            )
            unit_path = os.path.join(
                cell_prefix, "qv=alpha=1.0",
                "split=test", "eval_results.json",
            )
            acc = _load_value(unit_path, TEST_METRIC_KEY)
            if acc is not None and (best_unit_acc is None or acc > best_unit_acc):
                best_unit_acc = acc
                best_unit_donor = qv_dataset

        # Step 2: get that donor's best-λ accuracy
        best_transfer_acc = None
        if best_unit_donor is not None:
            cell_prefix = _qv_transfer_cell_prefix(
                model_dir, best_unit_donor, target_dataset, args.seed,
                optim_frag, qat_frag, ptq_frag,
            
                source_epoch_mult=args.source_epoch_mult, target_epoch_mult=args.target_epoch_mult,
            )
            best_alpha_path = os.path.join(cell_prefix, BEST_ALPHA_FILE)
            if os.path.exists(best_alpha_path):
                with open(best_alpha_path) as f:
                    info = json.load(f).get(BEST_ALPHA_KEY)
                if info is not None:
                    best_alpha_val = info["alpha"]
                    test_path = os.path.join(
                        cell_prefix, f"qv=alpha={best_alpha_val}",
                        "split=test", "eval_results.json",
                    )
                    best_transfer_acc = _load_value(test_path, TEST_METRIC_KEY)

        data[target_dataset] = {
            "fp_ptq": fp_ptq_acc,
            "qat_ptq": qat_ptq_acc,
            "qat_transfer_ptq": best_transfer_acc,
        }

    return data


# ---------------------------------------------------------------------------
# Radar plot
# ---------------------------------------------------------------------------
def plot_radar(all_data, model_labels, args, optim_frag, qat_frag):
    datasets = _swapped_dataset_order(DATASET_NAME_TO_EPOCHS)
    n = len(datasets)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles.append(angles[0])

    web_keys = ["fp_ptq", "qat_transfer_ptq", "qat_ptq"]
    web_labels = [r"\textbf{FT$+$PTQ}", r"\textbf{QV Patching$+$PTQ}", r"\textbf{QAT$+$PTQ}"]
    web_colors = ["#e07a5f", "#81b29a", "#3d405b"]
    web_markers = ["o", "s", "^"]

    fig, axes = plt.subplots(1, 3, figsize=(27, 10), subplot_kw=dict(polar=True))

    for ax, data, model_label in zip(axes, all_data, model_labels):
        for key, label, color, marker in zip(web_keys, web_labels, web_colors, web_markers):
            values = []
            for ds in datasets:
                v = data[ds].get(key)
                values.append(v if v is not None else np.nan)
            values.append(values[0])
            ax.plot(angles, values, color=color, linewidth=1.5, marker=marker,
                    markersize=4, label=label)
            ax.fill(angles, values, color=color, alpha=0.08)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(datasets, fontsize=16)
        ax.tick_params(axis="x", pad=20)
        ax.set_ylim(top=1.0)
        display_title = MODEL_DISPLAY_NAMES.get(model_label, model_label)
        ax.set_title(r"\textbf{" + display_title + "}", fontsize=24, pad=35)
        ax.tick_params(axis="y", labelsize=14)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=20,
               frameon=False, bbox_to_anchor=(0.5, 0.02))

    fig.tight_layout(rect=[0, 0.02, 1, 1], w_pad=5)

    # -- export ---------------------------------------------------------------
    out_dir = os.path.join(
        "plots", "vision", "ilharco_open_clip", "999_paper_stuff", "001_qat_transfer",
        "radar_plot", f"seed={args.seed}", f"smult={mult_tag(args.source_epoch_mult)}", f"tmult={mult_tag(args.target_epoch_mult)}", optim_frag, qat_frag,
        _ptq_frag(args),
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "radar_plot.pdf")

    fig.savefig(out_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    qat_frag = _qat_frag(args)
    ptq_frag = _ptq_frag(args)

    all_data = []
    model_labels = []
    for model_name, pretrained, bs in zip(args.model_names, args.pretrained_tags,
                                          args.batch_sizes):
        model_dir = sanitize_open_clip_model_name(model_name, pretrained)
        optim_frag = _optim_frag(args, bs)
        display_name = model_name.replace("-", "/")
        print(f"Loading data for {display_name} ({pretrained}) ...")

        data = load_radar_data(model_dir, args, optim_frag, qat_frag, ptq_frag)
        all_data.append(data)
        model_labels.append(display_name)

    plot_radar(all_data, model_labels, args, _optim_frag(args, args.batch_sizes[0]), qat_frag)


if __name__ == "__main__":
    main()
