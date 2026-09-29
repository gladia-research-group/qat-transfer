"""008 — PV-QV Transfer Heatmap (difference vs FP+PTQ) — timm supervised

The 008 counterpart of 001's heatmap. Loads PV-QV transfer results and the
baselines (pretrained, fp, fp_ptq, pv, pv_ptq) for all
(target_dataset x qv_dataset) pairs and produces heatmaps for both head
variants (FP head / PV head) where:

  rows  = target datasets  (y-axis)
  cols  = qv datasets      (x-axis)  +  baseline columns appended at right

Left-panel cell value =
    {split}_accuracy_<head_variant>_ptq[target,qv] - test_accuracy[fp_ptq,target]
Right-panel cell value =
    test_accuracy of the corresponding baseline (raw, in [0, 1]).

Two columns behave differently from their QAT analogues, both because
finetune_pv.py settles the model onto the quantization grid before saving:
`pv` and `pv_ptq` should carry identical numbers, and the transfer diagonal at
alpha=1 should reproduce `pv_ptq` exactly. A visible discrepancy in either is a
signal that some checkpoint was not written by the settle path, not a signal
about transfer.
"""

import argparse
import datetime
import json
import os
import sys

from pathlib import Path

# ---------------------------------------------------------------------------
# Make `from src.vision...` imports work when this script is run from the
# project root via `uv run python code/visualizations/.../qv_transfer_heatmap.py`.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

os.chdir(_PROJECT_ROOT)

from src.duration import mult_path_frag, mult_tag, role_path_frag
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.pv_tuning import pv_path_frag
from src.vision.data.common import DATASET_NAME_TO_EPOCHS
from src.vision.utils import sanitize_timm_model_name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT_BASELINES = "evaluations/vision/ilharco_timm_supervised/000_baselines/vision"
EVAL_ROOT_QV        = "evaluations/vision/ilharco_timm_supervised/008_pv_transfer/vision/qv_transfer_pv"

BASELINE_METHODS = ["pretrained", "pretrained_ptq", "fp", "fp_ptq", "random", "pv", "pv_ptq"]

BASELINE_METHOD_LABELS = {
    "pretrained":     "Pretrained",
    "pretrained_ptq": "Pretrained+PTQ",
    "fp":             "FP",
    "fp_ptq":         "FP+PTQ",
    "random":         "Random",
    "pv":             "PV",
    "pv_ptq":         "PV+PTQ",
}

# Number of classes per dataset, used to compute the random-chance baseline
# (1 / num_classes). Mirrors the class_names sizes in code/src/vision/data/*.py.
DATASET_NAME_TO_NUM_CLASSES = {
    "Cars":          196,
    "DTD":           47,
    "EuroSAT":       10,
    "GTSRB":         43,
    "MNIST":         10,
    "RESISC45":      45,
    "SUN397":        397,
    "SVHN":          10,
    "CIFAR10":       10,
    "CIFAR100":      100,
    "STL10":         10,
    "Food101":       101,
    "Flowers102":    102,
    "FER2013":       7,
    "PCAM":          2,
    "OxfordIIITPet": 37,
    "RenderedSST2":  2,
    "EMNIST":        26,
    "FashionMNIST":  10,
    "KMNIST":        10,
    "TinyImageNet":  200,
    "ImageNet":      1000,
}


def _qv_metric_keys(eval_split):
    return {
        "fp_head_ptq": f"{eval_split}_accuracy_fp_head_ptq",
        "pv_head_ptq": f"{eval_split}_accuracy_pv_head_ptq",
    }


QV_METRIC_LABELS = {
    "fp_head_ptq": "FP Head",
    "pv_head_ptq": "PV Head",
}

TEST_ACC_KEY = "test_accuracy"

HEATMAP_COLORSCALE_SEQUENTIAL = "Viridis"
HEATMAP_COLORSCALE_DIVERGING  = "RdYlGn"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name",     required=True,
                        help="timm model name, e.g. vit_base_patch16_224")
    parser.add_argument("--seed",           required=True, type=int)
    parser.add_argument("--source-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the DONOR checkpoints.")
    parser.add_argument("--target-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the RECEIVER checkpoints.")

    # optim path-fragment components
    parser.add_argument("--optim",          required=True, choices=["adamw", "sgd"])
    parser.add_argument("--lr",             required=True, type=float)
    parser.add_argument("--wd",             required=True, type=float)
    parser.add_argument("--ls",             required=True, type=float)
    parser.add_argument("--wl",             required=True, type=int)
    parser.add_argument("--max-grad-norm",  required=True, type=float)
    parser.add_argument("--batch-size",     required=True, type=int)

    # quantization path-fragment components
    parser.add_argument("--pv-bits",        required=True, type=int)
    parser.add_argument("--ptq-bits",       required=True, type=int)
    parser.add_argument("--granularity",    required=True, choices=["tensor", "channel"])
    parser.add_argument("--skip-modules",   required=True, nargs="+",
                        help="One or more module names to skip during quantization "
                             "(no default: must be specified explicitly).")

    # PV path-fragment components
    parser.add_argument("--pv-delta",       required=True, type=float)
    parser.add_argument("--pv-tau",         required=True, type=float)
    parser.add_argument("--pv-trust",       required=True,
                        help='PV trust_ratio, or "none" if it was disabled')
    parser.add_argument("--pv-pevery",      required=True, type=int)
    parser.add_argument("--pv-temp",        required=True, type=float)

    # qv path-fragment component
    parser.add_argument("--qv-alpha",       required=True, type=float)

    # eval split
    parser.add_argument("--eval-split",     default="test", choices=["val", "test"],
                        help="Which split the experiment was evaluated on (default: test).")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path-fragment helpers (must mirror what config/experiments/* write to disk)
# ---------------------------------------------------------------------------
def _skip_tag(skip_modules):
    return "-".join(sorted(skip_modules)) if len(skip_modules) > 0 else "none"


def _optim_frag(optim, lr, wd, ls, wl, mgn, bs):
    # NOTE: config/experiments/* hardcode "optim=adamw" in the path even though they
    # accept other optimizers in cfg. We mirror that exactly.
    del optim
    return f"optim=adamw_lr={lr}_wd={wd}_ls={ls}_wl={wl}_mgn={mgn}_bs={bs}"


def _pv_frag(args):
    # Built by the same library helper finetune_pv.py, the PV baselines and
    # qv_transfer_pv.py use, so the fragment cannot drift between writer and
    # reader.
    trust = None if args.pv_trust.lower() == "none" else float(args.pv_trust)
    return pv_path_frag(
        bits=args.pv_bits,
        granularity=args.granularity,
        skip_modules=args.skip_modules,
        delta_decay=args.pv_delta,
        max_code_change_per_step=args.pv_tau,
        trust_ratio=trust,
        p_every=args.pv_pevery,
        temperature=args.pv_temp,
    )


def _ptq_frag(bits, gran, skip_modules):
    return f"ptq=bits={bits}_gran={gran}_skip={_skip_tag(skip_modules)}"


def _qv_frag(alpha):
    return f"qv=alpha={alpha}"


# ---------------------------------------------------------------------------
# Per-baseline path builders
# ---------------------------------------------------------------------------
def _pretrained_path(model_dir, dataset, seed, *, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_BASELINES, "pretrained", model_dir, dataset,
        mult_path_frag(target_epoch_mult),
        f"seed={seed}", "eval_results.json",
    )


def _pretrained_ptq_path(model_dir, dataset, seed, ptq_frag, *, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_BASELINES, "pretrained_ptq", model_dir, dataset,
        mult_path_frag(target_epoch_mult),
        ptq_frag, f"seed={seed}", "eval_results.json",
    )


def _fp_path(model_dir, dataset, seed, optim_frag, *, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_BASELINES, "fp", model_dir, dataset,
        optim_frag, mult_path_frag(target_epoch_mult), f"seed={seed}", "eval_results.json",
    )


def _fp_ptq_path(model_dir, dataset, seed, optim_frag, ptq_frag, *, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_BASELINES, "fp_ptq", model_dir, dataset,
        optim_frag, mult_path_frag(target_epoch_mult), ptq_frag, f"seed={seed}", "eval_results.json",
    )


def _pv_path(model_dir, dataset, seed, optim_frag, pv_frag, *, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_BASELINES, "pv", model_dir, dataset,
        optim_frag, mult_path_frag(target_epoch_mult), pv_frag, f"seed={seed}", "eval_results.json",
    )


def _pv_ptq_path(model_dir, dataset, seed, optim_frag, pv_frag, ptq_frag, *, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_BASELINES, "pv_ptq", model_dir, dataset,
        optim_frag, mult_path_frag(target_epoch_mult), pv_frag, ptq_frag, f"seed={seed}", "eval_results.json",
    )


def _qv_transfer_path(model_dir, qv_dataset, target_dataset, seed,
                      optim_frag, pv_frag, ptq_frag, qv_frag, eval_split, *, source_epoch_mult, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_QV, model_dir,
        role_path_frag("src", qv_dataset, seed, source_epoch_mult),
        role_path_frag("tgt", target_dataset, seed, target_epoch_mult),
        optim_frag, pv_frag, ptq_frag, qv_frag,
        f"split={eval_split}",
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
def load_data(args):
    model_dir  = sanitize_timm_model_name(args.model_name)
    optim_frag = _optim_frag(args.optim, args.lr, args.wd, args.ls, args.wl,
                             args.max_grad_norm, args.batch_size)
    pv_frag    = _pv_frag(args)
    ptq_frag   = _ptq_frag(args.ptq_bits, args.granularity, args.skip_modules)
    qv_frag    = _qv_frag(args.qv_alpha)
    eval_split = args.eval_split
    qv_metric_keys = _qv_metric_keys(eval_split)

    datasets = sorted(DATASET_NAME_TO_EPOCHS.keys(), key=str.lower)

    data = {}
    for target_dataset in datasets:
        data[target_dataset] = {
            "pretrained": _load_value(
                _pretrained_path(model_dir, target_dataset, args.seed, target_epoch_mult=args.target_epoch_mult),
                TEST_ACC_KEY,
            ),
            "pretrained_ptq": _load_value(
                _pretrained_ptq_path(model_dir, target_dataset, args.seed, ptq_frag, target_epoch_mult=args.target_epoch_mult),
                TEST_ACC_KEY,
            ),
            "fp": _load_value(
                _fp_path(model_dir, target_dataset, args.seed, optim_frag, target_epoch_mult=args.target_epoch_mult),
                TEST_ACC_KEY,
            ),
            "fp_ptq": _load_value(
                _fp_ptq_path(model_dir, target_dataset, args.seed, optim_frag, ptq_frag, target_epoch_mult=args.target_epoch_mult),
                TEST_ACC_KEY,
            ),
            "pv": _load_value(
                _pv_path(model_dir, target_dataset, args.seed, optim_frag, pv_frag, target_epoch_mult=args.target_epoch_mult),
                TEST_ACC_KEY,
            ),
            "pv_ptq": _load_value(
                _pv_ptq_path(model_dir, target_dataset, args.seed,
                             optim_frag, pv_frag, ptq_frag, target_epoch_mult=args.target_epoch_mult),
                TEST_ACC_KEY,
            ),
            "random": (
                1.0 / DATASET_NAME_TO_NUM_CLASSES[target_dataset]
                if target_dataset in DATASET_NAME_TO_NUM_CLASSES else None
            ),
            "qv_transfer": {},
        }

        for qv_dataset in datasets:
            qv_path = _qv_transfer_path(
                model_dir, qv_dataset, target_dataset, args.seed,
                optim_frag, pv_frag, ptq_frag, qv_frag, eval_split,
            
                source_epoch_mult=args.source_epoch_mult, target_epoch_mult=args.target_epoch_mult,
            )
            data[target_dataset]["qv_transfer"][qv_dataset] = {
                metric_tag: _load_value(qv_path, metric_key)
                for metric_tag, metric_key in qv_metric_keys.items()
            }

    return data, model_dir, optim_frag, pv_frag, qv_frag


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _add_diagonal_borders(fig, datasets, color="black", width=2, xref="x", yref="y"):
    """Add a rectangular border around each diagonal cell (row i, col i)."""
    for i in range(len(datasets)):
        fig.add_shape(
            type="rect",
            xref=xref, yref=yref,
            x0=i - 0.5, x1=i + 0.5,
            y0=i - 0.5, y1=i + 0.5,
            line=dict(color=color, width=width),
            fillcolor="rgba(0,0,0,0)",
        )


def _finite_values(matrix):
    return [v for row in matrix for v in row if v is not None]


def _quantile(sorted_values, q):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _robust_symmetric_bounds(values, center, min_span=0.05, q_low=0.05, q_high=0.95):
    if not values:
        return center - min_span, center + min_span
    svals = sorted(values)
    ql = _quantile(svals, q_low)
    qh = _quantile(svals, q_high)
    span = max(abs(center - ql), abs(qh - center), min_span)
    return center - span, center + span


def _out_dir(args, model_dir, optim_frag, pv_frag, qv_frag):
    return os.path.join(
        "plots", "vision", "ilharco_timm_supervised", "008_pv_transfer",
        "qv_transfer_heatmap",
        model_dir, f"seed={args.seed}", f"smult={mult_tag(args.source_epoch_mult)}", f"tmult={mult_tag(args.target_epoch_mult)}", optim_frag, pv_frag,
        _ptq_frag(args.ptq_bits, args.granularity, args.skip_modules),
        qv_frag,
        f"split={args.eval_split}",
    )


def _subtitle(args):
    # NOTE: keep this short. Kaleido derives a scratch filename from the title
    # text even when plotly hands it an explicit output path, so a title over
    # ~255 characters fails with ENAMETOOLONG regardless of where the figure is
    # written.
    skip_str = ",".join(sorted(args.skip_modules))
    return (
        f"<sup>{args.model_name} | seed={args.seed} | "
        f"pv/ptq bits={args.pv_bits}/{args.ptq_bits} | "
        f"gran={args.granularity} | skip={skip_str} | "
        f"delta={args.pv_delta} | tau={args.pv_tau} | trust={args.pv_trust} | "
        f"alpha={args.qv_alpha}</sup>"
    )


def _baseline_columns(data, datasets):
    base_z, base_text = [], []
    for target_dataset in datasets:
        row_z, row_text = [], []
        for method in BASELINE_METHODS:
            val = data[target_dataset][method]
            if val is not None:
                row_z.append(val)
                row_text.append(f"{val:.2f}")
            else:
                row_z.append(None)
                row_text.append("")
        base_z.append(row_z)
        base_text.append(row_text)
    return base_z, base_text


def _coverage_note(qv_z):
    """Describe how much of the transfer grid is actually populated.

    A blank cell and a zero-valued cell look similar, so a partially-filled
    heatmap read as a finished one is actively misleading. This states the
    fill fraction on the figure itself rather than relying on whoever looks at
    it remembering that the sweep was still running.
    """
    total = sum(len(row) for row in qv_z)
    filled = len(_finite_values(qv_z))
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    status = "COMPLETE" if filled == total else "PARTIAL — sweep still running"
    return (
        f"{status}: {filled}/{total} transfer cells populated "
        f"({100.0 * filled / total:.0f}%) · generated {stamp}"
    )


def _render(fig, datasets, qv_col_labels, baseline_col_labels, title,
            qv_coloraxis, out_path, coverage=None):
    _add_diagonal_borders(fig, datasets, xref="x", yref="y")
    if coverage:
        fig.add_annotation(
            text=coverage,
            xref="paper", yref="paper", x=0.0, y=-0.13,
            showarrow=False, align="left",
            font=dict(size=11, color="#b00000" if "PARTIAL" in coverage else "#555555"),
        )
    fig.update_layout(
        title=title,
        coloraxis=qv_coloraxis,
        coloraxis2=dict(
            colorscale=HEATMAP_COLORSCALE_SEQUENTIAL,
            cmin=0, cmax=1,
            colorbar=dict(title="Baseline Acc", x=1.01, y=0.22, len=0.42),
        ),
        template="plotly_white",
        height=max(400, 60 * len(datasets) + 180),
        width=max(900, 55 * len(qv_col_labels) + 100 * len(baseline_col_labels) + 260),
        margin=dict(l=80, r=220, t=120, b=90),
    )
    fig.update_xaxes(
        title_text="Quantization Vector Dataset<br>(dataset the PV qv is computed from)",
        row=1, col=1, side="bottom",
    )
    fig.update_xaxes(title_text="Target Baselines", row=1, col=2, side="bottom")
    fig.update_yaxes(
        title_text="Target Dataset<br>(dataset the qv is applied to)",
        row=1, col=1, autorange="reversed",
    )
    fig.update_yaxes(row=1, col=2, showticklabels=False, autorange="reversed")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_image(out_path, scale=300 / 96)
    print(f"Saved: {out_path}")


def _build_figure(qv_z, qv_text, base_z, base_text, datasets,
                  qv_col_labels, baseline_col_labels, qv_hover_label):
    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.06,
        column_widths=[max(1, len(qv_col_labels)), len(baseline_col_labels)],
    )
    fig.add_trace(
        go.Heatmap(
            z=qv_z,
            x=qv_col_labels,
            y=datasets,
            text=qv_text,
            texttemplate="%{text}",
            coloraxis="coloraxis",
            xgap=1, ygap=1,
            hovertemplate=(
                "target=%{y}<br>qv=%{x}<br>" + qv_hover_label + "=%{z:.4f}<extra></extra>"
            ),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=base_z,
            x=baseline_col_labels,
            y=datasets,
            text=base_text,
            texttemplate="%{text}",
            coloraxis="coloraxis2",
            xgap=1, ygap=1,
            hovertemplate="target=%{y}<br>baseline=%{x}<br>acc=%{z:.4f}<extra></extra>",
        ),
        row=1, col=2,
    )
    return fig


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_heatmap(data, args, model_dir, optim_frag, pv_frag, qv_frag, metric_tag):
    """Raw PV-QV transfer+PTQ accuracy heatmap (sequential Viridis, cmin=0, cmax=1)."""
    datasets = sorted(data.keys(), key=str.lower)
    head_label = QV_METRIC_LABELS[metric_tag]

    qv_col_labels = datasets
    baseline_col_labels = [BASELINE_METHOD_LABELS[m] for m in BASELINE_METHODS]

    qv_z, qv_text = [], []
    for target_dataset in datasets:
        row_z, row_text = [], []
        for qv_dataset in qv_col_labels:
            qv_val = data[target_dataset]["qv_transfer"][qv_dataset][metric_tag]
            if qv_val is not None:
                row_z.append(qv_val)
                row_text.append(f"{qv_val:.2f}")
            else:
                row_z.append(None)
                row_text.append("")
        qv_z.append(row_z)
        qv_text.append(row_text)

    base_z, base_text = _baseline_columns(data, datasets)

    fig = _build_figure(qv_z, qv_text, base_z, base_text, datasets,
                        qv_col_labels, baseline_col_labels, "acc")

    title = f"PV-QV Transfer+PTQ ({head_label})<br>{_subtitle(args)}"
    qv_coloraxis = dict(
        colorscale=HEATMAP_COLORSCALE_SEQUENTIAL,
        cmin=0, cmax=1,
        colorbar=dict(title="Transfer Acc", x=1.01, y=0.78, len=0.42),
    )
    out_path = os.path.join(
        _out_dir(args, model_dir, optim_frag, pv_frag, qv_frag),
        f"heatmap_qv_transfer_{metric_tag}.png",
    )
    _render(fig, datasets, qv_col_labels, baseline_col_labels, title,
            qv_coloraxis, out_path, coverage=_coverage_note(qv_z))


def plot_difference_heatmap(data, args, model_dir, optim_frag, pv_frag, qv_frag,
                            metric_tag, subtractor="fp_ptq"):
    datasets = sorted(data.keys(), key=str.lower)
    head_label = QV_METRIC_LABELS[metric_tag]

    qv_col_labels = datasets
    baseline_col_labels = [BASELINE_METHOD_LABELS[m] for m in BASELINE_METHODS]

    qv_z, qv_text = [], []
    for target_dataset in datasets:
        row_z, row_text = [], []
        sub_val = data[target_dataset][subtractor]
        for qv_dataset in qv_col_labels:
            qv_val = data[target_dataset]["qv_transfer"][qv_dataset][metric_tag]
            if qv_val is not None and sub_val is not None:
                diff = qv_val - sub_val
                row_z.append(diff)
                row_text.append(f"{diff:.2f}")
            else:
                row_z.append(None)
                row_text.append("")
        qv_z.append(row_z)
        qv_text.append(row_text)

    base_z, base_text = _baseline_columns(data, datasets)

    fig = _build_figure(qv_z, qv_text, base_z, base_text, datasets,
                        qv_col_labels, baseline_col_labels, "delta")

    qv_cmin, qv_cmax = _robust_symmetric_bounds(
        _finite_values(qv_z), center=0.0, min_span=0.02,
    )
    title = (
        f"PV-QV Transfer ({head_label}, PV+PTQ) − "
        f"{BASELINE_METHOD_LABELS[subtractor]}<br>{_subtitle(args)}"
    )
    qv_coloraxis = dict(
        colorscale=HEATMAP_COLORSCALE_DIVERGING,
        cmin=qv_cmin,
        cmax=qv_cmax,
        cmid=0,
        colorbar=dict(
            title=f"Acc Δ (vs {BASELINE_METHOD_LABELS[subtractor]})",
            x=1.01, y=0.78, len=0.42,
        ),
    )
    out_path = os.path.join(
        _out_dir(args, model_dir, optim_frag, pv_frag, qv_frag),
        f"heatmap_qv_transfer_{metric_tag}_minus_{subtractor}.png",
    )
    _render(fig, datasets, qv_col_labels, baseline_col_labels, title,
            qv_coloraxis, out_path, coverage=_coverage_note(qv_z))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    data, model_dir, optim_frag, pv_frag, qv_frag = load_data(args)
    common = (data, args, model_dir, optim_frag, pv_frag, qv_frag)
    for metric_tag in _qv_metric_keys(args.eval_split):
        plot_heatmap(*common, metric_tag=metric_tag)
        plot_difference_heatmap(*common, metric_tag=metric_tag, subtractor="fp_ptq")


if __name__ == "__main__":
    main()
