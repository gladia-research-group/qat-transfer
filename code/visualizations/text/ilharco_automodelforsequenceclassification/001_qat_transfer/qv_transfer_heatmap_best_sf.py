"""001 — QV Transfer Best-Alpha Heatmap (difference vs FP+PTQ) — text

Loads QV-transfer results for all (target_dataset x qv_dataset) pairs and, for
each cell, picks the alpha that achieves the highest *val* accuracy across all
alpha values swept on disk.  Then loads the *test* result for that alpha.
Produces one heatmap per head variant (FP head / QAT head):

  heatmap_qv_transfer_<variant>_best_alpha_minus_fp_ptq.png
      Left-panel cell value  = test_acc_at_best_alpha[target, qv] - fp_ptq_acc[target]
      Right-panel cell value = test_accuracy of the corresponding baseline.

Cells where the best alpha differs from --qv-alpha (the fixed reference) are
annotated with a trailing '*' in the cell text.

  rows = target datasets  (y-axis)
  cols = qv datasets      (x-axis)  +  5 baseline columns appended at right
"""

import argparse
import glob
import json
import os
import re
import sys

from pathlib import Path

# ---------------------------------------------------------------------------
# Make `from src.text...` imports work when this script is run from the
# project root via `uv run python code/visualizations/.../qv_transfer_heatmap_best_sf.py`.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

os.chdir(_PROJECT_ROOT)

from src.duration import mult_path_frag, mult_tag, role_path_frag
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.text.data.common import DATASET_NAME_TO_EPOCHS
from src.vision.utils import sanitize_hf_model_name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT_BASELINES = "evaluations/text/ilharco_automodelforsequenceclassification/000_baselines/text"
EVAL_ROOT_QV        = "evaluations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/text/qv_transfer"

BASELINE_METHODS = ["pretrained", "fp", "fp_ptq", "random", "qat", "qat_ptq"]

BASELINE_METHOD_LABELS = {
    "pretrained": "Pretrained",
    "fp":         "FP",
    "fp_ptq":     "FP+PTQ",
    "random":     "Random",
    "qat":        "QAT",
    "qat_ptq":    "QAT+PTQ",
}

# Number of classes per dataset, used to compute the random-chance baseline
# (1 / num_classes).
DATASET_NAME_TO_NUM_CLASSES = {
    "Emotion":                      6,
    "IMDB":                         2,
    "Banking77":                   77,
    "AmazonReviewsClassification":  5,
    "AmazonCounterfactual":         2,
    "MassiveIntent":               60,
    "MassiveScenario":             18,
    "MTOPDomain":                  11,
    "MTOPIntent":                 113,
    "ToxicConversations":           2,
    "TweetSentimentExtraction":     3,
}

VAL_METRIC_KEYS = {
    "fp_head_ptq":  "val_accuracy_fp_head_ptq",
    "qat_head_ptq": "val_accuracy_qat_head_ptq",
}

TEST_METRIC_KEYS = {
    "fp_head_ptq":  "test_accuracy_fp_head_ptq",
    "qat_head_ptq": "test_accuracy_qat_head_ptq",
}

QV_METRIC_LABELS = {
    "fp_head_ptq":  "FP Head",
    "qat_head_ptq": "QAT Head",
}

TEST_ACC_KEY  = "test_accuracy"

HEATMAP_COLORSCALE_SEQUENTIAL = "Viridis"
HEATMAP_COLORSCALE_DIVERGING  = "RdYlGn"

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

    # optim path-fragment components
    parser.add_argument("--optim",          required=True, choices=["adamw", "sgd"])
    parser.add_argument("--lr",             required=True, type=float)
    parser.add_argument("--wd",             required=True, type=float)
    parser.add_argument("--ls",             required=True, type=float)
    parser.add_argument("--max-grad-norm",  required=True, type=float)
    parser.add_argument("--batch-size",     required=True, type=int)
    parser.add_argument("--max-length",     required=True, type=int)

    # quantization path-fragment components
    parser.add_argument("--qat-bits",       required=True, type=int)
    parser.add_argument("--ptq-bits",       required=True, type=int)
    parser.add_argument("--granularity",    required=True, choices=["tensor", "channel"])
    parser.add_argument("--skip-modules",   required=True, nargs="+",
                        help="One or more module names to skip during quantization "
                             "(no default: must be specified explicitly).")

    # fixed reference alpha (used only for the '*' marker on cells whose best
    # alpha differs from this value)
    parser.add_argument("--qv-alpha",       required=True, type=float,
                        help="Fixed reference alpha. Cells whose best alpha differs "
                             "from this value are annotated with a trailing '*'.")

    # Defaulted, unlike every other path-affecting argument, because the defaults
    # reproduce the selection protocol every existing result was produced under.
    # The lower bound matters: negative lambdas now exist on disk for bert-base
    # (998_rebuttal/003 sweeps the left arm of the sensitivity curve), and they
    # must not enter lambda* selection unless asked for explicitly.  Mirrors
    # pick_best_alpha.py, which selects the lambda* this figure must agree with.
    parser.add_argument("--min-alpha",      default=0.0, type=float,
                        help="lowest lambda considered for selection (default 0.0: "
                             "excludes the negative-lambda sensitivity sweep)")
    parser.add_argument("--max-alpha",      default=float("inf"), type=float,
                        help="highest lambda considered for selection (default: no bound)")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path-fragment helpers (must mirror what experiments write to disk)
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


# ---------------------------------------------------------------------------
# Per-baseline path builders
# ---------------------------------------------------------------------------
def _pretrained_path(model_dir, dataset, seed, *, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_BASELINES, "pretrained", model_dir, dataset,
        mult_path_frag(target_epoch_mult),
        f"seed={seed}", "eval_results.json",
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


def _qat_path(model_dir, dataset, seed, optim_frag, qat_frag, *, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_BASELINES, "qat", model_dir, dataset,
        optim_frag, mult_path_frag(target_epoch_mult), qat_frag, f"seed={seed}", "eval_results.json",
    )


def _qat_ptq_path(model_dir, dataset, seed, optim_frag, qat_frag, ptq_frag, *, target_epoch_mult):
    return os.path.join(
        EVAL_ROOT_BASELINES, "qat_ptq", model_dir, dataset,
        optim_frag, mult_path_frag(target_epoch_mult), qat_frag, ptq_frag, f"seed={seed}", "eval_results.json",
    )


def _qv_transfer_cell_prefix(model_dir, qv_dataset, target_dataset, seed,
                              optim_frag, qat_frag, ptq_frag, *, source_epoch_mult, target_epoch_mult):
    """Return the QV cell directory up to (but not including) the qv=alpha=* segment."""
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
# Data loading
# ---------------------------------------------------------------------------
def load_data(args):
    model_dir   = sanitize_hf_model_name(args.model_name)
    optim_frag  = _optim_frag(args.optim, args.lr, args.wd, args.ls,
                              args.max_grad_norm, args.batch_size, args.max_length)
    qat_frag    = _qat_frag(args.qat_bits, args.granularity, args.skip_modules)
    ptq_frag    = _ptq_frag(args.ptq_bits, args.granularity, args.skip_modules)

    datasets = sorted(DATASET_NAME_TO_EPOCHS.keys(), key=str.lower)

    data = {}
    for target_dataset in datasets:
        data[target_dataset] = {
            "pretrained": _load_value(
                _pretrained_path(model_dir, target_dataset, args.seed, target_epoch_mult=args.target_epoch_mult),
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
            "qat": _load_value(
                _qat_path(model_dir, target_dataset, args.seed, optim_frag, qat_frag, target_epoch_mult=args.target_epoch_mult),
                TEST_ACC_KEY,
            ),
            "qat_ptq": _load_value(
                _qat_ptq_path(model_dir, target_dataset, args.seed,
                              optim_frag, qat_frag, ptq_frag, target_epoch_mult=args.target_epoch_mult),
                TEST_ACC_KEY,
            ),
            "random": (
                1.0 / DATASET_NAME_TO_NUM_CLASSES[target_dataset]
                if target_dataset in DATASET_NAME_TO_NUM_CLASSES else None
            ),
            "qv_transfer": {},
        }

        for qv_dataset in datasets:
            cell_prefix = _qv_transfer_cell_prefix(
                model_dir, qv_dataset, target_dataset, args.seed,
                optim_frag, qat_frag, ptq_frag,
            
                source_epoch_mult=args.source_epoch_mult, target_epoch_mult=args.target_epoch_mult,
            )

            # Auto-discover all available alpha val results for this cell.
            val_pattern = os.path.join(cell_prefix, "qv=alpha=*", "split=val", "eval_results.json")
            val_files   = sorted(glob.glob(val_pattern))

            # Track best alpha on val independently per metric.
            best = {mt: {"val_acc": None, "alpha": None} for mt in VAL_METRIC_KEYS}

            for val_file in val_files:
                alpha_dir = os.path.basename(os.path.dirname(os.path.dirname(val_file)))
                m = re.match(r"^qv=alpha=(.+)$", alpha_dir)
                if m is None:
                    continue
                try:
                    alpha_val = float(m.group(1))
                except ValueError:
                    continue

                # Explicit grid restriction.  The default lower bound of 0.0 is
                # what keeps the negative-lambda sweep out of lambda* selection:
                # those runs exist to measure the left arm of the sensitivity
                # curve, not to widen the protocol the paper reports.
                if not (args.min_alpha <= alpha_val <= args.max_alpha):
                    continue

                for metric_tag, val_key in VAL_METRIC_KEYS.items():
                    acc = _load_value(val_file, val_key)
                    if acc is None:
                        continue
                    if best[metric_tag]["val_acc"] is None or acc > best[metric_tag]["val_acc"]:
                        best[metric_tag]["val_acc"] = acc
                        best[metric_tag]["alpha"]   = alpha_val

            if not val_files:
                print(f"  [NO VAL ALPHA FILES] {val_pattern}", file=sys.stderr)

            # Load test results for the val-selected best alpha (per metric).
            cell_data = {}
            for metric_tag in VAL_METRIC_KEYS:
                best_alpha = best[metric_tag]["alpha"]
                test_acc = None
                if best_alpha is not None:
                    test_path = os.path.join(
                        cell_prefix, f"qv=alpha={best_alpha}",
                        "split=test", "eval_results.json",
                    )
                    test_acc = _load_value(test_path, TEST_METRIC_KEYS[metric_tag])
                cell_data[metric_tag] = {
                    "best_alpha_acc": test_acc,
                    "best_alpha_val": best_alpha,
                }

            data[target_dataset]["qv_transfer"][qv_dataset] = cell_data

    return data, model_dir, optim_frag, qat_frag


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


# ---------------------------------------------------------------------------
# Plot: best-alpha minus FP+PTQ
# ---------------------------------------------------------------------------
def plot_best_alpha_minus_fp_ptq_heatmap(data, args, model_dir, optim_frag,
                                          qat_frag, metric_tag):
    datasets = sorted(data.keys(), key=str.lower)
    head_label = QV_METRIC_LABELS[metric_tag]

    display_datasets    = [DATASET_LABEL_RENAMES.get(ds, ds) for ds in datasets]
    qv_col_labels       = display_datasets
    baseline_col_labels = [BASELINE_METHOD_LABELS[m] for m in BASELINE_METHODS]

    qv_z, qv_text     = [], []
    base_z, base_text = [], []

    for target_dataset in datasets:
        qv_row_z, qv_row_text = [], []
        b_row_z,  b_row_text  = [], []

        fp_ptq_acc = data[target_dataset]["fp_ptq"]

        # Iterate the raw dataset names, not qv_col_labels: those are the
        # display labels produced by DATASET_LABEL_RENAMES, while `data` is
        # keyed by the raw names.  Any renamed dataset (AmazonCounterfactual ->
        # Counterfactual, and seven others here) raises KeyError otherwise.
        # qv_col_labels stays the axis labelling, which is all it is for.
        for qv_dataset in datasets:
            cell           = data[target_dataset]["qv_transfer"][qv_dataset][metric_tag]
            best_alpha_acc = cell["best_alpha_acc"]
            best_alpha_val = cell["best_alpha_val"]
            if best_alpha_acc is not None and fp_ptq_acc is not None:
                diff = best_alpha_acc - fp_ptq_acc
                star = "*" if (best_alpha_val is not None
                               and best_alpha_val != args.qv_alpha) else ""
                qv_row_z.append(diff)
                qv_row_text.append(f"{diff:.2f}{star}")
            else:
                qv_row_z.append(None)
                qv_row_text.append("")

        for method in BASELINE_METHODS:
            val = data[target_dataset][method]
            if val is not None:
                b_row_z.append(val)
                b_row_text.append(f"{val:.2f}")
            else:
                b_row_z.append(None)
                b_row_text.append("")

        qv_z.append(qv_row_z)
        qv_text.append(qv_row_text)
        base_z.append(b_row_z)
        base_text.append(b_row_text)

    qv_cmin, qv_cmax = _robust_symmetric_bounds(
        _finite_values(qv_z), center=0.0, min_span=0.02,
    )
    qv_colorbar_title = "Acc \u0394 (best \u03b1 \u2212 fp_ptq)"

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
            y=display_datasets,
            text=qv_text,
            texttemplate="%{text}",
            coloraxis="coloraxis",
            xgap=1, ygap=1,
            hovertemplate="target=%{y}<br>qv=%{x}<br>delta=%{z:.4f}<extra></extra>",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Heatmap(
            z=base_z,
            x=baseline_col_labels,
            y=display_datasets,
            text=base_text,
            texttemplate="%{text}",
            coloraxis="coloraxis2",
            xgap=1, ygap=1,
            hovertemplate="target=%{y}<br>baseline=%{x}<br>acc=%{z:.4f}<extra></extra>",
        ),
        row=1, col=2,
    )

    _add_diagonal_borders(fig, datasets, xref="x", yref="y")

    skip_str = ",".join(sorted(args.skip_modules))
    title = (
        f"QV Transfer ({head_label}, QAT+PTQ, Best Alpha) \u2212 FP+PTQ<br>"
        f"<sup>{args.model_name} | seed={args.seed} | "
        f"src_mult={mult_tag(args.source_epoch_mult)} | tgt_mult={mult_tag(args.target_epoch_mult)} | "
        f"optim={args.optim} | "
        f"qat_bits={args.qat_bits} | ptq_bits={args.ptq_bits} | granularity={args.granularity} | skip={skip_str} | "
        f"alpha=best</sup>"
    )

    fig.update_layout(
        title=title,
        coloraxis=dict(
            colorscale=HEATMAP_COLORSCALE_DIVERGING,
            cmin=qv_cmin,
            cmax=qv_cmax,
            cmid=0,
            colorbar=dict(title=qv_colorbar_title, x=1.01, y=0.78, len=0.42),
        ),
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
        title_text="Quantization Vector Dataset<br>(dataset the qv is computed from)",
        row=1, col=1, side="bottom",
    )
    fig.update_xaxes(title_text="Target Baselines", row=1, col=2, side="bottom")
    fig.update_yaxes(
        title_text="Target Dataset<br>(dataset the qv is applied to)",
        row=1, col=1, autorange="reversed",
    )
    fig.update_yaxes(row=1, col=2, showticklabels=False, autorange="reversed")

    out_dir = os.path.join(
        "plots", "text", "ilharco_automodelforsequenceclassification",
        "001_qat_transfer", "qv_transfer_heatmap",

        model_dir, f"seed={args.seed}", f"smult={mult_tag(args.source_epoch_mult)}", f"tmult={mult_tag(args.target_epoch_mult)}", optim_frag, qat_frag, _ptq_frag(args.ptq_bits, args.granularity, args.skip_modules),
        "qv=alpha=best",
        "split=test",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f"heatmap_qv_transfer_{metric_tag}_best_alpha_minus_fp_ptq.png",
    )
    fig.write_image(out_path, scale=300 / 96)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    data, model_dir, optim_frag, qat_frag = load_data(args)
    for metric_tag in VAL_METRIC_KEYS:
        plot_best_alpha_minus_fp_ptq_heatmap(
            data, args, model_dir, optim_frag, qat_frag, metric_tag,
        )


if __name__ == "__main__":
    main()
