"""001 — QV Transfer Best-Alpha Heatmap (difference vs FP+PTQ) — timm supervised

Loads QV-transfer results for all (target_dataset x qv_dataset) pairs and, for
each cell, picks the alpha that achieves the highest *val* accuracy across all
alpha values swept on disk.  Then loads the *test* result for that alpha.
Produces one heatmap per head variant (FP head / QAT head):

  heatmap_qv_transfer_<variant>_best_alpha_minus_fp_ptq.png
  heatmap_qv_transfer_<variant>_best_alpha_minus_awq.png   (with --awq-*)
  heatmap_qv_transfer_<variant>_best_alpha_minus_gptq.png  (with --gptq-*)
      Left-panel cell value  = test_acc_at_best_alpha[target, qv] - <subtractor>[target]
      Right-panel cell value = test_accuracy of the corresponding baseline.

The `awq` and `gptq` subtractors read the 000_baselines `fp_awq` / `fp_gptq`
trees and answer the rebuttal's Task-1 question at lambda* rather than at a
fixed lambda: is QV+RTN competitive with strong PTQ? Their arguments are
conditionally required, all-or-none per competitor; passing none of a
competitor's flags skips it, so omitting both reproduces this script's previous
output byte-for-byte at the same paths. See qv_transfer_heatmap.py for the full
rationale, including why the baselines come from 000_baselines rather than from
a transfer sweep's alpha=0 self-pair.

Cells where the best alpha differs from --qv-alpha (the fixed reference) are
annotated with a trailing '*' in the cell text.

  rows = target datasets  (y-axis)
  cols = qv datasets      (x-axis)  +  5 baseline columns appended at right
"""

import argparse
import json
import os
import sys

from pathlib import Path

# ---------------------------------------------------------------------------
# Make `from src.vision...` imports work when this script is run from the
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

from src.awq import awq_path_frag
from src.gptq import gptq_path_frag
from src.vision.data.common import DATASET_NAME_TO_EPOCHS
from src.vision.utils import sanitize_timm_model_name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT_BASELINES = "evaluations/vision/ilharco_timm_supervised/000_baselines/vision"
EVAL_ROOT_QV        = "evaluations/vision/ilharco_timm_supervised/001_qat_transfer/vision/qv_transfer"

BASELINE_METHODS = ["pretrained", "pretrained_ptq", "fp", "fp_ptq", "random", "qat", "qat_ptq"]

BASELINE_METHOD_LABELS = {
    "pretrained":     "Pretrained",
    "pretrained_ptq": "Pretrained+PTQ",
    "fp":             "FP",
    "fp_ptq":     "FP+PTQ",
    "random":     "Random",
    "qat":        "QAT",
    "qat_ptq":    "QAT+PTQ",
}

# Competitor PTQ baselines, mirroring qv_transfer_heatmap.py. Each is opt-in and
# therefore NOT in BASELINE_METHODS: a competitor joins the right panel only on
# the figure that subtracts it, so the pre-existing fp_ptq figure keeps
# identical content at an identical path.
COMPETITORS = {
    "awq": {
        "label":  "AWQ(FP)",
        "subdir": "fp_awq",
        "args":   ("awq_bits", "awq_ncal", "awq_ngrid", "awq_clip"),
        "frag":   lambda a: awq_path_frag(
            bits=a.awq_bits, granularity=a.granularity, skip_modules=a.skip_modules,
            num_calib_batches=a.awq_ncal, n_grid=a.awq_ngrid, clip=a.awq_clip,
        ),
        "title":  lambda a: (f" | awq_bits={a.awq_bits},ncal={a.awq_ncal},"
                             f"ngrid={a.awq_ngrid},clip={a.awq_clip}"),
    },
    "gptq": {
        "label":  "GPTQ(FP)",
        "subdir": "fp_gptq",
        "args":   ("gptq_bits", "gptq_ncal", "gptq_percdamp", "gptq_actorder"),
        "frag":   lambda a: gptq_path_frag(
            bits=a.gptq_bits, granularity=a.granularity, skip_modules=a.skip_modules,
            num_calib_batches=a.gptq_ncal, percdamp=a.gptq_percdamp,
            actorder=a.gptq_actorder,
        ),
        "title":  lambda a: (f" | gptq_bits={a.gptq_bits},ncal={a.gptq_ncal},"
                             f"percdamp={a.gptq_percdamp},actorder={a.gptq_actorder}"),
    },
}

BASELINE_METHOD_LABELS.update({k: v["label"] for k, v in COMPETITORS.items()})

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

BEST_ALPHA_FILES = {
    "fp_head_ptq":  "best_alpha_fp_head_ptq.json",
    "qat_head_ptq": "best_alpha_qat_head_ptq.json",
}

BEST_ALPHA_KEYS = {
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

    # AWQ baseline (000_baselines/fp_awq) — conditionally required: all or none.
    # Granularity and skip_modules deliberately reuse --granularity /
    # --skip-modules: the AWQ baseline must quantize the same layer set at the
    # same granularity as the RTN column or the comparison is not like-for-like.
    parser.add_argument("--awq-bits",        type=int)
    parser.add_argument("--awq-ncal",        type=int)
    parser.add_argument("--awq-ngrid",       type=int)
    parser.add_argument("--awq-clip",        choices=["True", "False"],
                        help="Rendered verbatim in the path fragment (Python bool str).")
    parser.add_argument("--gptq-bits",       type=int)
    parser.add_argument("--gptq-ncal",       type=int)
    parser.add_argument("--gptq-percdamp",   type=float)
    parser.add_argument("--gptq-actorder",   choices=["True", "False"],
                        help="Rendered verbatim in the path fragment (Python bool str).")

    args = parser.parse_args()

    _resolve_competitors(parser, args)

    return args


def _resolve_competitors(parser, args):
    """
    Decide which competitor baselines are enabled and precompute their path
    fragments, storing the result on *args*.

    All-or-none per competitor: a partially specified competitor is an error
    rather than a silent fallback, because its flags jointly name one sweep on
    disk and a missing one cannot be guessed.
    """
    args.enabled_competitors = []
    args.competitor_frags = {}
    for name, spec in COMPETITORS.items():
        dests = spec["args"]
        given = [d for d in dests if getattr(args, d) is not None]
        if given and len(given) != len(dests):
            missing = [d for d in dests if getattr(args, d) is None]
            parser.error(
                f"--{name}-* arguments are all-or-none (they name one sweep on "
                f"disk); got {sorted(given)} but missing {sorted(missing)}"
            )
        if given:
            args.enabled_competitors.append(name)
            args.competitor_frags[name] = spec["frag"](args)


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


def _competitor_path(subdir, model_dir, dataset, seed, optim_frag, frag, *, target_epoch_mult):
    """A 000_baselines competitor cell: the `ptq=` slot holds the method's own
    fragment (`awq=` / `gptq=`), since the method replaces RTN rather than
    following it."""
    return os.path.join(
        EVAL_ROOT_BASELINES, subdir, model_dir, dataset,
        optim_frag, mult_path_frag(target_epoch_mult), frag, f"seed={seed}", "eval_results.json",
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
    model_dir   = sanitize_timm_model_name(args.model_name)
    optim_frag  = _optim_frag(args.optim, args.lr, args.wd, args.ls, args.wl,
                              args.max_grad_norm, args.batch_size)
    qat_frag    = _qat_frag(args.qat_bits, args.granularity, args.skip_modules)
    ptq_frag    = _ptq_frag(args.ptq_bits, args.granularity, args.skip_modules)

    datasets = sorted(DATASET_NAME_TO_EPOCHS.keys())

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

        for name in args.enabled_competitors:
            data[target_dataset][name] = _load_value(
                _competitor_path(
                    COMPETITORS[name]["subdir"], model_dir, target_dataset,
                    args.seed, optim_frag, args.competitor_frags[name],
                target_epoch_mult=args.target_epoch_mult),
                TEST_ACC_KEY,
            )

        for qv_dataset in datasets:
            cell_prefix = _qv_transfer_cell_prefix(
                model_dir, qv_dataset, target_dataset, args.seed,
                optim_frag, qat_frag, ptq_frag,
            
                source_epoch_mult=args.source_epoch_mult, target_epoch_mult=args.target_epoch_mult,
            )

            cell_data = {}
            for metric_tag in TEST_METRIC_KEYS:
                best_alpha = None
                best_alpha_path = os.path.join(cell_prefix, BEST_ALPHA_FILES[metric_tag])
                if os.path.exists(best_alpha_path):
                    with open(best_alpha_path) as f:
                        info = json.load(f).get(BEST_ALPHA_KEYS[metric_tag])
                        if info is not None:
                            best_alpha = info["alpha"]

                test_acc = None
                if best_alpha is not None:
                    test_path = os.path.join(
                        cell_prefix, f"qv=alpha={best_alpha}",
                        "split=test", "eval_results.json",
                    )
                    test_acc = _load_value(test_path, TEST_METRIC_KEYS[metric_tag])
                else:
                    print(f"  [NO BEST ALPHA] {best_alpha_path}", file=sys.stderr)

                cell_data[metric_tag] = {
                    "best_alpha_acc": test_acc,
                    "best_alpha_val": best_alpha,
                }

            data[target_dataset]["qv_transfer"][qv_dataset] = cell_data

    return data, model_dir, optim_frag, qat_frag


# ---------------------------------------------------------------------------
# Plot helpers (ported verbatim from qv_transfer_heatmap.py)
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
def plot_best_alpha_minus_baseline_heatmap(data, args, model_dir, optim_frag,
                                           qat_frag, metric_tag,
                                           subtractor="fp_ptq"):
    datasets = sorted(data.keys())
    head_label = QV_METRIC_LABELS[metric_tag]

    qv_col_labels       = datasets
    baseline_methods    = BASELINE_METHODS + ([subtractor] if subtractor in COMPETITORS else [])
    baseline_col_labels = [BASELINE_METHOD_LABELS[m] for m in baseline_methods]

    qv_z, qv_text     = [], []
    base_z, base_text = [], []

    for target_dataset in datasets:
        qv_row_z, qv_row_text = [], []
        b_row_z,  b_row_text  = [], []

        sub_acc = data[target_dataset][subtractor]

        for qv_dataset in qv_col_labels:
            cell           = data[target_dataset]["qv_transfer"][qv_dataset][metric_tag]
            best_alpha_acc = cell["best_alpha_acc"]
            best_alpha_val = cell["best_alpha_val"]
            if best_alpha_acc is not None and sub_acc is not None:
                diff = best_alpha_acc - sub_acc
                star = "*" if (best_alpha_val is not None
                               and best_alpha_val != args.qv_alpha) else ""
                qv_row_z.append(diff)
                qv_row_text.append(f"{diff:.2f}{star}")
            else:
                qv_row_z.append(None)
                qv_row_text.append("")

        for method in baseline_methods:
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
    qv_colorbar_title = f"Acc \u0394 (best \u03b1 \u2212 {subtractor})"

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
            hovertemplate="target=%{y}<br>qv=%{x}<br>delta=%{z:.4f}<extra></extra>",
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

    _add_diagonal_borders(fig, datasets, xref="x", yref="y")

    skip_str = ",".join(sorted(args.skip_modules))
    competitor_str = (
        COMPETITORS[subtractor]["title"](args) if subtractor in COMPETITORS else ""
    )
    title = (
        f"QV Transfer ({head_label}, QAT+PTQ, Best Alpha) \u2212 "
        f"{BASELINE_METHOD_LABELS[subtractor]}<br>"
        f"<sup>{args.model_name} | seed={args.seed} | "
        f"src_mult={mult_tag(args.source_epoch_mult)} | tgt_mult={mult_tag(args.target_epoch_mult)} | "
        f"optim={args.optim} | "
        f"qat_bits={args.qat_bits} | ptq_bits={args.ptq_bits} | granularity={args.granularity} | skip={skip_str} | "
        f"alpha=best{competitor_str}</sup>"
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

    out_parts = [
        "plots", "vision", "ilharco_timm_supervised", "001_qat_transfer", "qv_transfer_heatmap",
        model_dir, f"seed={args.seed}",
        f"smult={mult_tag(args.source_epoch_mult)}", f"tmult={mult_tag(args.target_epoch_mult)}",
        optim_frag, qat_frag,
        _ptq_frag(args.ptq_bits, args.granularity, args.skip_modules),
    ]
    # Only a competitor's own figure is qualified by the sweep it subtracts;
    # adding the fragment unconditionally would relocate the pre-existing fp_ptq
    # figure the moment any competitor flag is passed.
    if subtractor in COMPETITORS:
        out_parts.append(args.competitor_frags[subtractor])
    out_parts += ["qv=alpha=best", "split=test"]
    out_dir = os.path.join(*out_parts)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"heatmap_qv_transfer_{metric_tag}_best_alpha_minus_{subtractor}.png",
    )
    fig.write_image(out_path, scale=300 / 96)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    data, model_dir, optim_frag, qat_frag = load_data(args)
    for metric_tag in TEST_METRIC_KEYS:
        plot_best_alpha_minus_baseline_heatmap(
            data, args, model_dir, optim_frag, qat_frag, metric_tag,
            subtractor="fp_ptq",
        )
        for name in args.enabled_competitors:
            plot_best_alpha_minus_baseline_heatmap(
                data, args, model_dir, optim_frag, qat_frag, metric_tag,
                subtractor=name,
            )


if __name__ == "__main__":
    main()
