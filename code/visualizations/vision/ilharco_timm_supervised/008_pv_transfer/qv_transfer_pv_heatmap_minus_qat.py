"""008 — Does the finetuner matter? PV-QV transfer minus QAT-QV transfer.

This is the figure the 008 phase exists to produce. Both phases patch the same
receiver FP checkpoint with a donor-derived quantization vector and evaluate
under the same round-to-nearest `apply_ptq_` at the same bit-width; the only
difference between them is how the donor was finetuned — straight-through
estimation in 001, PV-Tuning in 008. So

    cell[target, qv] = 008 {split}_accuracy_<head>_ptq[target, qv]
                     - 001 {split}_accuracy_<head>_ptq[target, qv]

isolates the effect of the finetuner on transferability, with everything else
held fixed by construction rather than by argument.

Reading it:

* Positive cells mean the PV-derived direction transfers better than the
  STE-derived one — the transferable content depends on how the QAT optimum was
  found, and PV finds a better one.
* Cells near zero mean the transferable content is a property of the
  quantization grid that any quantization-aware finetuner recovers. That is a
  negative result for "use a better finetuner", but a positive one for the
  paper's central claim: it says the QV is about the grid, not about the
  optimizer's idiosyncrasies.
* The **diagonal is not a transfer result** in either phase — it is each
  phase's own ceiling (PV_target vs QAT_target), so a diagonal difference is a
  finetuner-quality difference, not a transferability one. It is drawn with a
  border and must be read separately from the off-diagonal cells; the printed
  summary reports the two populations separately for that reason.

Because a cell requires both phases to have been run at matched
(seed, bits, granularity, ptq, alpha, split), missing cells are left blank
rather than filled — a half-populated difference map would otherwise read as a
result.

The FP-head variant is the one to trust for the transfer question: the PV-head
and QAT-head variants graft different receiver heads, so their difference mixes
the head in with the direction.
"""

import argparse
import datetime
import json
import os
import sys

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

os.chdir(_PROJECT_ROOT)

from src.duration import mult_path_frag, mult_tag, role_path_frag
import plotly.graph_objects as go

from src.pv_tuning import pv_path_frag
from src.vision.data.common import DATASET_NAME_TO_EPOCHS
from src.vision.utils import sanitize_timm_model_name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT_QV_PV  = "evaluations/vision/ilharco_timm_supervised/008_pv_transfer/vision/qv_transfer_pv"
EVAL_ROOT_QV_QAT = "evaluations/vision/ilharco_timm_supervised/001_qat_transfer/vision/qv_transfer"

# metric_tag -> (008 key suffix, 001 key suffix). The FP-head variant is the
# like-for-like comparison; the trained-head variant is kept because its
# absence would be conspicuous, but it grafts different heads on the two sides.
METRIC_TAGS = {
    "fp_head_ptq":      ("fp_head_ptq", "fp_head_ptq"),
    "trained_head_ptq": ("pv_head_ptq", "qat_head_ptq"),
}

METRIC_LABELS = {
    "fp_head_ptq":      "FP Head",
    "trained_head_ptq": "Trained Head (PV vs QAT)",
}

HEATMAP_COLORSCALE_DIVERGING = "RdYlGn"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name",     required=True)
    parser.add_argument("--seed",           required=True, type=int)
    parser.add_argument("--source-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the DONOR checkpoints.")
    parser.add_argument("--target-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the RECEIVER checkpoints.")

    parser.add_argument("--optim",          required=True, choices=["adamw", "sgd"])
    parser.add_argument("--lr",             required=True, type=float)
    parser.add_argument("--wd",             required=True, type=float)
    parser.add_argument("--ls",             required=True, type=float)
    parser.add_argument("--wl",             required=True, type=int)
    parser.add_argument("--max-grad-norm",  required=True, type=float)
    parser.add_argument("--batch-size",     required=True, type=int)

    # 001 locates its donor by qat=..., 008 by pv=...; both must be given
    # because this figure reads both trees.
    parser.add_argument("--qat-bits",       required=True, type=int)
    parser.add_argument("--pv-bits",        required=True, type=int)
    parser.add_argument("--ptq-bits",       required=True, type=int)
    parser.add_argument("--granularity",    required=True, choices=["tensor", "channel"])
    parser.add_argument("--skip-modules",   required=True, nargs="+")

    parser.add_argument("--pv-delta",       required=True, type=float)
    parser.add_argument("--pv-tau",         required=True, type=float)
    parser.add_argument("--pv-trust",       required=True,
                        help='PV trust_ratio, or "none" if it was disabled')
    parser.add_argument("--pv-pevery",      required=True, type=int)
    parser.add_argument("--pv-temp",        required=True, type=float)

    parser.add_argument("--qv-alpha",       required=True, type=float)

    parser.add_argument("--eval-split",     default="test", choices=["val", "test"])

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path-fragment helpers
# ---------------------------------------------------------------------------
def _skip_tag(skip_modules):
    return "-".join(sorted(skip_modules)) if len(skip_modules) > 0 else "none"


def _optim_frag(optim, lr, wd, ls, wl, mgn, bs):
    del optim
    return f"optim=adamw_lr={lr}_wd={wd}_ls={ls}_wl={wl}_mgn={mgn}_bs={bs}"


def _qat_frag(bits, gran, skip_modules):
    return f"qat=bits={bits}_gran={gran}_skip={_skip_tag(skip_modules)}"


def _pv_frag(args):
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


def _transfer_path(root, model_dir, qv_dataset, target_dataset, seed,
                   optim_frag, method_frag, ptq_frag, qv_frag, eval_split, *, source_epoch_mult, target_epoch_mult):
    return os.path.join(
        root, model_dir,
        role_path_frag("src", qv_dataset, seed, source_epoch_mult),
        role_path_frag("tgt", target_dataset, seed, target_epoch_mult),
        optim_frag, method_frag, ptq_frag, qv_frag,
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
    qat_frag   = _qat_frag(args.qat_bits, args.granularity, args.skip_modules)
    ptq_frag   = _ptq_frag(args.ptq_bits, args.granularity, args.skip_modules)
    qv_frag    = _qv_frag(args.qv_alpha)
    split      = args.eval_split

    datasets = sorted(DATASET_NAME_TO_EPOCHS.keys(), key=str.lower)

    # data[metric_tag][target][qv] = pv_value - qat_value, or None
    data = {tag: {} for tag in METRIC_TAGS}

    for target_dataset in datasets:
        for tag in METRIC_TAGS:
            data[tag][target_dataset] = {}

        for qv_dataset in datasets:
            pv_path = _transfer_path(
                EVAL_ROOT_QV_PV, model_dir, qv_dataset, target_dataset, args.seed,
                optim_frag, pv_frag, ptq_frag, qv_frag, split,
            
                source_epoch_mult=args.source_epoch_mult, target_epoch_mult=args.target_epoch_mult,
            )
            qat_path = _transfer_path(
                EVAL_ROOT_QV_QAT, model_dir, qv_dataset, target_dataset, args.seed,
                optim_frag, qat_frag, ptq_frag, qv_frag, split,
            
                source_epoch_mult=args.source_epoch_mult, target_epoch_mult=args.target_epoch_mult,
            )

            for tag, (pv_suffix, qat_suffix) in METRIC_TAGS.items():
                pv_value = _load_value(pv_path, f"{split}_accuracy_{pv_suffix}")
                qat_value = _load_value(qat_path, f"{split}_accuracy_{qat_suffix}")
                data[tag][target_dataset][qv_dataset] = (
                    pv_value - qat_value
                    if pv_value is not None and qat_value is not None
                    else None
                )

    return data, model_dir, optim_frag, pv_frag, qat_frag, qv_frag


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _add_diagonal_borders(fig, datasets, color="black", width=2):
    for i in range(len(datasets)):
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=i - 0.5, x1=i + 0.5,
            y0=i - 0.5, y1=i + 0.5,
            line=dict(color=color, width=width),
            fillcolor="rgba(0,0,0,0)",
        )


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


def _robust_symmetric_bounds(values, center=0.0, min_span=0.02, q_low=0.05, q_high=0.95):
    if not values:
        return center - min_span, center + min_span
    svals = sorted(values)
    ql = _quantile(svals, q_low)
    qh = _quantile(svals, q_high)
    span = max(abs(center - ql), abs(qh - center), min_span)
    return center - span, center + span


def _mean(values):
    return sum(values) / len(values) if values else None


def _summarize(data, datasets, tag):
    """Report off-diagonal and diagonal populations separately.

    They answer different questions: off-diagonal cells are transfer, the
    diagonal is each phase's own ceiling. Averaging them together would let a
    finetuner-quality difference masquerade as a transferability difference.
    """
    off_diagonal, diagonal = [], []
    for target_dataset in datasets:
        for qv_dataset in datasets:
            value = data[tag][target_dataset][qv_dataset]
            if value is None:
                continue
            (diagonal if qv_dataset == target_dataset else off_diagonal).append(value)

    print(f"\n### {METRIC_LABELS[tag]}  (PV-QV minus QAT-QV, accuracy points)")
    for label, population in (("cross-task (transfer)", off_diagonal),
                              ("same-task (ceiling)", diagonal)):
        if not population:
            print(f"  {label:<24} n=0")
            continue
        wins = sum(1 for v in population if v > 0)
        print(
            f"  {label:<24} n={len(population):<4} "
            f"mean={_mean(population):+.4f}  "
            f"PV better in {wins}/{len(population)} "
            f"({100.0 * wins / len(population):.1f}%)"
        )
    return off_diagonal, diagonal


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_difference(data, args, model_dir, optim_frag, pv_frag, qat_frag, qv_frag, tag):
    datasets = sorted(DATASET_NAME_TO_EPOCHS.keys(), key=str.lower)

    z, text = [], []
    for target_dataset in datasets:
        row_z, row_text = [], []
        for qv_dataset in datasets:
            value = data[tag][target_dataset][qv_dataset]
            if value is not None:
                row_z.append(value)
                row_text.append(f"{value:.2f}")
            else:
                row_z.append(None)
                row_text.append("")
        z.append(row_z)
        text.append(row_text)

    off_diagonal, _ = _summarize(data, datasets, tag)

    # Scale the colour axis on the cross-task cells only: the diagonal is a
    # different quantity and letting it set the range would flatten the cells
    # the figure is actually about.
    cmin, cmax = _robust_symmetric_bounds(off_diagonal)

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=datasets,
            y=datasets,
            text=text,
            texttemplate="%{text}",
            colorscale=HEATMAP_COLORSCALE_DIVERGING,
            zmin=cmin, zmax=cmax, zmid=0,
            xgap=1, ygap=1,
            colorbar=dict(title="Acc Δ<br>(PV − QAT)"),
            hovertemplate="target=%{y}<br>qv=%{x}<br>delta=%{z:.4f}<extra></extra>",
        )
    )
    _add_diagonal_borders(fig, datasets)

    skip_str = ",".join(sorted(args.skip_modules))
    # NOTE: keep the title short. Kaleido derives a scratch filename from the
    # title text even when plotly hands it an explicit output path, so a title
    # over ~255 characters fails with ENAMETOOLONG no matter where the figure
    # is being written. Long-form notes belong in an annotation, which is not
    # used for the filename.
    fig.add_annotation(
        text=(
            "bordered diagonal = each phase's own ceiling (PV vs QAT), not a "
            "transfer result; colour scale is set from the cross-task cells only"
        ),
        xref="paper", yref="paper", x=0.0, y=-0.16,
        showarrow=False, align="left", font=dict(size=11, color="#555555"),
    )

    # State the fill fraction on the figure. A blank cell and a zero-difference
    # cell look alike, so a partially-populated grid read as finished would
    # invite exactly the over-reading this phase has already suffered twice.
    total = len(datasets) * len(datasets)
    filled = sum(1 for row in z for v in row if v is not None)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    status = "COMPLETE" if filled == total else "PARTIAL — sweep still running"
    fig.add_annotation(
        text=(
            f"{status}: {filled}/{total} cells populated ({100.0*filled/total:.0f}%) · "
            f"cross-task n={len(off_diagonal)} · generated {stamp}"
        ),
        xref="paper", yref="paper", x=0.0, y=-0.21,
        showarrow=False, align="left",
        font=dict(size=11, color="#b00000" if filled != total else "#555555"),
    )
    fig.update_layout(
        title=(
            f"PV-QV − QAT-QV Transfer ({METRIC_LABELS[tag]})<br>"
            f"<sup>{args.model_name} | seed={args.seed} | "
            f"pv/qat/ptq bits={args.pv_bits}/{args.qat_bits}/{args.ptq_bits} | "
            f"gran={args.granularity} | skip={skip_str} | "
            f"delta={args.pv_delta} | tau={args.pv_tau} | "
            f"trust={args.pv_trust} | alpha={args.qv_alpha} | "
            f"split={args.eval_split}</sup>"
        ),
        template="plotly_white",
        height=max(400, 60 * len(datasets) + 200),
        width=max(900, 55 * len(datasets) + 320),
        margin=dict(l=140, r=160, t=150, b=185),
    )
    fig.update_xaxes(
        title_text="Quantization Vector Dataset<br>(dataset the qv is computed from)",
        side="bottom",
    )
    fig.update_yaxes(
        title_text="Target Dataset<br>(dataset the qv is applied to)",
        autorange="reversed",
    )

    out_dir = os.path.join(
        "plots", "vision", "ilharco_timm_supervised", "008_pv_transfer",
        "qv_transfer_pv_heatmap_minus_qat",
        model_dir, f"seed={args.seed}", f"smult={mult_tag(args.source_epoch_mult)}", f"tmult={mult_tag(args.target_epoch_mult)}", optim_frag, pv_frag, qat_frag,
        _ptq_frag(args.ptq_bits, args.granularity, args.skip_modules),
        qv_frag,
        f"split={args.eval_split}",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"heatmap_pv_minus_qat_{tag}.png")
    fig.write_image(out_path, scale=300 / 96)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    data, model_dir, optim_frag, pv_frag, qat_frag, qv_frag = load_data(args)
    for tag in METRIC_TAGS:
        plot_difference(data, args, model_dir, optim_frag, pv_frag, qat_frag,
                        qv_frag, tag)


if __name__ == "__main__":
    main()
