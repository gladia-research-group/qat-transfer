"""998 — Lambda sensitivity figure

Renders the figure behind the answer to "how sensitive is the method to the
choice of lambda", from the lambda_curves.json written by
aggregate_lambda_curves.py.  Two panels:

    left   the curve against lambda, one median trace per backbone, over the
           interquartile band pooled across every donor-receiver cell.  A flat
           trace is the claim: lambda barely matters.  The horizontal zero line
           is the data-free default, so the vertical distance from it is exactly
           what tuning lambda would buy.
    right  the distribution of the per-cell optimum lambda*, pooled over cells.
           Mass concentrated below 1 says the default overshoots -- the reading
           the paper asserts, checked here directly.

Only the pooled grid is plotted: models swept on different lambda grids are
restricted to the lambdas they share, since a trace that changed model set
partway along the axis would not be comparable across lambda.  The restriction
is computed by the aggregate, not here.

Reads the aggregate JSON; recomputes no statistic.  Writes an HTML figure to
plots/998_rebuttal/003_lambda_sensitivity/.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is the working directory and on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from src.duration import mult_tag

import os

os.chdir(_PROJECT_ROOT)

import argparse
import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT = "evaluations/998_rebuttal/003_lambda_sensitivity"
PLOT_ROOT = "plots/998_rebuttal/003_lambda_sensitivity"

BASELINE_FP_PTQ = "fp_ptq"
BASELINE_UNIT   = "unit"

COLOR_BAND = "rgba(47, 110, 143, 0.16)"
COLOR_MARK = "#5A5A5A"
COLOR_BARS = "#2F6E8F"

# One colour per backbone; vision-supervised warm, CLIP-style cool, text green,
# so a family reads as a group without needing the legend.
MODEL_COLORS = [
    "#B24C3F", "#C4703E", "#D19A4A", "#8A6E4B", "#6E7F5C",
    "#2F6E8F", "#4A8FA8", "#6FA8BF",
    "#5C6E9E", "#7A5C9E", "#9E5C8A", "#4F9E7A",
]

UNIT_ALPHA = 1.0
PP = 100.0  # accuracies are stored as fractions; report accuracy points


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",        required=True, type=int)
    parser.add_argument("--source-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the DONOR checkpoints "
                             "the aggregate was computed at.")
    parser.add_argument("--target-epoch-mult", required=True, type=float,
                        help="Training-budget multiplier of the RECEIVER checkpoints.")
    parser.add_argument("--qat-bits",    required=True, type=int)
    parser.add_argument("--ptq-bits",    required=True, type=int)
    parser.add_argument("--granularity", required=True, choices=["tensor", "channel"])
    parser.add_argument("--split",       required=True, choices=["val", "test"])
    parser.add_argument("--baseline",    required=True,
                        choices=[BASELINE_FP_PTQ, BASELINE_UNIT])
    parser.add_argument("--grid",        required=True, choices=["shared", "full"])
    parser.add_argument("--scope",       required=True, choices=["cross_task", "same_task"],
                        help="cross_task is the genuine transfer setting; same_task "
                             "is the QAT ceiling and not a transfer result.")
    parser.add_argument("--pdf", action="store_true",
                        help="Also write a PDF next to the HTML (needs kaleido).")
    parser.add_argument("--width",  type=int, default=1080)
    parser.add_argument("--height", type=int, default=460)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path fragments
# ---------------------------------------------------------------------------
def _qat_frag(args):
    return f"qat=bits={args.qat_bits}_gran={args.granularity}"


def _ptq_frag(args):
    """Evaluation-time PTQ fragment.

    Mandatory per the repo convention, so that runs sharing a QAT configuration
    but differing in PTQ cannot overwrite each other.  The skip-module tag is
    absent here because this figure is cross-family and the families use
    different skipped heads; a single tag would misdescribe the aggregate. Bits
    and granularity, which are the axes that actually collided, are kept.
    """
    return f"ptq=bits={args.ptq_bits}_gran={args.granularity}"


def _split_frag(args):
    return f"split={args.split}"


def _in_dir(args):
    return os.path.join(
        EVAL_ROOT,
        f"seed={args.seed}",
        f"smult={mult_tag(args.source_epoch_mult)}",
        f"tmult={mult_tag(args.target_epoch_mult)}",
        _qat_frag(args),
        _ptq_frag(args),
        _split_frag(args),
        f"baseline={args.baseline}",
        f"grid={args.grid}",
    )


def _out_dir(args):
    return os.path.join(
        PLOT_ROOT,
        f"seed={args.seed}",
        _qat_frag(args),
        _ptq_frag(args),
        _split_frag(args),
        f"baseline={args.baseline}",
        f"grid={args.grid}",
    )


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def _y_title(baseline):
    if baseline == BASELINE_FP_PTQ:
        return "Δ accuracy vs vanilla PTQ (pp)"
    return "accuracy relative to λ = 1 (pp)"


def _left_title(baseline):
    if baseline == BASELINE_FP_PTQ:
        return "Gain over vanilla PTQ against λ"
    return "Cost of moving off the data-free default λ = 1"


def _headline(results, args):
    micro = results["overall"][args.scope]["micro"]
    n_models = len(results["models"])
    parts = [f"{n_models} backbones", f"{micro['n']} donor–receiver pairs"]
    if micro["lambda_star"]["median"] is not None:
        parts.append(f"median λ* = {micro['lambda_star']['median']}")
    if args.baseline == BASELINE_UNIT:
        if micro["unit_regret"]["median"] is not None:
            parts.append(f"median regret of λ = 1 is "
                         f"{micro['unit_regret']['median'] * PP:.2f}pp")
    elif micro["unit_retention"]["median"] is not None:
        parts.append(f"median retention at λ = 1 is "
                     f"{micro['unit_retention']['median']:.2f}")
    if micro["plateau_width"]["median"] is not None:
        parts.append(f"median plateau {micro['plateau_width']['median']:.2f} wide")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build_figure(results, args):
    grid  = results["pooled_grid"]
    micro = results["overall"][args.scope]["micro"]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.62, 0.38],
        subplot_titles=(_left_title(args.baseline),
                        "Where the per-pair optimum λ* falls"),
        horizontal_spacing=0.11,
    )

    # -- left panel: pooled interquartile band, drawn first so lines sit on it --
    keys = [str(a) for a in grid]
    p25 = [micro["curve"][k]["p25"] * PP for k in keys]
    p75 = [micro["curve"][k]["p75"] * PP for k in keys]

    fig.add_trace(
        go.Scatter(x=grid, y=p25, mode="lines", line=dict(width=0),
                   hoverinfo="skip", showlegend=False),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=grid, y=p75, mode="lines", line=dict(width=0),
                   fill="tonexty", fillcolor=COLOR_BAND,
                   name="pooled IQR", hoverinfo="skip"),
        row=1, col=1,
    )

    # -- left panel: one median trace per backbone ---------------------------
    for i, model in enumerate(results["models"]):
        block = model.get(args.scope)
        if not block or not block.get("curve"):
            continue
        xs, ys = [], []
        for alpha, key in zip(grid, keys):
            point = block["curve"].get(key)
            if point and point["median"] is not None:
                xs.append(alpha)
                ys.append(point["median"] * PP)
        if not xs:
            continue
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                name=model["display_name"],
                line=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)], width=1.8),
                marker=dict(size=4),
                hovertemplate=(f"{model['display_name']}<br>"
                               "λ=%{x}<br>%{y:+.2f}pp<extra></extra>"),
            ),
            row=1, col=1,
        )

    # The reference level, and the default the reviewers care about.
    fig.add_hline(y=0.0, line=dict(color=COLOR_MARK, width=1, dash="dot"),
                  row=1, col=1)
    fig.add_vline(x=UNIT_ALPHA, line=dict(color=COLOR_MARK, width=1, dash="dash"),
                  row=1, col=1)

    # -- right panel: lambda* distribution ----------------------------------
    histogram = micro.get("lambda_star_histogram") or {}
    if histogram:
        total = sum(histogram.values())
        xs = [float(k) for k in histogram]
        ys = [v / total * 100.0 for v in histogram.values()]
        fig.add_trace(
            go.Bar(x=xs, y=ys, marker_color=COLOR_BARS, name="λ*",
                   showlegend=False,
                   hovertemplate="λ*=%{x}<br>%{y:.1f}% of pairs<extra></extra>"),
            row=1, col=2,
        )
        fig.add_vline(x=UNIT_ALPHA,
                      line=dict(color=COLOR_MARK, width=1, dash="dash"),
                      row=1, col=2)

    fig.update_xaxes(title_text="λ", row=1, col=1)
    fig.update_xaxes(title_text="λ*", row=1, col=2)
    fig.update_yaxes(title_text=_y_title(args.baseline), row=1, col=1)
    fig.update_yaxes(title_text="% of donor–receiver pairs", row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        width=args.width,
        height=args.height,
        margin=dict(l=76, r=30, t=96, b=120),
        bargap=0.15,
        legend=dict(orientation="h", yanchor="top", y=-0.20,
                    xanchor="left", x=0, font=dict(size=10)),
        title=dict(text=_headline(results, args), x=0.5, xanchor="center",
                   font=dict(size=13)),
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    results_path = os.path.join(_in_dir(args), "lambda_curves.json")
    assert os.path.exists(results_path), (
        f"{results_path} not found. Run aggregate_lambda_curves.py first with "
        f"matching --split/--baseline/--grid."
    )
    with open(results_path) as f:
        results = json.load(f)

    fig = build_figure(results, args)

    out_dir = _out_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    stem = f"lambda_curve_{args.scope}"
    html_path = os.path.join(out_dir, f"{stem}.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"wrote {html_path}")

    if args.pdf:
        pdf_path = os.path.join(out_dir, f"{stem}.pdf")
        fig.write_image(pdf_path)
        print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
