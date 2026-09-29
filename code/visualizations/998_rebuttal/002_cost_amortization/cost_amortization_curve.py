"""998 — Cost amortization curves for QV transfer against per-task QAT

Renders the figure the reviewer asked for: how the cost of the two pipelines
scales with the number N of deployed receivers, and where the one-off donor QAT
run is amortized.  Two panels, from the cost_amortization.json written by
compute_costs.py:

    left   cumulative cost against N.  Per-task QAT grows linearly; QV transfer
           pays the donor run once and is flat afterwards.  The shaded band
           spans the cheapest-first and costliest-first deployment orders, so it
           brackets every possible ordering, and the crossing is annotated with
           the break-even factor N*.
    right  per-deployment cost against N.  Per-task QAT is constant; QV transfer
           decays as 1/N towards the cost of a vector addition.

Writes an HTML figure to plots/998_rebuttal/002_cost_amortization/.
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

import os

os.chdir(_PROJECT_ROOT)

import argparse
import json

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT = "evaluations/998_rebuttal/002_cost_amortization"
PLOT_ROOT = "plots/998_rebuttal/002_cost_amortization"

COLOR_BASELINE = "#B24C3F"   # per-task QAT
COLOR_OURS     = "#2F6E8F"   # QV transfer
COLOR_BAND     = "rgba(178, 76, 63, 0.15)"
COLOR_MARK     = "#5A5A5A"

SAMPLES_PER_UNIT = 1e6       # curves are reported in millions of samples
UNIT_LABEL = "M training samples"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Log-scale both cost axes (useful when the donor dwarfs the receivers).",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also write a PDF next to the HTML (needs kaleido).",
    )
    parser.add_argument("--width",  type=int, default=1000)
    parser.add_argument("--height", type=int, default=420)
    return parser.parse_args()


def _scaled(values):
    return [v / SAMPLES_PER_UNIT for v in values]


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build_figure(results, args):
    curves = results["curves"]
    ns = curves["n"]
    donor = results["donor"]
    n_star = results["break_even"]["n_star_mean_exact"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Cumulative cost of deploying N quantized tasks",
            "Cost per deployed task",
        ),
        horizontal_spacing=0.12,
    )

    # -- left panel: cumulative cost ----------------------------------------
    # Ordering envelope, drawn first so the mean lines sit on top of it.
    fig.add_trace(
        go.Scatter(
            x=ns,
            y=_scaled(curves["baseline_cumulative_cheapest_first"]),
            mode="lines",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ns,
            y=_scaled(curves["baseline_cumulative_costliest_first"]),
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=COLOR_BAND,
            name="per-task QAT, deployment-order range",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ns,
            y=_scaled(curves["baseline_cumulative_mean"]),
            mode="lines",
            line=dict(color=COLOR_BASELINE, width=2.5),
            name="per-task QAT (one QAT run per task)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=ns,
            y=_scaled(curves["ours_cumulative"]),
            mode="lines",
            line=dict(color=COLOR_OURS, width=2.5),
            name=f"QV transfer (one QAT run on {donor}, then vector additions)",
        ),
        row=1,
        col=1,
    )

    # Break-even marker.
    fig.add_vline(
        x=n_star,
        line=dict(color=COLOR_MARK, width=1.5, dash="dash"),
        row=1,
        col=1,
    )
    fig.add_annotation(
        x=n_star,
        y=1.0,
        yref="y domain",
        text=f"N* = {n_star:.1f}",
        showarrow=False,
        xanchor="left",
        xshift=6,
        font=dict(size=12, color=COLOR_MARK),
        row=1,
        col=1,
    )

    # -- right panel: per-deployment cost -----------------------------------
    fig.add_trace(
        go.Scatter(
            x=ns,
            y=_scaled(curves["baseline_per_deployment"]),
            mode="lines",
            line=dict(color=COLOR_BASELINE, width=2.5),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=ns,
            y=_scaled(curves["ours_per_deployment"]),
            mode="lines",
            line=dict(color=COLOR_OURS, width=2.5),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_vline(
        x=n_star,
        line=dict(color=COLOR_MARK, width=1.5, dash="dash"),
        row=1,
        col=2,
    )

    axis_kwargs = dict(type="log") if args.log_y else {}
    fig.update_yaxes(title_text=UNIT_LABEL, row=1, col=1, **axis_kwargs)
    fig.update_yaxes(title_text=f"{UNIT_LABEL} / task", row=1, col=2, **axis_kwargs)
    fig.update_xaxes(title_text="number of deployed tasks N", row=1, col=1)
    fig.update_xaxes(title_text="number of deployed tasks N", row=1, col=2)

    speedup = results["at_full_suite"]["speedup"]
    fig.update_layout(
        template="plotly_white",
        width=args.width,
        height=args.height,
        margin=dict(l=70, r=30, t=90, b=110),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="left",
            x=0,
        ),
        title=dict(
            text=(
                f"One donor QAT run on {donor} amortizes after N* = {n_star:.1f} tasks "
                f"({speedup:.1f}x less compute across the full suite of "
                f"{results['n_receivers']})"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=14),
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    results_path = os.path.join(EVAL_ROOT, "cost_amortization.json")
    assert os.path.exists(results_path), (
        f"{results_path} not found. Run compute_costs.py first."
    )
    with open(results_path) as f:
        results = json.load(f)

    fig = build_figure(results, args)

    os.makedirs(PLOT_ROOT, exist_ok=True)
    html_path = os.path.join(PLOT_ROOT, "cost_amortization_curve.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"wrote {html_path}")

    if args.pdf:
        pdf_path = os.path.join(PLOT_ROOT, "cost_amortization_curve.pdf")
        fig.write_image(pdf_path)
        print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
