"""998 — Donor benefit against donor cost

One marker per donor dataset, from the donor_benefit.json written by
compute_donor_benefit.py:

    x   the compute spent producing that donor's quantization vector, i.e. one
        QAT run over the donor's training set, in millions of training samples.
    y   what that single QV buys, selected with --y:

        mean_delta      (default) the mean Top-1 gain over vanilla PTQ across
                        the donor's receivers and the family's backbones, in
                        percentage points.  This is the quantity the cost is
                        actually being paid for, and unlike a win count it keeps
                        the magnitude: a donor that wins narrowly on 18 tasks and
                        one that wins decisively on 18 tasks are different buys.
        median_delta    the same, robust to a single receiver carrying the mean.
        mean_wins       how many receivers the QV helps at all, averaged over
                        backbones — a binarization of the above, kept because it
                        is what the win/loss table reports.

Under the delta metrics a zero line is drawn, since donors do land below it: a
negative mean is a donor whose vector hurts the average receiver, which no win
count can express.

Each donor gets its own colour and a legend entry, since the markers crowd
together on the cost axis and in-plot text labels overlap.  Nothing else is
drawn: the reading of the plot — cheap donors that buy a lot sit top-left — is
left to the axes.

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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT = "evaluations/998_rebuttal/002_cost_amortization"
PLOT_ROOT = "plots/998_rebuttal/002_cost_amortization"

# Kelly's 22 colours of maximum contrast, with the near-white entry swapped for
# a cyan so every donor stays legible on the white template.
PALETTE = [
    "#00A0B0", "#222222", "#F3C300", "#875692", "#F38400", "#A1CAF1",
    "#BE0032", "#C2B280", "#848482", "#008856", "#E68FAC", "#0067A5",
    "#F99379", "#604E97", "#F6A600", "#B3446C", "#DCD300", "#882D17",
    "#8DB600", "#654522", "#E25822", "#2B3D26",
]

SAMPLES_PER_UNIT = 1e6       # axes are reported in millions of samples
UNIT_LABEL = "M training samples"

COLOR_ZERO = "#8A8A8A"       # the delta = 0 line

# What the donor's QAT run buys.  `scale` converts the stored value to the
# plotted unit (deltas are stored as fractions, reported as percentage points).
Y_METRICS = {
    "mean_delta": dict(
        key="mean_delta", scale=100.0, signed=True,
        title="mean Top-1 gain over PTQ (p.p.), across receivers and backbones",
    ),
    "median_delta": dict(
        key="median_delta", scale=100.0, signed=True,
        title="median Top-1 gain over PTQ (p.p.), across receivers and backbones",
    ),
    "mean_wins": dict(
        key="mean_wins", scale=1.0, signed=False,
        title="receivers helped (of {n_receivers}), mean over backbones",
    ),
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in-name",
        default="donor_benefit.json",
        help="File read from evaluations/998_rebuttal/002_cost_amortization/.",
    )
    parser.add_argument(
        "--y",
        default="mean_delta",
        choices=sorted(Y_METRICS),
        help="What the donor buys: the size of the gain (default) or a count of wins.",
    )
    parser.add_argument(
        "--log-x",
        action="store_true",
        help="Log-scale the cost axis (the donors bunch up under ImageNet otherwise).",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also write a PDF next to the HTML (needs kaleido).",
    )
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=560)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _scaled(value):
    return value / SAMPLES_PER_UNIT


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build_figure(results, args):
    metric = Y_METRICS[args.y]
    donors = [d for d in results["donors"] if d["cost_samples"] is not None]
    assert donors, "no donor has a cost entry; check dataset_sizes.json"
    assert len(donors) <= len(PALETTE), (
        f"{len(donors)} donors but only {len(PALETTE)} colours in PALETTE"
    )

    ys = [d[metric["key"]] * metric["scale"] for d in donors]
    x_max = _scaled(max(d["cost_samples"] for d in donors)) * 1.18
    # Signed metrics need room below zero: a donor with a negative mean gain is a
    # real outcome, not a clipping artefact.
    y_span = max(ys) - min(min(ys), 0.0)
    y_range = [min(min(ys), 0.0) - 0.08 * y_span, max(ys) + 0.10 * y_span]
    n_receivers = donors[0]["n_receivers"]

    fig = go.Figure()
    # One trace per donor: the trace name is what the legend keys the colour to.
    for d, y, color in zip(donors, ys, PALETTE):
        fig.add_trace(
            go.Scatter(
                x=[_scaled(d["cost_samples"])],
                y=[y],
                mode="markers",
                marker=dict(
                    color=color, size=12, line=dict(color="white", width=1)
                ),
                name=d["donor"],
            )
        )

    if metric["signed"]:
        fig.add_hline(y=0.0, line=dict(color=COLOR_ZERO, width=1, dash="dot"))

    fig.update_xaxes(
        title_text=f"cost of the donor's QAT run ({UNIT_LABEL})",
        range=None if args.log_x else [0, x_max],
        type="log" if args.log_x else "linear",
        zeroline=False,
    )
    fig.update_yaxes(
        title_text=metric["title"].format(n_receivers=n_receivers),
        range=y_range,
        zeroline=False,
    )

    fig.update_layout(
        template="plotly_white",
        width=args.width,
        height=args.height,
        margin=dict(l=70, r=20, t=30, b=60),
        legend=dict(
            title_text="",
            itemsizing="constant",
            font=dict(size=11),
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
        ),
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    results_path = os.path.join(EVAL_ROOT, args.in_name)
    assert os.path.exists(results_path), (
        f"{results_path} not found. Run compute_donor_benefit.py first."
    )
    with open(results_path) as f:
        results = json.load(f)

    fig = build_figure(results, args)

    os.makedirs(PLOT_ROOT, exist_ok=True)
    # The win-count figure keeps its original stem; the gain metrics add a
    # suffix so the two do not overwrite each other.
    stem = f"donor_benefit_scatter_{results['family']}_{results['settings']['alpha_regime']}"
    if args.y != "mean_wins":
        stem = f"{stem}_{args.y}"
    html_path = os.path.join(PLOT_ROOT, f"{stem}.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"wrote {html_path}")

    if args.pdf:
        pdf_path = os.path.join(PLOT_ROOT, f"{stem}.pdf")
        fig.write_image(pdf_path)
        print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
