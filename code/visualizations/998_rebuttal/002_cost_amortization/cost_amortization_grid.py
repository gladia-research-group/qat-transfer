"""998 — Cost amortization, one panel per donor

cost_amortization_curve.py answers the reviewer's N-receiver question for the
ImageNet donor.  This script repeats that exact analysis with every dataset in
turn playing the donor, and lays the results out as small multiples, so the
break-even factor N* can be compared across donors without reading 22 curves off
a single crowded axis.

Each panel is one donor.  Its receiver set is every *other* dataset, so both the
baseline and N* are computed from 21 receivers that exclude the donor itself.

    baseline (per-task QAT)   nothing is reused, so the cost of shipping N
                              receivers is the sum of their own QAT runs.  That
                              sum depends on which N are shipped and in which
                              order, so --orders selects which order is drawn:
                              cheapest-first accumulates the baseline as slowly
                              as possible and is therefore the worst case for our
                              claim (the default, and the number to quote),
                              costliest-first is the best case, and every other
                              order lies between the two.
    QV transfer               one donor QAT run, then a vector addition per
                              receiver, so the cumulative cost is flat in N.

    --view per-deployment   (default) cost per shipped receiver.  Per-task QAT
                            is flat, since nothing it does is reused, while QV
                            transfer decays as C_donor / N: the same donor run
                            split over more and more receivers.  Where the decay
                            drops under the flat line is N*, marked.
    --view cumulative       the same multiplied by N: total cost of shipping N
                            receivers.  The roles invert — the baseline climbs
                            and QV transfer is flat — but N* is unchanged.

Cost is training samples seen, epochs * train_size, exactly as in
compute_costs.py: both pipelines run the same QAT procedure on the same
backbone, so the per-sample cost cancels and only samples seen remain.

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
import itertools
import json
import math
import statistics

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT = "evaluations/998_rebuttal/002_cost_amortization"
PLOT_ROOT = "plots/998_rebuttal/002_cost_amortization"

COLOR_OURS = "#2F6E8F"       # QV transfer

# The per-task QAT baseline, one entry per deployment order.  Cheapest-first
# accumulates baseline cost as slowly as possible and so is the worst case for
# our claim; costliest-first is the best case; every other order lies between
# them, and the mean-receiver line is the typical one.
ORDERS = {
    "worst": [
        dict(key="base_cheap", n_star_key="n_star_cheapest_first", short="worst",
             label="per-task QAT, worst order (cheapest first)", color="#B24C3F"),
    ],
    "best": [
        dict(key="base_costly", n_star_key="n_star_costliest_first", short="best",
             label="per-task QAT, best order (costliest first)", color="#5C8A3C"),
    ],
    "mean": [
        dict(key="base_mean", n_star_key="n_star", short="mean",
             label="per-task QAT, mean receiver", color="#B24C3F"),
    ],
    "worst-best": [
        dict(key="base_cheap", n_star_key="n_star_cheapest_first", short="worst",
             label="per-task QAT, worst order (cheapest first)", color="#B24C3F"),
        dict(key="base_costly", n_star_key="n_star_costliest_first", short="best",
             label="per-task QAT, best order (costliest first)", color="#5C8A3C"),
    ],
}

SAMPLES_PER_UNIT = 1e6       # curves are reported in millions of samples
UNIT_LABEL = "M training samples"

N_COLS = 5
Y_HEADROOM = 2.6             # free-y panels span 0 .. Y_HEADROOM * donor cost


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes-name",
        default="dataset_sizes.json",
        help="File read from evaluations/998_rebuttal/002_cost_amortization/.",
    )
    parser.add_argument(
        "--donors",
        nargs="+",
        default=None,
        help="Restrict to these donors (default: every dataset in the sizes file).",
    )
    parser.add_argument(
        "--view",
        default="per-deployment",
        choices=["cumulative", "per-deployment"],
        help="Total cost of shipping N receivers, or that cost divided by N.",
    )
    parser.add_argument(
        "--patch-samples",
        type=float,
        default=0.0,
        help="Training samples charged per patched receiver (default 0: it is a vector addition).",
    )
    parser.add_argument(
        "--orders",
        default="worst",
        choices=sorted(ORDERS),
        help="Which deployment order(s) to draw the per-task QAT baseline for "
             "(default worst: the honest number to quote).",
    )
    parser.add_argument(
        "--n-max",
        type=int,
        default=8,
        help="Largest N shown; every crossing happens early, so the default window "
             "keeps the panels legible (pass 21 for the full suite).",
    )
    parser.add_argument(
        "--shared-y",
        action="store_true",
        help="One cost scale for every panel (default: each panel scales to its own donor cost).",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also write a PDF next to the HTML (needs kaleido).",
    )
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=1150)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _scaled(value):
    return value / SAMPLES_PER_UNIT


def _costs(sizes):
    """One QAT run over each dataset's train split, in training samples seen."""
    return {
        name: rec["epochs"] * rec["train_size"]
        for name, rec in sizes["datasets"].items()
    }


def _cumsum(values):
    return list(itertools.accumulate(values))


def _series(donor, donor_cost, receiver_costs, patch_samples):
    """One donor's cost against N, alongside the baseline it must beat.

    The baseline is reported three ways because the sum over receivers depends
    on the deployment order: the mean receiver (the line that is read), and the
    cheapest-first and costliest-first orders, which bracket every ordering.
    """
    ordered = sorted(receiver_costs.values())
    n_max = len(ordered)
    ns = list(range(1, n_max + 1))
    receiver_cost_mean = statistics.fmean(ordered)

    base_mean = [n * receiver_cost_mean for n in ns]
    base_cheap = _cumsum(ordered)
    base_costly = _cumsum(sorted(ordered, reverse=True))
    ours = [donor_cost + n * patch_samples for n in ns]

    def _n_star(base):
        return next((n for n, b in zip(ns, base) if b >= ours[n - 1]), None)

    return {
        "donor": donor,
        "donor_cost": donor_cost,
        "receiver_cost_mean": receiver_cost_mean,
        "n": ns,
        "ours": ours,
        "base_mean": base_mean,
        "base_cheap": base_cheap,
        "base_costly": base_costly,
        "n_star": _n_star(base_mean),
        "n_star_cheapest_first": _n_star(base_cheap),
        "n_star_costliest_first": _n_star(base_costly),
    }


def _viewed(values, ns, view):
    """Cumulative cost, or that cost per deployed receiver."""
    if view == "cumulative":
        return [_scaled(v) for v in values]
    return [_scaled(v / n) for v, n in zip(values, ns)]


def _window(s, view, n_max):
    """Restrict a series to N <= n_max, prepending N = 0 in the cumulative view.

    At N = 0 nothing is deployed, so the baseline has spent nothing and QV
    transfer has already paid its donor run: that is where the two curves start,
    and drawing it makes the crossing readable.  Per-deployment cost is undefined
    at N = 0, so the origin is only prepended for the cumulative view.
    """
    keep = [i for i, n in enumerate(s["n"]) if n <= n_max]
    ns = [s["n"][i] for i in keep]
    out = {"n": ns}
    for key in ("ours", "base_mean", "base_cheap", "base_costly"):
        out[key] = _viewed([s[key][i] for i in keep], ns, view)
    if view == "cumulative":
        out["n"] = [0] + ns
        out["ours"] = [_scaled(s["donor_cost"])] + out["ours"]
        for key in ("base_mean", "base_cheap", "base_costly"):
            out[key] = [0.0] + out[key]
    return out


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def build_figure(series, args):
    n_rows = math.ceil(len(series) / N_COLS)
    orders = ORDERS[args.orders]
    titles = [
        f"{s['donor']}  —  "
        + " / ".join(f"N* = {s[o['n_star_key']]}" for o in orders)
        for s in series
    ]

    fig = make_subplots(
        rows=n_rows,
        cols=N_COLS,
        subplot_titles=titles,
        shared_xaxes=False,
        shared_yaxes=False,
        horizontal_spacing=0.035,
        vertical_spacing=0.075,
    )

    windows = [_window(s, args.view, args.n_max) for s in series]
    y_max_shared = max(
        max(max(w[o["key"]]) for o in orders) for w in windows
    )

    for i, (s, w) in enumerate(zip(series, windows)):
        row, col = divmod(i, N_COLS)
        row, col = row + 1, col + 1
        first = i == 0
        ns = w["n"]
        ours = w["ours"]

        # One baseline line per requested deployment order.  They are drawn as
        # explicit lines rather than a shaded band: the band read as clutter and
        # its two edges are exactly these curves.
        for o in orders:
            fig.add_trace(
                go.Scatter(
                    x=ns, y=w[o["key"]], mode="lines",
                    line=dict(color=o["color"], width=2, dash="dash"),
                    name=o["label"],
                    legendgroup=o["key"], showlegend=first,
                    hovertemplate=(
                        f"{o['label']}<br>N = %{{x}}<br>%{{y:.3f}} {UNIT_LABEL}<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )
        fig.add_trace(
            go.Scatter(
                x=ns, y=ours, mode="lines",
                line=dict(color=COLOR_OURS, width=2.5),
                name="QV transfer",
                legendgroup="ours", showlegend=first,
                hovertemplate=(
                    f"<b>{s['donor']}</b> donor"
                    f" ({_scaled(s['donor_cost']):.3f} {UNIT_LABEL})"
                    f"<br>N = %{{x}}<br>%{{y:.3f}} {UNIT_LABEL}<extra></extra>"
                ),
            ),
            row=row, col=col,
        )
        for o in orders:
            n = s[o["n_star_key"]]
            if n is None or n > args.n_max:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[n], y=[ours[ns.index(n)]], mode="markers",
                    marker=dict(color=o["color"], size=9, symbol="circle",
                                line=dict(color="white", width=1.5)),
                    name=f"break-even N* ({o['short']})",
                    legendgroup=f"star_{o['key']}", showlegend=first,
                    hovertemplate=(
                        f"<b>{s['donor']}</b><br>{o['label']}"
                        f"<br>break-even N* = {n}<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )

        if args.shared_y:
            y_max = y_max_shared
        elif args.view == "cumulative":
            y_max = Y_HEADROOM * _scaled(s["donor_cost"])
        else:
            # The N = 1 point charges the entire donor run to one receiver and
            # is the tallest thing in the panel; scale to it.
            y_max = 1.12 * max(max(ours), max(max(w[o["key"]]) for o in orders))
        fig.update_yaxes(range=[0, y_max], zeroline=False, row=row, col=col)
        fig.update_xaxes(range=[ns[0], ns[-1]], dtick=2, zeroline=False, row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text=UNIT_LABEL, title_font=dict(size=10), row=row, col=col)
        if row == n_rows or (row == n_rows - 1 and i + N_COLS >= len(series)):
            fig.update_xaxes(title_text="N receivers", title_font=dict(size=10), row=row, col=col)

    view_label = (
        "cumulative cost of deploying N receivers"
        if args.view == "cumulative"
        else "cost per deployed receiver"
    )
    fig.update_layout(
        template="plotly_white",
        width=args.width,
        height=args.height,
        margin=dict(l=60, r=20, t=90, b=50),
        title=dict(
            text=(
                f"{view_label} — each panel uses that dataset as donor,"
                " the other 21 as receivers"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=15),
        ),
        font=dict(size=11),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
            title_text="",
        ),
    )
    fig.update_annotations(font_size=11)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    sizes_path = os.path.join(EVAL_ROOT, args.sizes_name)
    assert os.path.exists(sizes_path), (
        f"{sizes_path} not found. Run collect_dataset_sizes.py first."
    )
    with open(sizes_path) as f:
        sizes = json.load(f)

    costs = _costs(sizes)
    donors = args.donors if args.donors else sorted(costs)
    missing = [d for d in donors if d not in costs]
    assert not missing, f"no cost entry in {args.sizes_name} for: {', '.join(missing)}"

    series = [
        _series(
            d,
            costs[d],
            {k: v for k, v in costs.items() if k != d},
            args.patch_samples,
        )
        for d in donors
    ]
    series.sort(key=lambda s: s["donor_cost"])

    fig = build_figure(series, args)

    os.makedirs(PLOT_ROOT, exist_ok=True)
    stem = (
        f"cost_amortization_grid_{args.view.replace('-', '_')}"
        f"_{args.orders.replace('-', '_')}"
    )
    html_path = os.path.join(PLOT_ROOT, f"{stem}.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"wrote {html_path}")

    if args.pdf:
        pdf_path = os.path.join(PLOT_ROOT, f"{stem}.pdf")
        fig.write_image(pdf_path)
        print(f"wrote {pdf_path}")

    header = (
        f"{'donor':<22}{'cost (M)':>10}{'mean rcv (M)':>14}"
        f"{'N*':>5}{'worst':>7}{'best':>6}"
    )
    print(header)
    print("-" * len(header))
    for s in series:
        print(
            f"{s['donor']:<22}{_scaled(s['donor_cost']):>10.3f}"
            f"{_scaled(s['receiver_cost_mean']):>14.3f}"
            f"{s['n_star']:>5}{s['n_star_cheapest_first']:>7}"
            f"{s['n_star_costliest_first']:>6}"
        )


if __name__ == "__main__":
    main()
