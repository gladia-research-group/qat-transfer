"""998 — Per-model zero-shot reframing table

Renders the per-backbone comparison of unit scaling (lambda=1, data-free)
against validation-selected scaling (lambda*), from the win_loss.json written by
aggregate_win_loss.py.  Columns:

    Model | Win% & Mean at lambda=1 | Win% & Mean at lambda* | Recovery of the
    QAT ceiling at both | median lambda* | % of pairs with lambda* < 1 |
    failures at lambda=1 rescued by rescaling

Writes a LaTeX table to plots/ and prints the same table as plain text.
"""

import argparse
import json
import os
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

os.chdir(_PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT = "evaluations/998_rebuttal/001_zero_shot_reframing"
PLOT_ROOT = "plots/998_rebuttal/001_zero_shot_reframing"


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
    return parser.parse_args()


def _run_dir(args):
    return os.path.join(
        f"seed={args.seed}",
        f"smult={mult_tag(args.source_epoch_mult)}",
        f"tmult={mult_tag(args.target_epoch_mult)}",
        f"qat=bits={args.qat_bits}_gran={args.granularity}",
        f"ptq=bits={args.ptq_bits}_gran={args.granularity}",
        "split=test",
    )


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def _fmt_delta(val, latex):
    if val is None:
        return "--"
    pp = val * 100
    if pp >= 0:
        return f"+{pp:.1f}"
    return f"$-${abs(pp):.1f}" if latex else f"-{abs(pp):.1f}"


def _fmt_pct(val):
    return "--" if val is None else f"{val * 100:.1f}"


def _fmt_ratio(val):
    return "--" if val is None else f"{val:.2f}"


def _fmt_alpha(val):
    return "--" if val is None else f"{val:.2f}"


def _fmt_rescued(alignment):
    n = alignment.get("n_failing_at_unit", 0)
    if not n:
        return "--"
    return f"{alignment['n_recovered_at_best']}/{n}"


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------
COLUMNS = [
    "Win\\%", "Mean", "Win\\%", "Mean", "$\\lambda{=}1$", "$\\lambda^\\star$",
    "Med.\\ $\\lambda^\\star$", "$\\lambda^\\star{<}1$", "Rescued",
]

TEXT_COLUMNS = [
    "win@1", "mean@1", "win@a*", "mean@a*", "rec@1", "rec@a*",
    "med a*", "a*<1", "rescued",
]


def _cells(cross, best, recovery, recovery_best, alpha_best, alignment, latex):
    return [
        _fmt_pct(cross["win_rate"]),
        _fmt_delta(cross["mean"], latex),
        _fmt_pct(best["win_rate"]),
        _fmt_delta(best["mean"], latex),
        _fmt_ratio(recovery["median"]),
        _fmt_ratio(recovery_best["median"]),
        _fmt_alpha(alpha_best.get("median")),
        _fmt_pct(alpha_best.get("frac_below_1")),
        _fmt_rescued(alignment),
    ]


def model_cells(m, latex):
    return _cells(m["cross_task"], m["cross_task_best"], m["recovery"],
                  m["recovery_best"], m["alpha_best"], m["alignment"], latex)


def summary_cells(summary, latex):
    """Win rate and mean are per-model averages; the rest are pooled over pairs."""
    return _cells(summary["cross_task"]["macro"], summary["cross_task_best"]["macro"],
                  summary["recovery"], summary["recovery_best"],
                  summary["alpha_best"], summary["alignment"], latex)


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def format_latex_table(data):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(
        r"\caption{Cross-task QV transfer at unit scaling ($\lambda=1$, requiring no "
        r"receiver data) against validation-selected scaling ($\lambda^\star$). "
        r"Win\% is the proportion of donor--receiver pairs improving on vanilla PTQ; "
        r"Mean is the Top-1 change (p.p.) against the same baseline. Recovery is the "
        r"median fraction of the receiver's own QAT gain that transfer reproduces, the "
        r"quantity Proposition~1 predicts. $\lambda^\star{<}1$ is the proportion of "
        r"pairs whose optimum lies below unit scaling. Rescued counts pairs failing at "
        r"$\lambda=1$ that improve at some positive $\lambda$, which implies "
        r"$\lambda^\star>0$ and therefore a positively aligned donor direction. "
        r"Averages weight each backbone equally for Win\% and Mean, and pool over "
        r"pairs for the remaining columns.}"
    )
    lines.append(r"\label{tab:zero_shot_reframing_per_model}")
    lines.append(r"\begin{tabular}{l cc cc cc ccc}")
    lines.append(r"\toprule")
    lines.append(r"& \multicolumn{2}{c}{$\lambda=1$} & \multicolumn{2}{c}{$\lambda^\star$}"
                 r" & \multicolumn{2}{c}{Recovery} & & & \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    lines.append("Model & " + " & ".join(COLUMNS) + r" \\")

    for family in data["families"].values():
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{10}{l}{\textit{" + family["display_name"] + r"}} \\")
        for m in family["models"]:
            lines.append(f"\\quad {m['display_name']} & "
                         + " & ".join(model_cells(m, latex=True)) + r" \\")
        lines.append(r"\quad \textit{Average} & "
                     + " & ".join(summary_cells(family["summary"], latex=True))
                     + r" \\")

    lines.append(r"\midrule")
    lines.append(r"\textbf{All backbones} & "
                 + " & ".join(summary_cells(data["overall"], latex=True)) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------
def format_text_table(data):
    widths = [22] + [8] * (len(TEXT_COLUMNS) - 1) + [11]

    def row(label, cells):
        out = f"{label:<{widths[0]}}"
        for cell, w in zip(cells, widths[1:]):
            out += f"{cell:>{w}}"
        return out

    lines = [row("Model", TEXT_COLUMNS), "-" * sum(widths)]
    for family in data["families"].values():
        lines.append(f"{family['display_name']}:")
        for m in family["models"]:
            lines.append(row("  " + m["display_name"], model_cells(m, latex=False)))
        lines.append(row("  Average", summary_cells(family["summary"], latex=False)))
    lines.append("-" * sum(widths))
    lines.append(row("All backbones", summary_cells(data["overall"], latex=False)))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    run_dir = _run_dir(args)

    in_path = os.path.join(EVAL_ROOT, run_dir, "win_loss.json")
    if not os.path.exists(in_path):
        print(f"[MISSING] {in_path} — run aggregate_win_loss.py first.", file=sys.stderr)
        sys.exit(1)

    with open(in_path) as f:
        data = json.load(f)

    print(format_text_table(data))

    out_dir = os.path.join(PLOT_ROOT, run_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "win_loss_table.tex")
    with open(out_path, "w") as f:
        f.write(format_latex_table(data))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
