"""Render the unchanged 005 association panels using row-wise alignment."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _rowwise_common import (
    default_plot_root,
    load_statistics,
    output_directory,
    save_figure,
)


def _format_stat(value: Any) -> str:
    return "undefined" if value is None else f"{float(value):.3f}"


def _annotation(comparison: Dict[str, Any]) -> str:
    spearman = comparison["observed"]["spearman"]["coefficient"]
    pearson = comparison["observed"]["pearson"]["coefficient"]
    p_value = comparison["qap"]["spearman"]["p_two_sided"]
    p_text = "undefined" if p_value is None else f"{float(p_value):.4g}"
    return (
        rf"Spearman $\rho$ = {_format_stat(spearman)}" "\n"
        rf"Pearson $r$ = {_format_stat(pearson)}" "\n"
        rf"QAP $p_{{\rho}}$ = {p_text}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--plot-root", type=Path, default=default_plot_root())
    args = parser.parse_args()

    data = load_statistics(args.statistics)
    points = data["points"]
    comparisons = data["statistics"]["comparisons"]
    panels = (
        {
            "comparison": "signed_cosine_vs_delta",
            "x": "cosine",
            "y": "delta",
            "xlabel": "Mean signed matching-row Euclidean cosine",
            "ylabel": r"Unit-scale test accuracy gain $\Delta(D,R)$",
            "title": "Reviewer-literal association (primary)",
        },
        {
            "comparison": "cosine_sq_vs_recovery_best",
            "x": "cosine_sq",
            "y": "recovery_best",
            "xlabel": "Squared mean matching-row Euclidean cosine",
            "ylabel": "Validation-selected test recovery",
            "title": "Theory-adjacent association (secondary)",
        },
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), constrained_layout=True)
    for ax, panel in zip(axes, panels):
        x = np.asarray([point[panel["x"]] for point in points], dtype=np.float64)
        y = np.asarray([point[panel["y"]] for point in points], dtype=np.float64)
        ax.scatter(x, y, s=18, alpha=0.45, color="#3f5263", edgecolors="none")
        ax.axhline(0.0, color="#999999", linewidth=0.8, linestyle="--", zorder=0)
        if panel["x"] == "cosine":
            ax.axvline(0.0, color="#999999", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_xlabel(panel["xlabel"])
        ax.set_ylabel(panel["ylabel"])
        ax.set_title(panel["title"])
        ax.grid(alpha=0.18, linewidth=0.6)
        ax.text(
            0.03,
            0.97,
            _annotation(comparisons[panel["comparison"]]),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )

    fig.suptitle(
        "QV alignment versus transfer across 462 cross-task cells\n"
        "Cells share donors and receivers; significance uses task-label QAP, not IID tests",
        fontsize=12,
    )
    output_dir = output_directory(data, Path(__file__).stem, args.plot_root)
    outputs = save_figure(fig, output_dir, "rowwise_alignment_associations")
    plt.close(fig)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
