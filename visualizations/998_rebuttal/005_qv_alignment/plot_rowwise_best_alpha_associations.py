"""Render row-wise QV associations at validation-selected best alpha."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _rowwise_common import (
    default_plot_root,
    load_statistics,
    output_directory,
    require_best_alpha_payload,
    save_figure,
)


def _format_stat(value: Any) -> str:
    return "undefined" if value is None else f"{float(value):.3f}"


def _annotation(comparison: Mapping[str, Any]) -> str:
    observed = comparison.get("observed")
    if not isinstance(observed, Mapping):
        raise ValueError("row-wise best-alpha comparison has no observed statistics")
    try:
        spearman = observed["spearman"]["coefficient"]
        pearson = observed["pearson"]["coefficient"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "row-wise best-alpha comparison has incomplete correlations"
        ) from error
    return (
        rf"Spearman $\rho$ = {_format_stat(spearman)}" "\n"
        rf"Pearson $r$ = {_format_stat(pearson)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--plot-root", type=Path, default=default_plot_root())
    args = parser.parse_args()

    data = load_statistics(args.statistics)
    comparisons = require_best_alpha_payload(data)
    points = data["points"]
    panels = (
        {
            "comparison": "signed_cosine_vs_delta_best",
            "x": "cosine",
            "y": "delta_best",
            "xlabel": "Mean signed matching-row Euclidean cosine",
            "ylabel": r"Validation-selected test gain $\Delta_{\mathrm{best}}(D,R)$",
            "title": "Best-scale accuracy association",
        },
        {
            "comparison": "cosine_sq_vs_recovery_best",
            "x": "cosine_sq",
            "y": "recovery_best",
            "xlabel": "Squared mean matching-row Euclidean cosine",
            "ylabel": "Validation-selected test recovery",
            "title": "Best-scale recovery association",
        },
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), constrained_layout=True)
    for ax, panel in zip(axes, panels):
        x = np.asarray([point[panel["x"]] for point in points], dtype=np.float64)
        y = np.asarray([point[panel["y"]] for point in points], dtype=np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"non-finite values in row-wise best-alpha panel {panel}")
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
        "Mean matching-row QV alignment versus validation-selected transfer "
        "across 462 cross-task cells",
        fontsize=12,
    )
    output_dir = output_directory(data, Path(__file__).stem, args.plot_root)
    outputs = save_figure(
        fig,
        output_dir,
        "rowwise_best_alpha_alignment_associations",
    )
    plt.close(fig)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
