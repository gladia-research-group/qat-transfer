"""Render the global QV-alignment heatmap for the best-alpha result suite."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from _common import (
    default_plot_root,
    load_statistics,
    matrix_in_001_disposition,
    output_directory,
    save_figure,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--plot-root", type=Path, default=default_plot_root())
    args = parser.parse_args()

    data = load_statistics(args.statistics)
    comparisons = data.get("statistics", {}).get("comparisons", {})
    comparison_name = "signed_cosine_vs_delta_best"
    if comparison_name not in comparisons:
        raise ValueError(f"statistics artifact lacks best-alpha comparison {comparison_name}")
    for index, point in enumerate(data["points"]):
        value = point.get("delta_best")
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            raise ValueError(
                f"point {index} has invalid best-alpha field delta_best: {value!r}"
            )

    tasks, cosine = matrix_in_001_disposition(data, "cosine")
    diagonal_mask = np.eye(len(tasks), dtype=bool)
    off_diagonal = cosine[~diagonal_mask]
    color_limit = float(np.max(np.abs(off_diagonal)))
    if color_limit == 0.0:
        color_limit = 1.0
    displayed = np.ma.array(cosine, mask=diagonal_mask)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#d9d9d9")

    fig, ax = plt.subplots(figsize=(12.2, 10.8))
    image = ax.imshow(
        displayed,
        cmap=cmap,
        vmin=-color_limit,
        vmax=color_limit,
        interpolation="nearest",
        aspect="equal",
    )
    for index in range(len(tasks)):
        ax.add_patch(
            Rectangle(
                (index - 0.5, index - 0.5),
                1,
                1,
                facecolor="#d9d9d9",
                edgecolor="#555555",
                linewidth=0.5,
            )
        )
        ax.text(index, index, "1", ha="center", va="center", fontsize=5.5, color="#333333")

    ax.set_xticks(np.arange(len(tasks)), labels=tasks, rotation=58, ha="right")
    ax.set_yticks(np.arange(len(tasks)), labels=tasks)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.set_xlabel("Donor task (QV dataset)")
    ax.set_ylabel("Receiver task (target dataset)")
    ax.set_title(
        "Global Euclidean alignment used in the validation-selected best-scale analysis\n"
        "Diagonal is identically 1 and excluded from the off-diagonal color range"
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(r"Signed cosine $c_I(D,R)$ (off diagonal)")

    output_dir = output_directory(data, Path(__file__).stem, args.plot_root)
    outputs = save_figure(fig, output_dir, "best_alpha_alignment_heatmap")
    plt.close(fig)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
