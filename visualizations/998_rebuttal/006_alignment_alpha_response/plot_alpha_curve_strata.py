"""Plot median transfer curves stratified by signed cosine quartile."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import default_plot_root, load_statistics, output_directory, save_figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--plot-root", type=Path, default=default_plot_root())
    args = parser.parse_args()
    data = load_statistics(args.statistics)
    strata = data["statistics"]["cosine_quartile_curves"]
    alpha = np.asarray(strata["alphas"], dtype=np.float64)
    colors = ("#5f0f40", "#9a4c95", "#4d908e", "#277da1")
    fig, ax = plt.subplots(figsize=(7.7, 5.2), constrained_layout=True)
    for group, color in zip(strata["groups"], colors):
        ax.plot(
            alpha, group["median_delta"], marker="o", color=color,
            label=f"Cosine Q{group['quartile']} (n={group['n']})",
        )
    ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
    ax.axvline(1.0, color="#999999", linewidth=0.8, linestyle=":", label=r"Unit $\alpha$")
    ax.set_xlabel(r"QV scale $\alpha$")
    ax.set_ylabel("Median validation accuracy gain")
    ax.set_title("Alpha-response curves by signed Euclidean alignment quartile")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    outputs = save_figure(
        fig, output_directory(data, Path(__file__).stem, args.plot_root),
        "alpha_curve_strata",
    )
    plt.close(fig)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
