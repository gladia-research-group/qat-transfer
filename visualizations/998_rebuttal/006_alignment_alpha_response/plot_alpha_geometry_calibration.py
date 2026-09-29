"""Plot Euclidean predicted alpha against tie-aware empirical grid optima."""

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
    points = data["points"]
    raw = np.asarray([point["alpha_predicted_raw"] for point in points])
    clipped = np.asarray([point["alpha_predicted_clipped"] for point in points])
    empirical = np.asarray([point["alpha_best_midpoint"] for point in points])
    low = np.asarray([point["alpha_best_low"] for point in points])
    high = np.asarray([point["alpha_best_high"] for point in points])
    errors = np.vstack([empirical - low, high - empirical])
    scale = data["statistics"]["scale_calibration"]["qap"]["clipped"]
    rho = scale["spearman"]["observed"]
    p_value = scale["spearman"]["p_two_sided"]
    cosine_sq = np.asarray([point["cosine_sq"] for point in points])
    recovery = np.asarray([point["recovery_best_grid"] for point in points])
    secondary = data["statistics"]["secondary"]["qap"]["spearman"]

    fig, axes = plt.subplots(2, 2, figsize=(11.3, 9.0), constrained_layout=True)
    raw_ax, clipped_ax, full_ax, zoom_ax = axes.flat
    raw_ax.scatter(raw, empirical, s=17, alpha=0.42, color="#3f5263", edgecolors="none")
    raw_ax.axhline(0.0, color="#999999", linewidth=0.8)
    raw_ax.axhline(1.5, color="#999999", linewidth=0.8)
    raw_ax.set_xlabel(r"Raw Euclidean $\hat\alpha=\langle\rho_D,\rho_R\rangle/\|\rho_D\|^2$")
    raw_ax.set_ylabel("Empirical best-alpha tie midpoint")
    raw_ax.set_title("Raw scale prediction (all points)")
    raw_ax.grid(alpha=0.18)

    clipped_ax.errorbar(
        clipped, empirical, yerr=errors, fmt="o", markersize=3.2, alpha=0.35,
        color="#26547c", ecolor="#8aa8bd", elinewidth=0.6, capsize=0,
    )
    clipped_ax.plot([0, 1.5], [0, 1.5], color="#b23a48", linewidth=1.0, linestyle="--")
    clipped_ax.set_xlim(-0.04, 1.54)
    clipped_ax.set_ylim(-0.04, 1.54)
    clipped_ax.set_xlabel("Euclidean predicted alpha, clipped to grid")
    clipped_ax.set_ylabel("Empirical optimum (interval if tied)")
    clipped_ax.set_title(rf"Grid calibration: $\rho$={rho:.3f}, QAP $p$={p_value:.3g}")
    clipped_ax.grid(alpha=0.18)

    full_ax.scatter(cosine_sq, recovery, s=17, alpha=0.42, color="#3f5263", edgecolors="none")
    full_ax.axhline(0.0, color="#999999", linewidth=0.8, linestyle="--")
    full_ax.set_xlabel(r"Squared Euclidean cosine $c_I^2$")
    full_ax.set_ylabel("Grid-best validation recovery")
    full_ax.set_title(
        rf"Theory-adjacent, full range: $\rho$={secondary['observed']:.3f}, "
        rf"QAP $p$={secondary['p_two_sided']:.3g}"
    )
    full_ax.grid(alpha=0.18)

    lower, upper = np.quantile(recovery, [0.01, 0.99])
    zoom_ax.scatter(cosine_sq, recovery, s=17, alpha=0.42, color="#26547c", edgecolors="none")
    zoom_ax.axhline(0.0, color="#999999", linewidth=0.8, linestyle="--")
    if upper > lower:
        margin = 0.05 * (upper - lower)
        zoom_ax.set_ylim(lower - margin, upper + margin)
    zoom_ax.set_xlabel(r"Squared Euclidean cosine $c_I^2$")
    zoom_ax.set_ylabel("Grid-best validation recovery")
    zoom_ax.set_title("Theory-adjacent bulk zoom (1st–99th percentiles)")
    zoom_ax.grid(alpha=0.18)
    outputs = save_figure(
        fig, output_directory(data, Path(__file__).stem, args.plot_root),
        "alpha_geometry_calibration",
    )
    plt.close(fig)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
