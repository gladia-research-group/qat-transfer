"""Plot alignment/outcome correlation as a function of alpha."""

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
    primary = data["statistics"]["primary"]
    alpha = np.asarray(primary["alphas"], dtype=np.float64)
    spearman = primary["qap_profile"]["spearman"]
    pearson = primary["qap_profile"]["pearson"]
    rho = np.asarray([item["observed"] for item in spearman["pointwise"]])
    r_value = np.asarray([item["observed"] for item in pearson["pointwise"]])
    pointwise = np.asarray([item["p_two_sided"] for item in spearman["pointwise"]])
    familywise = np.asarray([item["p_max_abs"] for item in spearman["familywise_max_abs"]])

    fig, (ax, p_ax) = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True, constrained_layout=True)
    ax.plot(alpha, rho, marker="o", color="#26547c", label=r"Spearman $\rho$")
    ax.plot(alpha, r_value, marker="s", color="#ef8354", label=r"Pearson $r$")
    ax.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
    ax.axvline(1.0, color="#999999", linewidth=0.8, linestyle=":", label=r"Unit $\alpha$")
    ax.set_ylabel("Alignment–gain correlation")
    ax.set_ylim(-1.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend()
    global_p = spearman["p_any_alpha_max_abs"]
    ax.set_title(
        "Does Euclidean QV alignment predict transfer across alpha?\n"
        rf"Global max-$|\rho|$ QAP $p={global_p:.3g}$"
    )

    p_ax.plot(alpha, pointwise, marker="o", color="#26547c", label="Pointwise QAP")
    p_ax.plot(alpha, familywise, marker="s", color="#6c757d", label=r"Max-$|\rho|$ QAP")
    p_ax.axhline(0.05, color="#b23a48", linewidth=1.0, linestyle="--", label="0.05")
    p_ax.axvline(1.0, color="#999999", linewidth=0.8, linestyle=":")
    p_ax.set_xlabel(r"QV scale $\alpha$")
    p_ax.set_ylabel("Two-sided p-value")
    p_ax.set_ylim(-0.02, 1.02)
    p_ax.grid(alpha=0.2)
    p_ax.legend()
    outputs = save_figure(
        fig, output_directory(data, Path(__file__).stem, args.plot_root),
        "alpha_correlation_profile",
    )
    plt.close(fig)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
