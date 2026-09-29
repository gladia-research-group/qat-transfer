"""Render leave-one-task-out robustness for the primary Spearman coefficient."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import default_plot_root, load_statistics, output_directory, save_figure


def _series(
    records: List[Dict[str, Any]], task_key: str
) -> Tuple[List[str], np.ndarray, List[str]]:
    tasks = []
    values = []
    undefined = []
    for record in records:
        task = str(record[task_key])
        value = record["spearman"]["coefficient"]
        tasks.append(task)
        if value is None:
            values.append(np.nan)
            undefined.append(task)
        else:
            values.append(float(value))
    return tasks, np.asarray(values, dtype=np.float64), undefined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--statistics", required=True, type=Path)
    parser.add_argument("--plot-root", type=Path, default=default_plot_root())
    args = parser.parse_args()

    data = load_statistics(args.statistics)
    primary_name = data["statistics"]["primary_comparison"]
    primary = data["statistics"]["comparisons"][primary_name]
    full = primary["observed"]["spearman"]["coefficient"]
    if full is None:
        raise ValueError("primary full-matrix Spearman coefficient is undefined")
    influence = primary["influence"]
    panels = (
        ("leave_one_receiver_out", "omitted_receiver", "Receiver omitted"),
        ("leave_one_donor_out", "omitted_donor", "Donor omitted"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 7.4), sharex=True, constrained_layout=True)
    for ax, (field, task_key, title) in zip(axes, panels):
        tasks, values, undefined = _series(influence[field], task_key)
        positions = np.arange(len(tasks))
        finite = np.isfinite(values)
        ax.scatter(values[finite], positions[finite], s=34, color="#3f5263", zorder=3)
        ax.axvline(float(full), color="#b23a48", linewidth=1.5, label="All 462 cells")
        ax.axvline(0.0, color="#999999", linewidth=0.8, linestyle="--", zorder=0)
        ax.set_yticks(positions, labels=tasks)
        ax.invert_yaxis()
        ax.set_xlim(-1.0, 1.0)
        ax.set_xlabel(r"Spearman $\rho$")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.2, linewidth=0.6)
        ax.legend(loc="lower right", fontsize=8)
        if undefined:
            ax.text(
                0.02,
                0.02,
                "Undefined: " + ", ".join(undefined),
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=7,
            )

    fig.suptitle(
        "Influence sensitivity of signed cosine versus unit-scale transfer gain\n"
        "Each point recomputes the primary coefficient after omitting one task role",
        fontsize=12,
    )
    output_dir = output_directory(data, Path(__file__).stem, args.plot_root)
    outputs = save_figure(fig, output_dir, "alignment_influence")
    plt.close(fig)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
