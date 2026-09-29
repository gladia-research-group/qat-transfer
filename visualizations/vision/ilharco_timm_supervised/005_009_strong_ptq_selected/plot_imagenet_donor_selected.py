"""Export ImageNet-donor validation-selected QV results over AWQ and GPTQ."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
from statistics import mean, median
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qat-transfer-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVALUATION_ROOT = PROJECT_ROOT / "evaluations" / "vision" / "ilharco_timm_supervised"
DEFAULT_PLOT_ROOT = PROJECT_ROOT / "plots"
EXPERIMENT = Path("vision/ilharco_timm_supervised/005_009_strong_ptq_selected")
ALPHAS = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00, 1.05, 1.20, 1.35, 1.50)
METHODS = ("awq", "gptq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="vit_base_patch16_224_orig_in21k")
    parser.add_argument("--src", default="ImageNet")
    parser.add_argument("--sseed", type=int, default=2038)
    parser.add_argument("--tseed", type=int, default=2038)
    parser.add_argument("--wave", default="20260802-212527")
    parser.add_argument("--plot-root", type=Path, default=DEFAULT_PLOT_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def method_root(method: str, args: argparse.Namespace) -> Path:
    if method == "awq":
        return (
            EVALUATION_ROOT
            / "009_qat_transfer_awq"
            / "vision"
            / "qv_transfer_awq"
            / f"model={args.model}"
            / f"src={args.src}-seed{args.sseed}"
        )
    return (
        EVALUATION_ROOT
        / "005_qat_transfer_gptq"
        / "vision"
        / "qv_transfer_gptq"
        / args.model
        / f"src={args.src}_seed={args.sseed}"
    )


def metric(method: str, split: str) -> str:
    return f"{split}_accuracy_fp_head_{method}"


def load_results(
    method: str, args: argparse.Namespace
) -> dict[tuple[str, str, float], tuple[dict[str, Any], Path]]:
    root = method_root(method, args)
    records: dict[tuple[str, str, float], tuple[dict[str, Any], Path]] = {}
    for path in sorted(root.rglob("eval_results.json")):
        data = load_json(path)
        if data.get("experiment") not in {"qv_transfer_awq", "qv_transfer_gptq"}:
            continue
        source = data.get("source", {})
        target = data.get("target", {})
        if source.get("dataset_name") != args.src or int(source.get("seed", -1)) != args.sseed:
            continue
        if int(target.get("seed", -1)) != args.tseed:
            continue
        target_name = str(target.get("dataset_name"))
        if target_name == args.src:
            continue
        split = str(data.get("eval_split"))
        if split not in {"val", "test"}:
            continue
        alpha = float(data["qv"]["alpha"])
        if alpha not in ALPHAS:
            continue
        key = (target_name, split, alpha)
        if key in records:
            raise ValueError(f"duplicate {method} result {key}: {path}")
        if not isinstance(data.get(metric(method, split)), (int, float)):
            raise ValueError(f"missing {metric(method, split)} in {path}")
        records[key] = (data, path)
    return records


def baseline_path(method: str, target: str, args: argparse.Namespace) -> Path:
    optim = "optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128"
    if method == "awq":
        quant = "awq=bits=3_gran=channel_skip=head_ncal=4_ngrid=20_clip=True"
        family = "fp_awq"
    else:
        quant = "gptq=bits=3_gran=channel_skip=head_ncal=4_percdamp=0.01_actorder=False"
        family = "fp_gptq"
    return (
        EVALUATION_ROOT
        / "000_baselines"
        / "vision"
        / family
        / args.model
        / target
        / optim
        / quant
        / f"seed={args.tseed}"
        / "eval_results.json"
    )


def selected_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected_targets: set[str] | None = None
    for method in METHODS:
        records = load_results(method, args)
        targets = {target for target, split, _ in records if split == "val"}
        if len(targets) != 21:
            raise ValueError(f"expected 21 {method} validation targets, found {len(targets)}")
        if expected_targets is None:
            expected_targets = targets
        elif targets != expected_targets:
            raise ValueError("AWQ and GPTQ target sets differ")
        for target in sorted(targets, key=str.casefold):
            candidates = []
            for alpha in ALPHAS:
                key = (target, "val", alpha)
                if key not in records:
                    raise FileNotFoundError(f"missing {method} validation cell {key}")
                candidates.append((alpha, float(records[key][0][metric(method, "val")])) )
            selected_alpha, selected_val = max(candidates, key=lambda item: (item[1], -item[0]))
            test_key = (target, "test", selected_alpha)
            alpha1_key = (target, "test", 1.0)
            if test_key not in records or alpha1_key not in records:
                raise FileNotFoundError(f"missing {method} selected/alpha=1 test result for {target}")
            test_data, test_path = records[test_key]
            alpha1_data, alpha1_path = records[alpha1_key]
            status_path = test_path.with_name(".status.json")
            status = load_json(status_path)
            if status.get("state") != "done" or status.get("wave_id") != args.wave:
                raise ValueError(f"invalid selected-test status: {status_path}")
            baseline_file = baseline_path(method, target, args)
            baseline = float(load_json(baseline_file)["test_accuracy"])
            selected_test = float(test_data[metric(method, "test")])
            alpha1_test = float(alpha1_data[metric(method, "test")])
            rows.append(
                {
                    "method": method.upper(),
                    "target": target,
                    "selected_alpha": selected_alpha,
                    "validation_accuracy": selected_val,
                    "baseline_accuracy": baseline,
                    "selected_test_accuracy": selected_test,
                    "selected_delta": selected_test - baseline,
                    "alpha1_test_accuracy": alpha1_test,
                    "alpha1_delta": alpha1_test - baseline,
                    "selected_artifact": str(test_path.relative_to(PROJECT_ROOT)),
                    "alpha1_artifact": str(alpha1_path.relative_to(PROJECT_ROOT)),
                    "baseline_artifact": str(baseline_file.relative_to(PROJECT_ROOT)),
                }
            )
    return rows


def aggregate(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
        "wins": sum(value > 0.0 for value in values),
        "ties": sum(value == 0.0 for value in values),
        "losses": sum(value < 0.0 for value in values),
    }


def aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for method in ("AWQ", "GPTQ"):
        selected = [row for row in rows if row["method"] == method]
        result[method] = {
            "selected_delta": aggregate([float(row["selected_delta"]) for row in selected]),
            "alpha1_delta": aggregate([float(row["alpha1_delta"]) for row in selected]),
            "selected_alpha": aggregate([float(row["selected_alpha"]) for row in selected]),
        }
    return result


def output_directory(args: argparse.Namespace) -> Path:
    return (
        args.plot_root
        / EXPERIMENT
        / Path(__file__).stem
        / f"model={args.model}"
        / f"src={args.src}"
        / f"sseed={args.sseed}"
        / f"tseed={args.tseed}"
        / f"wave={args.wave}"
        / "split=test"
    )


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def matrix(rows: list[dict[str, Any]], key: str) -> tuple[list[str], np.ndarray]:
    targets = sorted({str(row["target"]) for row in rows}, key=str.casefold)
    lookup = {(row["target"], row["method"]): float(row[key]) for row in rows}
    values = np.asarray([[lookup[(target, method)] for method in ("AWQ", "GPTQ")] for target in targets])
    return targets, values


def interactive_heatmap(
    rows: list[dict[str, Any]], key: str, title: str, *, percentage_points: bool
) -> go.Figure:
    targets, values = matrix(rows, key)
    if percentage_points:
        values = 100.0 * values
        limit = float(np.max(np.abs(values)))
        colorscale, zmin, zmax, color_title, fmt = "RdYlGn", -limit, limit, "Δ accuracy (pp)", "+.2f"
    else:
        colorscale, zmin, zmax, color_title, fmt = "Viridis", 0.0, 1.5, "Selected α", ".2f"
    figure = go.Figure(
        go.Heatmap(
            z=values,
            x=["AWQ", "GPTQ"],
            y=targets,
            text=[[format(value, fmt) for value in row] for row in values],
            texttemplate="%{text}",
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            zmid=0.0 if percentage_points else None,
            colorbar={"title": color_title},
            hovertemplate="target=%{y}<br>method=%{x}<br>value=%{z}<extra></extra>",
        )
    )
    figure.update_layout(title=title, template="plotly_white", width=720, height=980)
    figure.update_yaxes(autorange="reversed", title="Receiver dataset")
    return figure


def static_heatmap(
    rows: list[dict[str, Any]], key: str, title: str, *, percentage_points: bool
) -> Any:
    targets, values = matrix(rows, key)
    if percentage_points:
        values = 100.0 * values
        limit = float(np.max(np.abs(values)))
        cmap, vmin, vmax, fmt, label = "RdYlGn", -limit, limit, "+.2f", "Δ accuracy (pp)"
    else:
        cmap, vmin, vmax, fmt, label = "viridis", 0.0, 1.5, ".2f", "Selected α"
    figure, axis = plt.subplots(figsize=(6.6, 10.0))
    image = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            axis.text(column_index, row_index, format(values[row_index, column_index], fmt), ha="center", va="center", fontsize=7)
    axis.set_xticks(np.arange(2), labels=("AWQ", "GPTQ"))
    axis.set_yticks(np.arange(len(targets)), labels=targets)
    axis.set_ylabel("Receiver dataset")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.05, pad=0.04, label=label)
    figure.tight_layout()
    return figure


def interactive_bars(rows: list[dict[str, Any]], title: str) -> go.Figure:
    targets = sorted({str(row["target"]) for row in rows}, key=str.casefold)
    figure = go.Figure()
    for method, color in (("AWQ", "#4c78a8"), ("GPTQ", "#f58518")):
        lookup = {row["target"]: row for row in rows if row["method"] == method}
        figure.add_trace(go.Bar(name=method, orientation="h", y=targets, x=[100.0 * float(lookup[target]["selected_delta"]) for target in targets], marker_color=color))
    figure.add_vline(x=0.0, line_width=1.0, line_color="black")
    figure.update_layout(title=title, template="plotly_white", barmode="group", width=1000, height=1050)
    figure.update_xaxes(title="Selected QV − strong PTQ baseline (percentage points)")
    figure.update_yaxes(autorange="reversed", title="Receiver dataset")
    return figure


def static_bars(rows: list[dict[str, Any]], title: str) -> Any:
    targets = sorted({str(row["target"]) for row in rows}, key=str.casefold)
    positions = np.arange(len(targets), dtype=float)
    figure, axis = plt.subplots(figsize=(10.5, 10.5))
    for index, (method, color) in enumerate((("AWQ", "#4c78a8"), ("GPTQ", "#f58518"))):
        lookup = {row["target"]: row for row in rows if row["method"] == method}
        axis.barh(positions + (index - 0.5) * 0.35, [100.0 * float(lookup[target]["selected_delta"]) for target in targets], height=0.35, label=method, color=color)
    axis.axvline(0.0, linewidth=1.0, color="black")
    axis.set_yticks(positions, labels=targets)
    axis.invert_yaxis()
    axis.set_xlabel("Selected QV − strong PTQ baseline (percentage points)")
    axis.set_ylabel("Receiver dataset")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def export_figure(interactive: go.Figure, static: Any, output_dir: Path, name: str) -> None:
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        path = output_dir / f"{name}.{suffix}"
        temporary = output_dir / f".{name}.{suffix}.tmp"
        static.savefig(temporary, format=suffix, bbox_inches="tight", **kwargs)
        temporary.replace(path)
    plt.close(static)
    temporary_html = output_dir / f".{name}.html.tmp"
    interactive.write_html(temporary_html, include_plotlyjs=True, full_html=True)
    temporary_html.replace(output_dir / f"{name}.html")


def markdown_summary(args: argparse.Namespace, stats: dict[str, Any]) -> str:
    lines = [
        "# ImageNet-donor QV over strong PTQ",
        "",
        f"Wave `{args.wave}`; validation-selected alpha; held-out test; exact ties use the smallest alpha.",
        "",
        "All deltas are accuracy percentage points against the receiver's matching 3-bit strong PTQ baseline.",
        "",
        "| method | protocol | mean | median | min | max | wins | ties | losses |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in ("AWQ", "GPTQ"):
        for key, label in (("selected_delta", "validation-selected"), ("alpha1_delta", "alpha=1")):
            values = stats[method][key]
            lines.append(
                f"| {method} | {label} | {100 * values['mean']:+.3f} | {100 * values['median']:+.3f} | "
                f"{100 * values['min']:+.3f} | {100 * values['max']:+.3f} | {values['wins']}/{values['n']} | "
                f"{values['ties']}/{values['n']} | {values['losses']}/{values['n']} |"
            )
    return "\n".join(lines) + "\n"


def write_index(output_dir: Path) -> None:
    figures = ("selected_delta_heatmap", "selected_alpha_heatmap", "selected_delta_bar")
    links = "\n".join(
        f'<li><a href="{name}.html">{html.escape(name)}</a> (<a href="{name}.png">PNG</a>, <a href="{name}.pdf">PDF</a>, <a href="{name}.svg">SVG</a>)</li>'
        for name in figures
    )
    atomic_text(
        output_dir / "index.html",
        f"<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Strong PTQ selected-alpha exports</title></head><body><h1>Strong PTQ selected-alpha exports</h1><ul>{links}<li><a href=\"summary.csv\">CSV</a></li><li><a href=\"summary.json\">JSON</a></li><li><a href=\"summary.md\">Markdown</a></li></ul></body></html>\n",
    )


def main() -> None:
    args = parse_args()
    rows = selected_rows(args)
    stats = aggregates(rows)
    output_dir = output_directory(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "strong_ptq_selected_v1",
        "wave": args.wave,
        "selection_split": "val",
        "evaluation_split": "test",
        "tie_break": "smallest_alpha",
        "alpha_grid": list(ALPHAS),
        "model": args.model,
        "source": args.src,
        "source_seed": args.sseed,
        "target_seed": args.tseed,
    }
    write_csv(output_dir / "summary.csv", rows)
    atomic_text(output_dir / "summary.json", json.dumps({"metadata": metadata, "aggregates": stats, "rows": rows}, indent=2) + "\n")
    atomic_text(output_dir / "summary.md", markdown_summary(args, stats))
    title = "ImageNet donor: validation-selected QV over receiver strong PTQ"
    export_figure(interactive_heatmap(rows, "selected_delta", title, percentage_points=True), static_heatmap(rows, "selected_delta", title, percentage_points=True), output_dir, "selected_delta_heatmap")
    export_figure(interactive_heatmap(rows, "selected_alpha", "Validation-selected QV magnitude", percentage_points=False), static_heatmap(rows, "selected_alpha", "Validation-selected QV magnitude", percentage_points=False), output_dir, "selected_alpha_heatmap")
    export_figure(interactive_bars(rows, title), static_bars(rows, title), output_dir, "selected_delta_bar")
    write_index(output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
