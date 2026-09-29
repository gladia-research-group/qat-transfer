"""Compare the matched 001 QV+PTQ and 009 QV+AWQ transfer grids."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean, median
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qat-transfer-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EVALUATION_ROOT = PROJECT_ROOT / "evaluations" / "vision" / "ilharco_timm_supervised"
DEFAULT_PLOT_ROOT = PROJECT_ROOT / "plots"
EXPERIMENT = Path("vision/ilharco_timm_supervised/001_009_qat_transfer_comparison")
HEADS = ("fp", "qat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="vit_base_patch16_224.orig_in21k")
    parser.add_argument("--source-seed", type=int, default=2038)
    parser.add_argument("--target-seed", type=int, default=2038)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--ls", type=float, default=0.0)
    parser.add_argument("--wl", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--qat-bits", type=int, default=3)
    parser.add_argument("--ptq-bits", type=int, default=3)
    parser.add_argument("--awq-bits", type=int, default=3)
    parser.add_argument("--granularity", choices=("tensor", "channel"), default="channel")
    parser.add_argument("--skip-modules", nargs="+", default=["head"])
    parser.add_argument("--awq-num-calib-batches", type=int, default=4)
    parser.add_argument("--awq-n-grid", type=int, default=20)
    parser.add_argument(
        "--awq-clip", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--qv-alpha", type=float, default=1.0)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--expected-receivers", type=int, default=22)
    parser.add_argument("--plot-root", type=Path, default=DEFAULT_PLOT_ROOT)
    return parser.parse_args()


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_").replace(".", "_")


def skip_tag(skip_modules: list[str]) -> str:
    return "-".join(sorted(skip_modules)) if skip_modules else "none"


def optim_fragment(args: argparse.Namespace) -> str:
    return (
        f"optim=adamw_lr={args.lr}_wd={args.wd}_ls={args.ls}_wl={args.wl}"
        f"_mgn={args.max_grad_norm}_bs={args.batch_size}"
    )


def qat_fragment(args: argparse.Namespace) -> str:
    return (
        f"qat=bits={args.qat_bits}_gran={args.granularity}"
        f"_skip={skip_tag(args.skip_modules)}"
    )


def ptq_fragment(args: argparse.Namespace) -> str:
    return (
        f"ptq=bits={args.ptq_bits}_gran={args.granularity}"
        f"_skip={skip_tag(args.skip_modules)}"
    )


def awq_fragment(args: argparse.Namespace) -> str:
    return (
        f"awq=b{args.awq_bits}-g{args.granularity}-s{skip_tag(args.skip_modules)}"
        f"-n{args.awq_num_calib_batches}-grid{args.awq_n_grid}"
        f"-clip{int(args.awq_clip)}"
    )


def compact_optim(args: argparse.Namespace) -> str:
    return (
        f"lr{args.lr}-wd{args.wd}-ls{args.ls}-wl{args.wl}"
        f"-mgn{args.max_grad_norm}-bs{args.batch_size}"
    )


def compact_qat(args: argparse.Namespace) -> str:
    return f"b{args.qat_bits}-g{args.granularity}-s{skip_tag(args.skip_modules)}"


def qv_fragment(args: argparse.Namespace) -> str:
    return f"qv=alpha={args.qv_alpha}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def require_equal(actual: Any, expected: Any, label: str, path: Path) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch in {path}: expected {expected!r}, got {actual!r}")


def require_number(data: dict[str, Any], key: str, path: Path) -> float:
    value = data.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"missing or non-finite {key} in {path}")
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{key} outside [0, 1] in {path}: {value}")
    return value


def validate_common(data: dict[str, Any], path: Path, args: argparse.Namespace) -> None:
    require_equal(data.get("model_name"), args.model_name, "model_name", path)
    require_equal(data.get("eval_split"), args.split, "eval_split", path)
    require_equal(data.get("lr"), args.lr, "lr", path)
    require_equal(data.get("wd"), args.wd, "wd", path)
    require_equal(data.get("ls"), args.ls, "ls", path)
    require_equal(data.get("wl"), args.wl, "wl", path)
    require_equal(data.get("max_grad_norm"), args.max_grad_norm, "max_grad_norm", path)
    require_equal(data.get("batch_size"), args.batch_size, "batch_size", path)
    require_equal(data.get("qv", {}).get("alpha"), args.qv_alpha, "qv.alpha", path)
    require_equal(data.get("qat", {}).get("bits"), args.qat_bits, "qat.bits", path)
    require_equal(
        data.get("qat", {}).get("granularity"), args.granularity, "qat.granularity", path
    )
    require_equal(
        sorted(data.get("qat", {}).get("skip_modules", [])),
        sorted(args.skip_modules),
        "qat.skip_modules",
        path,
    )
    source = data.get("source", {})
    target = data.get("target", {})
    require_equal(source.get("seed"), args.source_seed, "source.seed", path)
    require_equal(target.get("seed"), args.target_seed, "target.seed", path)


def validate_quantizer(
    data: dict[str, Any], path: Path, method: str, args: argparse.Namespace
) -> None:
    quantizer = data.get(method, {})
    bits = args.ptq_bits if method == "ptq" else args.awq_bits
    require_equal(quantizer.get("bits"), bits, f"{method}.bits", path)
    require_equal(
        quantizer.get("granularity"), args.granularity, f"{method}.granularity", path
    )
    require_equal(
        sorted(quantizer.get("skip_modules", [])),
        sorted(args.skip_modules),
        f"{method}.skip_modules",
        path,
    )
    if method == "awq":
        require_equal(
            quantizer.get("num_calib_batches"),
            args.awq_num_calib_batches,
            "awq.num_calib_batches",
            path,
        )
        require_equal(quantizer.get("n_grid"), args.awq_n_grid, "awq.n_grid", path)
        require_equal(quantizer.get("clip"), args.awq_clip, "awq.clip", path)


def result_paths(method: str, args: argparse.Namespace) -> list[Path]:
    model = sanitize_model_name(args.model_name)
    if method == "ptq":
        root = EVALUATION_ROOT / "001_qat_transfer" / "vision" / "qv_transfer" / model
        pattern = "/".join(
            (
                f"src=*_seed={args.source_seed}",
                f"tgt=*_seed={args.target_seed}",
                optim_fragment(args),
                qat_fragment(args),
                ptq_fragment(args),
                qv_fragment(args),
                f"split={args.split}",
                "eval_results.json",
            )
        )
    else:
        root = (
            EVALUATION_ROOT
            / "009_qat_transfer_awq"
            / "vision"
            / "qv_transfer_awq"
            / f"model={model}"
        )
        pattern = "/".join(
            (
                f"src=*-seed{args.source_seed}",
                f"tgt=*-seed{args.target_seed}",
                f"optim={compact_optim(args)}",
                f"qat={compact_qat(args)}",
                awq_fragment(args),
                f"qv=a{args.qv_alpha}",
                f"split={args.split}",
                "eval_results.json",
            )
        )
    return sorted(root.glob(pattern))


def load_grid(
    method: str, args: argparse.Namespace
) -> dict[tuple[str, str], tuple[dict[str, Any], Path]]:
    experiment = "qv_transfer" if method == "ptq" else "qv_transfer_awq"
    records: dict[tuple[str, str], tuple[dict[str, Any], Path]] = {}
    for path in result_paths(method, args):
        data = load_json(path)
        require_equal(data.get("experiment"), experiment, "experiment", path)
        validate_common(data, path, args)
        validate_quantizer(data, path, method, args)
        source = str(data["source"]["dataset_name"])
        target = str(data["target"]["dataset_name"])
        key = (source, target)
        if key in records:
            raise ValueError(f"duplicate {method} result {key}: {records[key][1]} and {path}")
        for head in HEADS:
            require_number(data, f"{args.split}_accuracy_{head}_head_{method}", path)
        records[key] = (data, path)
    return records


def validate_grids(
    ptq: dict[tuple[str, str], tuple[dict[str, Any], Path]],
    awq: dict[tuple[str, str], tuple[dict[str, Any], Path]],
    args: argparse.Namespace,
) -> tuple[list[str], list[str], dict[str, list[str]]]:
    """Restrict both grids to the donor rows the AWQ sweep covers completely.

    The 001 PTQ grid is a complete square; the 009 AWQ sweep is a strict subset of
    it, so the comparable region is rectangular (covered donors x all receivers)
    rather than square. A donor whose row is short is dropped whole rather than
    padded, so no reported cell is ever imputed, and every dropped donor is named
    in the return value and in summary.json.
    """
    unmatched = sorted(set(awq) - set(ptq))
    if unmatched:
        raise ValueError(
            f"AWQ cells without a matching PTQ cell: {unmatched[:5]} ({len(unmatched)} total)"
        )
    receivers = {target for _, target in awq}
    if len(receivers) != args.expected_receivers:
        raise ValueError(
            f"expected {args.expected_receivers} receivers, got {len(receivers)}: "
            f"{sorted(receivers, key=str.casefold)}"
        )
    covered = {source for source, _ in awq}
    donors: list[str] = []
    excluded: dict[str, list[str]] = {}
    for donor in sorted(covered, key=str.casefold):
        missing = sorted(
            (receiver for receiver in receivers if (donor, receiver) not in awq),
            key=str.casefold,
        )
        if missing:
            excluded[donor] = missing
        else:
            donors.append(donor)
    for donor in sorted(receivers - covered, key=str.casefold):
        excluded[donor] = sorted(receivers, key=str.casefold)
    if not donors:
        raise ValueError("no donor row is complete over the receiver axis")
    for donor, missing in sorted(excluded.items(), key=lambda item: item[0].casefold()):
        print(
            f"warning: dropping donor {donor} — missing {len(missing)} of "
            f"{len(receivers)} receivers: {', '.join(missing)}"
        )
    kept = {(donor, receiver) for donor in donors for receiver in receivers}
    if not kept <= set(awq):
        raise ValueError("kept keys do not form a complete Cartesian grid")
    for key in sorted(kept):
        ptq_data, ptq_path = ptq[key]
        awq_data, awq_path = awq[key]
        ptq_modules = ptq_data.get("ptq_quantized_modules")
        awq_modules = awq_data.get("awq_quantized_modules")
        if ptq_modules != awq_modules:
            raise ValueError(
                f"quantized module coverage differs for {key}: {ptq_path} vs {awq_path}"
            )
    return donors, sorted(receivers, key=str.casefold), excluded


def matrices(
    head: str,
    donors: list[str],
    receivers: list[str],
    ptq: dict[tuple[str, str], tuple[dict[str, Any], Path]],
    awq: dict[tuple[str, str], tuple[dict[str, Any], Path]],
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ptq_metric = f"{split}_accuracy_{head}_head_ptq"
    awq_metric = f"{split}_accuracy_{head}_head_awq"
    ptq_values = np.asarray(
        [
            [float(ptq[(source, target)][0][ptq_metric]) for source in donors]
            for target in receivers
        ]
    )
    awq_values = np.asarray(
        [
            [float(awq[(source, target)][0][awq_metric]) for source in donors]
            for target in receivers
        ]
    )
    delta_pp = 100.0 * np.subtract(awq_values, ptq_values)
    if not np.isfinite(delta_pp).all():
        raise ValueError("delta matrix contains a non-finite value")
    return ptq_values, awq_values, delta_pp


def aggregate(values: np.ndarray) -> dict[str, float | int]:
    flat = [float(value) for value in values.ravel()]
    return {
        "n": len(flat),
        "mean_pp": mean(flat),
        "median_pp": median(flat),
        "min_pp": min(flat),
        "max_pp": max(flat),
        "wins": sum(value > 0.0 for value in flat),
        "ties": sum(value == 0.0 for value in flat),
        "losses": sum(value < 0.0 for value in flat),
        "positive_rate": sum(value > 0.0 for value in flat) / len(flat),
    }


def cross_task_values(
    values: np.ndarray, donors: list[str], receivers: list[str]
) -> np.ndarray:
    """Drop the same-task cells, which are not transfer results.

    At alpha = 1 a same-task "patch" is algebraically the receiver's own QAT
    checkpoint, so its column in either grid measures the QAT ceiling rather than
    donor-to-receiver transfer. Every transfer statement quoted from this file is
    therefore over the cross-task cells, and reporting them separately keeps that
    number in the artifact instead of leaving it to be re-derived by hand from
    `delta_pp`.
    """
    kept = [
        float(values[row, column])
        for row, receiver in enumerate(receivers)
        for column, donor in enumerate(donors)
        if donor != receiver
    ]
    return np.asarray(kept)


def output_directory(args: argparse.Namespace) -> Path:
    model = sanitize_model_name(args.model_name)
    compact_qat = f"qat=b{args.qat_bits}-g{args.granularity}-s{skip_tag(args.skip_modules)}"
    compact_ptq = f"ptq=b{args.ptq_bits}-g{args.granularity}-s{skip_tag(args.skip_modules)}"
    compact_awq = (
        f"awq=b{args.awq_bits}-g{args.granularity}-s{skip_tag(args.skip_modules)}"
        f"-n{args.awq_num_calib_batches}-grid{args.awq_n_grid}"
        f"-clip{int(args.awq_clip)}"
    )
    return (
        args.plot_root
        / EXPERIMENT
        / Path(__file__).stem
        / f"model={model}"
        / f"sseed={args.source_seed}"
        / f"tseed={args.target_seed}"
        / optim_fragment(args)
        / compact_qat
        / compact_ptq
        / compact_awq
        / f"qv=a{args.qv_alpha}"
        / f"split={args.split}"
    )


def annotate(axis: Any, values: np.ndarray, *, delta: bool) -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = float(values[row, column])
            label = f"{value:+.1f}" if delta else f"{value:.2f}"
            threshold = 0.55 if not delta else 0.45 * float(np.max(np.abs(values)))
            color = "white" if (abs(value) > threshold if delta else value < threshold) else "black"
            axis.text(column, row, label, ha="center", va="center", fontsize=4.1, color=color)


def outline_diagonal(axis: Any, donors: list[str], receivers: list[str]) -> None:
    """Outline the same-task cells, which on a rectangular grid are not a diagonal."""
    for column, donor in enumerate(donors):
        if donor not in receivers:
            continue
        row = receivers.index(donor)
        axis.add_patch(
            Rectangle(
                (column - 0.5, row - 0.5),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=0.9,
            )
        )


def configure_axis(
    axis: Any, donors: list[str], receivers: list[str], *, show_ylabels: bool
) -> None:
    axis.set_xticks(np.arange(len(donors)), labels=donors, rotation=58, ha="right", fontsize=7)
    axis.set_yticks(np.arange(len(receivers)))
    axis.set_yticklabels(receivers if show_ylabels else [], fontsize=7)
    axis.set_xlabel("QV donor dataset")
    if show_ylabels:
        axis.set_ylabel("Target dataset")
    axis.set_xticks(np.arange(-0.5, len(donors), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(receivers), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=0.25)
    axis.tick_params(which="minor", bottom=False, left=False)


def plot_head(
    head: str,
    donors: list[str],
    receivers: list[str],
    ptq_values: np.ndarray,
    awq_values: np.ndarray,
    delta_pp: np.ndarray,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(25.5, 11.5), constrained_layout=True)
    raw_ptq = axes[0].imshow(ptq_values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    axes[1].imshow(awq_values, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
    delta_limit = max(float(np.max(np.abs(delta_pp))), 1e-12)
    raw_delta = axes[2].imshow(
        delta_pp, cmap="RdYlGn", vmin=-delta_limit, vmax=delta_limit, aspect="equal"
    )
    titles = (
        "001: QV transfer + naive PTQ",
        "009: QV transfer + AWQ",
        "009 − 001",
    )
    for index, axis in enumerate(axes):
        axis.set_title(titles[index], fontsize=12)
        configure_axis(axis, donors, receivers, show_ylabels=index == 0)
        outline_diagonal(axis, donors, receivers)
    annotate(axes[0], ptq_values, delta=False)
    annotate(axes[1], awq_values, delta=False)
    annotate(axes[2], delta_pp, delta=True)
    figure.colorbar(raw_ptq, ax=axes[:2], fraction=0.025, pad=0.015, label="Accuracy")
    figure.colorbar(raw_delta, ax=axes[2], fraction=0.045, pad=0.015, label="AWQ − PTQ (pp)")
    head_label = "FP target head" if head == "fp" else "QAT target head"
    figure.suptitle(
        f"QV transfer: AWQ versus naive PTQ — {head_label}\n"
        f"{args.model_name}, {args.qat_bits}-bit QAT vector, α={args.qv_alpha}, "
        f"{args.split} split — {len(donors)} donors × {len(receivers)} receivers "
        f"({len(donors) * len(receivers)} pairs)",
        fontsize=15,
    )
    name = f"heatmap_qv_awq_vs_ptq_{head}_head"
    for suffix, kwargs in (("png", {"dpi": 300}), ("pdf", {})):
        path = output_dir / f"{name}.{suffix}"
        temporary = output_dir / f".{name}.{suffix}.tmp"
        figure.savefig(temporary, format=suffix, bbox_inches="tight", **kwargs)
        temporary.replace(path)
    plt.close(figure)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    ptq = load_grid("ptq", args)
    awq = load_grid("awq", args)
    donors, receivers, excluded = validate_grids(ptq, awq, args)
    output_dir = output_directory(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    heads: dict[str, Any] = {}
    for head in HEADS:
        ptq_values, awq_values, delta_pp = matrices(
            head, donors, receivers, ptq, awq, args.split
        )
        heads[head] = {
            "ptq_accuracy": ptq_values.tolist(),
            "awq_accuracy": awq_values.tolist(),
            "delta_pp": delta_pp.tolist(),
            "aggregate": aggregate(delta_pp),
            "aggregate_cross_task": aggregate(
                cross_task_values(delta_pp, donors, receivers)
            ),
            "aggregate_ptq_accuracy_cross_task": aggregate(
                100.0 * cross_task_values(ptq_values, donors, receivers)
            ),
            "aggregate_awq_accuracy_cross_task": aggregate(
                100.0 * cross_task_values(awq_values, donors, receivers)
            ),
        }
        plot_head(
            head, donors, receivers, ptq_values, awq_values, delta_pp, args, output_dir
        )
    metadata = {
        "schema_version": "qv_awq_vs_ptq_heatmap_v1",
        "formula": "009 QV+AWQ accuracy - 001 QV+PTQ accuracy",
        "model_name": args.model_name,
        "source_seed": args.source_seed,
        "target_seed": args.target_seed,
        "split": args.split,
        "qv_alpha": args.qv_alpha,
        "donors": donors,
        "receivers": receivers,
        "shape": [len(receivers), len(donors)],
        "pairs": len(donors) * len(receivers),
        "cross_task_pairs": sum(
            1 for donor in donors for receiver in receivers if donor != receiver
        ),
        # The 009 AWQ sweep covers every donor only once wave 20260804-131438 has
        # landed; before that the comparison is a rectangular subset of the 22x22
        # grid 001 covers. Any number read off this file is over `pairs` (or over
        # `cross_task_pairs` for the aggregate_*_cross_task blocks), so check both
        # against `excluded_donors` before quoting one.
        "excluded_donors": {
            donor: missing
            for donor, missing in sorted(
                excluded.items(), key=lambda item: item[0].casefold()
            )
        },
        "source_artifact_counts": {"001_qv_ptq": len(ptq), "009_qv_awq": len(awq)},
        "source_roots": {
            "001_qv_ptq": str(
                (EVALUATION_ROOT / "001_qat_transfer" / "vision" / "qv_transfer").relative_to(
                    PROJECT_ROOT
                )
            ),
            "009_qv_awq": str(
                (
                    EVALUATION_ROOT
                    / "009_qat_transfer_awq"
                    / "vision"
                    / "qv_transfer_awq"
                ).relative_to(PROJECT_ROOT)
            ),
        },
        "selectors": {
            "optim": optim_fragment(args),
            "qat": qat_fragment(args),
            "ptq": ptq_fragment(args),
            "awq": awq_fragment(args),
            "qv": qv_fragment(args),
        },
    }
    atomic_text(
        output_dir / "summary.json",
        json.dumps({"metadata": metadata, "heads": heads}, indent=2) + "\n",
    )
    print(output_dir)
    for head in HEADS:
        stats = heads[head]["aggregate"]
        cross = heads[head]["aggregate_cross_task"]
        print(
            f"{head}: all n={stats['n']} mean={stats['mean_pp']:+.3f} pp, "
            f"median={stats['median_pp']:+.3f} pp, "
            f"wins/ties/losses={stats['wins']}/{stats['ties']}/{stats['losses']}"
        )
        print(
            f"{head}: cross-task n={cross['n']} mean={cross['mean_pp']:+.3f} pp, "
            f"median={cross['median_pp']:+.3f} pp, "
            f"positive={100.0 * cross['positive_rate']:.1f}%"
        )


if __name__ == "__main__":
    main()
