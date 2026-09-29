"""Shared, render-only helpers for the 005 Euclidean figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = "998_rebuttal/005_qv_alignment"
SCHEMA_VERSION = "euclidean_statistics_v1"


def load_statistics(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"statistics JSON root must be an object: {path}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"statistics schema={data.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
        )
    if data.get("analysis_spec") != "qv_alignment_v1":
        raise ValueError(f"unsupported analysis spec: {data.get('analysis_spec')!r}")
    if data.get("missing") != []:
        raise ValueError(f"statistics artifact reports missing records: {data.get('missing')}")
    if len(data.get("points", [])) != 462:
        raise ValueError("statistics artifact must contain exactly 462 cross-task points")
    if len(data.get("diagonal_audit", [])) != 22:
        raise ValueError("statistics artifact must contain exactly 22 diagonal records")
    provenance = data.get("provenance")
    if not isinstance(provenance, dict) or len(provenance.get("task_order", [])) != 22:
        raise ValueError("statistics artifact has invalid task-order provenance")
    return data


def _safe_run_path(text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe run-id path in provenance: {text!r}")
    return path


def output_directory(data: Dict[str, Any], script_stem: str, root: Path) -> Path:
    provenance = data["provenance"]
    producer = _safe_run_path(str(provenance["producer_run_id_path"]))
    analyzer = _safe_run_path(str(provenance["analyzer_run_id_path"]))
    return root / EXPERIMENT / script_stem / producer / "analysis" / analyzer


def matrix_from_records(
    data: Dict[str, Any], field: str
) -> Tuple[List[str], "numpy.ndarray"]:
    # Imported lazily so JSON validation stays lightweight.
    import numpy as np

    tasks = list(data["provenance"]["task_order"])
    index = {task: position for position, task in enumerate(tasks)}
    matrix = np.full((len(tasks), len(tasks)), np.nan, dtype=np.float64)
    records = list(data["points"]) + list(data["diagonal_audit"])
    seen = set()
    for record in records:
        key = (record.get("donor"), record.get("receiver"))
        if key in seen:
            raise ValueError(f"duplicate plotted pair: {key}")
        if key[0] not in index or key[1] not in index:
            raise ValueError(f"unknown task in plotted pair: {key}")
        value = record.get(field)
        if not isinstance(value, (int, float)):
            raise ValueError(f"non-numeric {field} for pair {key}: {value!r}")
        matrix[index[key[0]], index[key[1]]] = float(value)
        seen.add(key)
    if len(seen) != len(tasks) ** 2 or not np.isfinite(matrix).all():
        raise ValueError(f"could not construct complete matrix for {field}")
    return tasks, matrix


def matrix_in_001_disposition(
    data: Dict[str, Any], field: str
) -> Tuple[List[str], "numpy.ndarray"]:
    """Return receiver rows and donor columns in the 001 heatmap task order."""
    import numpy as np

    tasks, donor_by_receiver = matrix_from_records(data, field)
    display_tasks = sorted(tasks, key=str.lower)
    positions = [tasks.index(task) for task in display_tasks]
    receiver_by_donor = donor_by_receiver[np.ix_(positions, positions)].T
    return display_tasks, receiver_by_donor


def save_figure(fig: Any, output_dir: Path, basename: str) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, kwargs in (
        ("pdf", {"format": "pdf"}),
        ("png", {"format": "png", "dpi": 300}),
    ):
        path = output_dir / f"{basename}.{suffix}"
        tmp = output_dir / f".{basename}.{suffix}.tmp"
        fig.savefig(tmp, bbox_inches="tight", **kwargs)
        tmp.replace(path)
        outputs.append(path)
    return outputs


def default_plot_root() -> Path:
    return PROJECT_ROOT / "plots"
