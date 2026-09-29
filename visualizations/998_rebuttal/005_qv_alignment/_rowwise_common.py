"""Validation and output routing for row-wise 005 figure replicas."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

from _common import (
    PROJECT_ROOT,
    EXPERIMENT,
    _safe_run_path,
    default_plot_root,
    matrix_from_records,
    matrix_in_001_disposition,
    save_figure,
)


SCHEMA_VERSION = "rowwise_statistics_v1"
ANALYSIS_SPEC = "qv_alignment_rowwise_v1"


def load_statistics(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"statistics JSON root must be an object: {path}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"statistics schema={data.get('schema_version')!r}, "
            f"expected {SCHEMA_VERSION!r}"
        )
    if data.get("analysis_spec") != ANALYSIS_SPEC:
        raise ValueError(f"unsupported analysis spec: {data.get('analysis_spec')!r}")
    if data.get("alignment_aggregation") != "row_cosine_mean_v1":
        raise ValueError("statistics artifact is not the approved row-wise aggregation")
    if data.get("missing") != []:
        raise ValueError(
            f"statistics artifact reports missing records: {data.get('missing')}"
        )
    if len(data.get("points", [])) != 462:
        raise ValueError("statistics artifact must contain exactly 462 cross-task points")
    if len(data.get("diagonal_audit", [])) != 22:
        raise ValueError("statistics artifact must contain exactly 22 diagonal records")
    provenance = data.get("provenance")
    if not isinstance(provenance, dict) or len(provenance.get("task_order", [])) != 22:
        raise ValueError("statistics artifact has invalid task-order provenance")
    if provenance.get("producer_stage") != "rowwise_alignment":
        raise ValueError("statistics artifact has invalid producer-stage provenance")
    return data


def output_directory(data: Dict[str, Any], script_stem: str, root: Path) -> Path:
    provenance = data["provenance"]
    producer = _safe_run_path(str(provenance["producer_run_id_path"]))
    analyzer = _safe_run_path(str(provenance["analyzer_run_id_path"]))
    return (
        root
        / EXPERIMENT
        / script_stem
        / "rowwise_alignment"
        / producer
        / "analysis"
        / analyzer
    )


def require_best_alpha_payload(data: Dict[str, Any]) -> Mapping[str, Any]:
    comparisons = data.get("statistics", {}).get("comparisons")
    if not isinstance(comparisons, Mapping):
        raise ValueError("statistics artifact has no comparison mapping")
    required_comparisons = (
        "signed_cosine_vs_delta_best",
        "cosine_sq_vs_recovery_best",
    )
    missing = [name for name in required_comparisons if name not in comparisons]
    if missing:
        raise ValueError(
            f"statistics artifact lacks row-wise best-alpha comparisons: {missing}"
        )
    for index, point in enumerate(data["points"]):
        for field in ("best_alpha", "delta_best", "recovery_best"):
            value = point.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    f"point {index} has invalid row-wise best-alpha field "
                    f"{field}: {value!r}"
                )
    return comparisons


__all__ = [
    "PROJECT_ROOT",
    "default_plot_root",
    "load_statistics",
    "matrix_from_records",
    "matrix_in_001_disposition",
    "output_directory",
    "require_best_alpha_payload",
    "save_figure",
]
