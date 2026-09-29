"""Shared render-only helpers for the 006 alpha-response figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = "998_rebuttal/006_alignment_alpha_response"
SCHEMA_VERSION = "alpha_response_statistics_v1"


def load_statistics(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"statistics JSON root must be an object: {path}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported statistics schema: {data.get('schema_version')!r}")
    if data.get("analysis_spec") != "qv_alignment_alpha_v1":
        raise ValueError(f"unsupported analysis spec: {data.get('analysis_spec')!r}")
    if data.get("missing") != []:
        raise ValueError(f"statistics artifact reports missing records: {data.get('missing')}")
    if len(data.get("points", [])) != 462 or len(data.get("diagonal_audit", [])) != 22:
        raise ValueError("statistics artifact must contain 462 cross-task and 22 diagonal records")
    primary = data.get("statistics", {}).get("primary", {})
    if len(primary.get("alphas", [])) != 11:
        raise ValueError("primary profile must contain the 11 measured alpha values")
    return data


def _safe_run_path(text: str) -> Path:
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe run-id path in provenance: {text!r}")
    return path


def output_directory(data: Dict[str, Any], script_stem: str, root: Path) -> Path:
    run_path = _safe_run_path(str(data["provenance"]["run_id_path"]))
    return root / EXPERIMENT / script_stem / run_path


def save_figure(fig: Any, output_dir: Path, basename: str) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix, options in (("pdf", {"format": "pdf"}), ("png", {"format": "png", "dpi": 300})):
        destination = output_dir / f"{basename}.{suffix}"
        temporary = output_dir / f".{basename}.{suffix}.tmp"
        fig.savefig(temporary, bbox_inches="tight", **options)
        temporary.replace(destination)
        outputs.append(destination)
    return outputs


def default_plot_root() -> Path:
    return PROJECT_ROOT / "plots"
