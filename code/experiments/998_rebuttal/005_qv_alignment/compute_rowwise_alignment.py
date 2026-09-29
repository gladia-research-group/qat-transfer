"""Compute mean matching-row Euclidean QV alignment.

This is a strict one-variable replication of ``compute_euclidean_alignment``.
Checkpoint selection, QV construction, task order, and selected parameters are
imported from the global-cosine producer.  The sole numerical change is that
each matching output row is normalized independently before all row cosines
are averaged with equal weight.
"""

from __future__ import annotations

import json
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = PROJECT_ROOT / "code"
EXPERIMENT_DIR = Path(__file__).resolve().parent
for search_root in (CODE_ROOT, EXPERIMENT_DIR):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import hydra
import numpy as np
import timm
import torch
from omegaconf import DictConfig, OmegaConf

import compute_euclidean_alignment as global_alignment
from common.run_id import guard_run_config, run_id_path
from common.status import StatusWriter


EXPERIMENT = global_alignment.EXPERIMENT
STAGE = "rowwise_alignment"
SCHEMA_VERSION = "rowwise_alignment_v1"
GOLDEN_ARTIFACT = "rowwise_alignment.json"
AGGREGATION_SPEC = "row_cosine_mean_v1"
RUN_ID_PARAMS = [*global_alignment.RUN_ID_PARAMS, "aggregation_spec"]
TASKS = global_alignment.TASKS
EPOCH_BY_TASK = global_alignment.EPOCH_BY_TASK

# Bound one task's float64 block to roughly eight megabytes; across 22 tasks
# the largest temporary block is therefore roughly 176 MiB.
ROW_COORDINATE_BLOCK_SIZE = 1_000_000


def _resolved_config(cfg: DictConfig) -> Dict[str, Any]:
    value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved Hydra config must be a mapping")
    return value


def _validate_contract(cfg: DictConfig) -> None:
    # This call is the parity lock: every original scientific/configuration
    # requirement remains mandatory, and extra row-wise keys are ignored by
    # the original validator.
    global_alignment._validate_contract(cfg)
    if str(cfg.aggregation_spec) != AGGREGATION_SPEC:
        raise ValueError(
            f"aggregation_spec={cfg.aggregation_spec!r}, expected {AGGREGATION_SPEC!r}"
        )
    if not isinstance(cfg.smoke, bool):
        raise TypeError("smoke must be boolean")


def _mean_matching_row_cosines_float64(
    vectors: np.ndarray,
    parameters: Sequence[Mapping[str, Any]],
    task_names: Sequence[str],
    coordinate_block_size: int = ROW_COORDINATE_BLOCK_SIZE,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return the unweighted mean cosine across all matching output rows."""
    if coordinate_block_size <= 0:
        raise ValueError("coordinate_block_size must be positive")
    if vectors.ndim != 2 or vectors.shape[0] != len(task_names):
        raise ValueError(
            f"vectors shape {vectors.shape} is incompatible with {len(task_names)} tasks"
        )

    n_tasks = len(task_names)
    cosine_sum = np.zeros((n_tasks, n_tasks), dtype=np.float64)
    total_rows = 0
    zero_row_count = np.zeros(n_tasks, dtype=np.int64)
    minimum_norm = np.full(n_tasks, np.inf, dtype=np.float64)
    maximum_norm = np.zeros(n_tasks, dtype=np.float64)
    parameter_rows: Dict[str, int] = {}

    for parameter in parameters:
        key = str(parameter["key"])
        shape = tuple(int(value) for value in parameter["shape"])
        if len(shape) != 2:
            raise ValueError(f"selected weight must be a matrix: {key} shape={shape}")
        n_rows, row_width = shape
        start = int(parameter["offset_start"])
        stop = int(parameter["offset_stop"])
        if stop - start != n_rows * row_width:
            raise ValueError(f"parameter offsets do not match shape for {key}")
        if stop > vectors.shape[1]:
            raise ValueError(f"parameter offsets exceed vector width for {key}")

        parameter_rows[key] = n_rows
        view = vectors[:, start:stop].reshape(n_tasks, n_rows, row_width)
        rows_per_block = max(1, coordinate_block_size // row_width)
        for row_start in range(0, n_rows, rows_per_block):
            row_stop = min(n_rows, row_start + rows_per_block)
            block = np.asarray(
                view[:, row_start:row_stop, :], dtype=np.float64
            ).copy()
            norms = np.linalg.norm(block, axis=2)
            bad = np.argwhere(norms <= 0.0)
            if bad.size:
                examples = []
                for task_index, local_row in bad[:10]:
                    examples.append(
                        {
                            "task": str(task_names[int(task_index)]),
                            "parameter": key,
                            "row": int(row_start + local_row),
                        }
                    )
                    zero_row_count[int(task_index)] += 1
                raise ValueError(
                    "zero-norm QV rows make row cosine undefined: "
                    + json.dumps(examples, sort_keys=True)
                )

            minimum_norm = np.minimum(minimum_norm, np.min(norms, axis=1))
            maximum_norm = np.maximum(maximum_norm, np.max(norms, axis=1))
            block /= norms[:, :, None]
            flattened_normalized_rows = block.reshape(n_tasks, -1)
            cosine_sum += (
                flattened_normalized_rows @ flattened_normalized_rows.T
            )
            total_rows += row_stop - row_start

    if total_rows <= 0:
        raise ValueError("no selected output rows")
    cosine = cosine_sum / float(total_rows)
    diagnostics = {
        "n_rows": total_rows,
        "parameter_rows": parameter_rows,
        "zero_row_count_by_task": {
            str(task): int(count) for task, count in zip(task_names, zero_row_count)
        },
        "minimum_row_norm_by_task": {
            str(task): float(value) for task, value in zip(task_names, minimum_norm)
        },
        "maximum_row_norm_by_task": {
            str(task): float(value) for task, value in zip(task_names, maximum_norm)
        },
        "row_coordinate_block_size": coordinate_block_size,
        "accumulation_dtype": "float64",
        "cosine_symmetry_max_abs": float(np.max(np.abs(cosine - cosine.T))),
        "cosine_diagonal_max_abs_deviation_from_one": float(
            np.max(np.abs(np.diag(cosine) - 1.0))
        ),
        "cosine_min": float(np.min(cosine)),
        "cosine_max": float(np.max(cosine)),
    }
    return cosine, diagnostics


def _run_smoke() -> None:
    # Three tasks and two unequal row groups exercise the real aggregation
    # function without resolving checkpoints or touching evaluation paths.
    layer_a = np.asarray(
        [
            [[1.0, 2.0], [3.0, 1.0]],
            [[2.0, 1.0], [1.0, 4.0]],
            [[-1.0, 2.0], [2.0, 3.0]],
        ],
        dtype=np.float32,
    )
    layer_b = np.asarray(
        [
            [[1.0, 0.0, 2.0]],
            [[2.0, 1.0, 1.0]],
            [[1.0, 3.0, 2.0]],
        ],
        dtype=np.float32,
    )
    vectors = np.concatenate(
        [layer_a.reshape(3, -1), layer_b.reshape(3, -1)], axis=1
    )
    parameters = [
        {"key": "layer_a.weight", "shape": [2, 2], "offset_start": 0, "offset_stop": 4},
        {"key": "layer_b.weight", "shape": [1, 3], "offset_start": 4, "offset_stop": 7},
    ]
    cosine, diagnostics = _mean_matching_row_cosines_float64(
        vectors, parameters, ("task_a", "task_b", "task_c"), coordinate_block_size=3
    )
    if not np.isfinite(cosine).all():
        raise RuntimeError("smoke produced non-finite cosine values")
    if float(np.max(np.abs(cosine - cosine.T))) > 1e-12:
        raise RuntimeError("smoke symmetry check failed")
    if float(np.max(np.abs(np.diag(cosine) - 1.0))) > 1e-12:
        raise RuntimeError("smoke unit-diagonal check failed")
    if diagnostics["n_rows"] != 3:
        raise RuntimeError("smoke row-count check failed")
    print(
        "[smoke] row_cosine_mean_v1 passed: "
        "finite symmetric unit-diagonal 3x3 matrix over 3 matching rows",
        flush=True,
    )


@hydra.main(
    config_path="../../../../config/experiments/998_rebuttal/005_qv_alignment",
    config_name="compute_rowwise_alignment",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    _validate_contract(cfg)
    if bool(cfg.smoke):
        _run_smoke()
        return

    resolved = _resolved_config(cfg)
    eval_dir = (
        PROJECT_ROOT
        / str(cfg.evaluation_root)
        / EXPERIMENT
        / STAGE
        / run_id_path(resolved, RUN_ID_PARAMS)
    )
    guard_run_config(resolved, RUN_ID_PARAMS, eval_dir)
    golden = eval_dir / GOLDEN_ARTIFACT
    if golden.exists():
        raise FileExistsError(
            f"golden artifact already exists; refusing to overwrite: {golden}"
        )

    paths = global_alignment._checkpoint_paths(cfg)
    parameters = global_alignment._selected_parameters(cfg)
    total_parameters = sum(int(parameter["numel"]) for parameter in parameters)
    expected_rows = sum(int(parameter["shape"][0]) for parameter in parameters)

    with StatusWriter(eval_dir) as status:
        provenance: Dict[str, Dict[str, Any]] = {}
        with tempfile.TemporaryDirectory(prefix=".qv_memmap_", dir=eval_dir) as tmp_dir:
            mmap_path = Path(tmp_dir) / "qvs.float32.mmap"
            vectors = np.memmap(
                mmap_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(TASKS), total_parameters),
            )
            for index, task in enumerate(TASKS):
                task_paths = paths[task]
                provenance[task] = {}
                for kind, path in task_paths.items():
                    provenance[task][kind] = {
                        "path": global_alignment._relative_or_absolute(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": global_alignment._sha256(path),
                    }
                global_alignment._write_qv_row(
                    vectors,
                    index,
                    task_paths["fp"],
                    task_paths["qat"],
                    parameters,
                )
                status.heartbeat(progress=f"vectors {index + 1}/{len(TASKS)}")

            cosine, diagnostics = _mean_matching_row_cosines_float64(
                vectors, parameters, TASKS
            )
            del vectors

        if diagnostics["n_rows"] != expected_rows:
            raise RuntimeError(
                f"row count mismatch: {diagnostics['n_rows']} vs {expected_rows}"
            )
        if not np.isfinite(cosine).all():
            raise ValueError("non-finite values in row-wise cosine matrix")
        if diagnostics["cosine_symmetry_max_abs"] > 1e-12:
            raise RuntimeError(f"cosine symmetry check failed: {diagnostics}")
        if diagnostics["cosine_diagonal_max_abs_deviation_from_one"] > 1e-12:
            raise RuntimeError(f"cosine diagonal check failed: {diagnostics}")

        run_path = run_id_path(resolved, RUN_ID_PARAMS)
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "stage": STAGE,
            "run_id_params": RUN_ID_PARAMS,
            "run_id_path": str(run_path),
            "config": resolved,
            "task_order": list(TASKS),
            "task_epochs": EPOCH_BY_TASK,
            "module_selector": {
                "name": cfg.module_selector,
                "semantics": (
                    "sorted nn.Linear module names reached by apply_ptq_ recursion; "
                    "only each module's .weight tensor"
                ),
                "n_modules": len(parameters),
                "n_parameters": total_parameters,
                "n_rows": expected_rows,
                "parameters": parameters,
            },
            "aggregation": {
                "spec": AGGREGATION_SPEC,
                "semantics": (
                    "cosine each matching output row within the same selected module "
                    "and row index, then take the unweighted arithmetic mean over all rows"
                ),
                "n_rows": expected_rows,
                "row_weighting": "uniform",
                "layer_weighting": "by_number_of_rows_only",
                "cosine_sq_downstream": "square_of_aggregated_cosine",
                "zero_row_policy": "error",
            },
            "checkpoint_provenance": provenance,
            "row_cosine_sum_matrix": (cosine * float(expected_rows)).tolist(),
            "cosine_matrix": cosine.tolist(),
            "diagnostics": diagnostics,
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "timm": timm.__version__,
            },
        }
        global_alignment._atomic_json(golden, artifact)
        status.heartbeat(
            progress=f"rows {expected_rows}/{expected_rows}; artifact written"
        )
        print(f"[complete] {global_alignment._relative_or_absolute(golden)}", flush=True)


if __name__ == "__main__":
    main()
