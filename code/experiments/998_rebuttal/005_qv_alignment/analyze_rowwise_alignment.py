"""Run qv_alignment_v1 unchanged on mean matching-row QV cosine."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = PROJECT_ROOT / "code"
EXPERIMENT_DIR = Path(__file__).resolve().parent
for search_root in (CODE_ROOT, EXPERIMENT_DIR):
    if str(search_root) not in sys.path:
        sys.path.insert(0, str(search_root))

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import analyze_euclidean_alignment as global_analysis
import compute_rowwise_alignment as rowwise_producer
from common.run_id import guard_run_config, run_id_path
from common.status import StatusWriter


EXPERIMENT = global_analysis.EXPERIMENT
PRODUCER_STAGE = rowwise_producer.STAGE
PRODUCER_SCHEMA = rowwise_producer.SCHEMA_VERSION
PRODUCER_ARTIFACT = rowwise_producer.GOLDEN_ARTIFACT
SCHEMA_VERSION = "rowwise_statistics_v1"
GOLDEN_ARTIFACT = "rowwise_statistics.json"
ANALYSIS_SPEC = "qv_alignment_rowwise_v1"
PRODUCER_RUN_ID_PARAMS = rowwise_producer.RUN_ID_PARAMS
RUN_ID_PARAMS = list(global_analysis.RUN_ID_PARAMS)
TASKS = global_analysis.TASKS
COMPARISONS = global_analysis.COMPARISONS


def _resolved_config(cfg: DictConfig) -> Dict[str, Any]:
    value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved Hydra config must be a mapping")
    return value


def _validate_contract(cfg: DictConfig) -> None:
    # Replace only the provenance token while passing the complete config
    # through the original analyzer's contract validator.  This locks every
    # substantive analyzer option to qv_alignment_v1.
    compatibility = OmegaConf.create(_resolved_config(cfg))
    compatibility.analysis_spec = "qv_alignment_v1"
    global_analysis._validate_contract(compatibility)
    if str(cfg.analysis_spec) != ANALYSIS_SPEC:
        raise ValueError(
            f"analysis_spec={cfg.analysis_spec!r}, expected {ANALYSIS_SPEC!r}"
        )
    if str(cfg.aggregation_spec) != rowwise_producer.AGGREGATION_SPEC:
        raise ValueError(
            f"aggregation_spec={cfg.aggregation_spec!r}, expected "
            f"{rowwise_producer.AGGREGATION_SPEC!r}"
        )


def _producer_artifact_path(
    cfg: DictConfig, resolved: Mapping[str, Any]
) -> Path:
    return (
        PROJECT_ROOT
        / str(cfg.evaluation_root)
        / EXPERIMENT
        / PRODUCER_STAGE
        / run_id_path(resolved, PRODUCER_RUN_ID_PARAMS)
        / PRODUCER_ARTIFACT
    )


def _validate_alignment(data: Mapping[str, Any], cfg: DictConfig) -> np.ndarray:
    if data.get("schema_version") != PRODUCER_SCHEMA:
        raise ValueError(
            f"alignment schema={data.get('schema_version')!r}, "
            f"expected {PRODUCER_SCHEMA!r}"
        )
    if data.get("stage") != PRODUCER_STAGE:
        raise ValueError(f"alignment stage mismatch: {data.get('stage')!r}")
    if tuple(data.get("task_order", [])) != TASKS:
        raise ValueError("alignment task order differs from the locked 22-task order")
    config = data.get("config")
    if not isinstance(config, Mapping):
        raise TypeError("alignment artifact has no config mapping")
    expected = {key: _resolved_config(cfg)[key] for key in PRODUCER_RUN_ID_PARAMS}
    mismatches = {
        key: {"observed": config.get(key), "expected": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError("alignment metadata mismatch: " + json.dumps(mismatches))

    aggregation = data.get("aggregation")
    if not isinstance(aggregation, Mapping):
        raise TypeError("alignment artifact has no aggregation mapping")
    expected_aggregation = {
        "spec": rowwise_producer.AGGREGATION_SPEC,
        "n_rows": 82_944,
        "row_weighting": "uniform",
        "cosine_sq_downstream": "square_of_aggregated_cosine",
        "zero_row_policy": "error",
    }
    aggregation_mismatches = {
        key: {"observed": aggregation.get(key), "expected": value}
        for key, value in expected_aggregation.items()
        if aggregation.get(key) != value
    }
    if aggregation_mismatches:
        raise ValueError(
            "row aggregation mismatch: " + json.dumps(aggregation_mismatches)
        )
    diagnostics = data.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise TypeError("alignment artifact has no diagnostics mapping")
    zero_counts = diagnostics.get("zero_row_count_by_task")
    if not isinstance(zero_counts, Mapping) or any(
        int(zero_counts.get(task, -1)) != 0 for task in TASKS
    ):
        raise ValueError("alignment artifact does not prove zero valid-row omissions")

    cosine = np.asarray(data.get("cosine_matrix"), dtype=np.float64)
    if cosine.shape != (len(TASKS), len(TASKS)):
        raise ValueError(f"alignment matrix has shape {cosine.shape}, expected (22, 22)")
    if not np.isfinite(cosine).all():
        raise ValueError("alignment matrix contains non-finite values")
    if float(np.max(np.abs(cosine - cosine.T))) > 1e-12:
        raise ValueError("row-wise alignment matrix is not symmetric")
    if float(np.max(np.abs(np.diag(cosine) - 1.0))) > 1e-12:
        raise ValueError("row-wise alignment diagonal is not one")
    return cosine


def _joined_record(
    donor: str,
    receiver: str,
    cosine_value: float,
    pair: Mapping[str, Any],
    unit_alpha: float,
) -> Dict[str, Any]:
    return {
        "donor": donor,
        "receiver": receiver,
        "cosine": cosine_value,
        # Exactly the original downstream rule: square the one aggregated
        # similarity, never average squared row cosines.
        "cosine_sq": cosine_value * cosine_value,
        "delta": float(pair["delta"]),
        "delta_best": float(pair["delta_best"]),
        "recovery": float(pair["recovery"]),
        "recovery_best": float(pair["recovery_best"]),
        "unit_alpha": unit_alpha,
        "best_alpha": float(pair["alpha_best"]),
        "baseline_acc": float(pair["baseline_acc"]),
        "ceiling_delta": float(pair["ceiling_delta"]),
    }


@hydra.main(
    config_path="../../../../config/experiments/998_rebuttal/005_qv_alignment",
    config_name="analyze_rowwise_alignment",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    _validate_contract(cfg)
    resolved = _resolved_config(cfg)
    alignment_path = _producer_artifact_path(cfg, resolved)
    outcome_path = global_analysis._outcome_path(cfg)
    missing_sources = [
        str(path) for path in (alignment_path, outcome_path) if not path.is_file()
    ]
    if missing_sources:
        raise FileNotFoundError(
            "missing analyzer source artifacts:\n" + "\n".join(missing_sources)
        )

    producer_path = run_id_path(resolved, PRODUCER_RUN_ID_PARAMS)
    analyzer_path = run_id_path(resolved, RUN_ID_PARAMS)
    eval_dir = (
        PROJECT_ROOT
        / str(cfg.evaluation_root)
        / EXPERIMENT
        / PRODUCER_STAGE
        / producer_path
        / "analysis"
        / analyzer_path
    )
    guard_run_config(resolved, RUN_ID_PARAMS, eval_dir)
    golden = eval_dir / GOLDEN_ARTIFACT
    if golden.exists():
        raise FileExistsError(
            f"golden artifact already exists; refusing to overwrite: {golden}"
        )

    with StatusWriter(eval_dir) as status:
        alignment_data = global_analysis._load_json(alignment_path)
        outcome_data = global_analysis._load_json(outcome_path)
        cosine = _validate_alignment(alignment_data, cfg)
        outcome_by_key = global_analysis._validate_outcome(outcome_data, cfg)
        status.heartbeat(progress="sources validated")

        task_index = {task: index for index, task in enumerate(TASKS)}
        points = []
        diagonal = []
        all_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for donor in TASKS:
            for receiver in TASKS:
                pair = outcome_by_key[(donor, receiver)]
                cosine_value = float(cosine[task_index[donor], task_index[receiver]])
                record = _joined_record(
                    donor,
                    receiver,
                    cosine_value,
                    pair,
                    float(cfg.unit_alpha),
                )
                all_by_key[(donor, receiver)] = record
                (diagonal if donor == receiver else points).append(record)
        if len(points) != 462 or len(diagonal) != 22 or len(all_by_key) != 484:
            raise RuntimeError(
                f"joined population invariant failed: points={len(points)}, "
                f"diagonal={len(diagonal)}, all={len(all_by_key)}"
            )
        status.heartbeat(progress="joined cells 484/484")

        rng = np.random.default_rng(int(cfg.permutation_seed))
        permutations = np.stack(
            [rng.permutation(len(TASKS)) for _ in range(int(cfg.n_permutations))]
        ).astype(np.uint8, copy=False)
        permutation_digest = hashlib.sha256(permutations.tobytes()).hexdigest()
        mask = ~np.eye(len(TASKS), dtype=bool)
        matrices = {
            "cosine": cosine,
            "cosine_sq": cosine * cosine,
            "delta": global_analysis._matrix_from_points(all_by_key, "delta"),
            "delta_best": global_analysis._matrix_from_points(all_by_key, "delta_best"),
            "recovery": global_analysis._matrix_from_points(all_by_key, "recovery"),
            "recovery_best": global_analysis._matrix_from_points(
                all_by_key, "recovery_best"
            ),
        }

        comparison_results: Dict[str, Any] = {}
        for comparison_index, spec in enumerate(COMPARISONS):
            name = str(spec["name"])
            x_field = str(spec["x_field"])
            y_field = str(spec["y_field"])
            observed = global_analysis._subset_correlations(points, x_field, y_field)
            qap = {
                method: global_analysis._qap_result(
                    matrices[x_field],
                    matrices[y_field],
                    mask,
                    permutations,
                    method,
                    int(cfg.permutation_seed),
                    permutation_digest,
                )
                for method in ("spearman", "pearson")
            }
            for method in ("spearman", "pearson"):
                coefficient = observed[method]["coefficient"]
                qap_observed = qap[method]["observed"]
                if coefficient is None or qap_observed is None:
                    if coefficient != qap_observed:
                        raise RuntimeError(
                            f"observed/QAP null mismatch for {name}/{method}"
                        )
                elif not math.isclose(
                    float(coefficient),
                    float(qap_observed),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise RuntimeError(
                        f"observed/QAP coefficient mismatch for {name}/{method}: "
                        f"{coefficient} vs {qap_observed}"
                    )
            comparison_results[name] = {
                "role": spec["role"],
                "x_field": x_field,
                "y_field": y_field,
                "observed": observed,
                "qap": qap,
                "influence": global_analysis._influence(points, x_field, y_field),
            }
            status.heartbeat(
                progress=f"comparisons {comparison_index + 1}/{len(COMPARISONS)}"
            )

        artifact = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "analysis_spec": cfg.analysis_spec,
            "alignment_aggregation": rowwise_producer.AGGREGATION_SPEC,
            "provenance": {
                "resolved_config": resolved,
                "producer_stage": PRODUCER_STAGE,
                "producer_run_id_params": PRODUCER_RUN_ID_PARAMS,
                "producer_run_id_path": str(producer_path),
                "analyzer_run_id_params": RUN_ID_PARAMS,
                "analyzer_run_id_path": str(analyzer_path),
                "alignment_source": {
                    "path": global_analysis._relative_or_absolute(alignment_path),
                    "sha256": global_analysis._sha256(alignment_path),
                    "schema_version": alignment_data.get("schema_version"),
                },
                "outcome_source": {
                    "path": global_analysis._relative_or_absolute(outcome_path),
                    "sha256": global_analysis._sha256(outcome_path),
                },
                "task_order": list(TASKS),
                "software": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                },
            },
            "points": points,
            "diagonal_audit": diagonal,
            "statistics": {
                "primary_comparison": "signed_cosine_vs_delta",
                "primary_coefficient": "spearman",
                "qap": {
                    "method": "simultaneous donor/receiver task-label permutation",
                    "outcome_matrix_fixed": True,
                    "shared_permutations_across_comparisons": True,
                    "n_permutations": int(cfg.n_permutations),
                    "seed": int(cfg.permutation_seed),
                    "generator": "numpy.random.Generator(PCG64)",
                    "permutation_digest_sha256": permutation_digest,
                    "two_sided_monte_carlo_formula": (
                        "(1 + count(abs(null) >= abs(observed))) / "
                        "(n_permutations + 1)"
                    ),
                },
                "comparisons": comparison_results,
            },
            "missing": [],
        }
        global_analysis._atomic_json(golden, artifact)
        status.heartbeat(progress="comparisons 4/4; artifact written")
        print(f"[complete] {global_analysis._relative_or_absolute(golden)}", flush=True)


if __name__ == "__main__":
    main()
