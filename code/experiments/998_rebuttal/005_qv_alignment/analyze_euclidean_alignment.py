"""Join Euclidean QV geometry to transfer outcomes and run qv_alignment_v1."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
os.chdir(PROJECT_ROOT)

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from common.run_id import guard_run_config, run_id_path
from common.status import StatusWriter


EXPERIMENT = "998_rebuttal/005_qv_alignment"
PRODUCER_STAGE = "euclidean_alignment"
PRODUCER_SCHEMA = "euclidean_alignment_v1"
SCHEMA_VERSION = "euclidean_statistics_v1"
GOLDEN_ARTIFACT = "euclidean_statistics.json"
PRODUCER_RUN_ID_PARAMS = [
    "family",
    "model_name",
    "seed",
    "optim",
    "lr",
    "wd",
    "ls",
    "wl",
    "max_grad_norm",
    "batch_size",
    "qat_bits",
    "qat_granularity",
    "qat_skip_modules",
    "ptq_skip_modules",
    "checkpoint_kind",
    "epoch_policy",
    "vector_scope",
    "module_selector",
    "accumulation_dtype",
]
RUN_ID_PARAMS = [
    "ptq_bits",
    "ptq_granularity",
    "outcome_protocol",
    "outcome_split",
    "unit_alpha",
    "analysis_spec",
    "n_permutations",
    "permutation_seed",
]
TASKS: Tuple[str, ...] = (
    "CIFAR10",
    "CIFAR100",
    "Cars",
    "DTD",
    "EMNIST",
    "EuroSAT",
    "FER2013",
    "FashionMNIST",
    "Flowers102",
    "Food101",
    "GTSRB",
    "ImageNet",
    "KMNIST",
    "MNIST",
    "OxfordIIITPet",
    "PCAM",
    "RESISC45",
    "RenderedSST2",
    "STL10",
    "SUN397",
    "SVHN",
    "TinyImageNet",
)

COMPARISONS = (
    {
        "name": "signed_cosine_vs_delta",
        "role": "primary",
        "x_field": "cosine",
        "y_field": "delta",
    },
    {
        "name": "cosine_sq_vs_recovery_best",
        "role": "theory_adjacent_secondary",
        "x_field": "cosine_sq",
        "y_field": "recovery_best",
    },
    {
        "name": "signed_cosine_vs_delta_best",
        "role": "descriptive_diagnostic",
        "x_field": "cosine",
        "y_field": "delta_best",
    },
    {
        "name": "cosine_sq_vs_recovery",
        "role": "descriptive_diagnostic",
        "x_field": "cosine_sq",
        "y_field": "recovery",
    },
)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    tmp.replace(path)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _resolved_config(cfg: DictConfig) -> Dict[str, Any]:
    value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved Hydra config must be a mapping")
    return value


def _validate_contract(cfg: DictConfig) -> None:
    expected = {
        "family": "ilharco_timm_supervised",
        "model_name": "vit_base_patch16_224.orig_in21k",
        "seed": 2038,
        "optim": "adamw",
        "lr": 1.0e-5,
        "wd": 0.1,
        "ls": 0.0,
        "wl": 500,
        "max_grad_norm": 1.0,
        "batch_size": 128,
        "qat_bits": 3,
        "qat_granularity": "channel",
        "checkpoint_kind": "classifier",
        "epoch_policy": "dataset_final",
        "vector_scope": "quantized_linear_weight",
        "module_selector": "apply_ptq_linear_v1",
        "accumulation_dtype": "float64",
        "ptq_bits": 3,
        "ptq_granularity": "channel",
        "outcome_protocol": "full_qv",
        "outcome_split": "test",
        "unit_alpha": 1.0,
        "analysis_spec": "qv_alignment_v1",
        "n_permutations": 10_000,
        "permutation_seed": 2038,
        "use_wandb": False,
    }
    resolved = _resolved_config(cfg)
    mismatches = {
        key: {"observed": resolved.get(key), "expected": value}
        for key, value in expected.items()
        if resolved.get(key) != value
    }
    for key in ("qat_skip_modules", "ptq_skip_modules"):
        observed = list(resolved.get(key, []))
        if observed != ["head"]:
            mismatches[key] = {"observed": observed, "expected": ["head"]}
    if mismatches:
        raise ValueError(
            "configuration violates qv_alignment_v1: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _producer_artifact_path(cfg: DictConfig, resolved: Mapping[str, Any]) -> Path:
    return (
        PROJECT_ROOT
        / str(cfg.evaluation_root)
        / EXPERIMENT
        / PRODUCER_STAGE
        / run_id_path(resolved, PRODUCER_RUN_ID_PARAMS)
        / "euclidean_alignment.json"
    )


def _outcome_path(cfg: DictConfig) -> Path:
    path = Path(str(cfg.outcome_path))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _validate_alignment(data: Mapping[str, Any], cfg: DictConfig) -> np.ndarray:
    if data.get("schema_version") != PRODUCER_SCHEMA:
        raise ValueError(
            f"alignment schema={data.get('schema_version')!r}, expected {PRODUCER_SCHEMA!r}"
        )
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
    cosine = np.asarray(data.get("cosine_matrix"), dtype=np.float64)
    if cosine.shape != (len(TASKS), len(TASKS)):
        raise ValueError(f"alignment matrix has shape {cosine.shape}, expected (22, 22)")
    if not np.isfinite(cosine).all():
        raise ValueError("alignment matrix contains non-finite values")
    if float(np.max(np.abs(cosine - cosine.T))) > 1e-12:
        raise ValueError("Euclidean alignment matrix is not symmetric")
    if float(np.max(np.abs(np.diag(cosine) - 1.0))) > 1e-12:
        raise ValueError("Euclidean alignment diagonal is not one")
    return cosine


def _validate_outcome(
    data: Mapping[str, Any], cfg: DictConfig
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if data.get("modality") != "vision":
        raise ValueError(f"outcome modality mismatch: {data.get('modality')!r}")
    if data.get("family") != cfg.family:
        raise ValueError(f"outcome family mismatch: {data.get('family')!r}")
    if tuple(data.get("datasets", [])) != TASKS:
        raise ValueError("outcome dataset order differs from the locked 22-task order")
    source_config = data.get("config")
    if not isinstance(source_config, Mapping):
        raise TypeError("outcome source has no config mapping")
    expected_config = {
        "seed": cfg.seed,
        "optim": cfg.optim,
        "lr": cfg.lr,
        "wd": cfg.wd,
        "ls": cfg.ls,
        "wl": cfg.wl,
        "max_grad_norm": cfg.max_grad_norm,
        "qat_bits": cfg.qat_bits,
        "ptq_bits": cfg.ptq_bits,
        "granularity": cfg.ptq_granularity,
        "skip_modules": list(cfg.ptq_skip_modules),
        "alpha": cfg.unit_alpha,
        "eval_split": cfg.outcome_split,
        "metric_key": "test_accuracy_fp_head_ptq",
        "best_alpha_file": "best_alpha_fp_head_ptq.json",
        "best_alpha_key": "val_accuracy_fp_head_ptq",
        "baseline": "fp_ptq",
    }
    mismatches = {
        key: {"observed": source_config.get(key), "expected": value}
        for key, value in expected_config.items()
        if source_config.get(key) != value
    }
    if mismatches:
        raise ValueError("outcome config mismatch: " + json.dumps(mismatches))

    models = data.get("models")
    if not isinstance(models, Mapping) or cfg.model_name not in models:
        raise KeyError(f"outcome source has no model {cfg.model_name!r}")
    model = models[cfg.model_name]
    if not isinstance(model, Mapping):
        raise TypeError("outcome model entry is not a mapping")
    if model.get("n_datasets") != 22 or model.get("n_cells_expected") != 484:
        raise ValueError("outcome model does not declare the complete 22x22 matrix")
    if model.get("missing") != []:
        raise ValueError(f"outcome model reports missing cells: {model.get('missing')}")
    if model.get("batch_size") != cfg.batch_size:
        raise ValueError("outcome model batch-size metadata mismatch")
    if model.get("skip_modules") != list(cfg.ptq_skip_modules):
        raise ValueError("outcome model skip-module metadata mismatch")

    raw_pairs = model.get("pairs")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != 484:
        raise ValueError("outcome source must contain exactly 484 pair records")
    required_numeric = (
        "baseline_acc",
        "transfer_acc",
        "delta",
        "alpha_best",
        "transfer_acc_best",
        "delta_best",
        "ceiling_delta",
        "recovery",
        "recovery_best",
    )
    pair_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for raw in raw_pairs:
        if not isinstance(raw, Mapping):
            raise TypeError("outcome pair record is not a mapping")
        donor, receiver = raw.get("donor"), raw.get("receiver")
        if donor not in TASKS or receiver not in TASKS:
            raise ValueError(f"outcome pair has unknown task: {(donor, receiver)}")
        key = (str(donor), str(receiver))
        if key in pair_by_key:
            raise ValueError(f"duplicate outcome pair: {key}")
        record = dict(raw)
        for field in required_numeric:
            value = record.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"invalid outcome field {field} for {key}: {value!r}")
            record[field] = float(value)
        if bool(record.get("same_task")) != (donor == receiver):
            raise ValueError(f"same_task flag mismatch for {key}")
        pair_by_key[key] = record

    expected_keys = {(donor, receiver) for donor in TASKS for receiver in TASKS}
    observed_keys = set(pair_by_key)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise ValueError(f"outcome pair-set mismatch; missing={missing}, extra={extra}")
    return pair_by_key


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        # Average of one-based ranks start+1 through stop.
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def _correlation(x: Sequence[float], y: Sequence[float], method: str) -> Dict[str, Any]:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if x_values.shape != y_values.shape:
        raise ValueError(f"correlation shape mismatch: {x_values.shape} vs {y_values.shape}")
    if x_values.ndim != 1 or len(x_values) < 2:
        return {"coefficient": None, "n": int(x_values.size), "reason": "fewer_than_two_points"}
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        return {"coefficient": None, "n": int(x_values.size), "reason": "non_finite_input"}
    if method == "spearman":
        x_values = _average_ranks(x_values)
        y_values = _average_ranks(y_values)
    elif method != "pearson":
        raise ValueError(f"unknown correlation method: {method}")
    x_centered = x_values - x_values.mean()
    y_centered = y_values - y_values.mean()
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator == 0.0:
        return {"coefficient": None, "n": int(x_values.size), "reason": "constant_vector"}
    coefficient = float(
        np.clip(np.dot(x_centered, y_centered) / denominator, -1.0, 1.0)
    )
    return {"coefficient": coefficient, "n": int(x_values.size), "reason": None}


def _standardized(values: np.ndarray) -> Optional[np.ndarray]:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    norm = float(np.linalg.norm(centered))
    return None if norm == 0.0 else centered / norm


def _rank_matrix_off_diagonal(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ranked = np.full(matrix.shape, np.nan, dtype=np.float64)
    ranked[mask] = _average_ranks(matrix[mask])
    return ranked


def _qap_result(
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    mask: np.ndarray,
    permutations: np.ndarray,
    method: str,
    seed: int,
    permutation_digest: str,
) -> Dict[str, Any]:
    if method == "spearman":
        x_work = _rank_matrix_off_diagonal(x_matrix, mask)
        y_values = _average_ranks(y_matrix[mask])
    elif method == "pearson":
        x_work = x_matrix
        y_values = y_matrix[mask]
    else:
        raise ValueError(f"unknown QAP correlation method: {method}")

    x_reference = _standardized(x_work[mask])
    y_standard = _standardized(y_values)
    if x_reference is None or y_standard is None:
        return {
            "observed": None,
            "exceedance_count": None,
            "p_two_sided": None,
            "n_permutations": int(len(permutations)),
            "seed": seed,
            "generator": "numpy.random.Generator(PCG64)",
            "permutation_digest_sha256": permutation_digest,
            "quantiles": {},
            "null_distribution": [],
            "reason": "constant_vector",
        }
    observed = float(np.clip(np.dot(x_reference, y_standard), -1.0, 1.0))
    null = np.empty(len(permutations), dtype=np.float64)
    for index, permutation in enumerate(permutations):
        permuted = x_work[np.ix_(permutation, permutation)][mask]
        x_standard = _standardized(permuted)
        if x_standard is None:
            raise RuntimeError("QAP permutation unexpectedly produced a constant vector")
        null[index] = np.clip(np.dot(x_standard, y_standard), -1.0, 1.0)
    exceedance = int(np.count_nonzero(np.abs(null) >= abs(observed)))
    probabilities = (0.001, 0.01, 0.025, 0.05, 0.5, 0.95, 0.975, 0.99, 0.999)
    return {
        "observed": observed,
        "exceedance_count": exceedance,
        "p_two_sided": float((1 + exceedance) / (len(null) + 1)),
        "n_permutations": int(len(null)),
        "seed": seed,
        "generator": "numpy.random.Generator(PCG64)",
        "permutation_digest_sha256": permutation_digest,
        "quantiles": {
            str(probability): float(value)
            for probability, value in zip(probabilities, np.quantile(null, probabilities))
        },
        "null_distribution": null.tolist(),
        "reason": None,
    }


def _subset_correlations(records: Sequence[Mapping[str, Any]], x_field: str, y_field: str) -> Dict[str, Any]:
    x = [float(record[x_field]) for record in records]
    y = [float(record[y_field]) for record in records]
    return {
        method: _correlation(x, y, method) for method in ("spearman", "pearson")
    }


def _influence(
    points: Sequence[Mapping[str, Any]], x_field: str, y_field: str
) -> Dict[str, Any]:
    leave_receiver = []
    leave_donor = []
    per_receiver = []
    per_donor = []
    for task in TASKS:
        without_receiver = [point for point in points if point["receiver"] != task]
        without_donor = [point for point in points if point["donor"] != task]
        receiver_points = [point for point in points if point["receiver"] == task]
        donor_points = [point for point in points if point["donor"] == task]
        leave_receiver.append(
            {
                "omitted_receiver": task,
                **_subset_correlations(without_receiver, x_field, y_field),
            }
        )
        leave_donor.append(
            {
                "omitted_donor": task,
                **_subset_correlations(without_donor, x_field, y_field),
            }
        )
        per_receiver.append(
            {
                "receiver": task,
                **_subset_correlations(receiver_points, x_field, y_field),
            }
        )
        per_donor.append(
            {
                "donor": task,
                **_subset_correlations(donor_points, x_field, y_field),
            }
        )
    return {
        "leave_one_receiver_out": leave_receiver,
        "leave_one_donor_out": leave_donor,
        "per_receiver": per_receiver,
        "per_donor": per_donor,
    }


def _matrix_from_points(
    points_by_key: Mapping[Tuple[str, str], Mapping[str, Any]], field: str
) -> np.ndarray:
    return np.asarray(
        [
            [float(points_by_key[(donor, receiver)][field]) for receiver in TASKS]
            for donor in TASKS
        ],
        dtype=np.float64,
    )


@hydra.main(
    config_path="../../../../config/experiments/998_rebuttal/005_qv_alignment",
    config_name="analyze_euclidean_alignment",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    _validate_contract(cfg)
    resolved = _resolved_config(cfg)
    alignment_path = _producer_artifact_path(cfg, resolved)
    outcome_path = _outcome_path(cfg)
    missing_sources = [str(path) for path in (alignment_path, outcome_path) if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError("missing analyzer source artifacts:\n" + "\n".join(missing_sources))

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
        alignment_data = _load_json(alignment_path)
        outcome_data = _load_json(outcome_path)
        cosine = _validate_alignment(alignment_data, cfg)
        outcome_by_key = _validate_outcome(outcome_data, cfg)
        status.heartbeat(progress="sources validated")

        task_index = {task: index for index, task in enumerate(TASKS)}
        points: List[Dict[str, Any]] = []
        diagonal: List[Dict[str, Any]] = []
        all_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for donor in TASKS:
            for receiver in TASKS:
                pair = outcome_by_key[(donor, receiver)]
                cosine_value = float(cosine[task_index[donor], task_index[receiver]])
                record = {
                    "donor": donor,
                    "receiver": receiver,
                    "cosine": cosine_value,
                    "cosine_sq": cosine_value * cosine_value,
                    "delta": float(pair["delta"]),
                    "delta_best": float(pair["delta_best"]),
                    "recovery": float(pair["recovery"]),
                    "recovery_best": float(pair["recovery_best"]),
                    "unit_alpha": float(cfg.unit_alpha),
                    "best_alpha": float(pair["alpha_best"]),
                    "baseline_acc": float(pair["baseline_acc"]),
                    "ceiling_delta": float(pair["ceiling_delta"]),
                }
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
            "delta": _matrix_from_points(all_by_key, "delta"),
            "delta_best": _matrix_from_points(all_by_key, "delta_best"),
            "recovery": _matrix_from_points(all_by_key, "recovery"),
            "recovery_best": _matrix_from_points(all_by_key, "recovery_best"),
        }
        comparison_results: Dict[str, Any] = {}
        for comparison_index, spec in enumerate(COMPARISONS):
            name = str(spec["name"])
            x_field = str(spec["x_field"])
            y_field = str(spec["y_field"])
            observed = _subset_correlations(points, x_field, y_field)
            qap = {
                method: _qap_result(
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
                        raise RuntimeError(f"observed/QAP null mismatch for {name}/{method}")
                elif not math.isclose(
                    float(coefficient), float(qap_observed), rel_tol=0.0, abs_tol=1e-12
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
                "influence": _influence(points, x_field, y_field),
            }
            status.heartbeat(
                progress=f"comparisons {comparison_index + 1}/{len(COMPARISONS)}"
            )

        artifact = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "analysis_spec": cfg.analysis_spec,
            "provenance": {
                "resolved_config": resolved,
                "producer_run_id_params": PRODUCER_RUN_ID_PARAMS,
                "producer_run_id_path": str(producer_path),
                "analyzer_run_id_params": RUN_ID_PARAMS,
                "analyzer_run_id_path": str(analyzer_path),
                "alignment_source": {
                    "path": _relative_or_absolute(alignment_path),
                    "sha256": _sha256(alignment_path),
                    "schema_version": alignment_data.get("schema_version"),
                },
                "outcome_source": {
                    "path": _relative_or_absolute(outcome_path),
                    "sha256": _sha256(outcome_path),
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
                        "(1 + count(abs(null) >= abs(observed))) / (n_permutations + 1)"
                    ),
                },
                "comparisons": comparison_results,
            },
            "missing": [],
        }
        _atomic_json(golden, artifact)
        status.heartbeat(progress="comparisons 4/4; artifact written")
        print(f"[complete] {_relative_or_absolute(golden)}", flush=True)


if __name__ == "__main__":
    main()
