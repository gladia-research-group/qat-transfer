"""Analyze Euclidean QV alignment across the complete validation alpha response."""

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


EXPERIMENT = "998_rebuttal/006_alignment_alpha_response"
SCHEMA_VERSION = "alpha_response_statistics_v1"
GOLDEN_ARTIFACT = "alpha_response_statistics.json"
RUN_ID_PARAMS = [
    "model_name",
    "curve_split",
    "curve_baseline",
    "curve_grid",
    "analysis_spec",
    "n_permutations",
    "permutation_seed",
]
TASKS: Tuple[str, ...] = (
    "CIFAR10", "CIFAR100", "Cars", "DTD", "EMNIST", "EuroSAT",
    "FER2013", "FashionMNIST", "Flowers102", "Food101", "GTSRB",
    "ImageNet", "KMNIST", "MNIST", "OxfordIIITPet", "PCAM",
    "RESISC45", "RenderedSST2", "STL10", "SUN397", "SVHN",
    "TinyImageNet",
)
MEASURED_ALPHAS: Tuple[float, ...] = (
    0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00, 1.05, 1.20, 1.35, 1.50,
)
ANALYSIS_ALPHAS: Tuple[float, ...] = (0.0,) + MEASURED_ALPHAS


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolved_config(cfg: DictConfig) -> Dict[str, Any]:
    value = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("resolved Hydra config must be a mapping")
    return value


def _source_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


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
        "ptq_bits": 3,
        "ptq_granularity": "channel",
        "vector_scope": "quantized_linear_weight",
        "module_selector": "apply_ptq_linear_v1",
        "accumulation_dtype": "float64",
        "curve_split": "val",
        "curve_baseline": "fp_ptq",
        "curve_grid": "shared",
        "analysis_spec": "qv_alignment_alpha_v1",
        "n_permutations": 10_000,
        "permutation_seed": 2038,
        "unit_alpha": 1.0,
        "alpha_min": 0.0,
        "alpha_max": 1.5,
        "tie_tolerance": 1.0e-12,
        "use_wandb": False,
    }
    resolved = _resolved_config(cfg)
    mismatches = {
        key: {"observed": resolved.get(key), "expected": expected_value}
        for key, expected_value in expected.items()
        if resolved.get(key) != expected_value
    }
    for key in ("qat_skip_modules", "ptq_skip_modules"):
        if list(resolved.get(key, [])) != ["head"]:
            mismatches[key] = {"observed": resolved.get(key), "expected": ["head"]}
    if mismatches:
        raise ValueError(
            "configuration violates qv_alignment_alpha_v1: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _validate_alignment(
    data: Mapping[str, Any], cfg: DictConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if data.get("schema_version") != "euclidean_alignment_v1":
        raise ValueError("unsupported Euclidean alignment schema")
    if tuple(data.get("task_order", [])) != TASKS:
        raise ValueError("alignment task order differs from the locked 22-task order")
    source_cfg = data.get("config")
    if not isinstance(source_cfg, Mapping):
        raise TypeError("alignment artifact has no config mapping")
    expected = {
        "family": cfg.family,
        "model_name": cfg.model_name,
        "seed": cfg.seed,
        "optim": cfg.optim,
        "lr": cfg.lr,
        "wd": cfg.wd,
        "ls": cfg.ls,
        "wl": cfg.wl,
        "max_grad_norm": cfg.max_grad_norm,
        "batch_size": cfg.batch_size,
        "qat_bits": cfg.qat_bits,
        "qat_granularity": cfg.qat_granularity,
        "qat_skip_modules": list(cfg.qat_skip_modules),
        "ptq_skip_modules": list(cfg.ptq_skip_modules),
        "vector_scope": cfg.vector_scope,
        "module_selector": cfg.module_selector,
        "accumulation_dtype": cfg.accumulation_dtype,
    }
    mismatches = {
        key: {"observed": source_cfg.get(key), "expected": expected_value}
        for key, expected_value in expected.items()
        if source_cfg.get(key) != expected_value
    }
    if mismatches:
        raise ValueError("alignment metadata mismatch: " + json.dumps(mismatches))

    cosine = np.asarray(data.get("cosine_matrix"), dtype=np.float64)
    dot = np.asarray(data.get("dot_product_matrix"), dtype=np.float64)
    norms_map = data.get("qv_norms")
    if cosine.shape != (22, 22) or dot.shape != (22, 22):
        raise ValueError("alignment cosine and dot-product matrices must both be 22x22")
    if not isinstance(norms_map, Mapping) or set(norms_map) != set(TASKS):
        raise ValueError("alignment qv_norms must cover the exact locked task set")
    norms = np.asarray([float(norms_map[task]) for task in TASKS], dtype=np.float64)
    if not np.isfinite(cosine).all() or not np.isfinite(dot).all():
        raise ValueError("alignment matrices contain non-finite values")
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError("QV norms must be finite and strictly positive")
    if np.max(np.abs(cosine - cosine.T)) > 1e-12:
        raise ValueError("Euclidean cosine matrix is not symmetric")
    if np.max(np.abs(dot - dot.T)) > 1e-9:
        raise ValueError("Euclidean dot-product matrix is not symmetric")
    if np.max(np.abs(np.diag(cosine) - 1.0)) > 1e-12:
        raise ValueError("Euclidean cosine diagonal is not one")
    reconstructed = cosine * np.outer(norms, norms)
    tolerance = 1e-9 * np.maximum(1.0, np.abs(dot))
    if np.any(np.abs(dot - reconstructed) > tolerance):
        raise ValueError("dot products, cosines, and QV norms are algebraically inconsistent")
    return cosine, dot, norms


def _curve_key(alpha: float) -> str:
    return str(float(alpha))


def _validate_curves(
    data: Mapping[str, Any], cfg: DictConfig
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if data.get("modality") != "vision" or data.get("family") != cfg.family:
        raise ValueError("curve modality/family mismatch")
    if tuple(data.get("datasets", [])) != TASKS:
        raise ValueError("curve task order differs from the locked 22-task order")
    source_cfg = data.get("config")
    if not isinstance(source_cfg, Mapping):
        raise TypeError("curve artifact has no config mapping")
    expected = {
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
        "grid_mode": cfg.curve_grid,
        "eval_split": cfg.curve_split,
        "metric_key": "val_accuracy_fp_head_ptq",
        "baseline": cfg.curve_baseline,
    }
    mismatches = {
        key: {"observed": source_cfg.get(key), "expected": expected_value}
        for key, expected_value in expected.items()
        if source_cfg.get(key) != expected_value
    }
    if mismatches:
        raise ValueError("curve metadata mismatch: " + json.dumps(mismatches))
    models = data.get("models")
    if not isinstance(models, Mapping) or cfg.model_name not in models:
        raise KeyError(f"curve source has no model {cfg.model_name!r}")
    model = models[cfg.model_name]
    if not isinstance(model, Mapping):
        raise TypeError("curve model entry is not a mapping")
    if model.get("n_datasets") != 22 or model.get("n_cells_expected") != 484:
        raise ValueError("curve source does not declare the complete 22x22 matrix")
    if model.get("missing") != []:
        raise ValueError(f"curve source reports missing cells: {model.get('missing')}")
    if tuple(float(value) for value in model.get("grid", [])) != MEASURED_ALPHAS:
        raise ValueError("curve grid differs from the locked 11 measured alphas")
    if model.get("batch_size") != cfg.batch_size:
        raise ValueError("curve batch-size metadata mismatch")
    if model.get("skip_modules") != list(cfg.ptq_skip_modules):
        raise ValueError("curve skip-module metadata mismatch")

    raw_pairs = model.get("pairs")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != 484:
        raise ValueError("curve source must contain exactly 484 pair records")
    pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    expected_curve_keys = {_curve_key(alpha) for alpha in MEASURED_ALPHAS}
    for raw in raw_pairs:
        if not isinstance(raw, Mapping):
            raise TypeError("curve pair record is not a mapping")
        donor, receiver = str(raw.get("donor")), str(raw.get("receiver"))
        key = (donor, receiver)
        if donor not in TASKS or receiver not in TASKS or key in pairs:
            raise ValueError(f"unknown or duplicate curve pair: {key}")
        if bool(raw.get("same_task")) != (donor == receiver):
            raise ValueError(f"same_task mismatch for {key}")
        curve = raw.get("curve")
        if not isinstance(curve, Mapping) or set(curve) != expected_curve_keys:
            raise ValueError(f"curve keys mismatch for {key}")
        source_stats = raw.get("stats")
        if not isinstance(source_stats, Mapping) or not isinstance(source_stats.get("unimodal"), bool):
            raise ValueError(f"curve stats/unimodality diagnostic missing for {key}")
        values = {name: float(value) for name, value in curve.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"non-finite curve value for {key}")
        record = dict(raw)
        record["curve"] = values
        pairs[key] = record
    expected_pairs = {(donor, receiver) for donor in TASKS for receiver in TASKS}
    if set(pairs) != expected_pairs:
        raise ValueError("curve pair set is not the complete locked Cartesian product")
    return pairs


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
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return ranks


def _standardized(values: np.ndarray) -> Optional[np.ndarray]:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    norm = float(np.linalg.norm(centered))
    return None if norm == 0.0 else centered / norm


def _correlation(x: Sequence[float], y: Sequence[float], method: str) -> Dict[str, Any]:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if method == "spearman":
        x_values, y_values = _average_ranks(x_values), _average_ranks(y_values)
    elif method != "pearson":
        raise ValueError(f"unknown correlation method: {method}")
    x_standard, y_standard = _standardized(x_values), _standardized(y_values)
    if x_standard is None or y_standard is None:
        return {"coefficient": None, "n": int(len(x_values)), "reason": "constant_vector"}
    return {
        "coefficient": float(np.clip(np.dot(x_standard, y_standard), -1.0, 1.0)),
        "n": int(len(x_values)),
        "reason": None,
    }


def _qap_profile(
    x_matrix: np.ndarray,
    y_matrices: Sequence[np.ndarray],
    mask: np.ndarray,
    permutations: np.ndarray,
    method: str,
    seed: int,
    digest: str,
) -> Dict[str, Any]:
    x_work = np.full_like(x_matrix, np.nan) if method == "spearman" else x_matrix.copy()
    if method == "spearman":
        x_work[mask] = _average_ranks(x_matrix[mask])
    y_work = [
        _average_ranks(matrix[mask]) if method == "spearman" else matrix[mask]
        for matrix in y_matrices
    ]
    x_observed = _standardized(x_work[mask])
    y_standard = [_standardized(values) for values in y_work]
    if x_observed is None or any(value is None for value in y_standard):
        raise ValueError("profile QAP requires nonconstant alignment and outcome vectors")
    y_stack = np.stack([value for value in y_standard if value is not None])
    observed = np.clip(y_stack @ x_observed, -1.0, 1.0)
    null = np.empty((len(permutations), len(y_matrices)), dtype=np.float64)
    for index, permutation in enumerate(permutations):
        permuted = x_work[np.ix_(permutation, permutation)][mask]
        x_standard = _standardized(permuted)
        if x_standard is None:
            raise RuntimeError("QAP permutation unexpectedly produced a constant vector")
        null[index] = np.clip(y_stack @ x_standard, -1.0, 1.0)
    pointwise = []
    for column, coefficient in enumerate(observed):
        values = null[:, column]
        exceedance = int(np.count_nonzero(np.abs(values) >= abs(coefficient)))
        pointwise.append({
            "observed": float(coefficient),
            "exceedance_count": exceedance,
            "p_two_sided": float((1 + exceedance) / (len(values) + 1)),
            "null_distribution": values.tolist(),
        })
    max_null = np.max(np.abs(null), axis=1)
    observed_max = float(np.max(np.abs(observed)))
    global_exceedance = int(np.count_nonzero(max_null >= observed_max))
    familywise = []
    for coefficient in observed:
        exceedance = int(np.count_nonzero(max_null >= abs(coefficient)))
        familywise.append({
            "observed_abs": float(abs(coefficient)),
            "exceedance_count": exceedance,
            "p_max_abs": float((1 + exceedance) / (len(max_null) + 1)),
        })
    return {
        "method": method,
        "pointwise": pointwise,
        "familywise_max_abs": familywise,
        "observed_max_abs": observed_max,
        "global_exceedance_count": global_exceedance,
        "p_any_alpha_max_abs": float((1 + global_exceedance) / (len(max_null) + 1)),
        "max_abs_null_distribution": max_null.tolist(),
        "n_permutations": int(len(permutations)),
        "seed": seed,
        "generator": "numpy.random.Generator(PCG64)",
        "permutation_digest_sha256": digest,
    }


def _qap_single(
    x_matrix: np.ndarray,
    y_matrix: np.ndarray,
    mask: np.ndarray,
    permutations: np.ndarray,
    method: str,
    seed: int,
    digest: str,
) -> Dict[str, Any]:
    profile = _qap_profile(
        x_matrix, [y_matrix], mask, permutations, method, seed, digest
    )
    result = dict(profile["pointwise"][0])
    result.update({
        "n_permutations": profile["n_permutations"],
        "seed": seed,
        "generator": profile["generator"],
        "permutation_digest_sha256": digest,
    })
    return result


def _level_interval(
    x: np.ndarray, y: np.ndarray, level: float, maximizing: np.ndarray
) -> Optional[Dict[str, Any]]:
    if not np.any(y >= level) or len(maximizing) == 0:
        return None
    left, right = int(maximizing.min()), int(maximizing.max())
    while left > 0 and y[left - 1] >= level:
        left -= 1
    while right + 1 < len(y) and y[right + 1] >= level:
        right += 1
    if left == 0:
        lo = float(x[0])
        left_censored = bool(y[0] > level)
    else:
        x0, x1, y0, y1 = x[left - 1], x[left], y[left - 1], y[left]
        lo = float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
        left_censored = False
    if right == len(y) - 1:
        hi = float(x[-1])
        right_censored = bool(y[-1] > level)
    else:
        x0, x1, y0, y1 = x[right], x[right + 1], y[right], y[right + 1]
        hi = float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))
        right_censored = False
    return {
        "lo": lo,
        "hi": hi,
        "width": hi - lo,
        "level": float(level),
        "left_censored": left_censored,
        "right_censored": right_censored,
    }


def _quadratic_fit(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    coefficient = np.polyfit(x, y, 2)
    prediction = np.polyval(coefficient, x)
    residual_ss = float(np.sum((y - prediction) ** 2))
    total_ss = float(np.sum((y - float(np.mean(y))) ** 2))
    r_squared = None if total_ss == 0.0 else float(1.0 - residual_ss / total_ss)
    a, b, c = (float(value) for value in coefficient)
    vertex = None if a >= 0.0 else -b / (2.0 * a)
    return {
        "a": a,
        "b": b,
        "c": c,
        "r_squared": r_squared,
        "concave": bool(a < 0.0),
        "vertex_raw": vertex,
        "vertex_clipped": None if vertex is None else float(np.clip(vertex, x[0], x[-1])),
        "vertex_in_grid_range": None if vertex is None else bool(x[0] <= vertex <= x[-1]),
    }


def _curve_record(
    donor: str,
    receiver: str,
    source: Mapping[str, Any],
    cosine: float,
    dot_product: float,
    donor_norm: float,
    receiver_norm: float,
    ceiling: float,
    tolerance: float,
) -> Dict[str, Any]:
    x = np.asarray(ANALYSIS_ALPHAS, dtype=np.float64)
    y = np.asarray(
        [0.0] + [float(source["curve"][_curve_key(alpha)]) for alpha in MEASURED_ALPHAS],
        dtype=np.float64,
    )
    maximum = float(np.max(y))
    maximizing = np.flatnonzero(np.abs(y - maximum) <= tolerance)
    maximizing_alphas = [float(x[index]) for index in maximizing]
    midpoint = 0.5 * (min(maximizing_alphas) + max(maximizing_alphas))
    unit_index = ANALYSIS_ALPHAS.index(1.0)
    alpha_predicted = float(dot_product / (donor_norm * donor_norm))
    sign_values = np.sign(y)
    nonzero_signs = sign_values[sign_values != 0]
    sign_changes = int(np.count_nonzero(nonzero_signs[1:] != nonzero_signs[:-1]))
    first_max, last_max = int(maximizing.min()), int(maximizing.max())
    unimodal = bool(
        np.all(np.diff(y[: first_max + 1]) >= -tolerance)
        and np.all(np.diff(y[last_max:]) <= tolerance)
    )
    positive_interval = (
        None if maximum <= 0.0 else _level_interval(x, y, 0.0, maximizing)
    )
    plateau = (
        None if maximum <= 0.0 else _level_interval(x, y, 0.9 * maximum, maximizing)
    )
    return {
        "donor": donor,
        "receiver": receiver,
        "same_task": donor == receiver,
        "cosine": float(cosine),
        "cosine_sq": float(cosine * cosine),
        "dot_product": float(dot_product),
        "donor_qv_norm": float(donor_norm),
        "receiver_qv_norm": float(receiver_norm),
        "alpha_predicted_raw": alpha_predicted,
        "alpha_predicted_clipped": float(np.clip(alpha_predicted, x[0], x[-1])),
        "alphas": x.tolist(),
        "deltas": y.tolist(),
        "delta_unit": float(y[unit_index]),
        "delta_best_grid": maximum,
        "unit_regret": maximum - float(y[unit_index]),
        "maximizing_alphas": maximizing_alphas,
        "n_maximizing_alphas": len(maximizing_alphas),
        "alpha_best_low": min(maximizing_alphas),
        "alpha_best_high": max(maximizing_alphas),
        "alpha_best_midpoint": midpoint,
        "predicted_raw_absolute_error": abs(alpha_predicted - midpoint),
        "predicted_clipped_absolute_error": abs(float(np.clip(alpha_predicted, x[0], x[-1])) - midpoint),
        "predicted_clipped_inside_maximizing_interval": bool(
            min(maximizing_alphas) - tolerance
            <= float(np.clip(alpha_predicted, x[0], x[-1]))
            <= max(maximizing_alphas) + tolerance
        ),
        "best_touches_lower_boundary": bool(0.0 in maximizing_alphas),
        "best_touches_upper_boundary": bool(1.5 in maximizing_alphas),
        "same_task_unit_ceiling": float(ceiling),
        "recovery_best_grid": maximum / ceiling,
        "positive_interval": positive_interval,
        "plateau_90pct": plateau,
        "predicted_clipped_inside_positive_interval": bool(
            positive_interval is not None
            and positive_interval["lo"] - tolerance
            <= float(np.clip(alpha_predicted, x[0], x[-1]))
            <= positive_interval["hi"] + tolerance
        ),
        "predicted_clipped_inside_plateau": bool(
            plateau is not None
            and plateau["lo"] - tolerance
            <= float(np.clip(alpha_predicted, x[0], x[-1]))
            <= plateau["hi"] + tolerance
        ),
        "n_sign_changes": sign_changes,
        "unimodal_on_grid": unimodal,
        "source_003_unimodal_on_measured_grid": bool(source.get("stats", {}).get("unimodal", False)),
        "quadratic_fit": _quadratic_fit(x, y),
    }


def _matrix(records: Mapping[Tuple[str, str], Mapping[str, Any]], field: str) -> np.ndarray:
    return np.asarray(
        [[float(records[(donor, receiver)][field]) for receiver in TASKS] for donor in TASKS],
        dtype=np.float64,
    )


def _summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def _scale_subgroup(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "n": len(records),
        "raw_absolute_error": _summary([float(p["predicted_raw_absolute_error"]) for p in records]),
        "clipped_absolute_error": _summary([float(p["predicted_clipped_absolute_error"]) for p in records]),
        "raw_correlation": {
            method: _correlation(
                [float(p["alpha_predicted_raw"]) for p in records],
                [float(p["alpha_best_midpoint"]) for p in records],
                method,
            )
            for method in ("spearman", "pearson")
        },
        "clipped_correlation": {
            method: _correlation(
                [float(p["alpha_predicted_clipped"]) for p in records],
                [float(p["alpha_best_midpoint"]) for p in records],
                method,
            )
            for method in ("spearman", "pearson")
        },
    }


@hydra.main(
    config_path="../../../../config/experiments/998_rebuttal/006_alignment_alpha_response",
    config_name="analyze_alpha_response",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    _validate_contract(cfg)
    resolved = _resolved_config(cfg)
    alignment_path, curve_path = _source_path(cfg.alignment_path), _source_path(cfg.curve_path)
    missing = [str(path) for path in (alignment_path, curve_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing analyzer source artifacts:\n" + "\n".join(missing))
    analyzer_path = run_id_path(resolved, RUN_ID_PARAMS)
    eval_dir = PROJECT_ROOT / str(cfg.evaluation_root) / EXPERIMENT / analyzer_path
    guard_run_config(resolved, RUN_ID_PARAMS, eval_dir)
    golden = eval_dir / GOLDEN_ARTIFACT
    if golden.exists():
        raise FileExistsError(f"golden artifact already exists; refusing to overwrite: {golden}")

    with StatusWriter(eval_dir) as status:
        alignment_data, curve_data = _load_json(alignment_path), _load_json(curve_path)
        cosine, dot, norms = _validate_alignment(alignment_data, cfg)
        curves = _validate_curves(curve_data, cfg)
        status.heartbeat(progress="sources validated 2/2")

        index = {task: position for position, task in enumerate(TASKS)}
        ceilings = {
            receiver: float(curves[(receiver, receiver)]["curve"][_curve_key(1.0)])
            for receiver in TASKS
        }
        invalid_ceilings = {task: value for task, value in ceilings.items() if value <= 0.0}
        if invalid_ceilings:
            raise ValueError(f"same-task unit validation ceilings must be positive: {invalid_ceilings}")
        all_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
        points: List[Dict[str, Any]] = []
        diagonal: List[Dict[str, Any]] = []
        for donor in TASKS:
            for receiver in TASKS:
                i, j = index[donor], index[receiver]
                record = _curve_record(
                    donor, receiver, curves[(donor, receiver)], cosine[i, j], dot[i, j],
                    norms[i], norms[j], ceilings[receiver], float(cfg.tie_tolerance),
                )
                all_records[(donor, receiver)] = record
                (diagonal if donor == receiver else points).append(record)
        if len(points) != 462 or len(diagonal) != 22:
            raise RuntimeError("joined population must be 462 cross-task plus 22 diagonal cells")
        status.heartbeat(progress="joined alpha responses 484/484")

        mask = ~np.eye(len(TASKS), dtype=bool)
        rng = np.random.default_rng(int(cfg.permutation_seed))
        permutations = np.stack(
            [rng.permutation(len(TASKS)) for _ in range(int(cfg.n_permutations))]
        ).astype(np.uint8, copy=False)
        digest = hashlib.sha256(permutations.tobytes()).hexdigest()
        delta_matrices = [
            np.asarray(
                [[all_records[(d, r)]["deltas"][k] for r in TASKS] for d in TASKS],
                dtype=np.float64,
            )
            for k in range(1, len(ANALYSIS_ALPHAS))
        ]
        profile = {
            method: _qap_profile(
                cosine, delta_matrices, mask, permutations, method,
                int(cfg.permutation_seed), digest,
            )
            for method in ("spearman", "pearson")
        }
        status.heartbeat(progress="alignment profile QAP 2/4")

        recovery = _matrix(all_records, "recovery_best_grid")
        alpha_predicted_raw = _matrix(all_records, "alpha_predicted_raw")
        alpha_predicted_clipped = _matrix(all_records, "alpha_predicted_clipped")
        alpha_empirical = _matrix(all_records, "alpha_best_midpoint")
        secondary = {
            method: _qap_single(
                cosine * cosine, recovery, mask, permutations, method,
                int(cfg.permutation_seed), digest,
            )
            for method in ("spearman", "pearson")
        }
        scale = {
            variant: {
                method: _qap_single(
                    matrix, alpha_empirical, mask, permutations, method,
                    int(cfg.permutation_seed), digest,
                )
                for method in ("spearman", "pearson")
            }
            for variant, matrix in (
                ("raw", alpha_predicted_raw),
                ("clipped", alpha_predicted_clipped),
            )
        }

        cosine_values = np.asarray([record["cosine"] for record in points], dtype=np.float64)
        quartile_edges = np.quantile(cosine_values, [0.25, 0.5, 0.75])
        quartile_curves = []
        for quartile in range(4):
            selected = [
                record for record in points
                if int(np.searchsorted(quartile_edges, record["cosine"], side="right")) == quartile
            ]
            quartile_curves.append({
                "quartile": quartile + 1,
                "n": len(selected),
                "median_delta": [
                    float(np.median([record["deltas"][k] for record in selected]))
                    for k in range(len(ANALYSIS_ALPHAS))
                ],
            })
        boundary_points = [
            p for p in points
            if p["best_touches_lower_boundary"] or p["best_touches_upper_boundary"]
        ]
        interior_points = [p for p in points if p not in boundary_points]
        if not boundary_points or not interior_points:
            raise ValueError("boundary/interior scale diagnostics require both nonempty groups")

        artifact = {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "analysis_spec": cfg.analysis_spec,
            "provenance": {
                "resolved_config": resolved,
                "run_id_params": RUN_ID_PARAMS,
                "run_id_path": str(analyzer_path),
                "task_order": list(TASKS),
                "alignment_source": {
                    "path": _relative_or_absolute(alignment_path),
                    "sha256": _sha256(alignment_path),
                    "schema_version": alignment_data.get("schema_version"),
                },
                "curve_source": {
                    "path": _relative_or_absolute(curve_path),
                    "sha256": _sha256(curve_path),
                },
                "software": {"python": platform.python_version(), "numpy": np.__version__},
            },
            "points": points,
            "diagonal_audit": diagonal,
            "statistics": {
                "primary": {
                    "description": "signed Euclidean cosine versus validation gain at each measured alpha",
                    "alphas": list(MEASURED_ALPHAS),
                    "primary_coefficient": "spearman",
                    "qap_profile": profile,
                },
                "secondary": {
                    "description": "squared cosine versus grid-best validation recovery; selection-optimistic",
                    "qap": secondary,
                },
                "scale_calibration": {
                    "description": "raw and clipped Euclidean predicted alpha versus tie-midpoint empirical grid optimum",
                    "qap": scale,
                    "raw_absolute_error": _summary([p["predicted_raw_absolute_error"] for p in points]),
                    "clipped_absolute_error": _summary([p["predicted_clipped_absolute_error"] for p in points]),
                    "n_predicted_inside_maximizing_interval": sum(p["predicted_clipped_inside_maximizing_interval"] for p in points),
                    "n_predicted_inside_positive_interval": sum(p["predicted_clipped_inside_positive_interval"] for p in points),
                    "n_predicted_inside_plateau": sum(p["predicted_clipped_inside_plateau"] for p in points),
                    "subgroups": {
                        "boundary_censored": _scale_subgroup(boundary_points),
                        "interior": _scale_subgroup(interior_points),
                    },
                },
                "curve_diagnostics": {
                    "n_cross_task": 462,
                    "n_best_at_lower_boundary": sum(p["best_touches_lower_boundary"] for p in points),
                    "n_best_at_upper_boundary": sum(p["best_touches_upper_boundary"] for p in points),
                    "n_unimodal": sum(p["unimodal_on_grid"] for p in points),
                    "n_source_003_unimodal": sum(p["source_003_unimodal_on_measured_grid"] for p in points),
                    "n_concave_quadratic": sum(p["quadratic_fit"]["concave"] for p in points),
                    "n_concave_vertex_outside_grid": sum(
                        p["quadratic_fit"]["concave"]
                        and not p["quadratic_fit"]["vertex_in_grid_range"]
                        for p in points
                    ),
                    "unit_regret": _summary([p["unit_regret"] for p in points]),
                    "quadratic_r_squared": _summary([
                        p["quadratic_fit"]["r_squared"] for p in points
                        if p["quadratic_fit"]["r_squared"] is not None
                    ]),
                },
                "cosine_quartile_curves": {
                    "edges": quartile_edges.tolist(),
                    "alphas": list(ANALYSIS_ALPHAS),
                    "groups": quartile_curves,
                },
                "qap": {
                    "method": "simultaneous donor/receiver task-label permutation",
                    "outcome_matrices_fixed": True,
                    "shared_permutations_across_all_tests": True,
                    "n_permutations": int(cfg.n_permutations),
                    "seed": int(cfg.permutation_seed),
                    "generator": "numpy.random.Generator(PCG64)",
                    "permutation_digest_sha256": digest,
                    "two_sided_formula": "(1 + count(abs(null) >= abs(observed))) / (B + 1)",
                    "familywise_formula": "(1 + count(max_alpha(abs(null)) >= abs(observed_alpha))) / (B + 1)",
                },
            },
            "missing": [],
        }
        _atomic_json(golden, artifact)
        status.heartbeat(progress="statistics and artifact 4/4; artifact written")
        print(f"[complete] {_relative_or_absolute(golden)}", flush=True)


if __name__ == "__main__":
    main()
