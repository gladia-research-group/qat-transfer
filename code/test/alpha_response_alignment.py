"""Targeted contract tests for rebuttal experiment 006."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "code/experiments/998_rebuttal/006_alignment_alpha_response"
CONFIG = ROOT / "config/experiments/998_rebuttal/006_alignment_alpha_response/analyze_alpha_response.yaml"
VIS_DIR = ROOT / "visualizations/998_rebuttal/006_alignment_alpha_response"


def _load_module():
    spec = importlib.util.spec_from_file_location("alpha_response_analyzer", EXPERIMENT_DIR / "analyze_alpha_response.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYZER = _load_module()


def test_locked_config_and_existing_sources_are_complete() -> None:
    cfg = OmegaConf.load(CONFIG)
    ANALYZER._validate_contract(cfg)
    alignment_path = ANALYZER._source_path(cfg.alignment_path)
    curve_path = ANALYZER._source_path(cfg.curve_path)
    cosine, dot, norms = ANALYZER._validate_alignment(ANALYZER._load_json(alignment_path), cfg)
    curves = ANALYZER._validate_curves(ANALYZER._load_json(curve_path), cfg)
    assert cosine.shape == dot.shape == (22, 22)
    assert norms.shape == (22,)
    assert len(curves) == 484
    assert len({value for pair in curves.values() for value in pair["curve"]}) == 11


def test_predicted_alpha_uses_directional_donor_norm_and_ties_include_zero() -> None:
    source = {"curve": {str(alpha): -0.1 for alpha in ANALYZER.MEASURED_ALPHAS}}
    record = ANALYZER._curve_record(
        "CIFAR10", "CIFAR100", source, cosine=0.5, dot_product=8.0,
        donor_norm=2.0, receiver_norm=8.0, ceiling=0.5, tolerance=1e-12,
    )
    assert record["alpha_predicted_raw"] == 2.0
    assert record["alpha_predicted_clipped"] == 1.5
    assert record["maximizing_alphas"] == [0.0]
    assert record["alpha_best_midpoint"] == 0.0
    assert record["delta_best_grid"] == 0.0
    assert record["recovery_best_grid"] == 0.0

    tied = {"curve": {str(alpha): (0.2 if alpha in (0.3, 0.45) else 0.0) for alpha in ANALYZER.MEASURED_ALPHAS}}
    tied_record = ANALYZER._curve_record(
        "CIFAR10", "CIFAR100", tied, 0.1, 1.0, 2.0, 5.0, 0.4, 1e-12,
    )
    assert tied_record["maximizing_alphas"] == [0.3, 0.45]
    assert tied_record["alpha_best_low"] == 0.3
    assert tied_record["alpha_best_high"] == 0.45
    assert tied_record["alpha_best_midpoint"] == 0.375


def test_quadratic_fit_reports_only_concave_vertices() -> None:
    x = np.asarray(ANALYZER.ANALYSIS_ALPHAS)
    concave = ANALYZER._quadratic_fit(x, -(x - 0.6) ** 2 + 1.0)
    convex = ANALYZER._quadratic_fit(x, (x - 0.6) ** 2)
    assert concave["concave"] is True
    assert abs(concave["vertex_raw"] - 0.6) < 1e-10
    assert concave["r_squared"] > 0.999999999
    assert convex["concave"] is False
    assert convex["vertex_raw"] is None


def test_shared_qap_profile_is_deterministic_and_max_adjusted() -> None:
    x = np.asarray([
        [1.0, 0.1, 0.2, 0.3],
        [0.1, 1.0, 0.4, 0.5],
        [0.2, 0.4, 1.0, 0.6],
        [0.3, 0.5, 0.6, 1.0],
    ])
    y1 = np.asarray([
        [0.0, 1.0, 2.0, 4.0], [3.0, 0.0, 5.0, 6.0],
        [7.0, 8.0, 0.0, 10.0], [11.0, 9.0, 12.0, 0.0],
    ])
    y2 = -y1 + np.asarray([[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
    rng = np.random.default_rng(2038)
    permutations = np.stack([rng.permutation(4) for _ in range(100)]).astype(np.uint8)
    digest = ANALYZER.hashlib.sha256(permutations.tobytes()).hexdigest()
    mask = ~np.eye(4, dtype=bool)
    first = ANALYZER._qap_profile(x, [y1, y2], mask, permutations, "spearman", 2038, digest)
    second = ANALYZER._qap_profile(x, [y1, y2], mask, permutations, "spearman", 2038, digest)
    assert first == second
    assert len(first["pointwise"]) == 2
    assert len(first["max_abs_null_distribution"]) == 100
    assert 0.0 < first["p_any_alpha_max_abs"] <= 1.0
    for pointwise, adjusted in zip(first["pointwise"], first["familywise_max_abs"]):
        assert adjusted["p_max_abs"] >= pointwise["p_two_sided"]


def _synthetic_statistics(path: Path) -> None:
    points = []
    for index in range(462):
        low = float((index % 11) * 0.15)
        high = min(1.5, low + (0.15 if index % 7 == 0 else 0.0))
        points.append({
            "alpha_predicted_raw": float((index % 31) / 20 - 0.1),
            "alpha_predicted_clipped": float(np.clip((index % 31) / 20 - 0.1, 0, 1.5)),
            "alpha_best_midpoint": 0.5 * (low + high),
            "alpha_best_low": low,
            "alpha_best_high": high,
            "cosine_sq": float((index % 23) / 22) ** 2,
            "recovery_best_grid": float((index % 37) / 10 - 0.5),
        })
    alphas = list(ANALYZER.MEASURED_ALPHAS)
    profile = {
        "pointwise": [
            {"observed": float(0.2 * np.sin(alpha)), "p_two_sided": 0.2}
            for alpha in alphas
        ],
        "familywise_max_abs": [
            {"observed_abs": float(abs(0.2 * np.sin(alpha))), "p_max_abs": 0.5}
            for alpha in alphas
        ],
        "p_any_alpha_max_abs": 0.3,
    }
    artifact = {
        "schema_version": "alpha_response_statistics_v1",
        "analysis_spec": "qv_alignment_alpha_v1",
        "provenance": {"run_id_path": "model_name=synthetic/analysis_spec=test"},
        "points": points,
        "diagonal_audit": [{} for _ in range(22)],
        "statistics": {
            "primary": {"alphas": alphas, "qap_profile": {"spearman": profile, "pearson": profile}},
            "scale_calibration": {"qap": {"clipped": {"spearman": {"observed": 0.1, "p_two_sided": 0.4}}}},
            "secondary": {"qap": {"spearman": {"observed": 0.2, "p_two_sided": 0.3}}},
            "cosine_quartile_curves": {
                "alphas": list(ANALYZER.ANALYSIS_ALPHAS),
                "groups": [
                    {"quartile": q, "n": 115 + (q <= 2), "median_delta": [0.01 * q * alpha for alpha in ANALYZER.ANALYSIS_ALPHAS]}
                    for q in range(1, 5)
                ],
            },
        },
        "missing": [],
    }
    path.write_text(json.dumps(artifact))


def test_all_visualizations_render_pdf_and_png_from_statistics_only(tmp_path: Path) -> None:
    statistics = tmp_path / "statistics.json"
    plot_root = tmp_path / "plots"
    _synthetic_statistics(statistics)
    for script_name in (
        "plot_alpha_correlation_profile.py",
        "plot_alpha_geometry_calibration.py",
        "plot_alpha_curve_strata.py",
    ):
        subprocess.run(
            [sys.executable, str(VIS_DIR / script_name), "--statistics", str(statistics), "--plot-root", str(plot_root)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    assert len(list(plot_root.rglob("*.pdf"))) == 3
    assert len(list(plot_root.rglob("*.png"))) == 3
    assert all(path.stat().st_size > 0 for path in plot_root.rglob("*.*"))
