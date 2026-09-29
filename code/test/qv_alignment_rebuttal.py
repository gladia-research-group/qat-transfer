"""Synthetic and contract tests for rebuttal experiment 005.

These tests never resolve CHECKPOINT_BASE_PATH and never load a real model
checkpoint.  The only real experiment input they read is the already-existing
22x22 outcome summary, whose metadata and pair completeness are validated.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPERIMENT_DIR = ROOT / "code/experiments/998_rebuttal/005_qv_alignment"
PRODUCER = _load_module(
    "qv_alignment_producer", EXPERIMENT_DIR / "compute_euclidean_alignment.py"
)
ANALYZER = _load_module(
    "qv_alignment_analyzer", EXPERIMENT_DIR / "analyze_euclidean_alignment.py"
)
ROW_PRODUCER = _load_module(
    "qv_alignment_row_producer", EXPERIMENT_DIR / "compute_rowwise_alignment.py"
)
ROW_ANALYZER = _load_module(
    "qv_alignment_row_analyzer", EXPERIMENT_DIR / "analyze_rowwise_alignment.py"
)
CONFIG_DIR = ROOT / "config/experiments/998_rebuttal/005_qv_alignment"
VIS_DIR = ROOT / "visualizations/998_rebuttal/005_qv_alignment"
VIS_COMMON = _load_module("qv_alignment_visualization_common", VIS_DIR / "_common.py")
OUTCOME_PATH = ROOT / (
    "evaluations/998_rebuttal/001_zero_shot_reframing/seed=2038/"
    "qat=bits=3_gran=channel/ptq=bits=3_gran=channel/split=test/"
    "win_loss_ilharco_timm_supervised.json"
)


def test_configs_satisfy_frozen_contracts_and_run_id_order():
    producer_cfg = OmegaConf.load(CONFIG_DIR / "compute_euclidean_alignment.yaml")
    analyzer_cfg = OmegaConf.load(CONFIG_DIR / "analyze_euclidean_alignment.yaml")
    PRODUCER._validate_contract(producer_cfg)
    ANALYZER._validate_contract(analyzer_cfg)
    assert PRODUCER.RUN_ID_PARAMS == ANALYZER.PRODUCER_RUN_ID_PARAMS
    assert ANALYZER.RUN_ID_PARAMS == [
        "ptq_bits",
        "ptq_granularity",
        "outcome_protocol",
        "outcome_split",
        "unit_alpha",
        "analysis_spec",
        "n_permutations",
        "permutation_seed",
    ]


def test_rowwise_configs_differ_only_by_aggregation_and_provenance_tokens():
    producer_cfg = OmegaConf.load(CONFIG_DIR / "compute_euclidean_alignment.yaml")
    row_producer_cfg = OmegaConf.load(CONFIG_DIR / "compute_rowwise_alignment.yaml")
    analyzer_cfg = OmegaConf.load(CONFIG_DIR / "analyze_euclidean_alignment.yaml")
    row_analyzer_cfg = OmegaConf.load(CONFIG_DIR / "analyze_rowwise_alignment.yaml")

    ROW_PRODUCER._validate_contract(row_producer_cfg)
    ROW_ANALYZER._validate_contract(row_analyzer_cfg)

    producer_plain = OmegaConf.to_container(producer_cfg, resolve=True)
    row_producer_plain = OmegaConf.to_container(row_producer_cfg, resolve=True)
    assert isinstance(producer_plain, dict) and isinstance(row_producer_plain, dict)
    assert row_producer_plain.pop("aggregation_spec") == "row_cosine_mean_v1"
    assert row_producer_plain.pop("smoke") is False
    assert row_producer_plain == producer_plain

    analyzer_plain = OmegaConf.to_container(analyzer_cfg, resolve=True)
    row_analyzer_plain = OmegaConf.to_container(row_analyzer_cfg, resolve=True)
    assert isinstance(analyzer_plain, dict) and isinstance(row_analyzer_plain, dict)
    assert row_analyzer_plain.pop("aggregation_spec") == "row_cosine_mean_v1"
    assert row_analyzer_plain["analysis_spec"] == "qv_alignment_rowwise_v1"
    row_analyzer_plain["analysis_spec"] = "qv_alignment_v1"
    assert row_analyzer_plain == analyzer_plain

    assert ROW_PRODUCER.RUN_ID_PARAMS == [
        *PRODUCER.RUN_ID_PARAMS,
        "aggregation_spec",
    ]
    assert ROW_ANALYZER.RUN_ID_PARAMS == ANALYZER.RUN_ID_PARAMS
    assert ROW_ANALYZER.COMPARISONS is ROW_ANALYZER.global_analysis.COMPARISONS


def test_selector_matches_apply_ptq_recursion_on_toy_model():
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Linear(5, 4), nn.ReLU(), nn.Linear(4, 3))
            self.norm = nn.LayerNorm(3)
            self.head = nn.Sequential(nn.Linear(3, 2))

    from src.quantization import apply_ptq_

    selected = PRODUCER._walk_ptq_linear_names(Toy(), ["head"], prefix="")
    quantized = apply_ptq_(
        Toy(), bits=3, granularity="channel", skip_modules=frozenset({"head"})
    )
    assert selected == quantized == ["stem.0", "stem.2"]


def test_vit_selector_is_nonempty_and_excludes_the_head():
    cfg = OmegaConf.load(CONFIG_DIR / "compute_euclidean_alignment.yaml")
    parameters = PRODUCER._selected_parameters(cfg)
    keys = [parameter["key"] for parameter in parameters]
    assert len(keys) > 0
    assert len(keys) == len(set(keys))
    assert all(key.startswith("model.") and key.endswith(".weight") for key in keys)
    assert all(".head." not in key and key != "model.head.weight" for key in keys)
    assert sum(parameter["numel"] for parameter in parameters) > 0


def test_checkpoint_paths_match_the_frozen_fp_and_qat_templates(monkeypatch):
    cfg = OmegaConf.load(CONFIG_DIR / "compute_euclidean_alignment.yaml")
    monkeypatch.setenv("CHECKPOINT_BASE_PATH", "/synthetic-checkpoints")
    fp = PRODUCER._checkpoint_path(cfg, "Cars", "fp")
    qat = PRODUCER._checkpoint_path(cfg, "Cars", "qat")
    common = (
        "/synthetic-checkpoints/vision/ilharco_timm_supervised/{kind}/"
        "vit_base_patch16_224_orig_in21k/Cars/"
        "optim=adamw_lr=1e-05_wd=0.1_ls=0.0_wl=500_mgn=1.0_bs=128/"
    )
    assert str(fp) == common.format(kind="fp") + "seed=2038/classifier_epoch_35.pt"
    assert str(qat) == (
        common.format(kind="qat")
        + "qat=bits=3_gran=channel_skip=head/seed=2038/classifier_epoch_35.pt"
    )


def test_float64_gram_matches_one_globally_concatenated_vector():
    rng = np.random.default_rng(7)
    layer_a = rng.normal(size=(len(PRODUCER.TASKS), 3)).astype(np.float32)
    layer_b = rng.normal(size=(len(PRODUCER.TASKS), 8)).astype(np.float32)
    concatenated = np.concatenate([layer_a, layer_b], axis=1)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "vectors.mmap"
        vectors = np.memmap(
            path,
            mode="w+",
            dtype=np.float32,
            shape=concatenated.shape,
        )
        vectors[:] = concatenated
        vectors.flush()
        observed = PRODUCER._gram_float64(vectors, concatenated.shape[1])
    expected = concatenated.astype(np.float64) @ concatenated.astype(np.float64).T
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)


def test_rowwise_cosine_is_explicit_unweighted_mean_over_all_matching_rows():
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
        {"key": "a.weight", "shape": [2, 2], "offset_start": 0, "offset_stop": 4},
        {"key": "b.weight", "shape": [1, 3], "offset_start": 4, "offset_stop": 7},
    ]
    observed, diagnostics = ROW_PRODUCER._mean_matching_row_cosines_float64(
        vectors, parameters, ("a", "b", "c"), coordinate_block_size=2
    )

    explicit = np.zeros((3, 3), dtype=np.float64)
    all_rows = [layer_a[:, row, :] for row in range(2)] + [layer_b[:, 0, :]]
    for row in all_rows:
        row64 = row.astype(np.float64)
        normalized = row64 / np.linalg.norm(row64, axis=1)[:, None]
        explicit += normalized @ normalized.T
    explicit /= len(all_rows)
    np.testing.assert_allclose(observed, explicit, rtol=0.0, atol=1e-12)
    assert diagnostics["n_rows"] == 3

    global_cosine = vectors.astype(np.float64) @ vectors.astype(np.float64).T
    global_norms = np.sqrt(np.diag(global_cosine))
    global_cosine /= np.outer(global_norms, global_norms)
    assert not np.allclose(observed, global_cosine)

    layer_means = []
    for layer in (layer_a, layer_b):
        subtotal = np.zeros((3, 3), dtype=np.float64)
        for row_index in range(layer.shape[1]):
            row = layer[:, row_index, :].astype(np.float64)
            normalized = row / np.linalg.norm(row, axis=1)[:, None]
            subtotal += normalized @ normalized.T
        layer_means.append(subtotal / layer.shape[1])
    assert not np.allclose(observed, np.mean(layer_means, axis=0))


def test_rowwise_cosine_rejects_any_zero_norm_row():
    vectors = np.asarray([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32)
    parameters = [
        {"key": "toy.weight", "shape": [1, 2], "offset_start": 0, "offset_stop": 2}
    ]
    with pytest.raises(ValueError, match="zero-norm QV rows"):
        ROW_PRODUCER._mean_matching_row_cosines_float64(
            vectors, parameters, ("valid", "zero")
        )


def test_vit_rowwise_population_is_exactly_all_82944_selected_rows():
    cfg = OmegaConf.load(CONFIG_DIR / "compute_rowwise_alignment.yaml")
    parameters = ROW_PRODUCER.global_alignment._selected_parameters(cfg)
    assert len(parameters) == 48
    assert sum(int(parameter["shape"][0]) for parameter in parameters) == 82_944


def test_rowwise_real_entrypoint_smoke_is_read_only_and_meaningful():
    result = subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT_DIR / "compute_rowwise_alignment.py"),
            "smoke=true",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (
        "[smoke] row_cosine_mean_v1 passed: finite symmetric unit-diagonal "
        "3x3 matrix over 3 matching rows"
    ) in result.stdout


def test_rowwise_analyzer_squares_only_the_aggregated_similarity():
    pair = {
        "delta": 0.1,
        "delta_best": 0.2,
        "recovery": 0.3,
        "recovery_best": 0.4,
        "alpha_best": 0.5,
        "baseline_acc": 0.6,
        "ceiling_delta": 0.7,
    }
    record = ROW_ANALYZER._joined_record("D", "R", -0.25, pair, 1.0)
    assert record["cosine"] == -0.25
    assert record["cosine_sq"] == 0.0625
    assert record["delta"] == pair["delta"]
    assert record["delta_best"] == pair["delta_best"]
    assert record["recovery"] == pair["recovery"]
    assert record["recovery_best"] == pair["recovery_best"]
    assert record["best_alpha"] == pair["alpha_best"]
    assert record["baseline_acc"] == pair["baseline_acc"]
    assert record["ceiling_delta"] == pair["ceiling_delta"]


def test_tie_aware_ranks_and_correlations():
    ranks = ANALYZER._average_ranks(np.asarray([4.0, 1.0, 1.0, 9.0]))
    np.testing.assert_array_equal(ranks, np.asarray([3.0, 1.5, 1.5, 4.0]))
    perfect = ANALYZER._correlation([1, 2, 2, 4], [10, 20, 20, 40], "spearman")
    assert perfect["coefficient"] == 1.0
    constant = ANALYZER._correlation([1, 1, 1], [1, 2, 3], "pearson")
    assert constant == {"coefficient": None, "n": 3, "reason": "constant_vector"}


def test_qap_uses_simultaneous_axis_reindexing_and_persists_full_null():
    x = np.asarray(
        [
            [1.0, 0.2, -0.3, 0.5],
            [0.2, 1.0, 0.7, -0.1],
            [-0.3, 0.7, 1.0, 0.4],
            [0.5, -0.1, 0.4, 1.0],
        ]
    )
    y = np.asarray(
        [
            [0.0, 0.3, -0.2, 0.8],
            [-0.4, 0.0, 0.9, 0.1],
            [0.5, 0.2, 0.0, -0.6],
            [0.7, -0.3, 0.4, 0.0],
        ]
    )
    mask = ~np.eye(4, dtype=bool)
    permutations = np.asarray(
        [[0, 1, 2, 3], [1, 0, 2, 3], [3, 2, 1, 0]], dtype=np.uint8
    )
    result = ANALYZER._qap_result(
        x,
        y,
        mask,
        permutations,
        "pearson",
        seed=13,
        permutation_digest="synthetic",
    )
    expected_null = [
        ANALYZER._correlation(x[np.ix_(perm, perm)][mask], y[mask], "pearson")[
            "coefficient"
        ]
        for perm in permutations
    ]
    np.testing.assert_allclose(result["null_distribution"], expected_null, atol=1e-15)
    assert len(result["null_distribution"]) == len(permutations)
    assert result["exceedance_count"] == sum(
        abs(value) >= abs(result["observed"])
        for value in result["null_distribution"]
    )
    assert result["p_two_sided"] == (1 + result["exceedance_count"]) / 4


def _synthetic_points():
    points = []
    for donor_index, donor in enumerate(ANALYZER.TASKS):
        for receiver_index, receiver in enumerate(ANALYZER.TASKS):
            if donor == receiver:
                continue
            cosine = (donor_index - receiver_index) / len(ANALYZER.TASKS)
            points.append(
                {
                    "donor": donor,
                    "receiver": receiver,
                    "cosine": cosine,
                    "delta": 0.4 * cosine + 0.01 * receiver_index,
                }
            )
    return points


def test_influence_population_sizes_match_the_predeclared_matrix_design():
    influence = ANALYZER._influence(_synthetic_points(), "cosine", "delta")
    assert len(influence["leave_one_receiver_out"]) == 22
    assert len(influence["leave_one_donor_out"]) == 22
    assert len(influence["per_receiver"]) == 22
    assert len(influence["per_donor"]) == 22
    assert all(
        row["spearman"]["n"] == 441
        for row in influence["leave_one_receiver_out"]
    )
    assert all(
        row["spearman"]["n"] == 441 for row in influence["leave_one_donor_out"]
    )
    assert all(row["spearman"]["n"] == 21 for row in influence["per_receiver"])
    assert all(row["spearman"]["n"] == 21 for row in influence["per_donor"])


def test_existing_outcome_summary_is_a_complete_matching_22x22_source():
    cfg = OmegaConf.load(CONFIG_DIR / "analyze_euclidean_alignment.yaml")
    outcome = json.loads(OUTCOME_PATH.read_text())
    pairs = ANALYZER._validate_outcome(outcome, cfg)
    assert len(pairs) == 484
    assert set(pairs) == {
        (donor, receiver) for donor in ANALYZER.TASKS for receiver in ANALYZER.TASKS
    }


def _synthetic_statistics():
    tasks = list(ANALYZER.TASKS)
    points = []
    diagonal = []
    for donor_index, donor in enumerate(tasks):
        for receiver_index, receiver in enumerate(tasks):
            if donor == receiver:
                cosine = 1.0
            else:
                cosine = (donor_index + receiver_index - 21) / 30.0
            record = {
                "donor": donor,
                "receiver": receiver,
                "cosine": cosine,
                "cosine_sq": cosine * cosine,
                "delta": 0.2 * cosine + 0.01 * receiver_index,
                "delta_best": 0.25 * cosine + 0.01 * receiver_index,
                "recovery": 0.1 + 0.3 * cosine * cosine,
                "recovery_best": 0.2 + 0.4 * cosine * cosine,
                "unit_alpha": 1.0,
                "best_alpha": 0.5,
                "baseline_acc": 0.4,
                "ceiling_delta": 0.5,
            }
            (diagonal if donor == receiver else points).append(record)

    leave_receiver = []
    leave_donor = []
    for index, task in enumerate(tasks):
        coefficient = 0.2 + 0.01 * (index - 11)
        result = {"coefficient": coefficient, "n": 441, "reason": None}
        leave_receiver.append(
            {"omitted_receiver": task, "spearman": result, "pearson": result}
        )
        leave_donor.append({"omitted_donor": task, "spearman": result, "pearson": result})

    def comparison(role):
        correlation = {"coefficient": 0.25, "n": 462, "reason": None}
        return {
            "role": role,
            "observed": {"spearman": correlation, "pearson": correlation},
            "qap": {
                "spearman": {"p_two_sided": 0.02},
                "pearson": {"p_two_sided": 0.03},
            },
            "influence": {
                "leave_one_receiver_out": leave_receiver,
                "leave_one_donor_out": leave_donor,
            },
        }

    return {
        "schema_version": "euclidean_statistics_v1",
        "analysis_spec": "qv_alignment_v1",
        "provenance": {
            "task_order": tasks,
            "producer_run_id_path": "family=synthetic/model=vit",
            "analyzer_run_id_path": "analysis_spec=qv_alignment_v1",
        },
        "points": points,
        "diagonal_audit": diagonal,
        "statistics": {
            "primary_comparison": "signed_cosine_vs_delta",
            "comparisons": {
                "signed_cosine_vs_delta": comparison("primary"),
                "signed_cosine_vs_delta_best": comparison(
                    "descriptive_diagnostic"
                ),
                "cosine_sq_vs_recovery_best": comparison(
                    "theory_adjacent_secondary"
                ),
            },
        },
        "missing": [],
    }


def _synthetic_rowwise_statistics():
    data = _synthetic_statistics()
    data["schema_version"] = "rowwise_statistics_v1"
    data["analysis_spec"] = "qv_alignment_rowwise_v1"
    data["alignment_aggregation"] = "row_cosine_mean_v1"
    data["provenance"]["producer_stage"] = "rowwise_alignment"
    data["provenance"]["producer_run_id_path"] += (
        "/aggregation_spec=row_cosine_mean_v1"
    )
    data["provenance"]["analyzer_run_id_path"] = (
        "analysis_spec=qv_alignment_rowwise_v1"
    )
    return data


def test_alignment_heatmap_matches_001_dataset_disposition():
    data = _synthetic_statistics()
    tasks, receiver_by_donor = VIS_COMMON.matrix_in_001_disposition(data, "delta")
    expected_tasks = [
        "Cars",
        "CIFAR10",
        "CIFAR100",
        "DTD",
        "EMNIST",
        "EuroSAT",
        "FashionMNIST",
        "FER2013",
        "Flowers102",
        "Food101",
        "GTSRB",
        "ImageNet",
        "KMNIST",
        "MNIST",
        "OxfordIIITPet",
        "PCAM",
        "RenderedSST2",
        "RESISC45",
        "STL10",
        "SUN397",
        "SVHN",
        "TinyImageNet",
    ]
    assert tasks == expected_tasks

    records = {
        (record["donor"], record["receiver"]): record
        for record in data["points"] + data["diagonal_audit"]
    }
    expected = np.asarray(
        [
            [records[(donor, receiver)]["delta"] for donor in expected_tasks]
            for receiver in expected_tasks
        ]
    )
    np.testing.assert_allclose(receiver_by_donor, expected, rtol=0.0, atol=0.0)


def test_all_visualizations_render_pdf_and_png_from_statistics_only():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        statistics = tmp / "euclidean_statistics.json"
        statistics.write_text(json.dumps(_synthetic_statistics()))
        plot_root = tmp / "plots"
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(tmp / "matplotlib")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        scripts = (
            "plot_alignment_heatmap.py",
            "plot_alignment_associations.py",
            "plot_alignment_influence.py",
        )
        for script in scripts:
            subprocess.run(
                [
                    sys.executable,
                    str(VIS_DIR / script),
                    "--statistics",
                    str(statistics),
                    "--plot-root",
                    str(plot_root),
                ],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
        outputs = sorted(plot_root.rglob("*.pdf")) + sorted(plot_root.rglob("*.png"))
        assert len(outputs) == 6
        assert all(path.stat().st_size > 0 for path in outputs)


def test_best_alpha_visualizations_render_from_existing_statistics_only():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        statistics = tmp / "euclidean_statistics.json"
        statistics.write_text(json.dumps(_synthetic_statistics()))
        plot_root = tmp / "plots"
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(tmp / "matplotlib")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        scripts = (
            "plot_best_alpha_heatmap.py",
            "plot_best_alpha_associations.py",
            "plot_best_alpha_influence.py",
        )
        for script in scripts:
            subprocess.run(
                [
                    sys.executable,
                    str(VIS_DIR / script),
                    "--statistics",
                    str(statistics),
                    "--plot-root",
                    str(plot_root),
                ],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
        outputs = sorted(plot_root.rglob("*.pdf")) + sorted(plot_root.rglob("*.png"))
        assert len(outputs) == 6
        assert all(path.stat().st_size > 0 for path in outputs)


def test_best_alpha_visualizations_reject_incomplete_artifacts():
    scripts = (
        "plot_best_alpha_heatmap.py",
        "plot_best_alpha_associations.py",
        "plot_best_alpha_influence.py",
    )
    invalid_variants = []
    missing_field = _synthetic_statistics()
    del missing_field["points"][0]["delta_best"]
    invalid_variants.append((missing_field, "delta_best"))
    missing_comparison = _synthetic_statistics()
    del missing_comparison["statistics"]["comparisons"][
        "signed_cosine_vs_delta_best"
    ]
    invalid_variants.append((missing_comparison, "signed_cosine_vs_delta_best"))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(tmp / "matplotlib")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for variant_index, (data, expected_error) in enumerate(invalid_variants):
            statistics = tmp / f"invalid_{variant_index}.json"
            statistics.write_text(json.dumps(data))
            for script in scripts:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(VIS_DIR / script),
                        "--statistics",
                        str(statistics),
                        "--plot-root",
                        str(tmp / "plots"),
                    ],
                    cwd=ROOT,
                    env=env,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                assert result.returncode != 0
                assert expected_error in result.stderr


def test_all_rowwise_visualizations_are_exact_parallel_pdf_png_outputs():
    data = _synthetic_rowwise_statistics()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        statistics = tmp / "rowwise_statistics.json"
        statistics.write_text(json.dumps(data))
        plot_root = tmp / "plots"
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(tmp / "matplotlib")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        scripts = (
            "plot_rowwise_alignment_heatmap.py",
            "plot_rowwise_alignment_associations.py",
            "plot_rowwise_alignment_influence.py",
        )
        for script in scripts:
            subprocess.run(
                [
                    sys.executable,
                    str(VIS_DIR / script),
                    "--statistics",
                    str(statistics),
                    "--plot-root",
                    str(plot_root),
                ],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
        outputs = sorted(plot_root.rglob("*.pdf")) + sorted(plot_root.rglob("*.png"))
        assert len(outputs) == 6
        assert all(path.stat().st_size > 0 for path in outputs)


def test_rowwise_best_alpha_visualizations_render_to_full_provenance_paths():
    data = _synthetic_rowwise_statistics()
    scripts = (
        "plot_rowwise_best_alpha_heatmap.py",
        "plot_rowwise_best_alpha_associations.py",
        "plot_rowwise_best_alpha_influence.py",
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        statistics = tmp / "rowwise_statistics.json"
        statistics.write_text(json.dumps(data))
        plot_root = tmp / "plots"
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(tmp / "matplotlib")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        for script in scripts:
            subprocess.run(
                [
                    sys.executable,
                    str(VIS_DIR / script),
                    "--statistics",
                    str(statistics),
                    "--plot-root",
                    str(plot_root),
                ],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            script_stem = Path(script).stem
            expected_dir = (
                plot_root
                / "998_rebuttal/005_qv_alignment"
                / script_stem
                / "rowwise_alignment"
                / data["provenance"]["producer_run_id_path"]
                / "analysis"
                / data["provenance"]["analyzer_run_id_path"]
            )
            assert sorted(path.suffix for path in expected_dir.iterdir()) == [
                ".pdf",
                ".png",
            ]
        outputs = sorted(plot_root.rglob("*.pdf")) + sorted(plot_root.rglob("*.png"))
        assert len(outputs) == 6
        assert all(path.stat().st_size > 0 for path in outputs)


@pytest.mark.parametrize("field", ("best_alpha", "delta_best", "recovery_best"))
def test_rowwise_best_alpha_visualizations_reject_missing_outcome_fields(field):
    data = _synthetic_rowwise_statistics()
    del data["points"][0][field]
    _assert_rowwise_best_alpha_scripts_reject(data, field)


@pytest.mark.parametrize(
    "comparison",
    ("signed_cosine_vs_delta_best", "cosine_sq_vs_recovery_best"),
)
def test_rowwise_best_alpha_visualizations_reject_missing_comparisons(comparison):
    data = _synthetic_rowwise_statistics()
    del data["statistics"]["comparisons"][comparison]
    _assert_rowwise_best_alpha_scripts_reject(data, comparison)


def test_rowwise_best_alpha_visualizations_reject_global_schema():
    _assert_rowwise_best_alpha_scripts_reject(
        _synthetic_statistics(),
        "rowwise_statistics_v1",
    )


def test_rowwise_best_alpha_influence_requires_all_22_task_records():
    data = _synthetic_rowwise_statistics()
    comparison = data["statistics"]["comparisons"]["signed_cosine_vs_delta_best"]
    comparison["influence"]["leave_one_receiver_out"].pop()
    _assert_visualization_rejects(
        data,
        "plot_rowwise_best_alpha_influence.py",
        "must contain 22 records",
    )


def test_rowwise_best_alpha_output_rejects_unsafe_provenance_path():
    data = _synthetic_rowwise_statistics()
    data["provenance"]["producer_run_id_path"] = "../escape"
    _assert_visualization_rejects(
        data,
        "plot_rowwise_best_alpha_heatmap.py",
        "unsafe run-id path",
    )


def _assert_rowwise_best_alpha_scripts_reject(data, expected_error):
    for script in (
        "plot_rowwise_best_alpha_heatmap.py",
        "plot_rowwise_best_alpha_associations.py",
        "plot_rowwise_best_alpha_influence.py",
    ):
        _assert_visualization_rejects(data, script, expected_error)


def _assert_visualization_rejects(data, script, expected_error):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        statistics = tmp / "invalid.json"
        statistics.write_text(json.dumps(data))
        env = os.environ.copy()
        env["MPLCONFIGDIR"] = str(tmp / "matplotlib")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                str(VIS_DIR / script),
                "--statistics",
                str(statistics),
                "--plot-root",
                str(tmp / "plots"),
            ],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert expected_error in result.stderr
