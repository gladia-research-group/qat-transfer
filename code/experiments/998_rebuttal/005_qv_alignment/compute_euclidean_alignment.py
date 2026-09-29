"""Compute the reviewer-literal global Euclidean QV alignment matrix.

This producer reads the fixed 22 matched FP/QAT checkpoint pairs, projects
their QVs onto exactly the nn.Linear weights touched by ``apply_ptq_``, and
emits one atomic ``euclidean_alignment.json`` artifact.  It does not read any
transfer outcomes and it does not run model inference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
os.chdir(PROJECT_ROOT)

# Environment variables must be loaded before importing project data modules.
from dotenv import load_dotenv

load_dotenv()

import hydra
import numpy as np
import timm
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from common.run_id import guard_run_config, run_id_path
from common.status import StatusWriter
from src.duration import UNIT_MULT, mult_path_frag, mult_tag
from src.vision.data.common import DATASET_NAME_TO_EPOCHS
from src.vision.utils import sanitize_timm_model_name


EXPERIMENT = "998_rebuttal/005_qv_alignment"
STAGE = "euclidean_alignment"
SCHEMA_VERSION = "euclidean_alignment_v1"
GOLDEN_ARTIFACT = "euclidean_alignment.json"
RUN_ID_PARAMS = [
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
    "epoch_mult",
    "vector_scope",
    "module_selector",
    "accumulation_dtype",
]

# Fixed scientific invariant, deliberately not a Hydra option or run-id key.
TASK_EPOCHS: Tuple[Tuple[str, int], ...] = (
    ("CIFAR10", 6),
    ("CIFAR100", 6),
    ("Cars", 35),
    ("DTD", 76),
    ("EMNIST", 2),
    ("EuroSAT", 12),
    ("FER2013", 10),
    ("FashionMNIST", 5),
    ("Flowers102", 147),
    ("Food101", 4),
    ("GTSRB", 11),
    ("ImageNet", 1),
    ("KMNIST", 5),
    ("MNIST", 5),
    ("OxfordIIITPet", 82),
    ("PCAM", 1),
    ("RESISC45", 15),
    ("RenderedSST2", 39),
    ("STL10", 60),
    ("SUN397", 14),
    ("SVHN", 4),
    ("TinyImageNet", 4),
)
TASKS = tuple(task for task, _ in TASK_EPOCHS)
EPOCH_BY_TASK = dict(TASK_EPOCHS)

# Fixed implementation detail: changing it requires a module-selector version.
DOT_BLOCK_SIZE = 1_000_000


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
        # Which training budget the checkpoints were produced at. Orthogonal to
        # epoch_policy: the policy says *which* checkpoint of a run is taken
        # (the final one), the multiplier says how long that run was. Spelled
        # via mult_tag so this run-id component cannot drift from the mult=
        # directory grammar used everywhere else.
        "epoch_mult": mult_tag(UNIT_MULT),
        "vector_scope": "quantized_linear_weight",
        "module_selector": "apply_ptq_linear_v1",
        "accumulation_dtype": "float64",
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
            "configuration violates the approved Euclidean-pilot contract: "
            + json.dumps(mismatches, sort_keys=True)
        )

    live_epochs = {task: int(DATASET_NAME_TO_EPOCHS.get(task, -1)) for task in TASKS}
    if tuple(sorted(DATASET_NAME_TO_EPOCHS)) != TASKS:
        raise RuntimeError(
            "DATASET_NAME_TO_EPOCHS task set/order differs from the locked set; "
            "do not silently change the 005 population"
        )
    if live_epochs != EPOCH_BY_TASK:
        raise RuntimeError(
            "DATASET_NAME_TO_EPOCHS values differ from the locked final epochs: "
            + json.dumps(live_epochs, sort_keys=True)
        )


def _walk_ptq_linear_names(
    parent: nn.Module,
    skip_modules: Iterable[str],
    prefix: str,
) -> List[str]:
    """Mirror ``apply_ptq_`` recursion without modifying model weights."""
    skip = frozenset(skip_modules)
    selected: List[str] = []
    for child_name, child in parent.named_children():
        if child_name in skip:
            continue
        full_name = f"{prefix}{child_name}"
        if isinstance(child, nn.Linear):
            selected.append(full_name)
        else:
            selected.extend(
                _walk_ptq_linear_names(child, skip, prefix=full_name + ".")
            )
    return selected


def _selected_parameters(cfg: DictConfig) -> List[Dict[str, Any]]:
    # ``pretrained=False`` is intentional: only module types/shapes are needed.
    model = timm.create_model(str(cfg.model_name), pretrained=False, num_classes=1)
    module_names = _walk_ptq_linear_names(
        model,
        cfg.ptq_skip_modules,
        prefix="model.",  # ImageClassifier checkpoint wrapper prefix.
    )
    module_by_name = {"model." + name: module for name, module in model.named_modules()}

    offset = 0
    parameters: List[Dict[str, Any]] = []
    for module_name in sorted(module_names):
        module = module_by_name.get(module_name)
        if not isinstance(module, nn.Linear):
            raise RuntimeError(f"selector produced a non-Linear module: {module_name}")
        shape = tuple(int(value) for value in module.weight.shape)
        numel = math.prod(shape)
        parameters.append(
            {
                "module_name": module_name,
                "key": module_name + ".weight",
                "shape": list(shape),
                "numel": numel,
                "offset_start": offset,
                "offset_stop": offset + numel,
            }
        )
        offset += numel
    del model
    if not parameters:
        raise RuntimeError("module selector found no quantized Linear weights")
    return parameters


def _checkpoint_path(cfg: DictConfig, task: str, kind: str) -> Path:
    root_text = os.environ.get("CHECKPOINT_BASE_PATH")
    if not root_text:
        raise EnvironmentError(
            "CHECKPOINT_BASE_PATH is unset; define it in the environment or .env"
        )
    if kind not in ("fp", "qat"):
        raise ValueError(f"unsupported checkpoint kind: {kind}")
    optim_tag = (
        f"optim={cfg.optim}_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}"
        f"_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}"
    )
    path = (
        Path(root_text)
        / "vision"
        / cfg.family
        / kind
        / sanitize_timm_model_name(cfg.model_name)
        / task
        / optim_tag
        / mult_path_frag(cfg.epoch_mult)
    )
    if kind == "qat":
        skip_tag = "-".join(sorted(cfg.qat_skip_modules)) or "none"
        path /= (
            f"qat=bits={cfg.qat_bits}_gran={cfg.qat_granularity}_skip={skip_tag}"
        )
    return path / f"seed={cfg.seed}" / f"classifier_epoch_{EPOCH_BY_TASK[task]}.pt"


def _checkpoint_paths(cfg: DictConfig) -> Dict[str, Dict[str, Path]]:
    paths = {
        task: {
            "fp": _checkpoint_path(cfg, task, "fp"),
            "qat": _checkpoint_path(cfg, task, "qat"),
        }
        for task in TASKS
    }
    missing = [
        str(path)
        for task_paths in paths.values()
        for path in task_paths.values()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of 44 required checkpoints are missing:\n"
            + "\n".join(missing)
        )
    return paths


def _load_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint is not a state-dict mapping: {path}")
    return state


def _write_qv_row(
    destination: np.memmap,
    row_index: int,
    fp_path: Path,
    qat_path: Path,
    parameters: Sequence[Mapping[str, Any]],
) -> None:
    fp_state = _load_state_dict(fp_path)
    qat_state = _load_state_dict(qat_path)
    selected_keys = {str(parameter["key"]) for parameter in parameters}

    for state_name, state in (("FP", fp_state), ("QAT", qat_state)):
        missing = sorted(selected_keys - set(state))
        if missing:
            raise KeyError(
                f"{state_name} checkpoint {fp_path if state_name == 'FP' else qat_path} "
                f"is missing {len(missing)} selected keys; first={missing[:5]}"
            )

    for parameter in parameters:
        key = str(parameter["key"])
        expected_shape = tuple(parameter["shape"])
        fp_tensor = fp_state[key]
        qat_tensor = qat_state[key]
        if tuple(fp_tensor.shape) != expected_shape or tuple(qat_tensor.shape) != expected_shape:
            raise ValueError(
                f"shape mismatch for {key}: expected {expected_shape}, "
                f"FP={tuple(fp_tensor.shape)}, QAT={tuple(qat_tensor.shape)}"
            )
        if fp_tensor.dtype != torch.float32 or qat_tensor.dtype != torch.float32:
            raise TypeError(
                f"selected QV key must match the float32 transfer checkpoints: {key}; "
                f"FP={fp_tensor.dtype}, QAT={qat_tensor.dtype}"
            )
        qv = qat_tensor.detach() - fp_tensor.detach()
        if not bool(torch.isfinite(qv).all()):
            raise ValueError(f"non-finite QV coordinates in {key}")
        start = int(parameter["offset_start"])
        stop = int(parameter["offset_stop"])
        destination[row_index, start:stop] = qv.reshape(-1).numpy()
    destination.flush()


def _gram_float64(vectors: np.memmap, total_parameters: int) -> np.ndarray:
    gram = np.zeros((len(TASKS), len(TASKS)), dtype=np.float64)
    for start in range(0, total_parameters, DOT_BLOCK_SIZE):
        stop = min(total_parameters, start + DOT_BLOCK_SIZE)
        block = np.asarray(vectors[:, start:stop], dtype=np.float64)
        gram += block @ block.T
    return gram


def _relative_or_absolute(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


@hydra.main(
    config_path="../../../../config/experiments/998_rebuttal/005_qv_alignment",
    config_name="compute_euclidean_alignment",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    _validate_contract(cfg)
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

    paths = _checkpoint_paths(cfg)
    parameters = _selected_parameters(cfg)
    total_parameters = sum(int(parameter["numel"]) for parameter in parameters)

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
                        "path": _relative_or_absolute(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                _write_qv_row(
                    vectors,
                    index,
                    task_paths["fp"],
                    task_paths["qat"],
                    parameters,
                )
                status.heartbeat(progress=f"vectors {index + 1}/{len(TASKS)}")

            gram = _gram_float64(vectors, total_parameters)
            del vectors

        if not np.isfinite(gram).all():
            raise ValueError("non-finite values in Euclidean Gram matrix")
        norms = np.sqrt(np.diag(gram))
        zero_norm_tasks = [task for task, norm in zip(TASKS, norms) if norm <= 0.0]
        if zero_norm_tasks:
            raise ValueError(f"zero-norm projected QVs: {zero_norm_tasks}")
        cosine = gram / np.outer(norms, norms)
        if not np.isfinite(cosine).all():
            raise ValueError("non-finite values in Euclidean cosine matrix")

        eigenvalues = np.linalg.eigvalsh(gram)
        diagnostics = {
            "gram_symmetry_max_abs": float(np.max(np.abs(gram - gram.T))),
            "cosine_symmetry_max_abs": float(np.max(np.abs(cosine - cosine.T))),
            "cosine_diagonal_max_abs_deviation_from_one": float(
                np.max(np.abs(np.diag(cosine) - 1.0))
            ),
            "cosine_min": float(np.min(cosine)),
            "cosine_max": float(np.max(cosine)),
            "gram_min_eigenvalue": float(eigenvalues[0]),
            "gram_max_eigenvalue": float(eigenvalues[-1]),
            "dot_block_size": DOT_BLOCK_SIZE,
            "temporary_vector_dtype": "float32",
            "dot_accumulation_dtype": "float64",
        }
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
                "parameters": parameters,
            },
            "checkpoint_provenance": provenance,
            "qv_norms": {
                task: float(norm) for task, norm in zip(TASKS, norms)
            },
            "dot_product_matrix": gram.tolist(),
            "cosine_matrix": cosine.tolist(),
            "diagnostics": diagnostics,
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "timm": timm.__version__,
            },
        }
        _atomic_json(golden, artifact)
        status.heartbeat(progress=f"vectors {len(TASKS)}/{len(TASKS)}; artifact written")
        print(f"[complete] {_relative_or_absolute(golden)}", flush=True)


if __name__ == "__main__":
    main()
