"""Does a stronger quantization-aware finetuner produce a better-transferring QV?

Every quantization vector in this project is `QV = QAT_D - FP_D`, and every QAT
checkpoint behind it was produced by straight-through estimation. The central
claim -- that the QV is a task-agnostic robustness direction rather than a
task-specific displacement -- says nothing about *how* the QAT optimum was
reached, so it leaves an obvious question open: is the transferable content a
property of the quantization grid, which any quantization-aware finetuner would
find, or a property of STE's particular solution?

This phase answers it by swapping the finetuner and nothing else.
`QV_pv = PV_D - FP_D` is built from a PV-tuned donor (`src/pv_tuning.py`,
`finetune_pv.py`), patched onto the receiver's FP checkpoint the same way, and
evaluated under the same round-to-nearest `apply_ptq_` at the same bit-width as
`001_qat_transfer`. Every path fragment, alpha grid, head variant, and result
key mirrors 001, so the two phases' heatmaps are comparable cell by cell and
their difference is exactly the effect of the finetuner. That difference is the
figure this phase exists to produce
(`visualizations/.../008_pv_transfer/qv_transfer_pv_heatmap_minus_qat.py`).

The QV is built from the donor's **straight-through buffer** (the `latent`
entry of `pv_state_epoch_N.pt`), not from the saved checkpoint. A PV checkpoint
stores settled `q*s` weights, so `PV_ckpt - FP` is dominated by quantization
rounding error rather than by anything PV learned -- measured on MNIST, norm
293.8 against the QAT QV's 12.4, cosine 0.035, and the same orthogonality
appears when a *QAT* checkpoint is settled the same way. The buffer is the
exact analogue of what a QAT checkpoint stores (norm 12.37, cosine 0.909),
which is what makes this phase comparable to 001 at all. See `qv.weights`.

Two properties shape how the grid reads:

* The self-pair at alpha=1 is algebraically `FP_tgt + (B_tgt - FP_tgt) =
  B_tgt`, the donor's own buffer; `apply_ptq_` then rounds it to exactly the
  settled checkpoint, so the cell reproduces `000_baselines/pv_ptq`. That
  diagonal is the PV ceiling, not a transfer result, and must be excluded from
  transfer statistics for the same reason 001's diagonal is.
* The self-pair at alpha=0 is `ptq(FP_tgt)`, i.e. the `fp_ptq` baseline -- the
  thing one would otherwise ship, and the number a "win" is measured against.
  It is donor-independent, so it is computed once per receiver and skipped on
  cross-task pairs.

Alphas are swept in-process (`qv.alphas`) with the QV built once per donor
pair, and `skip_existing` makes the sweep resumable, following 005/006/007
rather than 001's external hydra sweep. Selection of lambda* on the validation
split is left to `pick_best_alpha.py` as a pure analysis step over the
`split=val` cells.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from dotenv import load_dotenv

load_dotenv()

import gc
import json
import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from tqdm import tqdm

from src.duration import checkpoint_epochs, mult_path_frag, role_path_frag
from src.pv_tuning import pv_path_frag
from src.quantization import apply_ptq_
from src.task_vectors import TaskVector
from src.vision.data.common import (
    DATASET_NAME_TO_EPOCHS,
    DATASET_NAME_TO_NUM_CLASSES,
    maybe_dictionarize,
)
from src.vision.data.registry import get_dataset
from src.vision.ilharco_timm_supervised.modeling import ImageClassifier
from src.vision.utils import (
    accuracy,
    random_tqdm_color,
    sanitize_timm_model_name,
    set_seed,
)


log = logging.getLogger(__name__)
IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)
HEAD_PREFIX = "model.head."


def _is_head_key(key: str) -> bool:
    return key.startswith(HEAD_PREFIX)


def _optim_frag(cfg: DictConfig) -> str:
    return (
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}"
        f"_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}"
    )


def _pv_frag(cfg: DictConfig) -> str:
    return pv_path_frag(
        bits=cfg.pv.bits,
        granularity=cfg.pv.granularity,
        skip_modules=cfg.pv.skip_modules,
        delta_decay=cfg.pv.delta_decay,
        max_code_change_per_step=cfg.pv.max_code_change_per_step,
        trust_ratio=cfg.pv.trust_ratio,
        p_every=cfg.pv.p_every,
        temperature=cfg.pv.temperature,
    )


def _fp_ckpt_dir(cfg: DictConfig, dataset_name: str, seed: int, epoch_mult) -> str:
    return os.path.join(
        os.environ["CHECKPOINT_BASE_PATH"],
        "vision",
        "ilharco_timm_supervised",
        "fp",
        sanitize_timm_model_name(cfg.model_name),
        dataset_name,
        _optim_frag(cfg),
        mult_path_frag(epoch_mult),
        f"seed={seed}",
    )


def _pv_ckpt_dir(cfg: DictConfig, dataset_name: str, seed: int, epoch_mult) -> str:
    return os.path.join(
        os.environ["CHECKPOINT_BASE_PATH"],
        "vision",
        "ilharco_timm_supervised",
        "pv",
        sanitize_timm_model_name(cfg.model_name),
        dataset_name,
        _optim_frag(cfg),
        mult_path_frag(epoch_mult),
        _pv_frag(cfg),
        f"seed={seed}",
    )


def _eval_dir(
    cfg: DictConfig,
    source_dataset_name: str,
    target_dataset_name: str,
    alpha: float,
    eval_split: str,
) -> str:
    ptq_skip_tag = (
        "-".join(sorted(cfg.ptq.skip_modules))
        if len(cfg.ptq.skip_modules) > 0
        else "none"
    )
    # A truncated run must never write into the real results tree. The older
    # transfer phases have no dryrun fragment here, so a `limit_num_batches`
    # smoke test silently overwrites real accuracies that the analysis scripts
    # then consume as if they were complete. Route them somewhere harmless
    # instead, matching what finetune_*.py already does with `*_dryrun`.
    is_dryrun = cfg.limit_num_batches is not None
    return os.path.join(
        os.environ["EVALUATION_BASE_PATH"],
        "vision",
        "ilharco_timm_supervised",
        "008_pv_transfer_dryrun" if is_dryrun else "008_pv_transfer",
        # The doubled modality segment is redundant but load-bearing for every
        # existing evaluation path in this repository; keep emitting it.
        "vision",
        "qv_transfer_pv",
        sanitize_timm_model_name(cfg.model_name),
        role_path_frag("src", source_dataset_name, cfg.source.seed, cfg.source.epoch_mult),
        role_path_frag("tgt", target_dataset_name, cfg.target.seed, cfg.target.epoch_mult),
        _optim_frag(cfg),
        _pv_frag(cfg),
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={ptq_skip_tag}",
        f"qv=alpha={alpha}",
        f"split={eval_split}",
    )


def evaluate(
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    split: str,
    limit_num_batches: int | None = None,
) -> float:
    if split == "test":
        loader = dataset.test_loader
    elif split == "val":
        loader = dataset.val_loader
    else:
        raise ValueError(f"Unsupported split {split!r}; expected 'val' or 'test'")

    num_batches = len(loader)
    effective_num_batches = (
        min(limit_num_batches, num_batches)
        if limit_num_batches is not None
        else num_batches
    )
    model.to(device=device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        batch_bar = tqdm(
            enumerate(loader),
            total=effective_num_batches,
            desc=f"Evaluating ({split})",
            colour=random_tqdm_color(),
            leave=False,
            **TQDM_KW,
        )
        for i, batch in batch_bar:
            if i >= effective_num_batches:
                break
            batch = maybe_dictionarize(batch)
            inputs = batch["images"].to(device=device)
            labels = batch["labels"].to(device=device, dtype=torch.long)
            logits = model(inputs)
            top1, = accuracy(logits, labels, topk=(1,))
            correct += top1
            total += labels.size(0)
            batch_bar.set_postfix(
                batch=f"{i}/{effective_num_batches}",
                acc=f"{100.0 * correct / total:.2f}%",
            )
    if total == 0:
        raise RuntimeError(f"No samples were evaluated on split={split!r}")
    return float(correct / total)


def _evaluate_head_variant(
    cfg: DictConfig,
    state_dict: dict[str, torch.Tensor],
    dataset,
    num_classes: int,
    device: torch.device,
    eval_split: str,
    head_label: str,
) -> tuple[float, float, list[str]]:
    classifier = ImageClassifier(model_name=cfg.model_name, num_classes=num_classes)
    classifier.load_state_dict(state_dict)
    classifier.to(device=device, dtype=torch.float32)

    accuracy_before = evaluate(
        dataset=dataset,
        model=classifier,
        device=device,
        split=eval_split,
        limit_num_batches=cfg.limit_num_batches,
    )
    quantized_names = apply_ptq_(
        model=classifier,
        bits=cfg.ptq.bits,
        granularity=cfg.ptq.granularity,
        skip_modules=frozenset(cfg.ptq.skip_modules),
    )
    accuracy_after = evaluate(
        dataset=dataset,
        model=classifier,
        device=device,
        split=eval_split,
        limit_num_batches=cfg.limit_num_batches,
    )
    if IS_SLURM:
        log.info(
            "%s head: pre-PTQ=%s, post-PTQ=%s, quantized_modules=%d",
            head_label,
            accuracy_before,
            accuracy_after,
            len(quantized_names),
        )
    else:
        print(
            f"{head_label} head: pre-PTQ={accuracy_before}, "
            f"post-PTQ={accuracy_after}, quantized_modules={len(quantized_names)}"
        )

    del classifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return accuracy_before, accuracy_after, quantized_names


def _run_pair_alpha(
    cfg: DictConfig,
    source_dataset_name: str,
    target_dataset_name: str,
    fp_tgt_sd: dict[str, torch.Tensor],
    pv_tgt_sd: dict[str, torch.Tensor],
    task_vector: TaskVector,
    alpha: float,
    dataset,
    num_classes: int,
    device: torch.device,
    src_epochs: int,
    tgt_epochs: int,
    eval_split: str,
) -> None:
    patched_backbone = {}
    with torch.no_grad():
        for key, target_value in fp_tgt_sd.items():
            if _is_head_key(key):
                continue
            if key in task_vector.vector:
                vector_value = task_vector.vector[key]
                if vector_value.shape != target_value.shape:
                    raise ValueError(
                        f"Shape mismatch on {key}: QV={tuple(vector_value.shape)} "
                        f"target={tuple(target_value.shape)}"
                    )
                patched_backbone[key] = target_value + alpha * vector_value
            else:
                patched_backbone[key] = target_value

    fp_head = {key: value for key, value in fp_tgt_sd.items() if _is_head_key(key)}
    pv_head = {key: value for key, value in pv_tgt_sd.items() if _is_head_key(key)}
    fp_state = {**patched_backbone, **fp_head}
    pv_state = {**patched_backbone, **pv_head}
    del patched_backbone

    fp_before, fp_after, quantized_names_fp = _evaluate_head_variant(
        cfg=cfg,
        state_dict=fp_state,
        dataset=dataset,
        num_classes=num_classes,
        device=device,
        eval_split=eval_split,
        head_label="FP",
    )
    del fp_state
    pv_before, pv_after, quantized_names_pv = _evaluate_head_variant(
        cfg=cfg,
        state_dict=pv_state,
        dataset=dataset,
        num_classes=num_classes,
        device=device,
        eval_split=eval_split,
        head_label="PV",
    )
    del pv_state
    if quantized_names_fp != quantized_names_pv:
        raise RuntimeError("FP- and PV-head variants quantized different module sets")

    fp_source_path = os.path.join(
        _fp_ckpt_dir(cfg, source_dataset_name, cfg.source.seed, cfg.source.epoch_mult),
        f"classifier_epoch_{src_epochs}.pt",
    )
    pv_source_path = os.path.join(
        _pv_ckpt_dir(cfg, source_dataset_name, cfg.source.seed, cfg.source.epoch_mult),
        f"classifier_epoch_{src_epochs}.pt",
    )
    fp_target_path = os.path.join(
        _fp_ckpt_dir(cfg, target_dataset_name, cfg.target.seed, cfg.target.epoch_mult),
        f"classifier_epoch_{tgt_epochs}.pt",
    )
    pv_target_path = os.path.join(
        _pv_ckpt_dir(cfg, target_dataset_name, cfg.target.seed, cfg.target.epoch_mult),
        f"classifier_epoch_{tgt_epochs}.pt",
    )

    num_classes_actual = len(dataset.class_names)
    results = {
        "experiment": "qv_transfer_pv",
        "model_name": cfg.model_name,
        "batch_size": cfg.batch_size,
        "eval_split": eval_split,
        "lr": cfg.lr,
        "wd": cfg.wd,
        "ls": cfg.ls,
        "wl": cfg.wl,
        "max_grad_norm": cfg.max_grad_norm,
        "limit_num_batches": cfg.limit_num_batches,
        "device": str(device),
        "source": {
            "dataset_name": source_dataset_name,
            "seed": cfg.source.seed,
            "limit_num_epochs": cfg.source.limit_num_epochs,
            "epochs": src_epochs,
            "fp_classifier_path": fp_source_path,
            "pv_classifier_path": pv_source_path,
        },
        "target": {
            "dataset_name": target_dataset_name,
            "seed": cfg.target.seed,
            "limit_num_epochs": cfg.target.limit_num_epochs,
            "epochs": tgt_epochs,
            "fp_classifier_path": fp_target_path,
            "pv_classifier_path": pv_target_path,
        },
        "pv": {
            "bits": cfg.pv.bits,
            "granularity": cfg.pv.granularity,
            "skip_modules": list(cfg.pv.skip_modules),
            "delta_decay": cfg.pv.delta_decay,
            "max_code_change_per_step": cfg.pv.max_code_change_per_step,
            "trust_ratio": cfg.pv.trust_ratio,
            "p_every": cfg.pv.p_every,
            "temperature": cfg.pv.temperature,
        },
        "qv": {
            "alpha": alpha,
            "num_keys_in_vector": len(task_vector.vector),
            "weights": cfg.qv.weights,
        },
        "ptq": {
            "bits": cfg.ptq.bits,
            "granularity": cfg.ptq.granularity,
            "skip_modules": list(cfg.ptq.skip_modules),
        },
        "ptq_quantized_modules": quantized_names_fp,
        f"{eval_split}_accuracy_fp_head": fp_before,
        f"{eval_split}_accuracy_fp_head_ptq": fp_after,
        f"{eval_split}_accuracy_pv_head": pv_before,
        f"{eval_split}_accuracy_pv_head_ptq": pv_after,
        "num_classes": num_classes_actual,
        "random_chance": 1.0 / num_classes_actual,
        "comparison_baseline_note": (
            "A win is measured against this receiver's self-pair alpha=0.0 cell, "
            "which is ptq(FP_target) and cross-checks against "
            "000_baselines/fp_ptq. The self-pair alpha=1.0 cell is algebraically "
            "PV_target itself and is the PV ceiling, not a transfer result; "
            "exclude the diagonal from transfer statistics. Cells here are "
            "directly comparable to 001_qat_transfer at matched seed, bits, "
            "granularity, ptq and alpha -- that comparison is the point of the "
            "phase."
        ),
    }
    eval_dir = _eval_dir(
        cfg,
        source_dataset_name,
        target_dataset_name,
        alpha,
        eval_split,
    )
    os.makedirs(eval_dir, exist_ok=True)
    eval_results_path = os.path.join(eval_dir, "eval_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(results, f, indent=2)
    if IS_SLURM:
        log.info("Results saved to: %s", eval_results_path)
    else:
        print(f"Results saved to: {eval_results_path}")


def _run_pair(
    cfg: DictConfig,
    source_dataset_name: str,
    target_dataset_name: str,
    fp_tgt_sd: dict[str, torch.Tensor],
    pv_tgt_sd: dict[str, torch.Tensor],
    dataset,
    num_classes: int,
    device: torch.device,
    tgt_epochs: int,
    eval_split: str,
) -> None:
    alphas = [float(value) for value in OmegaConf.to_container(cfg.qv.alphas, resolve=True)]
    # alpha=0 erases the donor, so the cell is ptq(FP_target) regardless of who
    # the donor is: compute it once, on the self-pair.
    if source_dataset_name != target_dataset_name:
        alphas = [alpha for alpha in alphas if alpha != 0.0]
    if cfg.skip_existing:
        alphas = [
            alpha
            for alpha in alphas
            if not os.path.exists(
                os.path.join(
                    _eval_dir(
                        cfg,
                        source_dataset_name,
                        target_dataset_name,
                        alpha,
                        eval_split,
                    ),
                    "eval_results.json",
                )
            )
        ]
    if not alphas:
        log.info(
            "Skipping source=%s target=%s: no alphas remain",
            source_dataset_name,
            target_dataset_name,
        )
        return

    src_epochs = checkpoint_epochs(
        source_dataset_name, DATASET_NAME_TO_EPOCHS, cfg.source.limit_num_epochs
    )
    fp_source_path = os.path.join(
        _fp_ckpt_dir(cfg, source_dataset_name, cfg.source.seed, cfg.source.epoch_mult),
        f"classifier_epoch_{src_epochs}.pt",
    )
    pv_source_path = os.path.join(
        _pv_ckpt_dir(cfg, source_dataset_name, cfg.source.seed, cfg.source.epoch_mult),
        f"classifier_epoch_{src_epochs}.pt",
    )
    for path in (fp_source_path, pv_source_path):
        if not os.path.exists(path):
            log.warning("Skipping source=%s: checkpoint missing: %s", source_dataset_name, path)
            return

    fp_src_sd = torch.load(fp_source_path, map_location="cpu")
    pv_src_sd = torch.load(pv_source_path, map_location="cpu")
    src_keys = {key for key in fp_src_sd if not _is_head_key(key)}
    pv_keys = {key for key in pv_src_sd if not _is_head_key(key)}
    tgt_keys = {key for key in fp_tgt_sd if not _is_head_key(key)}
    if src_keys != pv_keys or src_keys != tgt_keys:
        raise ValueError(
            "Backbone key sets differ across FP source, PV source, and FP target; "
            "refusing to construct a partial QV"
        )

    # Which donor weights the QV is built from. This is not a cosmetic choice.
    #
    # A PV checkpoint stores the *settled* weights q*s, so `PV_ckpt - FP` is
    # dominated by the quantization rounding error rather than by anything PV
    # learned: measured on MNIST, that vector has norm 293.8 against the QAT
    # QV's 12.4 and a cosine of 0.035 with it -- essentially orthogonal, and
    # the same orthogonality appears if a *QAT* checkpoint is settled the same
    # way, with no PV involved. Comparing it to 001 would compare a
    # displacement-into-the-quantized-set against a latent-space displacement.
    #
    # The straight-through buffer in the sidecar is the exact analogue of what
    # a QAT checkpoint stores, and gives norm 12.37 with cosine 0.909 to the
    # QAT QV -- same magnitude, comparable object, and a real 43% difference in
    # direction that is attributable to the finetuner. That is the experiment.
    #
    # `settled` is retained only as an ablation; it is expected to behave like
    # 007_gptq_transfer, which showed that transferring a quantization
    # displacement does not work.
    latent_by_key: dict[str, torch.Tensor] = {}
    if cfg.qv.weights == "latent":
        pv_state_path = os.path.join(
            _pv_ckpt_dir(cfg, source_dataset_name, cfg.source.seed, cfg.source.epoch_mult),
            f"pv_state_epoch_{src_epochs}.pt",
        )
        if not os.path.exists(pv_state_path):
            log.warning(
                "Skipping source=%s: PV sidecar missing: %s",
                source_dataset_name,
                pv_state_path,
            )
            return
        pv_sidecar = torch.load(pv_state_path, map_location="cpu")
        latent_by_key = {
            f"{module_name}.weight": entry["latent"]
            for module_name, entry in pv_sidecar.items()
        }
        unknown = set(latent_by_key) - src_keys
        if unknown:
            raise ValueError(
                f"PV sidecar carries {len(unknown)} keys absent from the FP backbone, "
                f"e.g. {sorted(unknown)[:3]}; refusing to construct a mismatched QV"
            )
        del pv_sidecar
    elif cfg.qv.weights != "settled":
        raise ValueError(
            f"Unsupported qv.weights {cfg.qv.weights!r}; expected 'latent' or 'settled'"
        )

    vector = {}
    with torch.no_grad():
        for key in src_keys:
            value = fp_src_sd[key]
            if value.dtype in (torch.int64, torch.uint8):
                continue
            # Non-quantized backbone params (norms, biases) are untouched by
            # settling, so the checkpoint and the buffer agree on them and the
            # .get() falls through to the checkpoint by design.
            trained = latent_by_key.get(key, pv_src_sd[key])
            vector[key] = trained - value
    task_vector = TaskVector(vector=vector)
    del fp_src_sd, pv_src_sd, latent_by_key

    for alpha in alphas:
        _run_pair_alpha(
            cfg=cfg,
            source_dataset_name=source_dataset_name,
            target_dataset_name=target_dataset_name,
            fp_tgt_sd=fp_tgt_sd,
            pv_tgt_sd=pv_tgt_sd,
            task_vector=task_vector,
            alpha=alpha,
            dataset=dataset,
            num_classes=num_classes,
            device=device,
            src_epochs=src_epochs,
            tgt_epochs=tgt_epochs,
            eval_split=eval_split,
        )
    del task_vector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/008_pv_transfer",
    config_name="qv_transfer_pv",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(dict(cfg), expand_all=True)

    source_dataset_names = OmegaConf.to_container(cfg.source.dataset_names, resolve=True)
    target_dataset_names = OmegaConf.to_container(cfg.target.dataset_names, resolve=True)
    if cfg.eval_split not in ("val", "test"):
        raise ValueError(f"Unsupported eval_split {cfg.eval_split!r}; expected 'val' or 'test'")

    set_seed(cfg.target.seed)
    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    num_workers = int(os.environ["TORCH_NUM_WORKERS"])
    total_pairs = len(source_dataset_names) * len(target_dataset_names)
    pair_idx = 0

    for target_idx, target_dataset_name in enumerate(target_dataset_names):
        if IS_SLURM:
            log.info(
                "=== Target %d/%d: %s ===",
                target_idx + 1,
                len(target_dataset_names),
                target_dataset_name,
            )
        tgt_epochs = checkpoint_epochs(
        target_dataset_name, DATASET_NAME_TO_EPOCHS, cfg.target.limit_num_epochs
    )
        num_classes = DATASET_NAME_TO_NUM_CLASSES[target_dataset_name]
        fp_target_path = os.path.join(
            _fp_ckpt_dir(cfg, target_dataset_name, cfg.target.seed, cfg.target.epoch_mult),
            f"classifier_epoch_{tgt_epochs}.pt",
        )
        pv_target_path = os.path.join(
            _pv_ckpt_dir(cfg, target_dataset_name, cfg.target.seed, cfg.target.epoch_mult),
            f"classifier_epoch_{tgt_epochs}.pt",
        )
        missing = [path for path in (fp_target_path, pv_target_path) if not os.path.exists(path)]
        if missing:
            for path in missing:
                log.warning("Skipping target=%s: checkpoint missing: %s", target_dataset_name, path)
            pair_idx += len(source_dataset_names)
            continue

        fp_tgt_sd = torch.load(fp_target_path, map_location="cpu")
        pv_tgt_sd = torch.load(pv_target_path, map_location="cpu")
        temporary_classifier = ImageClassifier(
            model_name=cfg.model_name,
            num_classes=num_classes,
        )
        dataset = get_dataset(
            dataset_name=target_dataset_name,
            preprocess_train=temporary_classifier.train_preprocess,
            preprocess_inference=temporary_classifier.val_preprocess,
            batch_size=cfg.batch_size,
            num_workers=num_workers,
            seed=cfg.target.seed,
        )
        del temporary_classifier

        for source_dataset_name in source_dataset_names:
            pair_idx += 1
            if IS_SLURM:
                log.info(
                    "--- Pair %d/%d: source=%s target=%s ---",
                    pair_idx,
                    total_pairs,
                    source_dataset_name,
                    target_dataset_name,
                )
            _run_pair(
                cfg=cfg,
                source_dataset_name=source_dataset_name,
                target_dataset_name=target_dataset_name,
                fp_tgt_sd=fp_tgt_sd,
                pv_tgt_sd=pv_tgt_sd,
                dataset=dataset,
                num_classes=num_classes,
                device=device,
                tgt_epochs=tgt_epochs,
                eval_split=cfg.eval_split,
            )

        del dataset, fp_tgt_sd, pv_tgt_sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if IS_SLURM:
        log.info("All %d pairs completed. Forcing exit.", total_pairs)
        os._exit(0)


if __name__ == "__main__":
    main()
