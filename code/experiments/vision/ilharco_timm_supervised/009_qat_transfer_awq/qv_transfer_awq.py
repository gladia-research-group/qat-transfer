# ==============================================================================
# Quantization Vector (QV, qv)
# ==============================================================================
# Let PT be a pre-trained model
#
# Let FP_{S1}^{D1} be a fine-tuning in full precision of PT on dataset D1
# with seed S1
#
# Let QAT_{S1,Q}^{D1} be a quantization-aware fine-tuning of PT on dataset D1
# with seed S1 and quantization configuration Q (i.e. the low precision dtype
# and the granularity of the quantization)
#
# Let QV be the displacement between QAT_{S1,Q}^{D1} and FP_{S1}^{D1}, i.e.
# QV = QAT_{S1,Q}^{D1} - FP_{S1}^{D1}
# ==============================================================================

# ==============================================================================
# QV Transfer under AWQ
# ==============================================================================
# Reviewers objected that the paper's only quantizer is vanilla RTN per-channel
# PTQ (`apply_ptq_`), and asked whether QV patching retains its benefit on top
# of a *strong* PTQ method (GPTQ, AWQ and SmoothQuant were named). Phase
# 005_qat_transfer_gptq answers it for GPTQ; this phase answers it for AWQ:
#
#     acc(awq(FP_{S2}^{D2} + alpha * QV))   vs   acc(awq(FP_{S2}^{D2}))
#
# i.e. 001_qat_transfer with the final quantizer swapped from `apply_ptq_` to
# `apply_awq_` (code/src/awq.py, Lin et al. MLSys 2024, on the project's own
# quantization grid so the RTN, GPTQ and AWQ columns differ only in method).
#
# Why a second strong-PTQ phase rather than resting on 005. GPTQ and AWQ attack
# quantization error from opposite ends: GPTQ compensates rounding error after
# the fact using second-order information about the layer's inputs, AWQ rescales
# input channels by activation salience so that less error is made in the first
# place. 005's headline finding was negative — at lambda=1 QV patching does not
# add gain on top of GPTQ, because GPTQ has already closed most of the RTN gap
# the QV was closing. Whether that is a fact about *strong PTQ in general* or a
# fact about *error compensation specifically* cannot be answered from one
# method. AWQ leaves a different residual (it never moves a weight off the grid
# to pay for another's error), so if the negative result reproduces here it is
# about the regime, and if it does not, the difference localizes the mechanism.
#
# Methodology notes — none is cosmetic:
#
# * alpha=0 IS the AWQ(FP) receiver baseline: FP_tgt + 0*QV = FP_tgt, so the
#   alpha=0 cell is AWQ applied to the receiver's own FP checkpoint. Running
#   it inside this sweep (rather than only in 000_baselines/evaluate_fp_awq)
#   guarantees the Task-2 delta compares cells that share bit-identical
#   calibration data and code path — the difference can only come from the QV
#   term. alpha=0 is donor-independent, so it is evaluated only on the
#   self-pair (source == target) and skipped for every other donor.
# * Calibration batches are materialized ONCE per receiver and reused for
#   every (donor, alpha, head-variant) AWQ call of that receiver.
#   Re-iterating a seeded shuffled DataLoader yields different batches each
#   pass; freezing the batch list is what makes "identical calibration across
#   compared cells" true rather than hoped-for. Calibration reads the
#   receiver's *training* split; labels are never used.
# * NO `apply_ptq_` runs after AWQ, exactly as 005 runs no RTN after GPTQ. AWQ
#   *is* the quantizer; its output is Q(W diag(s)) diag(1/s), whose per-row
#   absmax differs from the grid it was quantized on, so a following RTN pass
#   would re-round onto a fresh grid and destroy both the activation-aware
#   scaling and the searched clipping. See code/src/awq.py.
# * The QAT-head ceiling under this framing remains RTN(QAT) (001's qat_ptq),
#   never AWQ(QAT). 005 found GPTQ *destroys* the QAT checkpoint on the
#   diagonal — an objective mismatch, since a calibration-driven method
#   reconstructs the FP function, which for a QAT checkpoint is the wrong one.
#   AWQ's objective is the same FP-reconstruction objective, so the same
#   caveat applies to its diagonal and it must not be read as a ceiling.
# * Resume guard (deviation from 001): a (pair, alpha) cell whose
#   eval_results.json already exists is skipped when cfg.skip_existing is
#   true. AWQ makes each cell far more expensive than an RTN cell (two
#   sequential per-layer searches over n_grid scale candidates and the clip
#   grid), so a ~500-cell sweep must survive restarts without redoing
#   finished work.
# * The eval path carries an `awq=` fragment in place of 001's `ptq=` fragment,
#   built by `awq_path_frag` in code/src/awq.py rather than spelled here — the
#   `pv=` fragment's pattern, not the `gptq=` fragment's eight hand-written
#   copies.
# ==============================================================================

import gc
import json
import logging
import os
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)

IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)

from src.duration import checkpoint_epochs, mult_path_frag, mult_tag, role_path_frag
from src.vision.ilharco_timm_supervised.modeling import ImageClassifier
from src.vision.data.registry import get_dataset
from src.vision.data.common import maybe_dictionarize, DATASET_NAME_TO_EPOCHS, DATASET_NAME_TO_NUM_CLASSES
from src.vision.utils import (
    accuracy,
    random_tqdm_color,
    sanitize_timm_model_name,
    set_seed,
)
from src.awq import apply_awq_
from common.run_id import guard_run_config, run_id_path
from common.status import StatusWriter
from src.task_vectors import TaskVector

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from tqdm import tqdm
import torch
from torch import nn

# Keys whose prefix matches this belong to the classification head and are
# excluded from the QV (backbone-only transfer).
HEAD_PREFIX = "model.head."

RUN_ID_PARAMS = ["model", "src", "tgt", "optim", "qat", "awq", "qv", "split"]


def _is_head_key(k: str) -> bool:
    return k.startswith(HEAD_PREFIX)


def _run_identity(
    cfg: DictConfig,
    source_dataset_name: str,
    target_dataset_name: str,
    alpha: float,
    eval_split: str,
) -> dict:
    qat_skip = "-".join(sorted(cfg.qat.skip_modules)) or "none"
    awq_skip = "-".join(sorted(cfg.awq.skip_modules)) or "none"
    return {
        "model": sanitize_timm_model_name(cfg.model_name),
        "src": f"{source_dataset_name}-seed{cfg.source.seed}-mult{mult_tag(cfg.source.epoch_mult)}",
        "tgt": f"{target_dataset_name}-seed{cfg.target.seed}-mult{mult_tag(cfg.target.epoch_mult)}",
        "optim": (
            f"lr{cfg.lr}-wd{cfg.wd}-ls{cfg.ls}-wl{cfg.wl}"
            f"-mgn{cfg.max_grad_norm}-bs{cfg.batch_size}"
        ),
        "qat": (
            f"b{cfg.qat.bits}-g{cfg.qat.granularity}-s{qat_skip}"
        ),
        "awq": (
            f"b{cfg.awq.bits}-g{cfg.awq.granularity}-s{awq_skip}"
            f"-n{cfg.awq.num_calib_batches}-grid{cfg.awq.n_grid}"
            f"-clip{int(cfg.awq.clip)}"
        ),
        "qv": f"a{alpha}",
        "split": eval_split,
    }


def _fp_ckpt_dir(cfg: DictConfig, dataset_name: str, seed: int, epoch_mult) -> str:
    return os.path.join(
        os.environ['CHECKPOINT_BASE_PATH'],
        "vision",
        "ilharco_timm_supervised",
        "fp",
        sanitize_timm_model_name(cfg.model_name),
        dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        mult_path_frag(epoch_mult),
        f"seed={seed}",
    )


def _qat_ckpt_dir(cfg: DictConfig, dataset_name: str, seed: int, epoch_mult) -> str:
    skip_modules_sorted = sorted(cfg.qat.skip_modules)
    skip_tag = "-".join(skip_modules_sorted) if skip_modules_sorted else "none"
    return os.path.join(
        os.environ['CHECKPOINT_BASE_PATH'],
        "vision",
        "ilharco_timm_supervised",
        "qat",
        sanitize_timm_model_name(cfg.model_name),
        dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        mult_path_frag(epoch_mult),
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={skip_tag}",
        f"seed={seed}",
    )


def _eval_dir(
    cfg: DictConfig,
    source_dataset_name: str,
    target_dataset_name: str,
    alpha: float,
    eval_split: str,
) -> str:
    identity = _run_identity(
        cfg, source_dataset_name, target_dataset_name, alpha, eval_split
    )
    return str(Path(
        os.environ['EVALUATION_BASE_PATH'],
        "vision",
        "ilharco_timm_supervised",
        # Smoke runs must never share a tree with real results (008's guard,
        # which 001 lacks).
        "009_qat_transfer_awq_dryrun" if cfg.limit_num_batches is not None
        else "009_qat_transfer_awq",
        "vision",
        "qv_transfer_awq",
    ) / run_id_path(identity, RUN_ID_PARAMS))


def evaluate(
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    split: str,
    limit_num_batches: int = None,
):

    if split == "test":
        loader = dataset.test_loader
    elif split == "val":
        loader = dataset.val_loader
    else:
        raise ValueError(f"Unsupported eval_split: {split!r}. Must be 'val' or 'test'.")

    num_batches = len(loader)
    effective_num_batches = min(limit_num_batches, num_batches) if limit_num_batches is not None else num_batches

    model.to(device=device)
    model.eval()

    correct = 0
    total = 0

    batch_color = random_tqdm_color()

    with torch.no_grad():

        batch_bar = tqdm(
            enumerate(loader),
            total=effective_num_batches,
            desc=f"Evaluating ({split})",
            colour=batch_color,
            leave=False,
            **TQDM_KW,
        )

        for i, batch in batch_bar:

            if i >= effective_num_batches:
                break

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to(device=device)
            labels = batch['labels'].to(device=device, dtype=torch.long)

            logits = model(inputs)

            top1, = accuracy(logits, labels, topk=(1,))

            correct += top1
            total += labels.size(0)

            batch_bar.set_postfix(
                batch=f"{i}/{effective_num_batches}",
                acc=f"{100.0 * correct / total:.2f}%"
            )

    top1_acc = correct / total

    return top1_acc


def _run_pair(
    cfg: DictConfig,
    source_dataset_name: str,
    target_dataset_name: str,
    fp_tgt_sd: dict,
    qat_tgt_sd: dict,
    dataset,
    calib_batches: list,
    num_classes: int,
    device: str,
    tgt_epochs: int,
    eval_split: str,
):
    """Run QV transfer + AWQ for a single (source, target) pair, all alphas."""

    src_epochs = checkpoint_epochs(
        source_dataset_name, DATASET_NAME_TO_EPOCHS, cfg.source.limit_num_epochs
    )

    ############################################################################
    # BEGIN alpha selection (alpha=0 is donor-independent: self-pair only)
    ############################################################################

    alphas = [float(a) for a in OmegaConf.to_container(cfg.qv.alphas, resolve=True)]
    if source_dataset_name != target_dataset_name:
        alphas = [a for a in alphas if a != 0.0]

    if cfg.skip_existing:
        alphas = [
            a for a in alphas
            if not os.path.exists(os.path.join(
                _eval_dir(cfg, source_dataset_name, target_dataset_name, a, eval_split),
                "eval_results.json",
            ))
        ]

    if not alphas:
        if IS_SLURM:
            log.info(
                "Skipping source=%s target=%s: no alphas left to run "
                "(alpha=0 dedupe and/or results already on disk)",
                source_dataset_name, target_dataset_name,
            )
        else:
            print(
                f"Skipping source={source_dataset_name} target={target_dataset_name}: "
                "no alphas left to run (alpha=0 dedupe and/or results already on disk)"
            )
        return

    ############################################################################
    # END alpha selection
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN checkpoint paths
    ############################################################################

    fp_src_dir = _fp_ckpt_dir(cfg, source_dataset_name, cfg.source.seed, cfg.source.epoch_mult)
    qat_src_dir = _qat_ckpt_dir(cfg, source_dataset_name, cfg.source.seed, cfg.source.epoch_mult)

    fp_source_path = os.path.join(fp_src_dir, f"classifier_epoch_{src_epochs}.pt")
    qat_source_path = os.path.join(qat_src_dir, f"classifier_epoch_{src_epochs}.pt")

    if IS_SLURM:
        log.info("--- source=%s target=%s ---", source_dataset_name, target_dataset_name)
        log.info("FP source  classifier: %s", fp_source_path)
        log.info("QAT source classifier: %s", qat_source_path)
    else:
        print(f"\n--- source={source_dataset_name} target={target_dataset_name} ---")
        print(f"FP source  classifier: {fp_source_path}")
        print(f"QAT source classifier: {qat_source_path}")

    for path in (fp_source_path, qat_source_path):
        if not os.path.exists(path):
            log.warning("Skipping source=%s: checkpoint missing: %s", source_dataset_name, path)
            return

    ############################################################################
    # END checkpoint paths
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN QV construction (backbone only)
    ############################################################################

    fp_src_sd = torch.load(fp_source_path, map_location="cpu")
    qat_src_sd = torch.load(qat_source_path, map_location="cpu")

    src_backbone_keys = {k for k in fp_src_sd if not _is_head_key(k)}
    qat_backbone_keys = {k for k in qat_src_sd if not _is_head_key(k)}
    tgt_backbone_keys = {k for k in fp_tgt_sd if not _is_head_key(k)}
    if src_backbone_keys != qat_backbone_keys:
        log.warning(
            "fp_source and qat_source backbone key sets differ "
            f"(only-in-fp={sorted(src_backbone_keys - qat_backbone_keys)[:5]}..., "
            f"only-in-qat={sorted(qat_backbone_keys - src_backbone_keys)[:5]}...)"
        )
    if tgt_backbone_keys != src_backbone_keys:
        log.warning(
            "fp_target and fp_source backbone key sets differ "
            f"(only-in-tgt={sorted(tgt_backbone_keys - src_backbone_keys)[:5]}..., "
            f"only-in-src={sorted(src_backbone_keys - tgt_backbone_keys)[:5]}...)"
        )

    vector = {}
    num_dtype_filtered = 0
    num_head_filtered = 0
    with torch.no_grad():
        for k, v_src in fp_src_sd.items():
            if _is_head_key(k):
                num_head_filtered += 1
                continue
            if v_src.dtype in (torch.int64, torch.uint8):
                num_dtype_filtered += 1
                continue
            if k not in qat_src_sd:
                if IS_SLURM:
                    log.warning("key %s present in fp_source but missing in qat_source — skipping", k)
                else:
                    print(f"Warning: key {k} present in fp_source but missing in qat_source — skipping")
                continue
            vector[k] = qat_src_sd[k] - v_src

    tv = TaskVector(vector=vector)
    if IS_SLURM:
        log.info(
            "QV built (backbone only): %d keys in vector, %d head keys excluded, "
            "%d keys dtype-filtered (int64/uint8)",
            len(tv.vector), num_head_filtered, num_dtype_filtered,
        )
    else:
        print(
            f"\nQV built (backbone only): {len(tv.vector)} keys in vector, "
            f"{num_head_filtered} head keys excluded, "
            f"{num_dtype_filtered} keys dtype-filtered (int64/uint8)\n"
        )

    del fp_src_sd, qat_src_sd

    ############################################################################
    # END QV construction
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for alpha in alphas:
        _run_pair_alpha(
            cfg=cfg,
            source_dataset_name=source_dataset_name,
            target_dataset_name=target_dataset_name,
            fp_tgt_sd=fp_tgt_sd,
            qat_tgt_sd=qat_tgt_sd,
            tv=tv,
            alpha=alpha,
            dataset=dataset,
            calib_batches=calib_batches,
            num_classes=num_classes,
            device=device,
            src_epochs=src_epochs,
            tgt_epochs=tgt_epochs,
            eval_split=eval_split,
        )

    del tv
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_pair_alpha(
    cfg: DictConfig,
    source_dataset_name: str,
    target_dataset_name: str,
    fp_tgt_sd: dict,
    qat_tgt_sd: dict,
    tv: TaskVector,
    alpha: float,
    dataset,
    calib_batches: list,
    num_classes: int,
    device: str,
    src_epochs: int,
    tgt_epochs: int,
    eval_split: str,
):
    """Guard, signal, and execute one independently addressable cell."""
    eval_dir = _eval_dir(
        cfg, source_dataset_name, target_dataset_name, alpha, eval_split
    )
    identity = _run_identity(
        cfg, source_dataset_name, target_dataset_name, alpha, eval_split
    )
    resolved = {
        **identity,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    guard_run_config(resolved, RUN_ID_PARAMS, eval_dir)
    with StatusWriter(eval_dir) as status:
        status.heartbeat(progress="0/4")
        _run_pair_alpha_impl(
            cfg=cfg,
            source_dataset_name=source_dataset_name,
            target_dataset_name=target_dataset_name,
            fp_tgt_sd=fp_tgt_sd,
            qat_tgt_sd=qat_tgt_sd,
            tv=tv,
            alpha=alpha,
            dataset=dataset,
            calib_batches=calib_batches,
            num_classes=num_classes,
            device=device,
            src_epochs=src_epochs,
            tgt_epochs=tgt_epochs,
            eval_split=eval_split,
            status=status,
        )


def _run_pair_alpha_impl(
    cfg: DictConfig,
    source_dataset_name: str,
    target_dataset_name: str,
    fp_tgt_sd: dict,
    qat_tgt_sd: dict,
    tv: TaskVector,
    alpha: float,
    dataset,
    calib_batches: list,
    num_classes: int,
    device: str,
    src_epochs: int,
    tgt_epochs: int,
    eval_split: str,
    status: StatusWriter,
):
    """Evaluate one (source, target, alpha) cell: patch, AWQ, evaluate."""

    if IS_SLURM:
        log.info("--- alpha=%s ---", alpha)
    else:
        print(f"\n--- alpha={alpha} ---")

    ############################################################################
    # BEGIN patched-backbone assembly
    ############################################################################

    patched_backbone = {}
    with torch.no_grad():
        for k, v_tgt in fp_tgt_sd.items():
            if _is_head_key(k):
                continue
            if k in tv.vector:
                if tv.vector[k].shape != v_tgt.shape:
                    raise ValueError(
                        f"Shape mismatch on key {k}: tv.vector={tuple(tv.vector[k].shape)} vs "
                        f"fp_target={tuple(v_tgt.shape)}"
                    )
                patched_backbone[k] = v_tgt + alpha * tv.vector[k]
            else:
                patched_backbone[k] = v_tgt

        for k in tv.vector:
            if k not in fp_tgt_sd:
                if IS_SLURM:
                    log.warning("key %s present in QV but missing in fp_target — skipping", k)
                else:
                    print(f"Warning: key {k} present in QV but missing in fp_target — skipping")

    ############################################################################
    # END patched-backbone assembly
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN build patched state_dicts (backbone + head variants)
    ############################################################################

    fp_tgt_head = {k: v for k, v in fp_tgt_sd.items() if _is_head_key(k)}
    qat_tgt_head = {k: v for k, v in qat_tgt_sd.items() if _is_head_key(k)}

    patched_with_fp_head = {**patched_backbone, **fp_tgt_head}
    patched_with_qat_head = {**patched_backbone, **qat_tgt_head}

    del patched_backbone

    ############################################################################
    # END build patched state_dicts
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN Eval A: patched backbone + FP target head
    ############################################################################

    classifier_fp_head = ImageClassifier(model_name=cfg.model_name, num_classes=num_classes)
    classifier_fp_head.load_state_dict(patched_with_fp_head)
    classifier_fp_head.to(device)

    accuracy_fp_head = evaluate(
        dataset=dataset,
        model=classifier_fp_head,
        device=device,
        split=eval_split,
        limit_num_batches=cfg.limit_num_batches,
    )
    status.heartbeat(progress="1/4")

    if IS_SLURM:
        log.info("eval %s_accuracy (patched + FP head, before AWQ): %s", eval_split, accuracy_fp_head)
    else:
        print(f"\n    eval {eval_split}_accuracy (patched + FP head, before AWQ): {accuracy_fp_head}\n")

    skip_modules = frozenset(cfg.awq.skip_modules)

    all_linear_names_fp = [
        name for name, module in classifier_fp_head.named_modules()
        if isinstance(module, nn.Linear)
    ]

    quantized_names_fp = apply_awq_(
        model=classifier_fp_head,
        bits=cfg.awq.bits,
        granularity=cfg.awq.granularity,
        skip_modules=skip_modules,
        calib_loader=calib_batches,
        device=device,
        num_calib_batches=cfg.awq.num_calib_batches,
        n_grid=cfg.awq.n_grid,
        clip=cfg.awq.clip,
    )

    skipped_names_fp = sorted(set(all_linear_names_fp) - set(quantized_names_fp))

    if IS_SLURM:
        log.info(
            "AWQ config: bits=%s, granularity=%s, skip_modules=%s, ncal=%s, n_grid=%s, clip=%s",
            cfg.awq.bits, cfg.awq.granularity, list(cfg.awq.skip_modules),
            cfg.awq.num_calib_batches, cfg.awq.n_grid, cfg.awq.clip,
        )
        log.info(f"Quantized layers ({len(quantized_names_fp)}): {quantized_names_fp}")
        log.info(f"Skipped layers ({len(skipped_names_fp)}): {skipped_names_fp}")
    else:
        print(
            f"\nAWQ config: bits={cfg.awq.bits}, granularity={cfg.awq.granularity}, "
            f"skip_modules={list(cfg.awq.skip_modules)}, ncal={cfg.awq.num_calib_batches}, "
            f"n_grid={cfg.awq.n_grid}, clip={cfg.awq.clip}"
        )
        print(f"\nQuantized layers: {len(quantized_names_fp)}, skipped layers: {len(skipped_names_fp)}\n")

    accuracy_fp_head_awq = evaluate(
        dataset=dataset,
        model=classifier_fp_head,
        device=device,
        split=eval_split,
        limit_num_batches=cfg.limit_num_batches,
    )
    status.heartbeat(progress="2/4")

    if IS_SLURM:
        log.info("eval %s_accuracy (patched + FP head + AWQ): %s", eval_split, accuracy_fp_head_awq)
    else:
        print(f"\n    eval {eval_split}_accuracy (patched + FP head + AWQ): {accuracy_fp_head_awq}\n")

    del classifier_fp_head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ############################################################################
    # END Eval A
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN Eval B: patched backbone + QAT target head
    ############################################################################

    classifier_qat_head = ImageClassifier(model_name=cfg.model_name, num_classes=num_classes)
    classifier_qat_head.load_state_dict(patched_with_qat_head)
    classifier_qat_head.to(device)

    accuracy_qat_head = evaluate(
        dataset=dataset,
        model=classifier_qat_head,
        device=device,
        split=eval_split,
        limit_num_batches=cfg.limit_num_batches,
    )
    status.heartbeat(progress="3/4")

    if IS_SLURM:
        log.info("eval %s_accuracy (patched + QAT head, before AWQ): %s", eval_split, accuracy_qat_head)
    else:
        print(f"\n    eval {eval_split}_accuracy (patched + QAT head, before AWQ): {accuracy_qat_head}\n")

    all_linear_names_qat = [
        name for name, module in classifier_qat_head.named_modules()
        if isinstance(module, nn.Linear)
    ]

    quantized_names_qat = apply_awq_(
        model=classifier_qat_head,
        bits=cfg.awq.bits,
        granularity=cfg.awq.granularity,
        skip_modules=skip_modules,
        calib_loader=calib_batches,
        device=device,
        num_calib_batches=cfg.awq.num_calib_batches,
        n_grid=cfg.awq.n_grid,
        clip=cfg.awq.clip,
    )

    skipped_names_qat = sorted(set(all_linear_names_qat) - set(quantized_names_qat))

    accuracy_qat_head_awq = evaluate(
        dataset=dataset,
        model=classifier_qat_head,
        device=device,
        split=eval_split,
        limit_num_batches=cfg.limit_num_batches,
    )
    status.heartbeat(progress="4/4")

    if IS_SLURM:
        log.info("eval %s_accuracy (patched + QAT head + AWQ): %s", eval_split, accuracy_qat_head_awq)
    else:
        print(f"\n    eval {eval_split}_accuracy (patched + QAT head + AWQ): {accuracy_qat_head_awq}\n")

    del classifier_qat_head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ############################################################################
    # END Eval B
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN summary
    ############################################################################

    num_classes_actual = len(dataset.class_names)
    random_chance = 1.0 / num_classes_actual

    if IS_SLURM:
        log.info("random chance baseline: %s  (1 / %d classes)", random_chance, num_classes_actual)
    else:
        print(f"\n    random chance baseline : {random_chance}  (1 / {num_classes_actual} classes)\n")

    ############################################################################
    # END summary
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN save results
    ############################################################################

    fp_src_dir = _fp_ckpt_dir(cfg, source_dataset_name, cfg.source.seed, cfg.source.epoch_mult)
    qat_src_dir = _qat_ckpt_dir(cfg, source_dataset_name, cfg.source.seed, cfg.source.epoch_mult)
    fp_source_path_for_results = os.path.join(fp_src_dir, f"classifier_epoch_{src_epochs}.pt")
    qat_source_path_for_results = os.path.join(qat_src_dir, f"classifier_epoch_{src_epochs}.pt")

    fp_tgt_dir = _fp_ckpt_dir(cfg, target_dataset_name, cfg.target.seed, cfg.target.epoch_mult)
    qat_tgt_dir = _qat_ckpt_dir(cfg, target_dataset_name, cfg.target.seed, cfg.target.epoch_mult)
    fp_target_path_for_results = os.path.join(fp_tgt_dir, f"classifier_epoch_{tgt_epochs}.pt")
    qat_target_path_for_results = os.path.join(qat_tgt_dir, f"classifier_epoch_{tgt_epochs}.pt")

    eval_dir = _eval_dir(cfg, source_dataset_name, target_dataset_name, alpha, eval_split)

    accuracy_key_fp_head = f"{eval_split}_accuracy_fp_head"
    accuracy_key_fp_head_awq = f"{eval_split}_accuracy_fp_head_awq"
    accuracy_key_qat_head = f"{eval_split}_accuracy_qat_head"
    accuracy_key_qat_head_awq = f"{eval_split}_accuracy_qat_head_awq"

    results = {
        "experiment": "qv_transfer_awq",
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
            "fp_classifier_path": fp_source_path_for_results,
            "qat_classifier_path": qat_source_path_for_results,
        },
        "target": {
            "dataset_name": target_dataset_name,
            "seed": cfg.target.seed,
            "limit_num_epochs": cfg.target.limit_num_epochs,
            "epochs": tgt_epochs,
            "fp_classifier_path": fp_target_path_for_results,
            "qat_classifier_path": qat_target_path_for_results,
        },
        "qat": {
            "bits": cfg.qat.bits,
            "granularity": cfg.qat.granularity,
            "skip_modules": list(cfg.qat.skip_modules),
        },
        "qv": {
            "alpha": alpha,
            "num_keys_in_vector": len(tv.vector),
        },
        "awq": {
            "bits": cfg.awq.bits,
            "granularity": cfg.awq.granularity,
            "skip_modules": list(cfg.awq.skip_modules),
            "num_calib_batches": cfg.awq.num_calib_batches,
            "n_grid": cfg.awq.n_grid,
            "clip": cfg.awq.clip,
            "calibration_split": "train",
        },
        "awq_quantized_modules": quantized_names_fp,
        "awq_skipped_modules": skipped_names_fp,
        accuracy_key_fp_head: accuracy_fp_head,
        accuracy_key_fp_head_awq: accuracy_fp_head_awq,
        accuracy_key_qat_head: accuracy_qat_head,
        accuracy_key_qat_head_awq: accuracy_qat_head_awq,
        "num_classes": num_classes_actual,
        "random_chance": random_chance,
        "comparison_baseline_note": (
            "Task 2 delta: compare alpha>0 cells against the alpha=0.0 cell of the "
            "same receiver (self-pair src=tgt), which is AWQ(FP_tgt) under "
            "identical calibration. Cross-check against 000_baselines fp_awq."
        ),
    }

    os.makedirs(eval_dir, exist_ok=True)
    eval_results_path = os.path.join(eval_dir, "eval_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(results, f, indent=2)

    if IS_SLURM:
        log.info("Results saved to: %s", eval_results_path)
    else:
        print(f"\nResults saved to: {eval_results_path}")

    ############################################################################
    # END save results
    ############################################################################


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/009_qat_transfer_awq",
    config_name="qv_transfer_awq",
    version_base=None,
)
def main(cfg: DictConfig):

    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(dict(cfg), expand_all=True)

    source_dataset_names = OmegaConf.to_container(cfg.source.dataset_names, resolve=True)
    target_dataset_names = OmegaConf.to_container(cfg.target.dataset_names, resolve=True)

    set_seed(cfg.target.seed)

    eval_split = cfg.eval_split
    if eval_split not in ("val", "test"):
        raise ValueError(f"Unsupported eval_split: {eval_split!r}. Must be 'val' or 'test'.")

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    total_pairs = len(source_dataset_names) * len(target_dataset_names)
    pair_idx = 0

    for ti, target_dataset_name in enumerate(target_dataset_names):

        if IS_SLURM:
            log.info("=== Target %d/%d: %s ===", ti + 1, len(target_dataset_names), target_dataset_name)
        else:
            print(f"\n{'='*60}")
            print(f"  Target {ti + 1}/{len(target_dataset_names)}: {target_dataset_name}")
            print(f"{'='*60}")

        tgt_epochs = checkpoint_epochs(
        target_dataset_name, DATASET_NAME_TO_EPOCHS, cfg.target.limit_num_epochs
    )

        num_classes = DATASET_NAME_TO_NUM_CLASSES[target_dataset_name]

        ####################################################################
        # Load target checkpoints
        ####################################################################

        fp_tgt_dir = _fp_ckpt_dir(cfg, target_dataset_name, cfg.target.seed, cfg.target.epoch_mult)
        qat_tgt_dir = _qat_ckpt_dir(cfg, target_dataset_name, cfg.target.seed, cfg.target.epoch_mult)

        fp_target_path = os.path.join(fp_tgt_dir, f"classifier_epoch_{tgt_epochs}.pt")
        qat_target_path = os.path.join(qat_tgt_dir, f"classifier_epoch_{tgt_epochs}.pt")

        if IS_SLURM:
            log.info("FP target  classifier: %s", fp_target_path)
            log.info("QAT target classifier: %s", qat_target_path)
        else:
            print(f"\nFP target  classifier: {fp_target_path}")
            print(f"QAT target classifier: {qat_target_path}\n")

        target_missing = False
        for path in (fp_target_path, qat_target_path):
            if not os.path.exists(path):
                log.warning("Skipping target=%s: checkpoint missing: %s", target_dataset_name, path)
                target_missing = True
                break
        if target_missing:
            pair_idx += len(source_dataset_names)
            continue

        fp_tgt_sd = torch.load(fp_target_path, map_location="cpu")
        qat_tgt_sd = torch.load(qat_target_path, map_location="cpu")

        ####################################################################
        # Create dataset (target)
        ####################################################################

        _tmp_classifier = ImageClassifier(model_name=cfg.model_name, num_classes=num_classes)

        dataset = get_dataset(
            dataset_name=target_dataset_name,
            preprocess_train=_tmp_classifier.train_preprocess,
            preprocess_inference=_tmp_classifier.val_preprocess,
            batch_size=cfg.batch_size,
            num_workers=int(os.environ['TORCH_NUM_WORKERS']),
            seed=cfg.target.seed,
        )

        del _tmp_classifier

        ####################################################################
        # Materialize calibration batches (once per target)
        #
        # Re-iterating a seeded shuffled DataLoader yields different batches
        # each pass; freezing the first num_calib_batches here guarantees
        # every (donor, alpha, head-variant) AWQ call of this receiver
        # calibrates on bit-identical data — the property the Task-2 delta
        # rests on. Labels are never used by AWQ.
        ####################################################################

        calib_batches = []
        for batch in dataset.train_loader:
            if len(calib_batches) >= cfg.awq.num_calib_batches:
                break
            calib_batches.append(batch)

        if IS_SLURM:
            log.info(
                "Materialized %d calibration batches from the %s train split",
                len(calib_batches), target_dataset_name,
            )
        else:
            print(
                f"Materialized {len(calib_batches)} calibration batches "
                f"from the {target_dataset_name} train split"
            )

        ####################################################################
        # Iterate over source datasets
        ####################################################################

        for si, source_dataset_name in enumerate(source_dataset_names):
            pair_idx += 1

            if IS_SLURM:
                log.info("--- Pair %d/%d: source=%s target=%s ---", pair_idx, total_pairs, source_dataset_name, target_dataset_name)
            else:
                print(f"\n--- Pair {pair_idx}/{total_pairs}: source={source_dataset_name} target={target_dataset_name} ---")

            _run_pair(
                cfg=cfg,
                source_dataset_name=source_dataset_name,
                target_dataset_name=target_dataset_name,
                fp_tgt_sd=fp_tgt_sd,
                qat_tgt_sd=qat_tgt_sd,
                dataset=dataset,
                calib_batches=calib_batches,
                num_classes=num_classes,
                device=device,
                tgt_epochs=tgt_epochs,
                eval_split=eval_split,
            )

        ####################################################################
        # Cleanup between target iterations
        ####################################################################

        del dataset, fp_tgt_sd, qat_tgt_sd, calib_batches
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if IS_SLURM:
        log.info("All %d pairs completed. Forcing exit.", total_pairs)
        os._exit(0)


if __name__ == "__main__":
    main()
