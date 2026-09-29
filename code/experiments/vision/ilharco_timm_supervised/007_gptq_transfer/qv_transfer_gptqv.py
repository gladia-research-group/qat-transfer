# ==============================================================================
# GPTQ Vector (QV_gptq)
# ==============================================================================
# Let PT be a pre-trained model
#
# Let FP_{S1}^{D1} be a fine-tuning in full precision of PT on dataset D1
# with seed S1
#
# Let GPTQ_{Q}^{D1} = gptq(FP_{S1}^{D1}) be the GPTQ quantization of that
# checkpoint with quantization configuration Q, calibrated on D1's own
# training split (materialized once by compute_gptq_checkpoints.py)
#
# Let QV_gptq be the displacement between them, i.e.
# QV_gptq = GPTQ_{Q}^{D1} - FP_{S1}^{D1}
# ==============================================================================

# ==============================================================================
# GPTQ Transfer (phase 006)
# ==============================================================================
# Is GPTQ itself transferrable the way QAT is? The 001 phase shows the QAT
# displacement is largely task-agnostic. QV_gptq is a fundamentally different
# object — a pure weight-space quantization displacement (snap-to-grid plus
# Hessian error feedback), produced with no task gradient. This phase tests
# whether it nonetheless acts as a quantization of a *different* task:
#
#   acc(FP_{S2}^{D2} + alpha * QV_gptq^{D1})     — evaluated RAW, in FP.
#
# Crucially, NO quantizer is applied after patching: the displacement itself is
# the quantization being tested. On the self-pair at alpha=1 the patched model
# is algebraically GPTQ(FP^{D2}) — already on the 3-bit grid — so the diagonal
# reproduces the GPTQ(FP) baseline (005's alpha=0 cell, up to calibration-batch
# differences: 005 drew its own batches from the receiver's seeded train loader,
# this phase reads the checkpoints materialized by compute_gptq_checkpoints.py;
# expect approximate, not bit-exact, agreement). Off-diagonal the patched
# weights are only *near* a grid (the donor's scales, displaced by the
# receiver-donor weight difference), and the question is how much accuracy the
# foreign displacement costs relative to the receiver's own GPTQ run.
#
# Only the FP-head variant exists here: GPTQ skips the head, so the donor's
# GPTQ and FP checkpoints have bit-identical heads and QV_gptq has no head
# component by construction; the receiver keeps its own FP head.
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

from src.duration import checkpoint_epochs, mult_path_frag, role_path_frag
from src.vision.ilharco_timm_supervised.modeling import ImageClassifier
from src.vision.data.registry import get_dataset
from src.vision.data.common import maybe_dictionarize, DATASET_NAME_TO_EPOCHS, DATASET_NAME_TO_NUM_CLASSES
from src.vision.utils import (
    accuracy,
    random_tqdm_color,
    sanitize_timm_model_name,
    set_seed,
)
from src.task_vectors import TaskVector

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from tqdm import tqdm
import torch

# Keys whose prefix matches this belong to the classification head and are
# excluded from the QV (backbone-only transfer). For QV_gptq the head keys are
# bit-identical between the two donor checkpoints anyway (GPTQ skips the head),
# so this filter only removes exact zeros — kept for parity with 001.
HEAD_PREFIX = "model.head."


def _is_head_key(k: str) -> bool:
    return k.startswith(HEAD_PREFIX)


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


def _gptq_frag(cfg: DictConfig) -> str:
    skip_modules_sorted = sorted(cfg.gptq.skip_modules)
    skip_tag = "-".join(skip_modules_sorted) if skip_modules_sorted else "none"
    return (
        f"gptq=bits={cfg.gptq.bits}_gran={cfg.gptq.granularity}_skip={skip_tag}"
        f"_ncal={cfg.gptq.num_calib_batches}_percdamp={cfg.gptq.percdamp}_actorder={cfg.gptq.actorder}"
    )


def _gptq_ckpt_dir(cfg: DictConfig, dataset_name: str, seed: int, epoch_mult) -> str:
    return os.path.join(
        os.environ['CHECKPOINT_BASE_PATH'],
        "vision",
        "ilharco_timm_supervised",
        "gptq",
        sanitize_timm_model_name(cfg.model_name),
        dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        mult_path_frag(epoch_mult),
        _gptq_frag(cfg),
        f"seed={seed}",
    )


def _eval_dir(
    cfg: DictConfig,
    source_dataset_name: str,
    target_dataset_name: str,
    alpha: float,
    eval_split: str,
) -> str:
    return os.path.join(
        os.environ['EVALUATION_BASE_PATH'],
        "vision",
        "ilharco_timm_supervised",
        "007_gptq_transfer",
        "vision",
        "qv_transfer_gptqv",
        sanitize_timm_model_name(cfg.model_name),
        role_path_frag("src", source_dataset_name, cfg.source.seed, cfg.source.epoch_mult),
        role_path_frag("tgt", target_dataset_name, cfg.target.seed, cfg.target.epoch_mult),
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        _gptq_frag(cfg),
        f"qv=alpha={alpha}",
        f"split={eval_split}",
    )


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
    dataset,
    num_classes: int,
    device: str,
    tgt_epochs: int,
    eval_split: str,
):
    """Run GPTQ-vector transfer for a single (source, target) pair, all alphas."""

    src_epochs = checkpoint_epochs(
        source_dataset_name, DATASET_NAME_TO_EPOCHS, cfg.source.limit_num_epochs
    )

    ############################################################################
    # BEGIN alpha selection (resume guard)
    ############################################################################

    alphas = [float(a) for a in OmegaConf.to_container(cfg.qv.alphas, resolve=True)]

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
            log.info("Skipping source=%s target=%s: results already on disk",
                     source_dataset_name, target_dataset_name)
        else:
            print(f"Skipping source={source_dataset_name} target={target_dataset_name}: "
                  "results already on disk")
        return

    ############################################################################
    # END alpha selection
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN checkpoint paths
    ############################################################################

    fp_src_dir = _fp_ckpt_dir(cfg, source_dataset_name, cfg.source.seed, cfg.source.epoch_mult)
    gptq_src_dir = _gptq_ckpt_dir(cfg, source_dataset_name, cfg.source.seed, cfg.source.epoch_mult)

    fp_source_path = os.path.join(fp_src_dir, f"classifier_epoch_{src_epochs}.pt")
    gptq_source_path = os.path.join(gptq_src_dir, f"classifier_epoch_{src_epochs}.pt")

    if IS_SLURM:
        log.info("--- source=%s target=%s ---", source_dataset_name, target_dataset_name)
        log.info("FP source   classifier: %s", fp_source_path)
        log.info("GPTQ source classifier: %s", gptq_source_path)
    else:
        print(f"\n--- source={source_dataset_name} target={target_dataset_name} ---")
        print(f"FP source   classifier: {fp_source_path}")
        print(f"GPTQ source classifier: {gptq_source_path}")

    for path in (fp_source_path, gptq_source_path):
        if not os.path.exists(path):
            log.warning("Skipping source=%s: checkpoint missing: %s", source_dataset_name, path)
            return

    ############################################################################
    # END checkpoint paths
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN QV_gptq construction (backbone only)
    ############################################################################

    fp_src_sd = torch.load(fp_source_path, map_location="cpu")
    gptq_src_sd = torch.load(gptq_source_path, map_location="cpu")

    src_backbone_keys = {k for k in fp_src_sd if not _is_head_key(k)}
    gptq_backbone_keys = {k for k in gptq_src_sd if not _is_head_key(k)}
    tgt_backbone_keys = {k for k in fp_tgt_sd if not _is_head_key(k)}
    if src_backbone_keys != gptq_backbone_keys:
        log.warning(
            "fp_source and gptq_source backbone key sets differ "
            f"(only-in-fp={sorted(src_backbone_keys - gptq_backbone_keys)[:5]}..., "
            f"only-in-gptq={sorted(gptq_backbone_keys - src_backbone_keys)[:5]}...)"
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
            if k not in gptq_src_sd:
                if IS_SLURM:
                    log.warning("key %s present in fp_source but missing in gptq_source — skipping", k)
                else:
                    print(f"Warning: key {k} present in fp_source but missing in gptq_source — skipping")
                continue
            vector[k] = gptq_src_sd[k] - v_src

    tv = TaskVector(vector=vector)
    if IS_SLURM:
        log.info(
            "QV_gptq built (backbone only): %d keys in vector, %d head keys excluded, "
            "%d keys dtype-filtered (int64/uint8)",
            len(tv.vector), num_head_filtered, num_dtype_filtered,
        )
    else:
        print(
            f"\nQV_gptq built (backbone only): {len(tv.vector)} keys in vector, "
            f"{num_head_filtered} head keys excluded, "
            f"{num_dtype_filtered} keys dtype-filtered (int64/uint8)\n"
        )

    del fp_src_sd, gptq_src_sd

    ############################################################################
    # END QV_gptq construction
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    for alpha in alphas:

        ########################################################################
        # BEGIN patched state_dict assembly
        ########################################################################

        patched = {}
        with torch.no_grad():
            for k, v_tgt in fp_tgt_sd.items():
                if not _is_head_key(k) and k in tv.vector:
                    if tv.vector[k].shape != v_tgt.shape:
                        raise ValueError(
                            f"Shape mismatch on key {k}: tv.vector={tuple(tv.vector[k].shape)} vs "
                            f"fp_target={tuple(v_tgt.shape)}"
                        )
                    patched[k] = v_tgt + alpha * tv.vector[k]
                else:
                    patched[k] = v_tgt

            for k in tv.vector:
                if k not in fp_tgt_sd:
                    if IS_SLURM:
                        log.warning("key %s present in QV_gptq but missing in fp_target — skipping", k)
                    else:
                        print(f"Warning: key {k} present in QV_gptq but missing in fp_target — skipping")

        ########################################################################
        # END patched state_dict assembly
        ########################################################################

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        ########################################################################
        # BEGIN evaluation (RAW — no quantizer after patching, by design)
        ########################################################################

        if IS_SLURM:
            log.info("--- alpha=%s ---", alpha)
        else:
            print(f"\n--- alpha={alpha} ---")

        classifier = ImageClassifier(model_name=cfg.model_name, num_classes=num_classes)
        classifier.load_state_dict(patched)
        classifier.to(device)

        accuracy_fp_head = evaluate(
            dataset=dataset,
            model=classifier,
            device=device,
            split=eval_split,
            limit_num_batches=cfg.limit_num_batches,
        )

        if IS_SLURM:
            log.info("eval %s_accuracy (patched, raw): %s", eval_split, accuracy_fp_head)
        else:
            print(f"\n    eval {eval_split}_accuracy (patched, raw): {accuracy_fp_head}\n")

        del classifier, patched
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ########################################################################
        # END evaluation
        ########################################################################

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        ########################################################################
        # BEGIN save results
        ########################################################################

        num_classes_actual = len(dataset.class_names)
        random_chance = 1.0 / num_classes_actual

        fp_tgt_dir = _fp_ckpt_dir(cfg, target_dataset_name, cfg.target.seed, cfg.target.epoch_mult)
        fp_target_path_for_results = os.path.join(fp_tgt_dir, f"classifier_epoch_{tgt_epochs}.pt")

        eval_dir = _eval_dir(cfg, source_dataset_name, target_dataset_name, alpha, eval_split)

        accuracy_key_fp_head = f"{eval_split}_accuracy_fp_head"

        results = {
            "experiment": "qv_transfer_gptqv",
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
                "gptq_classifier_path": gptq_source_path,
            },
            "target": {
                "dataset_name": target_dataset_name,
                "seed": cfg.target.seed,
                "limit_num_epochs": cfg.target.limit_num_epochs,
                "epochs": tgt_epochs,
                "fp_classifier_path": fp_target_path_for_results,
            },
            "qv": {
                "alpha": alpha,
                "num_keys_in_vector": len(tv.vector),
                "num_head_keys_excluded": num_head_filtered,
                "num_keys_dtype_filtered": num_dtype_filtered,
            },
            "gptq": {
                "bits": cfg.gptq.bits,
                "granularity": cfg.gptq.granularity,
                "skip_modules": list(cfg.gptq.skip_modules),
                "num_calib_batches": cfg.gptq.num_calib_batches,
                "percdamp": cfg.gptq.percdamp,
                "actorder": cfg.gptq.actorder,
                "block_size": cfg.gptq.block_size,
                "calibration_split": "train",
                "role": "defines the donor-side QV; NO quantizer is applied after patching",
            },
            accuracy_key_fp_head: accuracy_fp_head,
            "num_classes": num_classes_actual,
            "random_chance": random_chance,
            "comparison_baseline_note": (
                "Raw FP evaluation of the patched model; nothing is quantized at "
                "eval time. Compare against the receiver's fp baseline "
                "(000_baselines) and its own GPTQ(FP) "
                "(006 diagonal, or 005 qv=alpha=0.0; the latter used different "
                "calibration batches, expect approximate agreement)."
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

        ########################################################################
        # END save results
        ########################################################################

    del tv
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/007_gptq_transfer",
    config_name="qv_transfer_gptqv",
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
        # Load target checkpoint
        ####################################################################

        fp_tgt_dir = _fp_ckpt_dir(cfg, target_dataset_name, cfg.target.seed, cfg.target.epoch_mult)
        fp_target_path = os.path.join(fp_tgt_dir, f"classifier_epoch_{tgt_epochs}.pt")

        if IS_SLURM:
            log.info("FP target classifier: %s", fp_target_path)
        else:
            print(f"\nFP target classifier: {fp_target_path}\n")

        if not os.path.exists(fp_target_path):
            log.warning("Skipping target=%s: checkpoint missing: %s", target_dataset_name, fp_target_path)
            pair_idx += len(source_dataset_names)
            continue

        fp_tgt_sd = torch.load(fp_target_path, map_location="cpu")

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
                dataset=dataset,
                num_classes=num_classes,
                device=device,
                tgt_epochs=tgt_epochs,
                eval_split=eval_split,
            )

        ####################################################################
        # Cleanup between target iterations
        ####################################################################

        del dataset, fp_tgt_sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if IS_SLURM:
        log.info("All %d pairs completed. Forcing exit.", total_pairs)
        os._exit(0)


if __name__ == "__main__":
    main()
