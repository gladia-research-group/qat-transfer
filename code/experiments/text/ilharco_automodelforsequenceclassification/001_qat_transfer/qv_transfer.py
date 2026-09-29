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
# QV Transfer
# ==============================================================================
# Can we transfer the benefit of QAT, supposedly captured by a QV, to a model
# that has been trained on a dataset D2 with seed S2?
#
# What does transfer mean?
#
# acc(ptq(QAT_{S2,Q}^{D2}) \approx ptq(FP_{S2}^{D2} + \alpha QV))
#
# Everything related to the dataset, seed and epoch to which we apply the qv
# is referred to as "patched"
#
# Please note that S1 can be either the same or different than S2.
# Please note that D1 can be either the same or different than D2.
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

SUPPORTED_MODELS = {
    "google-bert/bert-base-uncased",
    "google-bert/bert-large-uncased",
    "google/embeddinggemma-300m",
    "Qwen/Qwen3-Embedding-0.6B",
}
MODEL_NAME_TO_HEAD_MODULE = {
    "google-bert/bert-base-uncased": "classifier",
    "google-bert/bert-large-uncased": "classifier",
    "google/embeddinggemma-300m": "score",
    "Qwen/Qwen3-Embedding-0.6B": "score",
}

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.duration import checkpoint_epochs, mult_path_frag, role_path_frag
from src.quantization import apply_ptq_
from src.task_vectors import TaskVector
from src.text.data.common import DATASET_NAME_TO_EPOCHS
from src.text.data.registry import get_dataset
from src.vision.utils import (
    random_tqdm_color,
    sanitize_hf_model_name,
    set_seed,
)

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from tqdm import tqdm
import torch
from torch import nn

OmegaConf.register_new_resolver("sanitize_hf", sanitize_hf_model_name, replace=True)


def _fp_ckpt_dir(cfg: DictConfig, dataset_name: str, seed: int, epoch_mult) -> str:
    return os.path.join(
        os.environ['CHECKPOINT_BASE_PATH'],
        "text",
        "ilharco_automodelforsequenceclassification",
        "fp",
        sanitize_hf_model_name(cfg.model_name),
        dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}_ml={cfg.max_length}",
        mult_path_frag(epoch_mult),
        f"seed={seed}",
    )


def _qat_ckpt_dir(cfg: DictConfig, dataset_name: str, seed: int, epoch_mult) -> str:
    skip_modules_sorted = sorted(cfg.qat.skip_modules)
    skip_tag = "-".join(skip_modules_sorted) if skip_modules_sorted else "none"
    return os.path.join(
        os.environ['CHECKPOINT_BASE_PATH'],
        "text",
        "ilharco_automodelforsequenceclassification",
        "qat",
        sanitize_hf_model_name(cfg.model_name),
        dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}_ml={cfg.max_length}",
        mult_path_frag(epoch_mult),
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={skip_tag}",
        f"seed={seed}",
    )


def _resolve_alpha(cfg, source_dataset_name, target_dataset_name):
    """Return the numeric alpha to use.

    If cfg.qv.alpha is ``"best"``, reads best_alpha_*.json from disk.
    Otherwise returns ``float(cfg.qv.alpha)``.  Returns ``None`` when
    the best_alpha file is missing (caller should skip the pair).
    """
    raw = str(cfg.qv.alpha)
    if raw != "best":
        return float(raw)

    evaluation_base_path = os.environ["EVALUATION_BASE_PATH"]
    sanitized_model = sanitize_hf_model_name(cfg.model_name)
    qat_skip_tag = "-".join(sorted(cfg.qat.skip_modules)) if len(cfg.qat.skip_modules) > 0 else "none"
    ptq_skip_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) > 0 else "none"

    metric_key = cfg.qv.best_metric
    if metric_key is None:
        raise ValueError("qv.best_metric must be set when qv.alpha='best'")
    # Derive file label from metric key: "val_accuracy_fp_head_ptq" -> "fp_head_ptq"
    label = metric_key.replace("val_accuracy_", "").replace("test_accuracy_", "")

    best_alpha_path = os.path.join(
        evaluation_base_path,
        "text", "ilharco_automodelforsequenceclassification", "001_qat_transfer", "text", "qv_transfer",
        sanitized_model,
        role_path_frag("src", source_dataset_name, cfg.source.seed, cfg.source.epoch_mult),
        role_path_frag("tgt", target_dataset_name, cfg.target.seed, cfg.target.epoch_mult),
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}_ml={cfg.max_length}",
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={qat_skip_tag}",
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={ptq_skip_tag}",
        f"best_alpha_{label}.json",
    )

    if not os.path.exists(best_alpha_path):
        return None

    with open(best_alpha_path) as f:
        data = json.load(f)

    if len(data) == 1:
        return float(next(iter(data.values()))["alpha"])

    if metric_key in data:
        return float(data[metric_key]["alpha"])

    raise ValueError(
        f"best_alpha file has keys {list(data.keys())} but metric_key={metric_key!r} not found"
    )


def build_model_and_tokenizer(model_name: str, num_labels: int, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device=device, dtype=torch.float32)
    return model, tokenizer


def evaluate(
    dataset,
    model: torch.nn.Module,
    tokenizer,
    max_length: int,
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

            texts, labels = batch
            encoding = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            labels = labels.to(device=device, dtype=torch.long)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

            batch_bar.set_postfix(
                batch=f"{i}/{effective_num_batches}",
                acc=f"{100.0 * correct / max(total, 1):.2f}%",
            )

    return correct / max(total, 1)


def _run_pair(
    cfg: DictConfig,
    source_dataset_name: str,
    target_dataset_name: str,
    fp_tgt_backbone_sd: dict,
    fp_tgt_head_sd: dict,
    qat_tgt_head_sd: dict,
    dataset,
    num_classes: int,
    device: str,
    tgt_epochs: int,
    tokenizer,
    max_length: int,
    head_prefix: str,
    eval_split: str,
):
    """Run QV transfer for a single (source, target) pair."""

    metric_only = cfg.eval_mode == "fp_head_ptq_only"

    def _is_head_key(k: str) -> bool:
        return k.startswith(head_prefix)

    src_epochs = checkpoint_epochs(
        source_dataset_name, DATASET_NAME_TO_EPOCHS, cfg.source.limit_num_epochs
    )

    ############################################################################
    # BEGIN checkpoint paths
    ############################################################################

    fp_src_dir = _fp_ckpt_dir(cfg, source_dataset_name, cfg.source.seed, cfg.source.epoch_mult)
    qat_src_dir = _qat_ckpt_dir(cfg, source_dataset_name, cfg.source.seed, cfg.source.epoch_mult)

    fp_src_backbone_path = os.path.join(fp_src_dir, f"backbone_epoch_{src_epochs}.pt")
    qat_src_backbone_path = os.path.join(qat_src_dir, f"backbone_epoch_{src_epochs}.pt")

    if IS_SLURM:
        log.info("--- source=%s target=%s ---", source_dataset_name, target_dataset_name)
        log.info("FP source  backbone: %s", fp_src_backbone_path)
        log.info("QAT source backbone: %s", qat_src_backbone_path)
    else:
        print(f"\n--- source={source_dataset_name} target={target_dataset_name} ---")
        print(f"FP source  backbone: {fp_src_backbone_path}")
        print(f"QAT source backbone: {qat_src_backbone_path}")

    for path in (fp_src_backbone_path, qat_src_backbone_path):
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

    fp_src_sd = torch.load(fp_src_backbone_path, map_location="cpu")
    qat_src_sd = torch.load(qat_src_backbone_path, map_location="cpu")

    src_backbone_keys = {k for k in fp_src_sd if not _is_head_key(k)}
    qat_backbone_keys = {k for k in qat_src_sd if not _is_head_key(k)}
    tgt_backbone_keys = {k for k in fp_tgt_backbone_sd if not _is_head_key(k)}
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

    ############################################################################
    # BEGIN patched-backbone assembly
    ############################################################################

    alpha = _resolve_alpha(cfg, source_dataset_name, target_dataset_name)
    if alpha is None:
        log.warning("Skipping source=%s target=%s: best_alpha file missing", source_dataset_name, target_dataset_name)
        return

    patched_backbone = {}
    with torch.no_grad():
        for k, v_tgt in fp_tgt_backbone_sd.items():
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
            if k not in fp_tgt_backbone_sd:
                if IS_SLURM:
                    log.warning("key %s present in QV but missing in fp_target — skipping", k)
                else:
                    print(f"Warning: key {k} present in QV but missing in fp_target — skipping")

    del tv

    ############################################################################
    # END patched-backbone assembly
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN build patched state_dicts (backbone + head variants)
    ############################################################################

    patched_with_fp_head = {**patched_backbone, **fp_tgt_head_sd}
    patched_with_qat_head = (
        {**patched_backbone, **qat_tgt_head_sd}
        if qat_tgt_head_sd is not None else None
    )

    del patched_backbone

    ############################################################################
    # END build patched state_dicts
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN Eval A: patched backbone + FP target head
    ############################################################################

    model_fp_head, _ = build_model_and_tokenizer(cfg.model_name, num_classes, device)
    model_fp_head.load_state_dict(patched_with_fp_head, strict=False)
    model_fp_head.to(device)
    if IS_SLURM:
        log.info("model (patched backbone + FP head): %s", model_fp_head)
    else:
        print(f"\n\nmodel (patched backbone + FP head):")
        pprint(model_fp_head, expand_all=True)
        print(f"\n\n")

    accuracy_fp_head = None
    if not metric_only:
        accuracy_fp_head = evaluate(
            dataset=dataset,
            model=model_fp_head,
            tokenizer=tokenizer,
            max_length=max_length,
            device=device,
            split=eval_split,
            limit_num_batches=cfg.limit_num_batches,
        )

    if IS_SLURM and not metric_only:
        log.info("eval %s_accuracy (patched + FP head, before PTQ): %s", eval_split, accuracy_fp_head)
    elif not metric_only:
        print(f"\n    eval {eval_split}_accuracy (patched + FP head, before PTQ): {accuracy_fp_head}\n")

    skip_modules = frozenset(cfg.ptq.skip_modules)

    all_linear_names_fp = [
        name for name, module in model_fp_head.named_modules()
        if isinstance(module, nn.Linear)
    ]

    quantized_names_fp = apply_ptq_(
        model=model_fp_head,
        bits=cfg.ptq.bits,
        granularity=cfg.ptq.granularity,
        skip_modules=skip_modules,
    )

    skipped_names_fp = sorted(set(all_linear_names_fp) - set(quantized_names_fp))

    if IS_SLURM:
        log.info(
            "PTQ config: bits=%s, granularity=%s, skip_modules=%s",
            cfg.ptq.bits, cfg.ptq.granularity, list(cfg.ptq.skip_modules),
        )
        log.info(f"Quantized layers ({len(quantized_names_fp)}): {quantized_names_fp}")
        log.info(f"Skipped layers ({len(skipped_names_fp)}): {skipped_names_fp}")
    else:
        print(f"\nPTQ config: bits={cfg.ptq.bits}, granularity={cfg.ptq.granularity}, skip_modules={list(cfg.ptq.skip_modules)}")

        print(f"\nQuantized layers ({len(quantized_names_fp)}):")
        for name in quantized_names_fp:
            print(f"  - {name}")

        print(f"\nSkipped layers ({len(skipped_names_fp)}):")
        for name in skipped_names_fp:
            print(f"  - {name}")
        print()

    accuracy_fp_head_ptq = evaluate(
        dataset=dataset,
        model=model_fp_head,
        tokenizer=tokenizer,
        max_length=max_length,
        device=device,
        split=eval_split,
        limit_num_batches=cfg.limit_num_batches,
    )

    if IS_SLURM:
        log.info("eval %s_accuracy (patched + FP head + PTQ): %s", eval_split, accuracy_fp_head_ptq)
    else:
        print(f"\n    eval {eval_split}_accuracy (patched + FP head + PTQ): {accuracy_fp_head_ptq}\n")

    del model_fp_head
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ############################################################################
    # END Eval A
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN Eval B: patched backbone + QAT target head
    ############################################################################

    accuracy_qat_head = None
    accuracy_qat_head_ptq = None
    if not metric_only:
        model_qat_head, _ = build_model_and_tokenizer(cfg.model_name, num_classes, device)
        model_qat_head.load_state_dict(patched_with_qat_head, strict=False)
        model_qat_head.to(device)
        if IS_SLURM:
            log.info("model (patched backbone + QAT head): %s", model_qat_head)
        else:
            print(f"\n\nmodel (patched backbone + QAT head):")
            pprint(model_qat_head, expand_all=True)
            print(f"\n\n")

        accuracy_qat_head = evaluate(
            dataset=dataset,
            model=model_qat_head,
            tokenizer=tokenizer,
            max_length=max_length,
            device=device,
            split=eval_split,
            limit_num_batches=cfg.limit_num_batches,
        )

        if IS_SLURM:
            log.info("eval %s_accuracy (patched + QAT head, before PTQ): %s", eval_split, accuracy_qat_head)
        else:
            print(f"\n    eval {eval_split}_accuracy (patched + QAT head, before PTQ): {accuracy_qat_head}\n")

        all_linear_names_qat = [
            name for name, module in model_qat_head.named_modules()
            if isinstance(module, nn.Linear)
        ]
        quantized_names_qat = apply_ptq_(
            model=model_qat_head,
            bits=cfg.ptq.bits,
            granularity=cfg.ptq.granularity,
            skip_modules=skip_modules,
        )
        skipped_names_qat = sorted(set(all_linear_names_qat) - set(quantized_names_qat))

        accuracy_qat_head_ptq = evaluate(
            dataset=dataset,
            model=model_qat_head,
            tokenizer=tokenizer,
            max_length=max_length,
            device=device,
            split=eval_split,
            limit_num_batches=cfg.limit_num_batches,
        )

        if IS_SLURM:
            log.info("eval %s_accuracy (patched + QAT head + PTQ): %s", eval_split, accuracy_qat_head_ptq)
        else:
            print(f"\n    eval {eval_split}_accuracy (patched + QAT head + PTQ): {accuracy_qat_head_ptq}\n")

        del model_qat_head
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

    evaluation_base_path = os.environ['EVALUATION_BASE_PATH']

    qat_skip_tag = "-".join(sorted(cfg.qat.skip_modules)) if len(cfg.qat.skip_modules) > 0 else "none"
    ptq_skip_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) > 0 else "none"

    fp_src_dir = _fp_ckpt_dir(cfg, source_dataset_name, cfg.source.seed, cfg.source.epoch_mult)
    qat_src_dir = _qat_ckpt_dir(cfg, source_dataset_name, cfg.source.seed, cfg.source.epoch_mult)
    fp_src_backbone_path_for_results = os.path.join(fp_src_dir, f"backbone_epoch_{src_epochs}.pt")
    qat_src_backbone_path_for_results = os.path.join(qat_src_dir, f"backbone_epoch_{src_epochs}.pt")

    fp_tgt_dir = _fp_ckpt_dir(cfg, target_dataset_name, cfg.target.seed, cfg.target.epoch_mult)
    qat_tgt_dir = _qat_ckpt_dir(cfg, target_dataset_name, cfg.target.seed, cfg.target.epoch_mult)
    fp_tgt_backbone_path_for_results = os.path.join(fp_tgt_dir, f"backbone_epoch_{tgt_epochs}.pt")
    fp_tgt_head_path_for_results = os.path.join(fp_tgt_dir, f"head_epoch_{tgt_epochs}.pt")
    qat_tgt_head_path_for_results = os.path.join(qat_tgt_dir, f"head_epoch_{tgt_epochs}.pt")

    eval_dir = os.path.join(
        evaluation_base_path,
        "text",
        "ilharco_automodelforsequenceclassification",
        "001_qat_transfer",
        "text",
        "qv_transfer",
        sanitize_hf_model_name(cfg.model_name),
        role_path_frag("src", source_dataset_name, cfg.source.seed, cfg.source.epoch_mult),
        role_path_frag("tgt", target_dataset_name, cfg.target.seed, cfg.target.epoch_mult),
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}_ml={cfg.max_length}",
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={qat_skip_tag}",
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={ptq_skip_tag}",
        f"qv=alpha={alpha}",
        f"split={eval_split}",
    )

    results = {
        "experiment": "qv_transfer",
        "model_name": cfg.model_name,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "wd": cfg.wd,
        "ls": cfg.ls,
        "max_length": cfg.max_length,
        "max_grad_norm": cfg.max_grad_norm,
        "limit_num_batches": cfg.limit_num_batches,
        "eval_mode": cfg.eval_mode,
        "device": str(device),
        "source": {
            "dataset_name": source_dataset_name,
            "seed": cfg.source.seed,
            "limit_num_epochs": cfg.source.limit_num_epochs,
            "epochs": src_epochs,
            "fp_backbone_path": fp_src_backbone_path_for_results,
            "qat_backbone_path": qat_src_backbone_path_for_results,
        },
        "target": {
            "dataset_name": target_dataset_name,
            "seed": cfg.target.seed,
            "limit_num_epochs": cfg.target.limit_num_epochs,
            "epochs": tgt_epochs,
            "fp_backbone_path": fp_tgt_backbone_path_for_results,
            "fp_head_path": fp_tgt_head_path_for_results,
            "qat_head_path": qat_tgt_head_path_for_results,
        },
        "qat": {
            "bits": cfg.qat.bits,
            "granularity": cfg.qat.granularity,
            "skip_modules": list(cfg.qat.skip_modules),
        },
        "qv": {
            "alpha": alpha,
            "num_keys_in_vector": len(vector),
            "num_head_keys_excluded": num_head_filtered,
            "num_keys_dtype_filtered": num_dtype_filtered,
        },
        "ptq": {
            "bits": cfg.ptq.bits,
            "granularity": cfg.ptq.granularity,
            "skip_modules": list(cfg.ptq.skip_modules),
        },
        "ptq_quantized_modules": quantized_names_fp,
        "ptq_skipped_modules": skipped_names_fp,
        "eval_split": eval_split,
        f"{eval_split}_accuracy_fp_head": accuracy_fp_head,
        f"{eval_split}_accuracy_fp_head_ptq": accuracy_fp_head_ptq,
        f"{eval_split}_accuracy_qat_head": accuracy_qat_head,
        f"{eval_split}_accuracy_qat_head_ptq": accuracy_qat_head_ptq,
        "num_classes": num_classes_actual,
        "random_chance": random_chance,
        "comparison_baseline_note": (
            "Compare to PTQ(QAT_{S2,Q,D2}); NOT computed here. "
            "Run config/experiments/text/000_baselines/evaluate_fp_ptq on the QAT checkpoint to obtain it."
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
    config_path="../../../../../config/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer",
    config_name="qv_transfer",
    version_base=None,
)
def main(cfg: DictConfig):

    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(dict(cfg), expand_all=True)

    if cfg.model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_name={cfg.model_name!r}. Supported: {sorted(SUPPORTED_MODELS)}")
    if cfg.eval_mode not in ("full", "fp_head_ptq_only"):
        raise ValueError(
            f"Unsupported eval_mode={cfg.eval_mode!r}; expected 'full' or "
            "'fp_head_ptq_only'"
        )

    source_dataset_names = OmegaConf.to_container(cfg.source.dataset_names, resolve=True)
    target_dataset_names = OmegaConf.to_container(cfg.target.dataset_names, resolve=True)

    set_seed(cfg.target.seed)

    eval_split = cfg.eval_split
    if eval_split not in ("val", "test"):
        raise ValueError(f"Unsupported eval_split: {eval_split!r}. Must be 'val' or 'test'.")

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    head_module = MODEL_NAME_TO_HEAD_MODULE[cfg.model_name]
    head_prefix = head_module + "."
    max_length = cfg.max_length

    ########################################################################
    # Create tokenizer once (depends only on model_name)
    ########################################################################

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

        ####################################################################
        # Load target checkpoints
        ####################################################################

        fp_tgt_dir = _fp_ckpt_dir(cfg, target_dataset_name, cfg.target.seed, cfg.target.epoch_mult)
        qat_tgt_dir = _qat_ckpt_dir(cfg, target_dataset_name, cfg.target.seed, cfg.target.epoch_mult)

        fp_tgt_backbone_path = os.path.join(fp_tgt_dir, f"backbone_epoch_{tgt_epochs}.pt")
        fp_tgt_head_path = os.path.join(fp_tgt_dir, f"head_epoch_{tgt_epochs}.pt")
        qat_tgt_head_path = os.path.join(qat_tgt_dir, f"head_epoch_{tgt_epochs}.pt")

        if IS_SLURM:
            log.info("FP target  backbone: %s", fp_tgt_backbone_path)
            log.info("FP target  head:     %s", fp_tgt_head_path)
            log.info("QAT target head:     %s", qat_tgt_head_path)
        else:
            print(f"\nFP target  backbone: {fp_tgt_backbone_path}")
            print(f"FP target  head:     {fp_tgt_head_path}")
            print(f"QAT target head:     {qat_tgt_head_path}\n")

        target_missing = False
        required_target_paths = [fp_tgt_backbone_path, fp_tgt_head_path]
        if cfg.eval_mode == "full":
            required_target_paths.append(qat_tgt_head_path)
        for path in required_target_paths:
            if not os.path.exists(path):
                log.warning("Skipping target=%s: checkpoint missing: %s", target_dataset_name, path)
                target_missing = True
                break
        if target_missing:
            pair_idx += len(source_dataset_names)
            continue

        fp_tgt_backbone_sd = torch.load(fp_tgt_backbone_path, map_location="cpu")
        fp_tgt_head_sd = torch.load(fp_tgt_head_path, map_location="cpu")
        qat_tgt_head_sd = (
            torch.load(qat_tgt_head_path, map_location="cpu")
            if cfg.eval_mode == "full" else None
        )

        ####################################################################
        # Create dataset (target)
        ####################################################################

        dataset = get_dataset(
            dataset_name=target_dataset_name,
            batch_size=cfg.batch_size,
            num_workers=int(os.environ['TORCH_NUM_WORKERS']),
            seed=cfg.target.seed,
        )

        num_classes = len(dataset.class_names)

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
                fp_tgt_backbone_sd=fp_tgt_backbone_sd,
                fp_tgt_head_sd=fp_tgt_head_sd,
                qat_tgt_head_sd=qat_tgt_head_sd,
                dataset=dataset,
                num_classes=num_classes,
                device=device,
                tgt_epochs=tgt_epochs,
                tokenizer=tokenizer,
                max_length=max_length,
                head_prefix=head_prefix,
                eval_split=eval_split,
            )

        ####################################################################
        # Cleanup between target iterations
        ####################################################################

        del dataset, fp_tgt_backbone_sd, fp_tgt_head_sd, qat_tgt_head_sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if IS_SLURM:
        log.info("All %d pairs completed. Forcing exit.", total_pairs)
        os._exit(0)


if __name__ == "__main__":
    main()
