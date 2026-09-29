# ==============================================================================
# Quantization Vector (QV, qv) — Reversed Transfer
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
# QV Transfer (Reversed)
# ==============================================================================
# Can we transfer the benefit of QAT in the reversed direction?
#
# Instead of adding the QV to an FP checkpoint, we subtract it from a PTQ'd
# QAT checkpoint:
#
# patched = ptq(QAT_{S2,Q}^{D2}) - \alpha QV
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
from src.quantization import apply_ptq_
from src.task_vectors import TaskVector

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from tqdm import tqdm
import torch

# Keys whose prefix matches this belong to the classification head and are
# excluded from the QV (backbone-only transfer).
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
    sanitized_model = sanitize_timm_model_name(cfg.model_name)
    qat_skip_tag = "-".join(sorted(cfg.qat.skip_modules)) if len(cfg.qat.skip_modules) > 0 else "none"
    ptq_skip_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) > 0 else "none"

    metric_key = cfg.qv.best_metric
    if metric_key is None:
        raise ValueError("qv.best_metric must be set when qv.alpha='best'")
    # Derive file label from metric key: "val_accuracy_fp_head" -> "fp_head"
    label = metric_key.replace("val_accuracy_", "").replace("test_accuracy_", "")

    best_alpha_path = os.path.join(
        evaluation_base_path,
        "vision", "ilharco_timm_supervised", "002_qat_transfer_reversed", "vision", "qv_transfer",
        sanitized_model,
        role_path_frag("src", source_dataset_name, cfg.source.seed, cfg.source.epoch_mult),
        role_path_frag("tgt", target_dataset_name, cfg.target.seed, cfg.target.epoch_mult),
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
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
    num_classes: int,
    device: str,
    tgt_epochs: int,
    eval_split: str,
):
    """Run reversed QV transfer for a single (source, target) pair."""

    src_epochs = checkpoint_epochs(
        source_dataset_name, DATASET_NAME_TO_EPOCHS, cfg.source.limit_num_epochs
    )

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
    # BEGIN PTQ(QAT_tgt) construction
    ############################################################################

    skip_modules = frozenset(cfg.ptq.skip_modules)

    ptq_qat_tgt_classifier = ImageClassifier(model_name=cfg.model_name, num_classes=num_classes)
    ptq_qat_tgt_classifier.load_state_dict(qat_tgt_sd)

    apply_ptq_(
        model=ptq_qat_tgt_classifier,
        bits=cfg.ptq.bits,
        granularity=cfg.ptq.granularity,
        skip_modules=skip_modules,
    )

    ptq_qat_tgt_sd = ptq_qat_tgt_classifier.state_dict()

    del ptq_qat_tgt_classifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if IS_SLURM:
        log.info("PTQ(QAT_tgt) constructed: bits=%s, granularity=%s, skip_modules=%s",
                 cfg.ptq.bits, cfg.ptq.granularity, list(cfg.ptq.skip_modules))
    else:
        print(f"\nPTQ(QAT_tgt) constructed: bits={cfg.ptq.bits}, granularity={cfg.ptq.granularity}, skip_modules={list(cfg.ptq.skip_modules)}")

    ############################################################################
    # END PTQ(QAT_tgt) construction
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN QV construction (backbone only)
    ############################################################################

    fp_src_sd = torch.load(fp_source_path, map_location="cpu")
    qat_src_sd = torch.load(qat_source_path, map_location="cpu")

    src_backbone_keys = {k for k in fp_src_sd if not _is_head_key(k)}
    qat_backbone_keys = {k for k in qat_src_sd if not _is_head_key(k)}
    tgt_backbone_keys = {k for k in ptq_qat_tgt_sd if not _is_head_key(k)}
    if src_backbone_keys != qat_backbone_keys:
        log.warning(
            "fp_source and qat_source backbone key sets differ "
            f"(only-in-fp={sorted(src_backbone_keys - qat_backbone_keys)[:5]}..., "
            f"only-in-qat={sorted(qat_backbone_keys - src_backbone_keys)[:5]}...)"
        )
    if tgt_backbone_keys != src_backbone_keys:
        log.warning(
            "ptq_qat_target and fp_source backbone key sets differ "
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
        for k, v_tgt in ptq_qat_tgt_sd.items():
            if _is_head_key(k):
                continue
            if k in tv.vector:
                if tv.vector[k].shape != v_tgt.shape:
                    raise ValueError(
                        f"Shape mismatch on key {k}: tv.vector={tuple(tv.vector[k].shape)} vs "
                        f"ptq_qat_target={tuple(v_tgt.shape)}"
                    )
                patched_backbone[k] = v_tgt - alpha * tv.vector[k]
            else:
                patched_backbone[k] = v_tgt

        for k in tv.vector:
            if k not in ptq_qat_tgt_sd:
                if IS_SLURM:
                    log.warning("key %s present in QV but missing in ptq_qat_target — skipping", k)
                else:
                    print(f"Warning: key {k} present in QV but missing in ptq_qat_target — skipping")

    del tv, ptq_qat_tgt_sd

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
    if IS_SLURM:
        log.info("classifier (patched backbone + FP head): %s", classifier_fp_head)
    else:
        print(f"\n\nclassifier (patched backbone + FP head):")
        pprint(classifier_fp_head, expand_all=True)
        print(f"\n\n")

    accuracy_fp_head = evaluate(
        dataset=dataset,
        model=classifier_fp_head,
        device=device,
        split=eval_split,
        limit_num_batches=cfg.limit_num_batches,
    )

    if IS_SLURM:
        log.info("eval %s_accuracy (patched + FP head): %s", eval_split, accuracy_fp_head)
    else:
        print(f"\n    eval {eval_split}_accuracy (patched + FP head): {accuracy_fp_head}\n")

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
    if IS_SLURM:
        log.info("classifier (patched backbone + QAT head): %s", classifier_qat_head)
    else:
        print(f"\n\nclassifier (patched backbone + QAT head):")
        pprint(classifier_qat_head, expand_all=True)
        print(f"\n\n")

    accuracy_qat_head = evaluate(
        dataset=dataset,
        model=classifier_qat_head,
        device=device,
        split=eval_split,
        limit_num_batches=cfg.limit_num_batches,
    )

    if IS_SLURM:
        log.info("eval %s_accuracy (patched + QAT head): %s", eval_split, accuracy_qat_head)
    else:
        print(f"\n    eval {eval_split}_accuracy (patched + QAT head): {accuracy_qat_head}\n")

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

    evaluation_base_path = os.environ['EVALUATION_BASE_PATH']

    qat_skip_tag = "-".join(sorted(cfg.qat.skip_modules)) if len(cfg.qat.skip_modules) > 0 else "none"
    ptq_skip_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) > 0 else "none"

    fp_src_dir = _fp_ckpt_dir(cfg, source_dataset_name, cfg.source.seed, cfg.source.epoch_mult)
    qat_src_dir = _qat_ckpt_dir(cfg, source_dataset_name, cfg.source.seed, cfg.source.epoch_mult)
    fp_source_path_for_results = os.path.join(fp_src_dir, f"classifier_epoch_{src_epochs}.pt")
    qat_source_path_for_results = os.path.join(qat_src_dir, f"classifier_epoch_{src_epochs}.pt")

    fp_tgt_dir = _fp_ckpt_dir(cfg, target_dataset_name, cfg.target.seed, cfg.target.epoch_mult)
    qat_tgt_dir = _qat_ckpt_dir(cfg, target_dataset_name, cfg.target.seed, cfg.target.epoch_mult)
    fp_target_path_for_results = os.path.join(fp_tgt_dir, f"classifier_epoch_{tgt_epochs}.pt")
    qat_target_path_for_results = os.path.join(qat_tgt_dir, f"classifier_epoch_{tgt_epochs}.pt")

    eval_dir = os.path.join(
        evaluation_base_path,
        "vision",
        "ilharco_timm_supervised",
        "002_qat_transfer_reversed",
        "vision",
        "qv_transfer",
        sanitize_timm_model_name(cfg.model_name),
        role_path_frag("src", source_dataset_name, cfg.source.seed, cfg.source.epoch_mult),
        role_path_frag("tgt", target_dataset_name, cfg.target.seed, cfg.target.epoch_mult),
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={qat_skip_tag}",
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={ptq_skip_tag}",
        f"qv=alpha={alpha}",
        f"split={eval_split}",
    )

    accuracy_key_fp_head = f"{eval_split}_accuracy_fp_head"
    accuracy_key_qat_head = f"{eval_split}_accuracy_qat_head"

    results = {
        "experiment": "qv_transfer_reversed",
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
            "num_keys_in_vector": len(vector),
            "num_head_keys_excluded": num_head_filtered,
            "num_keys_dtype_filtered": num_dtype_filtered,
        },
        "ptq": {
            "bits": cfg.ptq.bits,
            "granularity": cfg.ptq.granularity,
            "skip_modules": list(cfg.ptq.skip_modules),
        },
        accuracy_key_fp_head: accuracy_fp_head,
        accuracy_key_qat_head: accuracy_qat_head,
        "num_classes": num_classes_actual,
        "random_chance": random_chance,
        "comparison_baseline_note": (
            "Compare to PTQ(QAT_{S2,Q,D2}); the base model here IS PTQ(QAT_tgt). "
            "The QV is subtracted: patched = PTQ(QAT_tgt) - alpha * QV. "
            "No further PTQ is applied after patching."
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
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/002_qat_transfer_reversed",
    config_name="qv_transfer",
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
                num_classes=num_classes,
                device=device,
                tgt_epochs=tgt_epochs,
                eval_split=eval_split,
            )

        ####################################################################
        # Cleanup between target iterations
        ####################################################################

        del dataset, fp_tgt_sd, qat_tgt_sd
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if IS_SLURM:
        log.info("All %d pairs completed. Forcing exit.", total_pairs)
        os._exit(0)


if __name__ == "__main__":
    main()
