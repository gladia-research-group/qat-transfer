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

from src.duration import checkpoint_epochs, mult_path_frag, role_path_frag
from src.vision.ilharco_open_clip.modeling import ImageClassifier, ImageEncoder
from src.vision.ilharco_open_clip.heads import get_classification_head
from src.vision.data.registry import get_dataset
from src.vision.data.common import maybe_dictionarize, DATASET_NAME_TO_EPOCHS
from src.vision.utils import (
    accuracy,
    random_tqdm_color,
    sanitize_open_clip_model_name,
    set_seed,
)
from src.quantization import apply_ptq_
from src.task_vectors import TaskVector

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from tqdm import tqdm
import torch
from torch import nn


def _fp_ckpt_path(cfg: DictConfig, dataset_name: str, seed: int, epochs: int, epoch_mult) -> str:
    sanitized_model = sanitize_open_clip_model_name(cfg.model_name, cfg.pretrained)
    return os.path.join(
        os.environ['CHECKPOINT_BASE_PATH'],
        "vision",
        "ilharco_open_clip",
        "fp",
        sanitized_model,
        dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        mult_path_frag(epoch_mult),
        f"seed={seed}",
        f"epoch_{epochs}.pt",
    )


def _qat_ckpt_path(cfg: DictConfig, dataset_name: str, seed: int, epochs: int, epoch_mult) -> str:
    sanitized_model = sanitize_open_clip_model_name(cfg.model_name, cfg.pretrained)
    skip_modules_sorted = sorted(cfg.qat.skip_modules)
    skip_tag = "-".join(skip_modules_sorted) if skip_modules_sorted else "none"
    return os.path.join(
        os.environ['CHECKPOINT_BASE_PATH'],
        "vision",
        "ilharco_open_clip",
        "qat",
        sanitized_model,
        dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        mult_path_frag(epoch_mult),
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={skip_tag}",
        f"seed={seed}",
        f"epoch_{epochs}.pt",
    )


def _resolve_alpha(cfg, source_dataset_name, target_dataset_name):
    """Return the numeric alpha to use.

    If cfg.qv.alpha is ``"best"``, reads best_alpha.json from disk.
    Otherwise returns ``float(cfg.qv.alpha)``.  Returns ``None`` when
    the best_alpha.json file is missing (caller should skip the pair).
    """
    raw = str(cfg.qv.alpha)
    if raw != "best":
        return float(raw)

    evaluation_base_path = os.environ["EVALUATION_BASE_PATH"]
    sanitized_model = sanitize_open_clip_model_name(cfg.model_name, cfg.pretrained)
    qat_skip_tag = "-".join(sorted(cfg.qat.skip_modules)) if len(cfg.qat.skip_modules) > 0 else "none"
    ptq_skip_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) > 0 else "none"

    best_alpha_path = os.path.join(
        evaluation_base_path,
        "vision", "ilharco_open_clip", "001_qat_transfer", "vision", "qv_transfer",
        sanitized_model,
        role_path_frag("src", source_dataset_name, cfg.source.seed, cfg.source.epoch_mult),
        role_path_frag("tgt", target_dataset_name, cfg.target.seed, cfg.target.epoch_mult),
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={qat_skip_tag}",
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={ptq_skip_tag}",
        "best_alpha.json",
    )

    if not os.path.exists(best_alpha_path):
        return None

    with open(best_alpha_path) as f:
        data = json.load(f)

    metric_key = cfg.qv.get("best_metric", None)
    if metric_key is None:
        if len(data) == 1:
            metric_key = next(iter(data))
        else:
            raise ValueError(
                f"best_alpha.json has {len(data)} metrics but qv.best_metric is not set. "
                f"Available: {list(data.keys())}"
            )

    return float(data[metric_key]["alpha"])


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
    classification_head,
    device: str,
    tgt_epochs: int,
    eval_split: str,
):
    """Run QV transfer for a single (source, target) pair."""

    src_epochs = checkpoint_epochs(
        source_dataset_name, DATASET_NAME_TO_EPOCHS, cfg.source.limit_num_epochs
    )

    ############################################################################
    # BEGIN checkpoint paths
    ############################################################################

    fp_source_path = _fp_ckpt_path(cfg, source_dataset_name, cfg.source.seed, src_epochs, cfg.source.epoch_mult)
    qat_source_path = _qat_ckpt_path(cfg, source_dataset_name, cfg.source.seed, src_epochs, cfg.source.epoch_mult)

    if IS_SLURM:
        log.info("--- source=%s target=%s ---", source_dataset_name, target_dataset_name)
        log.info("FP source  checkpoint: %s", fp_source_path)
        log.info("QAT source checkpoint: %s", qat_source_path)
    else:
        print(f"\n--- source={source_dataset_name} target={target_dataset_name} ---")
        print(f"FP source  checkpoint: {fp_source_path}")
        print(f"QAT source checkpoint: {qat_source_path}")

    for path in (fp_source_path, qat_source_path):
        if not os.path.exists(path):
            log.warning("Skipping source=%s: checkpoint missing: %s", source_dataset_name, path)
            return

    ############################################################################
    # END checkpoint paths
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN QV construction
    ############################################################################

    fp_src_sd = torch.load(fp_source_path, map_location="cpu")
    qat_src_sd = torch.load(qat_source_path, map_location="cpu")

    src_keys = set(fp_src_sd.keys())
    qat_keys = set(qat_src_sd.keys())
    tgt_keys = set(fp_tgt_sd.keys())
    if src_keys != qat_keys:
        log.warning(
            "fp_source and qat_source state_dict key sets differ "
            f"(only-in-fp={sorted(src_keys - qat_keys)[:5]}..., "
            f"only-in-qat={sorted(qat_keys - src_keys)[:5]}...)"
        )
    if tgt_keys != src_keys:
        log.warning(
            "fp_target and fp_source state_dict key sets differ "
            f"(only-in-tgt={sorted(tgt_keys - src_keys)[:5]}..., "
            f"only-in-src={sorted(src_keys - tgt_keys)[:5]}...)"
        )

    vector = {}
    num_dtype_filtered = 0
    with torch.no_grad():
        for k, v_src in fp_src_sd.items():
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
            "QV built: %d keys in vector, %d keys dtype-filtered (int64/uint8)",
            len(tv.vector), num_dtype_filtered,
        )
    else:
        print(
            f"\nQV built: {len(tv.vector)} keys in vector, "
            f"{num_dtype_filtered} keys dtype-filtered (int64/uint8)\n"
        )

    del fp_src_sd, qat_src_sd

    ############################################################################
    # END QV construction
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN patched-model assembly
    ############################################################################

    alpha = _resolve_alpha(cfg, source_dataset_name, target_dataset_name)
    if alpha is None:
        if IS_SLURM:
            log.warning(
                "Skipping source=%s target=%s: best_alpha.json not found",
                source_dataset_name, target_dataset_name,
            )
        else:
            print(
                f"Warning: Skipping source={source_dataset_name} "
                f"target={target_dataset_name}: best_alpha.json not found"
            )
        return
    patched = {}
    with torch.no_grad():
        for k, v_tgt in fp_tgt_sd.items():
            if k in tv.vector:
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
                    log.warning("key %s present in QV but missing in fp_target — skipping", k)
                else:
                    print(f"Warning: key {k} present in QV but missing in fp_target — skipping")

    del tv

    ############################################################################
    # END patched-model assembly
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN load patched encoder
    ############################################################################

    image_encoder = ImageEncoder(model_name=cfg.model_name, pretrained=cfg.pretrained)
    image_encoder.load_state_dict(patched)
    image_encoder.to(device)
    if IS_SLURM:
        log.info("image_encoder (patched): %s", image_encoder)
    else:
        print(f"\n\nimage_encoder (patched):")
        pprint(image_encoder, expand_all=True)
        print(f"\n\n")

    del patched

    ############################################################################
    # END load patched encoder
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN image classifier creation
    ############################################################################

    image_classifier = ImageClassifier(
        image_encoder=image_encoder,
        classification_head=classification_head
    )
    image_classifier.to(device)
    if IS_SLURM:
        log.info("image_classifier: %s", image_classifier)
    else:
        print(f"\n\nimage_classifier:")
        pprint(image_classifier, expand_all=True)
        print(f"\n\n")

    ############################################################################
    # END image classifier creation
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN evaluation (patched QAT, before PTQ)
    ############################################################################

    accuracy_patched_qat = evaluate(
        dataset=dataset,
        model=image_classifier,
        device=device,
        split=eval_split,
        limit_num_batches=cfg.limit_num_batches,
    )

    if IS_SLURM:
        log.info("eval %s_accuracy (patched QAT, FP_target + %s*QV): %s", eval_split, alpha, accuracy_patched_qat)
    else:
        print(f"\n    eval {eval_split}_accuracy (patched QAT, FP_target + {alpha}*QV): {accuracy_patched_qat}\n")

    ############################################################################
    # END evaluation (patched QAT)
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN PTQ
    ############################################################################

    skip_modules = frozenset(cfg.ptq.skip_modules)

    all_linear_names = [
        name for name, module in image_classifier.named_modules()
        if isinstance(module, nn.Linear)
    ]

    quantized_names = apply_ptq_(
        model=image_classifier,
        bits=cfg.ptq.bits,
        granularity=cfg.ptq.granularity,
        skip_modules=skip_modules,
    )

    skipped_names = sorted(set(all_linear_names) - set(quantized_names))

    if IS_SLURM:
        log.info(
            "PTQ config: bits=%s, granularity=%s, skip_modules=%s",
            cfg.ptq.bits, cfg.ptq.granularity, list(cfg.ptq.skip_modules),
        )
        log.info(f"Quantized layers ({len(quantized_names)}): {quantized_names}")
        log.info(f"Skipped layers ({len(skipped_names)}): {skipped_names}")
    else:
        print(f"\nPTQ config: bits={cfg.ptq.bits}, granularity={cfg.ptq.granularity}, skip_modules={list(cfg.ptq.skip_modules)}")

        print(f"\nQuantized layers ({len(quantized_names)}):")
        for name in quantized_names:
            print(f"  - {name}")

        print(f"\nSkipped layers ({len(skipped_names)}):")
        for name in skipped_names:
            print(f"  - {name}")
        print()

    ############################################################################
    # END PTQ
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN evaluation (patched QAT + PTQ)
    ############################################################################

    accuracy_patched_qat_ptq = evaluate(
        dataset=dataset,
        model=image_classifier,
        device=device,
        split=eval_split,
        limit_num_batches=cfg.limit_num_batches,
    )

    num_classes = len(dataset.class_names)
    random_chance = 1.0 / num_classes
    if IS_SLURM:
        log.info(
            "eval %s_accuracy (patched QAT + PTQ, FP_target + %s*QV): %s",
            eval_split, alpha, accuracy_patched_qat_ptq,
        )
        log.info("random chance baseline: %s  (1 / %d classes)", random_chance, num_classes)
    else:
        print(f"\n    eval {eval_split}_accuracy (patched QAT + PTQ, FP_target + {alpha}*QV): {accuracy_patched_qat_ptq}\n")
        print(f"    random chance baseline : {random_chance}  (1 / {num_classes} classes)\n")

    ############################################################################
    # END evaluation (patched QAT + PTQ)
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN save results
    ############################################################################

    evaluation_base_path = os.environ['EVALUATION_BASE_PATH']

    sanitized_model = sanitize_open_clip_model_name(cfg.model_name, cfg.pretrained)
    qat_skip_tag = "-".join(sorted(cfg.qat.skip_modules)) if len(cfg.qat.skip_modules) > 0 else "none"
    ptq_skip_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) > 0 else "none"

    fp_source_path = _fp_ckpt_path(cfg, source_dataset_name, cfg.source.seed, src_epochs, cfg.source.epoch_mult)
    qat_source_path = _qat_ckpt_path(cfg, source_dataset_name, cfg.source.seed, src_epochs, cfg.source.epoch_mult)
    fp_target_path = _fp_ckpt_path(cfg, target_dataset_name, cfg.target.seed, tgt_epochs, cfg.target.epoch_mult)

    eval_dir = os.path.join(
        evaluation_base_path,
        "vision",
        "ilharco_open_clip",
        "001_qat_transfer",
        "vision",
        "qv_transfer",
        sanitized_model,
        role_path_frag("src", source_dataset_name, cfg.source.seed, cfg.source.epoch_mult),
        role_path_frag("tgt", target_dataset_name, cfg.target.seed, cfg.target.epoch_mult),
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={qat_skip_tag}",
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={ptq_skip_tag}",
        f"qv=alpha={alpha}",
        f"split={eval_split}",
    )

    accuracy_key_patched_qat = f"{eval_split}_accuracy_patched_qat"
    accuracy_key_patched_qat_ptq = f"{eval_split}_accuracy_patched_qat_ptq"

    results = {
        "experiment": "qv_transfer",
        "model_name": cfg.model_name,
        "pretrained": cfg.pretrained,
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
            "fp_checkpoint_path": fp_source_path,
            "qat_checkpoint_path": qat_source_path,
        },
        "target": {
            "dataset_name": target_dataset_name,
            "seed": cfg.target.seed,
            "limit_num_epochs": cfg.target.limit_num_epochs,
            "epochs": tgt_epochs,
            "fp_checkpoint_path": fp_target_path,
        },
        "qat": {
            "bits": cfg.qat.bits,
            "granularity": cfg.qat.granularity,
            "skip_modules": list(cfg.qat.skip_modules),
        },
        "qv": {
            "alpha": alpha,
            "num_keys_in_vector": len(vector),
            "num_keys_dtype_filtered": num_dtype_filtered,
        },
        "ptq": {
            "bits": cfg.ptq.bits,
            "granularity": cfg.ptq.granularity,
            "skip_modules": list(cfg.ptq.skip_modules),
        },
        "ptq_quantized_modules": quantized_names,
        "ptq_skipped_modules": skipped_names,
        accuracy_key_patched_qat: accuracy_patched_qat,
        accuracy_key_patched_qat_ptq: accuracy_patched_qat_ptq,
        "num_classes": num_classes,
        "random_chance": random_chance,
        "comparison_baseline_note": (
            "Compare to PTQ(QAT_{S2,Q,D2}); NOT computed here. "
            "Run config/experiments/vision/000_baselines/evaluate_fp_ptq on the QAT checkpoint to obtain it."
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

    del image_classifier, image_encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ############################################################################
    # END save results
    ############################################################################


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_open_clip/001_qat_transfer",
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

        ####################################################################
        # Load target checkpoint
        ####################################################################

        fp_target_path = _fp_ckpt_path(cfg, target_dataset_name, cfg.target.seed, tgt_epochs, cfg.target.epoch_mult)

        if IS_SLURM:
            log.info("FP target  checkpoint: %s", fp_target_path)
        else:
            print(f"\nFP target  checkpoint: {fp_target_path}\n")

        if not os.path.exists(fp_target_path):
            log.warning("Skipping target=%s: checkpoint missing: %s", target_dataset_name, fp_target_path)
            pair_idx += len(source_dataset_names)
            continue

        fp_tgt_sd = torch.load(fp_target_path, map_location="cpu")

        ####################################################################
        # Create dataset and classification head (target)
        ####################################################################

        image_encoder = ImageEncoder(model_name=cfg.model_name, pretrained=cfg.pretrained)

        dataset = get_dataset(
            dataset_name=target_dataset_name,
            preprocess_train=image_encoder.train_preprocess,
            preprocess_inference=image_encoder.val_preprocess,
            batch_size=cfg.batch_size,
            num_workers=int(os.environ['TORCH_NUM_WORKERS']),
            seed=cfg.target.seed,
        )

        del image_encoder

        head_base_path = os.environ['HEAD_BASE_PATH']

        classification_head = get_classification_head(
            model_name=cfg.model_name,
            pretrained=cfg.pretrained,
            dataset_name=target_dataset_name,
            save_dir=head_base_path,
            device=device,
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
                dataset=dataset,
                classification_head=classification_head,
                device=device,
                tgt_epochs=tgt_epochs,
                eval_split=eval_split,
            )

        ####################################################################
        # Cleanup between target iterations
        ####################################################################

        del dataset, fp_tgt_sd, classification_head
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if IS_SLURM:
        log.info("All %d pairs completed. Forcing exit.", total_pairs)
        os._exit(0)


if __name__ == "__main__":
    main()
