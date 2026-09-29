import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

# MUST be the first thing that runs: HF libs (transformers, huggingface_hub,
# datasets) and our own src.vision.data.common all snapshot env vars at import
# time. Loading .env after those imports has no effect.
from dotenv import load_dotenv
load_dotenv()

import copy
import json
import logging
import os

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from torch.utils.flop_counter import FlopCounterMode
from tqdm import tqdm

log = logging.getLogger(__name__)

IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)
LOG_EVERY = 50
REFERENCE_BATCH_SIZE = 128

from src.quantization import (
    enable_qat_open_clip_,
    disable_qat_open_clip_,
)
from src.vision.data.common import DATASET_NAME_TO_EPOCHS, maybe_dictionarize
from src.vision.data.registry import get_dataset
from src.vision.ilharco_open_clip.heads import get_classification_head
from src.vision.ilharco_open_clip.modeling import ImageClassifier, ImageEncoder
from src.vision.utils import (
    LabelSmoothing,
    cosine_lr,
    random_tqdm_color,
    sanitize_open_clip_model_name,
    set_seed,
)

OmegaConf.register_new_resolver(
    "sanitize_open_clip", sanitize_open_clip_model_name, replace=True
)


def _get_flop_total(flop_counter: FlopCounterMode) -> int:
    """Extract total FLOPs from a FlopCounterMode instance."""
    return sum(flop_counter.get_flop_counts()["Global"].values())


def _build_classifier(cfg, device):
    """Build a fresh QAT-enabled classifier."""
    encoder = ImageEncoder(
        model_name=cfg.model_name, pretrained=cfg.pretrained, keep_lang=False
    )
    head = get_classification_head(
        model_name=cfg.model_name,
        pretrained=cfg.pretrained,
        dataset_name=cfg.dataset_name,
        save_dir=cfg.head_cache_dir,
        device=device,
    )
    classifier = ImageClassifier(encoder, head)
    classifier.freeze_head()
    classifier.to(device)

    skip_modules = frozenset(cfg.qat.skip_modules)
    enable_qat_open_clip_(
        classifier,
        bits=cfg.qat.bits,
        granularity=cfg.qat.granularity,
        skip_modules=skip_modules,
    )
    return classifier, skip_modules


def _run_training_loop(
    classifier,
    dataset,
    cfg,
    device,
    epochs,
    num_batches,
    accum_steps,
    limit_num_batches,
):
    """Run the full QAT training loop (mirrors finetune_qat.py)."""
    params = [p for p in classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = cosine_lr(
        optimizer, cfg.lr, cfg.wl, epochs * num_batches // accum_steps
    )
    loss_fn = (
        LabelSmoothing(cfg.ls) if cfg.ls > 0 else torch.nn.CrossEntropyLoss()
    )

    epoch_bar = tqdm(
        range(epochs), desc="epochs", colour=random_tqdm_color(), **TQDM_KW
    )
    for epoch in epoch_bar:
        classifier.train()
        train_bar = tqdm(
            dataset.train_loader,
            desc=f"train e{epoch}",
            colour=random_tqdm_color(),
            leave=False,
            **TQDM_KW,
        )
        optimizer.zero_grad()
        accum_loss = 0.0
        for i, batch in enumerate(train_bar):
            if limit_num_batches is not None and i >= limit_num_batches:
                break

            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device=device, dtype=torch.long)

            logits = classifier(images)
            loss: torch.Tensor = loss_fn(logits, labels) / accum_steps
            loss.backward()
            accum_loss += loss.item()

            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                optimizer.step()
                opt_step = (
                    (i + 1) // accum_steps
                    + epoch * (num_batches // accum_steps)
                    - 1
                )
                scheduler(opt_step)

                if IS_SLURM:
                    opt_step_in_epoch = (i + 1) // accum_steps
                    if opt_step_in_epoch % LOG_EVERY == 0 or i == num_batches - 1:
                        log.info(
                            "epoch %d step %d/%d loss=%.4f",
                            epoch,
                            opt_step_in_epoch,
                            num_batches // accum_steps,
                            accum_loss,
                        )
                else:
                    train_bar.set_postfix(loss=f"{accum_loss:.4f}")

                optimizer.zero_grad()
                accum_loss = 0.0


@hydra.main(
    config_path="../../../../../../config/experiments/vision/ilharco_open_clip/999_paper_stuff/001_qat_vs_qat_transfer_flops",
    config_name="compute_flops",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(dict(cfg), expand_all=True)
    set_seed(cfg.seed)

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    num_workers = int(os.environ["TORCH_NUM_WORKERS"])
    evaluation_base_path = os.environ["EVALUATION_BASE_PATH"]
    sanitized_model = sanitize_open_clip_model_name(cfg.model_name, cfg.pretrained)

    ############################################################################
    # BEGIN setup
    ############################################################################

    classifier, skip_modules = _build_classifier(cfg, device)

    dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        preprocess_train=classifier.train_preprocess,
        preprocess_inference=classifier.val_preprocess,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        seed=cfg.seed,
    )

    epochs = DATASET_NAME_TO_EPOCHS[cfg.dataset_name]
    effective_epochs = (
        min(epochs, cfg.limit_num_epochs)
        if cfg.limit_num_epochs is not None
        else epochs
    )
    num_batches = len(dataset.train_loader)
    effective_num_batches = (
        min(num_batches, cfg.limit_num_batches)
        if cfg.limit_num_batches is not None
        else num_batches
    )
    assert REFERENCE_BATCH_SIZE % cfg.batch_size == 0
    accum_steps = REFERENCE_BATCH_SIZE // cfg.batch_size
    total_steps = effective_num_batches * effective_epochs

    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(
        p.numel() for p in classifier.parameters() if p.requires_grad
    )
    encoder_sd = classifier.image_encoder.state_dict()
    encoder_fp_elements = sum(
        v.numel() for v in encoder_sd.values() if v.is_floating_point()
    )

    if IS_SLURM:
        log.info(
            "dataset=%s epochs=%d num_batches=%d total_steps=%d accum_steps=%d",
            cfg.dataset_name, effective_epochs, effective_num_batches, total_steps, accum_steps,
        )
    else:
        print(
            f"dataset={cfg.dataset_name} epochs={effective_epochs} "
            f"num_batches={effective_num_batches} total_steps={total_steps} "
            f"accum_steps={accum_steps}"
        )

    ############################################################################
    # END setup
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN QAT Training FLOPs — Theoretical
    ############################################################################

    if IS_SLURM:
        log.info("Measuring theoretical forward-pass FLOPs...")
    else:
        print("\n[1/4] Measuring theoretical forward-pass FLOPs...")

    # Get a single batch for the forward pass measurement
    sample_batch = next(iter(dataset.train_loader))
    sample_batch = maybe_dictionarize(sample_batch)
    sample_images = sample_batch["images"].to(device)

    classifier.eval()
    flop_counter = FlopCounterMode()
    with torch.no_grad(), flop_counter:
        classifier(sample_images)
    theoretical_forward_flops = _get_flop_total(flop_counter)
    theoretical_per_step_3x = theoretical_forward_flops * 3
    theoretical_total = theoretical_per_step_3x * total_steps

    if IS_SLURM:
        log.info(
            "theoretical: forward=%d per_step_3x=%d total=%d",
            theoretical_forward_flops, theoretical_per_step_3x, theoretical_total,
        )
    else:
        print(
            f"  forward_flops={theoretical_forward_flops:,} "
            f"per_step_3x={theoretical_per_step_3x:,} "
            f"total={theoretical_total:,}"
        )

    ############################################################################
    # END QAT Training FLOPs — Theoretical
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN QAT Training FLOPs — Empirical Extrapolated
    ############################################################################

    if IS_SLURM:
        log.info("Measuring empirical FLOPs (%d warmup steps)...", cfg.num_warmup_steps)
    else:
        print(f"\n[2/4] Measuring empirical FLOPs ({cfg.num_warmup_steps} warmup steps)...")

    # Save initial state so we can reset after warmup
    initial_sd = copy.deepcopy(classifier.state_dict())

    classifier.train()
    params = [p for p in classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    loss_fn = (
        LabelSmoothing(cfg.ls) if cfg.ls > 0 else torch.nn.CrossEntropyLoss()
    )

    warmup_flop_counter = FlopCounterMode()
    steps_done = 0
    with warmup_flop_counter:
        for batch in dataset.train_loader:
            if steps_done >= cfg.num_warmup_steps:
                break
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device=device, dtype=torch.long)

            logits = classifier(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            steps_done += 1

    empirical_warmup_total = _get_flop_total(warmup_flop_counter)
    empirical_per_step = empirical_warmup_total // cfg.num_warmup_steps
    empirical_extrapolated_total = empirical_per_step * total_steps

    if IS_SLURM:
        log.info(
            "empirical extrapolated: per_step=%d total=%d",
            empirical_per_step, empirical_extrapolated_total,
        )
    else:
        print(
            f"  per_step={empirical_per_step:,} "
            f"total_extrapolated={empirical_extrapolated_total:,}"
        )

    # Reset model state
    classifier.load_state_dict(initial_sd)
    del initial_sd, optimizer, params
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    ############################################################################
    # END QAT Training FLOPs — Empirical Extrapolated
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN QAT Training FLOPs — Empirical Full Training
    ############################################################################

    empirical_full_total = None

    if cfg.run_full_training:
        if IS_SLURM:
            log.info("Measuring empirical FLOPs (full training)...")
        else:
            print(f"\n[3/4] Measuring empirical FLOPs (full training, {effective_epochs} epochs)...")

        # Rebuild a fresh classifier for clean measurement
        del classifier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        classifier, skip_modules = _build_classifier(cfg, device)

        full_flop_counter = FlopCounterMode()
        with full_flop_counter:
            _run_training_loop(
                classifier=classifier,
                dataset=dataset,
                cfg=cfg,
                device=device,
                epochs=effective_epochs,
                num_batches=num_batches,
                accum_steps=accum_steps,
                limit_num_batches=cfg.limit_num_batches,
            )
        empirical_full_total = _get_flop_total(full_flop_counter)

        if IS_SLURM:
            log.info("empirical full training: total=%d", empirical_full_total)
        else:
            print(f"  total={empirical_full_total:,}")
    else:
        if IS_SLURM:
            log.info("Skipping full training measurement (run_full_training=false)")
        else:
            print("\n[3/4] Skipping full training measurement (run_full_training=false)")

    ############################################################################
    # END QAT Training FLOPs — Empirical Full Training
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN QV Transfer FLOPs — Empirical
    ############################################################################

    if IS_SLURM:
        log.info("Measuring QV transfer FLOPs (weight-space arithmetic)...")
    else:
        print("\n[4/4] Measuring QV transfer FLOPs (weight-space arithmetic)...")

    # Generate random state dicts matching the encoder's structure
    # (we only care about tensor shapes/ops, not values)
    sd_fp_source = {
        k: torch.randn_like(v) for k, v in encoder_sd.items() if v.is_floating_point()
    }
    sd_qat_source = {
        k: torch.randn_like(v) for k, v in encoder_sd.items() if v.is_floating_point()
    }
    sd_fp_target = {
        k: torch.randn_like(v) for k, v in encoder_sd.items() if v.is_floating_point()
    }

    alpha = float(cfg.qv_alpha)

    transfer_flop_counter = FlopCounterMode()
    with torch.no_grad(), transfer_flop_counter:
        # Step 1: Compute QV = QAT_source - FP_source
        qv = {k: sd_qat_source[k] - sd_fp_source[k] for k in sd_fp_source}
        # Step 2: Scale by alpha
        scaled_qv = {k: alpha * v for k, v in qv.items()}
        # Step 3: Apply to target: patched = FP_target + scaled_QV
        patched = {k: sd_fp_target[k] + scaled_qv[k] for k in sd_fp_target}

    transfer_flop_counter_total = _get_flop_total(transfer_flop_counter)

    # Manual calculation as fallback (FlopCounterMode may not count element-wise ops)
    manual_transfer_flops = encoder_fp_elements * 3  # sub + mul + add

    if IS_SLURM:
        log.info(
            "QV transfer: flop_counter=%d manual=%d",
            transfer_flop_counter_total, manual_transfer_flops,
        )
    else:
        print(
            f"  flop_counter_total={transfer_flop_counter_total:,} "
            f"manual_elementwise={manual_transfer_flops:,}"
        )

    del sd_fp_source, sd_qat_source, sd_fp_target, qv, scaled_qv, patched

    ############################################################################
    # END QV Transfer FLOPs — Empirical
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN Save results
    ############################################################################

    # Use manual transfer FLOPs for speedup if flop_counter returned 0
    transfer_flops_for_ratio = (
        transfer_flop_counter_total if transfer_flop_counter_total > 0
        else manual_transfer_flops
    )

    results = {
        "experiment": "qat_vs_qat_transfer_flops",
        "model_name": cfg.model_name,
        "pretrained": cfg.pretrained,
        "dataset_name": cfg.dataset_name,
        "seed": cfg.seed,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "wd": cfg.wd,
        "ls": cfg.ls,
        "wl": cfg.wl,
        "max_grad_norm": cfg.max_grad_norm,
        "qat": {
            "bits": cfg.qat.bits,
            "granularity": cfg.qat.granularity,
            "skip_modules": list(cfg.qat.skip_modules),
        },
        "dataset_info": {
            "num_train_samples": len(dataset.train_loader.dataset),
            "num_batches_per_epoch": num_batches,
            "effective_num_batches_per_epoch": effective_num_batches,
            "epochs": effective_epochs,
            "total_steps": total_steps,
            "accum_steps": accum_steps,
        },
        "parameter_info": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "encoder_floating_point_elements": encoder_fp_elements,
        },
        "qat_training_flops": {
            "theoretical": {
                "per_forward_pass": theoretical_forward_flops,
                "per_step_3x": theoretical_per_step_3x,
                "total": theoretical_total,
            },
            "empirical_extrapolated": {
                "num_warmup_steps": cfg.num_warmup_steps,
                "per_step_measured": empirical_per_step,
                "total": empirical_extrapolated_total,
            },
            "empirical_full_training": {
                "total": empirical_full_total,
            },
        },
        "qv_transfer_flops": {
            "flop_counter_mode_total": transfer_flop_counter_total,
            "manual_elementwise_total": manual_transfer_flops,
            "note": "3 ops (sub+mul+add) per floating-point element",
        },
        "speedup_ratio": {
            "theoretical_training_vs_transfer": (
                theoretical_total / transfer_flops_for_ratio
                if transfer_flops_for_ratio > 0 else None
            ),
            "empirical_extrapolated_vs_transfer": (
                empirical_extrapolated_total / transfer_flops_for_ratio
                if transfer_flops_for_ratio > 0 else None
            ),
            "empirical_full_vs_transfer": (
                empirical_full_total / transfer_flops_for_ratio
                if empirical_full_total is not None and transfer_flops_for_ratio > 0
                else None
            ),
        },
    }

    skip_modules_sorted = sorted(cfg.qat.skip_modules)
    skip_tag = "-".join(skip_modules_sorted) if skip_modules_sorted else "none"
    eval_dir = os.path.join(
        evaluation_base_path,
        "vision",
        "ilharco_open_clip",
        "999_paper_stuff",
        "001_qat_vs_qat_transfer_flops",
        sanitized_model,
        cfg.dataset_name,
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={skip_tag}",
        f"seed={cfg.seed}",
    )
    os.makedirs(eval_dir, exist_ok=True)
    results_path = os.path.join(eval_dir, "eval_results.json")

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    if IS_SLURM:
        log.info("Results saved to: %s", results_path)
    else:
        print(f"\nResults saved to: {results_path}")
        pprint(results, expand_all=True)

    ############################################################################
    # END Save results
    ############################################################################


if __name__ == "__main__":
    main()
