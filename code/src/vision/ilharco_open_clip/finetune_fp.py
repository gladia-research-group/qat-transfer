import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.duration import clamped_warmup, mult_path_frag, run_meta, training_budget

# MUST be the first thing that runs: HF libs (transformers, huggingface_hub,
# datasets) and our own src.vision.data.common all snapshot env vars at import
# time. Loading .env after those imports has no effect.
from dotenv import load_dotenv
load_dotenv()

import json
import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from tqdm import tqdm

log = logging.getLogger(__name__)

IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)
LOG_EVERY = 50
REFERENCE_BATCH_SIZE = 128

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


@hydra.main(
    config_path="../../../../config/src/vision/ilharco_open_clip",
    config_name="finetune_fp",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(dict(cfg), expand_all=True)
    set_seed(cfg.seed)

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    checkpoint_base_path = os.environ["CHECKPOINT_BASE_PATH"]
    num_workers = int(os.environ["TORCH_NUM_WORKERS"])

    sanitized_model = sanitize_open_clip_model_name(cfg.model_name, cfg.pretrained)

    is_dryrun = (
        cfg.limit_num_batches is not None or cfg.limit_num_epochs is not None
    )
    save_dir_parts = [
        checkpoint_base_path,
        "vision",
        "ilharco_open_clip",
        "fp_dryrun" if is_dryrun else "fp",
        sanitized_model,
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        mult_path_frag(cfg.epoch_mult),
        f"seed={cfg.seed}",
    ]
    if is_dryrun:
        lnb = cfg.limit_num_batches if cfg.limit_num_batches is not None else "all"
        lne = cfg.limit_num_epochs if cfg.limit_num_epochs is not None else "all"
        save_dir_parts.append(f"lnb={lnb}_lne={lne}")
    save_dir = os.path.join(*save_dir_parts)
    os.makedirs(save_dir, exist_ok=True)

    # Build encoder + frozen zero-shot classification head
    encoder = ImageEncoder(model_name=cfg.model_name, pretrained=cfg.pretrained, keep_lang=False)
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

    if IS_SLURM:
        log.info("state_dict keys: %s", list(classifier.state_dict().keys()))
        log.info("classifier: %s", classifier)
    else:
        pprint(list(classifier.state_dict().keys()), expand_all=True)
        pprint(classifier, expand_all=True)

    # Dataset (seeded with the run seed — not SPLIT_SEED)
    dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        preprocess_train=classifier.train_preprocess,
        preprocess_inference=classifier.val_preprocess,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        seed=cfg.seed,
    )
    num_batches = len(dataset.train_loader)
    assert REFERENCE_BATCH_SIZE % cfg.batch_size == 0, (
        f"batch_size={cfg.batch_size} must evenly divide {REFERENCE_BATCH_SIZE}"
    )
    accum_steps = REFERENCE_BATCH_SIZE // cfg.batch_size

    # Optimizer / scheduler / loss
    params = [p for p in classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    # Training budget. `epoch_mult` scales this dataset's schedule; mult=1.0 is
    # pinned to reproduce the pre-multiplier behaviour exactly, so the loop still
    # runs every epoch to completion and the max_steps break never fires.
    #
    # This also repairs a pre-existing dryrun inconsistency: the loop used to run
    # `effective_epochs` (clamped by limit_num_epochs) while the scheduler length
    # and the checkpoint filename used the *unclamped* table value -- and the
    # evaluators then rebuilt that filename from the limit, so a limited run
    # wrote one name and was read under another. All three now agree.
    budget = training_budget(
        cfg.dataset_name, cfg.epoch_mult, num_batches, accum_steps,
        DATASET_NAME_TO_EPOCHS, cfg.limit_num_epochs,
    )
    effective_epochs = budget.loop_epochs
    max_steps = budget.max_steps
    ckpt_epochs = budget.ckpt_epochs

    scheduler = cosine_lr(optimizer, cfg.lr, clamped_warmup(cfg.wl, max_steps), max_steps)
    loss_fn = (
        LabelSmoothing(cfg.ls) if cfg.ls > 0 else torch.nn.CrossEntropyLoss()
    )

    # Training loop
    epoch_bar = tqdm(
        range(effective_epochs), desc="epochs", colour=random_tqdm_color(), **TQDM_KW
    )
    budget_exhausted = False
    for epoch in epoch_bar:
        if budget_exhausted:
            break
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
                opt_step = (i + 1) // accum_steps + epoch * (num_batches // accum_steps) - 1
                scheduler(opt_step)

                if IS_SLURM:
                    opt_step_in_epoch = (i + 1) // accum_steps
                    if opt_step_in_epoch % LOG_EVERY == 0 or i == num_batches - 1:
                        log.info(
                            "epoch %d step %d/%d loss=%.4f",
                            epoch, opt_step_in_epoch, num_batches // accum_steps, accum_loss,
                        )
                else:
                    train_bar.set_postfix(loss=f"{accum_loss:.4f}")

                optimizer.zero_grad()
                accum_loss = 0.0

                # Stop mid-epoch once the budget is spent. At mult=1.0 max_steps
                # equals the full schedule, so this never fires.
                if opt_step + 1 >= max_steps:
                    budget_exhausted = True
                    break

        # Per-epoch validation
        classifier.eval()
        val_correct, val_total = 0, 0
        val_bar = tqdm(
            dataset.val_loader,
            desc=f"val e{epoch}",
            colour=random_tqdm_color(),
            leave=False,
            **TQDM_KW,
        )
        with torch.no_grad():
            for i, batch in enumerate(val_bar):
                if (
                    cfg.limit_num_batches is not None
                    and i >= cfg.limit_num_batches
                ):
                    break
                batch = maybe_dictionarize(batch)
                images = batch["images"].to(device)
                labels = batch["labels"].to(device=device, dtype=torch.long)
                logits = classifier(images)
                val_correct += (logits.argmax(dim=-1) == labels).sum().item()
                val_total += labels.size(0)
                if not IS_SLURM:
                    val_bar.set_postfix(
                        acc=f"{val_correct / max(val_total, 1):.4f}"
                    )

        val_acc = val_correct / max(val_total, 1)
        if IS_SLURM:
            log.info("epoch %d val_acc=%.4f", epoch, val_acc)
        else:
            epoch_bar.set_postfix(val_acc=f"{val_acc:.4f}")

    # Final test evaluation
    classifier.eval()
    test_correct, test_total = 0, 0
    test_bar = tqdm(
        dataset.test_loader, desc="test", colour=random_tqdm_color(), **TQDM_KW
    )
    with torch.no_grad():
        for i, batch in enumerate(test_bar):
            if (
                cfg.limit_num_batches is not None
                and i >= cfg.limit_num_batches
            ):
                break
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device=device, dtype=torch.long)
            logits = classifier(images)
            test_correct += (logits.argmax(dim=-1) == labels).sum().item()
            test_total += labels.size(0)
            if not IS_SLURM:
                test_bar.set_postfix(
                    acc=f"{test_correct / max(test_total, 1):.4f}"
                )

    test_acc = test_correct / max(test_total, 1)
    if IS_SLURM:
        log.info("final test accuracy: %.4f", test_acc)
    else:
        print(f"Final test accuracy: {test_acc:.4f}")

    # Save encoder checkpoint
    ckpt_path = os.path.join(save_dir, f"epoch_{ckpt_epochs}.pt")
    classifier.image_encoder.save(ckpt_path)

    # The realized budget is not recoverable from the path, so record it.
    meta = run_meta(budget, num_batches, accum_steps, cfg.wl)
    meta["final_test_accuracy"] = test_acc
    meta["encoder_path"] = ckpt_path
    with open(os.path.join(save_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
