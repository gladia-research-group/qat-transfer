import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.duration import mult_path_frag, run_meta, training_budget

# MUST be the first thing that runs: HF libs (transformers, huggingface_hub,
# datasets) and our own src.text.data.common all snapshot env vars at import
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

log = logging.getLogger(__name__)

IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)
LOG_EVERY = 50
REFERENCE_BATCH_SIZE = 32
SUPPORTED_MODELS = {
    "google-t5/t5-small",
    "google-t5/t5-base",
    "google-t5/t5-large",
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

from src.text.data.common import DATASET_NAME_TO_EPOCHS
from src.text.data.registry import get_dataset
from src.vision.utils import (
    LabelSmoothing,
    linear_lr,
    random_tqdm_color,
    sanitize_hf_model_name,
    set_seed,
)

OmegaConf.register_new_resolver(
    "sanitize_hf", sanitize_hf_model_name, replace=True
)


@hydra.main(
    config_path="../../../../config/src/text/ilharco_automodelforsequenceclassification",
    config_name="finetune_fp",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(dict(cfg), expand_all=True)
    set_seed(cfg.seed)

    if cfg.model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model_name={cfg.model_name!r}. Supported: {sorted(SUPPORTED_MODELS)}")

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    checkpoint_base_path = os.environ["CHECKPOINT_BASE_PATH"]
    num_workers = int(os.environ["TORCH_NUM_WORKERS"])

    sanitized_model = sanitize_hf_model_name(cfg.model_name)

    is_dryrun = (
        cfg.limit_num_batches is not None or cfg.limit_num_epochs is not None
    )
    save_dir_parts = [
        checkpoint_base_path,
        "text",
        "ilharco_automodelforsequenceclassification",
        "fp_dryrun" if is_dryrun else "fp",
        sanitized_model,
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}_ml={cfg.max_length}",
        mult_path_frag(cfg.epoch_mult),
        f"seed={cfg.seed}",
    ]
    if is_dryrun:
        lnb = cfg.limit_num_batches if cfg.limit_num_batches is not None else "all"
        lne = cfg.limit_num_epochs if cfg.limit_num_epochs is not None else "all"
        save_dir_parts.append(f"lnb={lnb}_lne={lne}")
    save_dir = os.path.join(*save_dir_parts)
    os.makedirs(save_dir, exist_ok=True)

    # Dataset (seeded with the run seed — not SPLIT_SEED)
    dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        seed=cfg.seed,
    )
    num_labels = len(dataset.class_names)

    # Model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=num_labels,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device=device, dtype=torch.float32)
    if cfg.model_name == "Qwen/Qwen3-Embedding-0.6B":
        model.gradient_checkpointing_enable()

    if IS_SLURM:
        log.info("state_dict keys: %s", list(model.state_dict().keys()))
        log.info("model: %s", model)
    else:
        pprint(list(model.state_dict().keys()), expand_all=True)
        pprint(model, expand_all=True)

    # Epochs, gradient accumulation
    num_batches = len(dataset.train_loader)
    assert REFERENCE_BATCH_SIZE % cfg.batch_size == 0, (
        f"batch_size={cfg.batch_size} must evenly divide {REFERENCE_BATCH_SIZE}"
    )
    accum_steps = REFERENCE_BATCH_SIZE // cfg.batch_size

    # Optimizer / scheduler / loss
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    # Training budget. `epoch_mult` scales this dataset's schedule; mult=1.0 is
    # pinned to reproduce the pre-multiplier behaviour exactly, so the loop still
    # runs every epoch to completion and the max_steps break never fires.
    #
    # This family has no warmup at all -- linear_lr decays from lr to 0 with no
    # warmup phase and the configs carry no `wl` -- so it is the one place the
    # multiplier cannot interact with warmup length.
    #
    # It also repairs a pre-existing dryrun inconsistency: the loop ran
    # `effective_epochs` (clamped by limit_num_epochs) while the scheduler length
    # and the checkpoint filename used the *unclamped* table value.
    budget = training_budget(
        cfg.dataset_name, cfg.epoch_mult, num_batches, accum_steps,
        DATASET_NAME_TO_EPOCHS, cfg.limit_num_epochs,
    )
    effective_epochs = budget.loop_epochs
    max_steps = budget.max_steps
    ckpt_epochs = budget.ckpt_epochs

    scheduler = linear_lr(optimizer, cfg.lr, max_steps)
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
        model.train()
        train_bar = tqdm(
            dataset.train_loader,
            desc=f"train e{epoch}",
            colour=random_tqdm_color(),
            leave=False,
            **TQDM_KW,
        )
        optimizer.zero_grad()
        accum_loss = 0.0
        train_correct, train_total = 0, 0
        for i, batch in enumerate(train_bar):
            texts, labels = batch
            encoding = tokenizer(texts, padding=True, truncation=True, max_length=cfg.max_length, return_tensors="pt")
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            labels = labels.to(device=device, dtype=torch.long)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss: torch.Tensor = loss_fn(logits, labels) / accum_steps
            loss.backward()
            accum_loss += loss.item()

            train_correct += (logits.detach().argmax(dim=-1) == labels).sum().item()
            train_total += labels.size(0)

            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                optimizer.step()
                opt_step = (i + 1) // accum_steps + epoch * (num_batches // accum_steps) - 1
                scheduler(opt_step)

                train_acc = train_correct / max(train_total, 1)
                if IS_SLURM:
                    opt_step_in_epoch = (i + 1) // accum_steps
                    if opt_step_in_epoch % LOG_EVERY == 0 or i == num_batches - 1:
                        log.info(
                            "epoch %d step %d/%d loss=%.4f acc=%.4f",
                            epoch, opt_step_in_epoch, num_batches // accum_steps, accum_loss, train_acc,
                        )
                else:
                    train_bar.set_postfix(loss=f"{accum_loss:.4f}", acc=f"{train_acc:.4f}")

                optimizer.zero_grad()
                accum_loss = 0.0

                # Stop mid-epoch once the budget is spent. At mult=1.0 max_steps
                # equals the full schedule, so this never fires.
                if opt_step + 1 >= max_steps:
                    budget_exhausted = True
                    break

        # Per-epoch validation
        model.eval()
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
                texts, labels = batch
                encoding = tokenizer(texts, padding=True, truncation=True, max_length=cfg.max_length, return_tensors="pt")
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)
                labels = labels.to(device=device, dtype=torch.long)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
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
    model.eval()
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
            texts, labels = batch
            encoding = tokenizer(texts, padding=True, truncation=True, max_length=cfg.max_length, return_tensors="pt")
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            labels = labels.to(device=device, dtype=torch.long)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
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

    # Save backbone and head checkpoints separately
    head_module = MODEL_NAME_TO_HEAD_MODULE[cfg.model_name]
    head_prefix = head_module + "."
    full_sd = model.state_dict()
    head_state = {k: v for k, v in full_sd.items() if k.startswith(head_prefix)}
    backbone_state = {k: v for k, v in full_sd.items() if not k.startswith(head_prefix)}

    backbone_path = os.path.join(save_dir, f"backbone_epoch_{ckpt_epochs}.pt")
    torch.save(backbone_state, backbone_path)

    head_path = os.path.join(save_dir, f"head_epoch_{ckpt_epochs}.pt")
    torch.save(head_state, head_path)


    # The realized budget is not recoverable from the path, so record it. This
    # family has no warmup, hence None.
    meta = run_meta(budget, num_batches, accum_steps, None)
    meta["final_test_accuracy"] = test_acc
    meta["backbone_path"] = backbone_path
    with open(os.path.join(save_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
