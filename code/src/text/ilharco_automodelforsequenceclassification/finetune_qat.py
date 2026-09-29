import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.duration import mult_path_frag, run_meta, training_budget

# MUST be the first thing that runs: HF libs (transformers, huggingface_hub,
# datasets) and our own src.text.data.common all snapshot env vars at import
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
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

log = logging.getLogger(__name__)

IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)
LOG_EVERY = 50
REFERENCE_BATCH_SIZE = 32
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

from src.quantization import QATLinear, apply_ptq_, disable_qat_, enable_qat_
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


def _evaluate_ptq(
    model: nn.Module,
    tokenizer,
    max_length: int,
    loader,
    device: torch.device,
    bits: int,
    granularity: str,
    skip_modules: frozenset[str],
    limit_num_batches,
    desc: str,
) -> float:
    """Deepcopy the model, strip QAT wrappers, apply PTQ, and evaluate.

    This mirrors the deployment path: training uses fake-quant (QATLinear),
    but the reported metric is what a true post-training-quantized model
    (apply_ptq_ on plain nn.Linear) would achieve on this checkpoint.

    Deepcopy safety: AutoModelForSequenceClassification is a plain nn.Module
    subclass wrapping a transformer backbone + an nn.Linear-based head. No
    forward/backward hooks or non-picklable attrs are registered. QATLinear
    stores only a child nn.Linear plus Python scalars. All components are
    therefore deepcopy-safe.
    """
    model.eval()
    # Move the live model to CPU briefly so the deepcopy is made on CPU and
    # peak VRAM never holds two full copies of the backbone simultaneously.
    eval_model = copy.deepcopy(model).to("cpu")
    disable_qat_(eval_model)  # QATLinear -> nn.Linear (in place)
    apply_ptq_(
        eval_model,
        bits=bits,
        granularity=granularity,
        skip_modules=skip_modules,
    )
    eval_model.to(device)
    eval_model.eval()

    correct, total = 0, 0
    bar = tqdm(
        loader, desc=desc, colour=random_tqdm_color(), leave=False, **TQDM_KW
    )
    with torch.no_grad():
        for i, batch in enumerate(bar):
            if limit_num_batches is not None and i >= limit_num_batches:
                break
            texts, labels = batch
            encoding = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            labels = labels.to(device=device, dtype=torch.long)
            outputs = eval_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
            if not IS_SLURM:
                bar.set_postfix(acc=f"{correct / max(total, 1):.4f}")

    del eval_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return correct / max(total, 1)


@hydra.main(
    config_path="../../../../config/src/text/ilharco_automodelforsequenceclassification",
    config_name="finetune_qat",
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
    skip_modules_sorted = sorted(cfg.qat.skip_modules)
    skip_tag = "-".join(skip_modules_sorted) if skip_modules_sorted else "none"
    save_dir_parts = [
        checkpoint_base_path,
        "text",
        "ilharco_automodelforsequenceclassification",
        "qat_dryrun" if is_dryrun else "qat",
        sanitized_model,
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}_ml={cfg.max_length}",
        mult_path_frag(cfg.epoch_mult),
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={skip_tag}",
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

    # Enable quantization-aware training: wraps nn.Linear layers in QATLinear
    # (STE fake-quant forward), skipping the named top-level children in
    # cfg.qat.skip_modules (typically ["classifier"]).
    skip_modules = frozenset(cfg.qat.skip_modules)
    all_linear_names_before = sorted(
        name
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear)
    )
    enable_qat_(
        model,
        bits=cfg.qat.bits,
        granularity=cfg.qat.granularity,
        skip_modules=skip_modules,
    )
    # After enable_qat_, the surviving plain nn.Linear modules are exactly
    # those that were skipped. QAT-wrapped linears now live at
    # "<parent>.linear" (child of QATLinear).
    qat_wrapped_parents = sorted(
        name for name, m in model.named_modules() if isinstance(m, QATLinear)
    )
    surviving_plain_linears = sorted(
        name
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and not name.endswith(".linear")
    )
    if IS_SLURM:
        log.info(
            "QAT enabled: wrapped=%d, skipped=%d (of %d original nn.Linear)",
            len(qat_wrapped_parents),
            len(surviving_plain_linears),
            len(all_linear_names_before),
        )
        log.info("QAT skipped layers: %s", surviving_plain_linears)
    else:
        print(
            f"QAT enabled: {len(qat_wrapped_parents)} wrapped, "
            f"{len(surviving_plain_linears)} skipped "
            f"(of {len(all_linear_names_before)} original nn.Linear)"
        )
        pprint({"qat_skipped": surviving_plain_linears}, expand_all=True)

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

    # Optimizer / scheduler / loss — built AFTER enable_qat_ so params are
    # collected from the QAT-wrapped model (QATLinear.linear.weight is still
    # a regular nn.Parameter; STE keeps the forward differentiable).
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

        # Per-epoch validation: evaluate a PTQ-converted deepcopy so the
        # reported accuracy reflects the deployed (truly quantized) model.
        val_acc = _evaluate_ptq(
            model=model,
            tokenizer=tokenizer,
            max_length=cfg.max_length,
            loader=dataset.val_loader,
            device=device,
            bits=cfg.qat.bits,
            granularity=cfg.qat.granularity,
            skip_modules=skip_modules,
            limit_num_batches=cfg.limit_num_batches,
            desc=f"val e{epoch}",
        )
        if IS_SLURM:
            log.info("epoch %d val_acc=%.4f", epoch, val_acc)
        else:
            epoch_bar.set_postfix(val_acc=f"{val_acc:.4f}")

    # Final test evaluation — same PTQ-converted deepcopy path.
    test_acc = _evaluate_ptq(
        model=model,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        loader=dataset.test_loader,
        device=device,
        bits=cfg.qat.bits,
        granularity=cfg.qat.granularity,
        skip_modules=skip_modules,
        limit_num_batches=cfg.limit_num_batches,
        desc="test",
    )
    if IS_SLURM:
        log.info("final test accuracy: %.4f", test_acc)
    else:
        print(f"Final test accuracy: {test_acc:.4f}")

    # Save backbone and head checkpoints separately: strip QAT wrappers so
    # the saved state_dict is identical in format to finetune_fp.py output
    # (plain nn.Linear keys) and can be consumed unchanged by downstream
    # scripts.
    disable_qat_(model)
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
