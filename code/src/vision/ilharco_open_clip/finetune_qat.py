import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.duration import clamped_warmup, mult_path_frag, run_meta, training_budget

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
from tqdm import tqdm

log = logging.getLogger(__name__)

IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)
LOG_EVERY = 50
REFERENCE_BATCH_SIZE = 128

from src.quantization import (
    QATLinear,
    QATMultiheadAttention,
    apply_ptq_open_clip_,
    disable_qat_open_clip_,
    enable_qat_open_clip_,
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


def _evaluate_ptq(
    classifier: nn.Module,
    loader,
    device: torch.device,
    bits: int,
    granularity: str,
    skip_modules: frozenset[str],
    limit_num_batches,
    desc: str,
) -> float:
    """Deepcopy the classifier, strip QAT wrappers, apply PTQ, and evaluate.

    This mirrors the deployment path: training uses fake-quant
    (QATLinear / QATMultiheadAttention), but the reported metric is what a
    true post-training-quantized model would achieve on this checkpoint.

    Deepcopy safety: ImageClassifier / ImageEncoder / ClassificationHead are
    plain nn.Module subclasses wrapping an open_clip model + an nn.Linear-based
    head. No forward/backward hooks or non-picklable attrs are registered.
    QATLinear and QATMultiheadAttention store only a child module plus Python
    scalars. All components are therefore deepcopy-safe.
    """
    classifier.eval()
    # Move the live model to CPU briefly so the deepcopy is made on CPU and
    # peak VRAM never holds two full copies of the backbone simultaneously.
    eval_model = copy.deepcopy(classifier).to("cpu")
    disable_qat_open_clip_(eval_model)  # QATLinear/QATMultiheadAttention -> originals
    apply_ptq_open_clip_(
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
            batch = maybe_dictionarize(batch)
            images = batch["images"].to(device)
            labels = batch["labels"].to(device=device, dtype=torch.long)
            logits = eval_model(images)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
            if not IS_SLURM:
                bar.set_postfix(acc=f"{correct / max(total, 1):.4f}")

    del eval_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return correct / max(total, 1)


@hydra.main(
    config_path="../../../../config/src/vision/ilharco_open_clip",
    config_name="finetune_qat",
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
    skip_modules_sorted = sorted(cfg.qat.skip_modules)
    skip_tag = "-".join(skip_modules_sorted) if skip_modules_sorted else "none"
    save_dir_parts = [
        checkpoint_base_path,
        "vision",
        "ilharco_open_clip",
        "qat_dryrun" if is_dryrun else "qat",
        sanitized_model,
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
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

    # Enable quantization-aware training: wraps nn.Linear layers in QATLinear
    # and nn.MultiheadAttention layers in QATMultiheadAttention (STE fake-quant
    # forward), skipping named top-level children in cfg.qat.skip_modules
    # (typically ["classification_head"]).
    skip_modules = frozenset(cfg.qat.skip_modules)
    all_linear_names_before = sorted(
        name
        for name, m in classifier.named_modules()
        if isinstance(m, nn.Linear)
    )
    all_mha_names_before = sorted(
        name
        for name, m in classifier.named_modules()
        if isinstance(m, nn.MultiheadAttention)
    )
    enable_qat_open_clip_(
        classifier,
        bits=cfg.qat.bits,
        granularity=cfg.qat.granularity,
        skip_modules=skip_modules,
    )
    # After enable_qat_open_clip_, the surviving plain nn.Linear modules are
    # those that were skipped. QAT-wrapped linears now live at
    # "<parent>.linear" (child of QATLinear) or inside
    # QATMultiheadAttention.mha (child of the MHA wrapper).
    qat_wrapped_linears = sorted(
        name for name, m in classifier.named_modules() if isinstance(m, QATLinear)
    )
    qat_wrapped_mhas = sorted(
        name for name, m in classifier.named_modules()
        if isinstance(m, QATMultiheadAttention)
    )
    surviving_plain_linears = sorted(
        name
        for name, m in classifier.named_modules()
        if isinstance(m, nn.Linear)
        and not name.endswith(".linear")
        and ".mha." not in name
    )
    if IS_SLURM:
        log.info(
            "QAT enabled: wrapped_linear=%d, wrapped_mha=%d, skipped=%d "
            "(of %d original nn.Linear + %d original nn.MHA)",
            len(qat_wrapped_linears),
            len(qat_wrapped_mhas),
            len(surviving_plain_linears),
            len(all_linear_names_before),
            len(all_mha_names_before),
        )
        log.info("QAT skipped layers: %s", surviving_plain_linears)
    else:
        print(
            f"QAT enabled: {len(qat_wrapped_linears)} wrapped linear, "
            f"{len(qat_wrapped_mhas)} wrapped MHA, "
            f"{len(surviving_plain_linears)} skipped "
            f"(of {len(all_linear_names_before)} original nn.Linear "
            f"+ {len(all_mha_names_before)} original nn.MHA)"
        )
        pprint({"qat_skipped": surviving_plain_linears}, expand_all=True)

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

    # Optimizer / scheduler / loss — built AFTER enable_qat_ so params are
    # collected from the QAT-wrapped model (QATLinear.linear.weight is still
    # a regular nn.Parameter; STE keeps the forward differentiable).
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

        # Per-epoch validation: evaluate a PTQ-converted deepcopy so the
        # reported accuracy reflects the deployed (truly quantized) model.
        val_acc = _evaluate_ptq(
            classifier=classifier,
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
        classifier=classifier,
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

    # Save encoder checkpoint: strip QAT wrappers so the saved state_dict is
    # identical in format to finetune_fp.py output (plain nn.Linear keys) and
    # can be consumed unchanged by downstream scripts (e.g. evaluate_fp_ptq).
    disable_qat_open_clip_(classifier)
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
