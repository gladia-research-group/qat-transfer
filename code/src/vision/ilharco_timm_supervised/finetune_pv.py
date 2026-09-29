import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

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

from src.duration import clamped_warmup, mult_path_frag, run_meta, training_budget
from src.pv_tuning import (
    PVLinear,
    disable_pv_,
    enable_pv_,
    pv_path_frag,
    pv_sidecar_state,
    pv_step_,
    settle_pv_,
)
from src.quantization import apply_ptq_
from src.vision.data.common import (
    DATASET_NAME_TO_EPOCHS,
    DATASET_NAME_TO_NUM_CLASSES,
    maybe_dictionarize,
)
from src.vision.data.registry import get_dataset
from src.vision.ilharco_timm_supervised.modeling import ImageClassifier
from src.vision.utils import (
    LabelSmoothing,
    cosine_lr,
    random_tqdm_color,
    sanitize_timm_model_name,
    set_seed,
)

OmegaConf.register_new_resolver(
    "sanitize_timm", sanitize_timm_model_name, replace=True
)


def _settle_and_unwrap_(model: nn.Module) -> None:
    """Collapse every PVLinear onto the grid, then strip the wrappers.

    Used by both the evaluation path and the saving path, because they need the
    same object: a plain nn.Linear model whose weights are the grid points PV
    actually converged to, not the straight-through buffer that merely rounds
    to them.
    """
    settle_pv_(model)
    disable_pv_(model)


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
    """Deepcopy the classifier, settle + strip PV wrappers, apply PTQ, evaluate.

    Structurally identical to finetune_qat.py's evaluation, and deliberately
    so: the reported metric must be what a truly post-training-quantized model
    would achieve, measured the same way for both finetuners.

    One thing differs, and it is the point of PV's checkpoint contract: after
    _settle_and_unwrap_ the weights are already exactly on the grid, so
    apply_ptq_ here is a no-op. It is still called, both to keep this path
    identical to the QAT one and because the no-op-ness is an invariant worth
    exercising every epoch rather than assuming.

    Deepcopy safety: ImageClassifier is a plain nn.Module subclass wrapping a
    timm model. No forward/backward hooks or non-picklable attrs are
    registered. PVLinear stores a child nn.Linear, two tensor buffers, and
    Python scalars. All components are therefore deepcopy-safe.
    """
    classifier.eval()
    # Move the live model to CPU briefly so the deepcopy is made on CPU and
    # peak VRAM never holds two full copies of the backbone simultaneously.
    eval_model = copy.deepcopy(classifier).to("cpu")
    _settle_and_unwrap_(eval_model)  # PVLinear -> nn.Linear, weights on-grid
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


def _save_checkpoints(classifier: nn.Module, save_dir: str, tag) -> None:
    """Write the PV checkpoint, its head, and the PV sidecar state.

    The sidecar is captured *before* settling: the codes and scale are
    recoverable from the settled checkpoint by re-quantizing it, but the
    pre-settle straight-through buffer is not, and it is what a latent-QV
    ablation (`QV = B - FP`, the exact analogue of what QAT checkpoints store)
    would need. Saving it here makes that ablation possible without re-running
    any training.

    Wrappers are stripped so the saved state_dict is identical in format to
    finetune_fp.py output (plain nn.Linear keys) and can be consumed unchanged
    by downstream scripts -- this is what makes `QV = PV - FP` a well-defined
    state-dict subtraction.
    """
    sidecar = pv_sidecar_state(classifier)
    _settle_and_unwrap_(classifier)

    classifier.save(os.path.join(save_dir, f"classifier_epoch_{tag}.pt"))
    torch.save(classifier.model.head, os.path.join(save_dir, f"head_epoch_{tag}.pt"))
    torch.save(sidecar, os.path.join(save_dir, f"pv_state_epoch_{tag}.pt"))


@hydra.main(
    config_path="../../../../config/src/vision/ilharco_timm_supervised",
    config_name="finetune_pv",
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

    is_dryrun = (
        cfg.limit_num_batches is not None or cfg.limit_num_epochs is not None
    )
    # The pv= fragment occupies the slot QAT's paths give to qat=. Built by the
    # library helper so this script, the baseline evaluators, the transfer
    # phase, and the plotting scripts cannot drift apart.
    pv_frag = pv_path_frag(
        bits=cfg.pv.bits,
        granularity=cfg.pv.granularity,
        skip_modules=cfg.pv.skip_modules,
        delta_decay=cfg.pv.delta_decay,
        max_code_change_per_step=cfg.pv.max_code_change_per_step,
        trust_ratio=cfg.pv.trust_ratio,
        p_every=cfg.pv.p_every,
        temperature=cfg.pv.temperature,
    )
    save_dir_parts = [
        checkpoint_base_path,
        "vision",
        "ilharco_timm_supervised",
        "pv_dryrun" if is_dryrun else "pv",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        mult_path_frag(cfg.epoch_mult),
        pv_frag,
        f"seed={cfg.seed}",
    ]
    if is_dryrun:
        lnb = cfg.limit_num_batches if cfg.limit_num_batches is not None else "all"
        lne = cfg.limit_num_epochs if cfg.limit_num_epochs is not None else "all"
        save_dir_parts.append(f"lnb={lnb}_lne={lne}")
    save_dir = os.path.join(*save_dir_parts)
    os.makedirs(save_dir, exist_ok=True)

    classifier = ImageClassifier(
        model_name=cfg.model_name,
        num_classes=DATASET_NAME_TO_NUM_CLASSES[cfg.dataset_name],
    )

    # Dataset (seeded with the run seed — not SPLIT_SEED)
    dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        preprocess_train=classifier.train_preprocess,
        preprocess_inference=classifier.val_preprocess,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        seed=cfg.seed,
    )

    classifier.to(device=device)
    if cfg.model_name in [
        "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        "deit3_large_patch16_224.fb_in1k"
    ]:
        classifier.model.set_grad_checkpointing(enable=True)

    # Enable PV-Tuning: wraps nn.Linear layers in PVLinear (STE fake-quant
    # forward through explicitly stored codes), skipping the named top-level
    # children in cfg.pv.skip_modules (typically ["head"]).
    skip_modules = frozenset(cfg.pv.skip_modules)
    all_linear_names_before = sorted(
        name
        for name, m in classifier.named_modules()
        if isinstance(m, nn.Linear)
    )
    enable_pv_(
        classifier,
        bits=cfg.pv.bits,
        granularity=cfg.pv.granularity,
        skip_modules=skip_modules,
    )
    # After enable_pv_, the surviving plain nn.Linear modules are exactly
    # those that were skipped. PV-wrapped linears now live at
    # "<parent>.linear" (child of PVLinear).
    pv_wrapped_parents = sorted(
        name for name, m in classifier.named_modules() if isinstance(m, PVLinear)
    )
    surviving_plain_linears = sorted(
        name
        for name, m in classifier.named_modules()
        if isinstance(m, nn.Linear) and not name.endswith(".linear")
    )
    if IS_SLURM:
        log.info(
            "PV enabled: wrapped=%d, skipped=%d (of %d original nn.Linear)",
            len(pv_wrapped_parents),
            len(surviving_plain_linears),
            len(all_linear_names_before),
        )
        log.info("PV skipped layers: %s", surviving_plain_linears)
    else:
        print(
            f"PV enabled: {len(pv_wrapped_parents)} wrapped, "
            f"{len(surviving_plain_linears)} skipped "
            f"(of {len(all_linear_names_before)} original nn.Linear)"
        )
        pprint({"pv_skipped": surviving_plain_linears}, expand_all=True)

    if IS_SLURM:
        log.info("state_dict keys: %s", list(classifier.state_dict().keys()))
        log.info("classifier: %s", classifier)
    else:
        pprint(list(classifier.state_dict().keys()), expand_all=True)
        pprint(classifier, expand_all=True)

    num_batches = len(dataset.train_loader)
    assert REFERENCE_BATCH_SIZE % cfg.batch_size == 0, (
        f"batch_size={cfg.batch_size} must evenly divide {REFERENCE_BATCH_SIZE}"
    )
    accum_steps = REFERENCE_BATCH_SIZE // cfg.batch_size

    # Training budget. `epoch_mult` scales this dataset's schedule; mult=1.0 is
    # pinned to reproduce the pre-multiplier behaviour exactly, so the loop still
    # runs every epoch to completion and the max_steps break never fires.
    budget = training_budget(
        cfg.dataset_name, cfg.epoch_mult, num_batches, accum_steps,
        DATASET_NAME_TO_EPOCHS, cfg.limit_num_epochs,
    )
    epochs = budget.loop_epochs
    max_steps = budget.max_steps
    ckpt_epochs = budget.ckpt_epochs

    # Optimizer / scheduler / loss — built AFTER enable_pv_ so params are
    # collected from the PV-wrapped model. PVLinear.linear.weight is the
    # straight-through buffer and is still a regular nn.Parameter; the codes
    # and scale are buffers, so AdamW never sees them. That split is the V/P
    # split: AdamW performs the V-step, pv_step_ performs the P-step.
    params = [p for p in classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = cosine_lr(optimizer, cfg.lr, clamped_warmup(cfg.wl, max_steps), max_steps)
    loss_fn = (
        LabelSmoothing(cfg.ls) if cfg.ls > 0 else torch.nn.CrossEntropyLoss()
    )

    # Training loop
    num_opt_steps = 0
    pv_stats = {}
    epoch_bar = tqdm(
        range(epochs), desc="epochs", colour=random_tqdm_color(), **TQDM_KW
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
            if (
                cfg.limit_num_batches is not None
                and i >= cfg.limit_num_batches
            ):
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
                opt_step = (i + 1) // accum_steps + epoch * (num_batches // accum_steps) - 1
                scheduler(opt_step)

                # The P-step, after the V-step has landed. At delta_decay=0,
                # max_code_change_per_step=1 and p_every=1 this makes the model
                # bitwise equivalent to finetune_qat.py's STE — that corner is
                # the control condition, not a degenerate configuration.
                num_opt_steps += 1
                if num_opt_steps % cfg.pv.p_every == 0:
                    pv_stats = pv_step_(
                        classifier,
                        delta_decay=cfg.pv.delta_decay,
                        max_code_change_per_step=cfg.pv.max_code_change_per_step,
                        trust_ratio=cfg.pv.trust_ratio,
                        temperature=cfg.pv.temperature,
                    )

                if IS_SLURM:
                    opt_step_in_epoch = (i + 1) // accum_steps
                    if opt_step_in_epoch % LOG_EVERY == 0 or i == num_batches - 1:
                        log.info(
                            "epoch %d step %d/%d loss=%.4f pv_changed=%.0f/%.0f (%.4f)",
                            epoch, opt_step_in_epoch, num_batches // accum_steps, accum_loss,
                            pv_stats.get("pv_codes_changed", 0.0),
                            pv_stats.get("pv_codes", 0.0),
                            pv_stats.get("pv_code_change_fraction", 0.0),
                        )
                else:
                    train_bar.set_postfix(
                        loss=f"{accum_loss:.4f}",
                        pv=f"{pv_stats.get('pv_code_change_fraction', 0.0):.4f}",
                    )

                optimizer.zero_grad()
                accum_loss = 0.0

                # Stop mid-epoch once the budget is spent. At mult=1.0 max_steps
                # equals the full schedule, so this never fires and the run is
                # identical to the pre-multiplier behaviour.
                if opt_step + 1 >= max_steps:
                    budget_exhausted = True
                    break

        # Per-epoch validation: evaluate a settled, PTQ-converted deepcopy so
        # the reported accuracy reflects the deployed (truly quantized) model.
        val_acc = _evaluate_ptq(
            classifier=classifier,
            loader=dataset.val_loader,
            device=device,
            bits=cfg.pv.bits,
            granularity=cfg.pv.granularity,
            skip_modules=skip_modules,
            limit_num_batches=cfg.limit_num_batches,
            desc=f"val e{epoch}",
        )
        if IS_SLURM:
            log.info("epoch %d val_acc=%.4f", epoch, val_acc)
        else:
            epoch_bar.set_postfix(val_acc=f"{val_acc:.4f}")

        # Save classifier, head and sidecar checkpoints, if limit_num_epochs.
        # Settling mutates the model, so this happens on a deepcopy — the live
        # model must keep its stale codes to continue training.
        if cfg.limit_num_epochs:
            save_copy = copy.deepcopy(classifier)
            _save_checkpoints(save_copy, save_dir, epoch + 1)
            del save_copy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final test evaluation — same settled, PTQ-converted deepcopy path.
    test_acc = _evaluate_ptq(
        classifier=classifier,
        loader=dataset.test_loader,
        device=device,
        bits=cfg.pv.bits,
        granularity=cfg.pv.granularity,
        skip_modules=skip_modules,
        limit_num_batches=cfg.limit_num_batches,
        desc="test",
    )
    if IS_SLURM:
        log.info("final test accuracy: %.4f", test_acc)
    else:
        print(f"Final test accuracy: {test_acc:.4f}")

    # The tag names the 1x reference schedule, not the realized one -- see
    # resolve_duration. The budget lives in the mult= directory component.
    _save_checkpoints(classifier, save_dir, ckpt_epochs)

    # The realized budget is not recoverable from the path, so record it.
    meta = run_meta(budget, num_batches, accum_steps, cfg.wl)
    meta["final_test_accuracy"] = test_acc
    with open(os.path.join(save_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
