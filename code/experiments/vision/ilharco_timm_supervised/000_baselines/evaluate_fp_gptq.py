# ==============================================================================
# GPTQ(FP) baseline
# ==============================================================================
# Reviewers objected that the paper's only quantization baseline is vanilla RTN
# per-channel PTQ (`apply_ptq_`) and asked for a comparison against strong
# recent PTQ (GPTQ, VPTQ, SliM-LLM were named; GPTQ is the one that is
# architecture-agnostic and ports cleanly to ViTs). This script produces the
# GPTQ column of that comparison for the FP checkpoints:
#
#     acc(gptq(FP_{S}^{D}))   vs   the recorded acc(rtn(FP_{S}^{D}))  (fp_ptq)
#
# i.e. `evaluate_fp_ptq.py` with the quantizer swapped from `apply_ptq_` to
# `apply_gptq_` (code/src/gptq.py, Frantar et al. ICLR 2023). GPTQ runs on the
# project's own quantize/dequantize grid, so the RTN and GPTQ columns differ
# only in error compensation — "3-bit per-channel" means exactly the same
# thing in both.
#
# Methodology notes — none is cosmetic:
#
# * Calibration reads the first `num_calib_batches` batches of the dataset's
#   OWN training split (labels never used) via the existing `get_dataset`
#   registry. Using the task's own training data is the standard GPTQ setting
#   and the strongest version of the baseline; giving the competitor weaker
#   calibration data would be a strawman.
# * The eval path carries a `gptq=` fragment in place of the `ptq=` fragment,
#   with bits/gran/skip + ncal/percdamp/actorder. `block_size` is deliberately
#   excluded: it is result-invariant solver batching and would split identical
#   results across paths. It is still recorded in the results JSON.
# * The alpha=0 self-pair cells of 005_qat_transfer_gptq compute the same
#   quantity inside the transfer sweep (bit-identical calibration across its
#   compared cells); this script exists so the number also lives in the
#   000_baselines grammar every analysis and visualization script reads, and
#   agreement between the two trees is an end-to-end correctness check.
# ==============================================================================

import json
import logging
import os
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)

from src.duration import checkpoint_epochs, mult_path_frag
from src.vision.ilharco_timm_supervised.modeling import ImageClassifier
from src.vision.data.registry import get_dataset
from src.vision.data.common import (
    DATASET_NAME_TO_NUM_CLASSES,
    maybe_dictionarize,
    DATASET_NAME_TO_EPOCHS
)
from src.vision.utils import (
    accuracy,
    random_tqdm_color,
    sanitize_timm_model_name,
    set_seed,
)
from src.gptq import apply_gptq_

import hydra
from omegaconf import DictConfig
from rich.pretty import pprint
from tqdm import tqdm
import torch
from torch import nn


def evaluate(
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    limit_num_batches: int = None,
    split: str = "test",
):

    loader = dataset.val_loader if split == "val" else dataset.test_loader

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
            leave=False
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


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/000_baselines",
    config_name="evaluate_fp_gptq",
    version_base=None,
)
def main(cfg: DictConfig):

    set_seed(cfg.seed)

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    epochs = checkpoint_epochs(
        cfg.dataset_name, DATASET_NAME_TO_EPOCHS, cfg.limit_num_epochs
    )

    ############################################################################
    # BEGIN checkpoint loading
    ############################################################################

    checkpoint_base_path = os.environ['CHECKPOINT_BASE_PATH']

    is_dryrun = (
        cfg.limit_num_batches is not None or cfg.limit_num_epochs is not None
    )
    checkpoint_dir_parts = [
        checkpoint_base_path,
        "vision",
        "ilharco_timm_supervised",
        "fp_dryrun" if is_dryrun else "fp",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        mult_path_frag(cfg.epoch_mult),
        f"seed={cfg.seed}"
    ]
    if is_dryrun:
        lnb = cfg.limit_num_batches if cfg.limit_num_batches is not None else "all"
        lne = cfg.limit_num_epochs if cfg.limit_num_epochs is not None else "all"
        checkpoint_dir_parts.append(f"lnb={lnb}_lne={lne}")
    checkpoint_dir = os.path.join(*checkpoint_dir_parts)
    classifier_path = os.path.join(checkpoint_dir, f"classifier_epoch_{epochs}.pt")
    head_path = os.path.join(checkpoint_dir, f"head_epoch_{epochs}.pt")

    print(f"\nLoading encoder from: {classifier_path}")
    image_classifier = ImageClassifier.load(
        model_name=cfg.model_name,
        num_classes=DATASET_NAME_TO_NUM_CLASSES[cfg.dataset_name],
        filename=classifier_path,
    )
    image_classifier.to(device)
    print(f"\n\nimage_classifier:")
    pprint(image_classifier, expand_all=True)
    print(f"\n\n")
    if cfg.log_to_file:
        log.info(f"image_classifier:\n{image_classifier}")

    ############################################################################
    # END checkpoint loading
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN dataset creation
    ############################################################################

    dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        preprocess_train=image_classifier.train_preprocess,
        preprocess_inference=image_classifier.val_preprocess,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ['TORCH_NUM_WORKERS']),
        seed=cfg.seed,
    )

    ############################################################################
    # END dataset creation
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN GPTQ
    ############################################################################

    skip_modules = frozenset(cfg.gptq.skip_modules)

    all_linear_names = [
        name for name, module in image_classifier.named_modules()
        if isinstance(module, nn.Linear)
    ]

    quantized_names = apply_gptq_(
        model=image_classifier,
        bits=cfg.gptq.bits,
        granularity=cfg.gptq.granularity,
        skip_modules=skip_modules,
        calib_loader=dataset.train_loader,
        device=device,
        num_calib_batches=cfg.gptq.num_calib_batches,
        percdamp=cfg.gptq.percdamp,
        actorder=cfg.gptq.actorder,
        block_size=cfg.gptq.block_size,
    )

    skipped_names = sorted(set(all_linear_names) - set(quantized_names))

    print(
        f"\nGPTQ config: bits={cfg.gptq.bits}, granularity={cfg.gptq.granularity}, "
        f"skip_modules={list(cfg.gptq.skip_modules)}, ncal={cfg.gptq.num_calib_batches}, "
        f"percdamp={cfg.gptq.percdamp}, actorder={cfg.gptq.actorder}, "
        f"block_size={cfg.gptq.block_size}"
    )

    print(f"\nQuantized layers ({len(quantized_names)}):")
    for name in quantized_names:
        print(f"  - {name}")

    print(f"\nSkipped layers ({len(skipped_names)}):")
    for name in skipped_names:
        print(f"  - {name}")
    print()

    if cfg.log_to_file:
        log.info(f"Quantized layers ({len(quantized_names)}): {quantized_names}")
        log.info(f"Skipped layers ({len(skipped_names)}): {skipped_names}")

    ############################################################################
    # END GPTQ
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN evaluation
    ############################################################################

    accuracy_value = evaluate(
        dataset=dataset,
        model=image_classifier,
        device=device,
        limit_num_batches=cfg.limit_num_batches,
        split=cfg.split,
    )

    print(f"\n    eval {cfg.split}_accuracy (GPTQ): {accuracy_value}\n")

    num_classes = len(dataset.class_names)
    random_chance = 1.0 / num_classes
    print(f"    random chance baseline : {random_chance}  (1 / {num_classes} classes)\n")

    ############################################################################
    # END evaluation
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN save results
    ############################################################################

    evaluation_base_path = os.environ['EVALUATION_BASE_PATH']

    skip_modules_tag = "-".join(sorted(cfg.gptq.skip_modules)) if len(cfg.gptq.skip_modules) > 0 else "none"

    eval_dir_parts = [
        evaluation_base_path,
        "vision",
        "ilharco_timm_supervised",
        "000_baselines",
        "vision",
        "fp_gptq_dryrun" if is_dryrun else "fp_gptq",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        mult_path_frag(cfg.epoch_mult),
        f"gptq=bits={cfg.gptq.bits}_gran={cfg.gptq.granularity}_skip={skip_modules_tag}"
        f"_ncal={cfg.gptq.num_calib_batches}_percdamp={cfg.gptq.percdamp}_actorder={cfg.gptq.actorder}",
        f"seed={cfg.seed}",
    ]
    # Test-split baselines predate any notion of a split in this path and are
    # read by 001, 002, 003 and every visualization, so they stay exactly where
    # they are; other splits get their own leaf rather than shadowing them.
    if cfg.split != "test":
        eval_dir_parts.append(f"split={cfg.split}")
    eval_dir = os.path.join(*eval_dir_parts)

    results = {
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "wd": cfg.wd,
        "ls": cfg.ls,
        "wl": cfg.wl,
        "max_grad_norm": cfg.max_grad_norm,
        "seed": cfg.seed,
        "limit_num_epochs": cfg.limit_num_epochs,
        "limit_num_batches": cfg.limit_num_batches,
        "epochs": epochs,
        "device": str(device),
        "eval_split": cfg.split,
        f"{cfg.split}_accuracy": accuracy_value,
        "random_chance": random_chance,
        "num_classes": num_classes,
        "encoder_path": classifier_path,
        "head_path": head_path,
        "gptq_bits": cfg.gptq.bits,
        "gptq_granularity": cfg.gptq.granularity,
        "gptq_skip_modules": list(cfg.gptq.skip_modules),
        "gptq_num_calib_batches": cfg.gptq.num_calib_batches,
        "gptq_percdamp": cfg.gptq.percdamp,
        "gptq_actorder": cfg.gptq.actorder,
        "gptq_block_size": cfg.gptq.block_size,
        "quantized_layers": quantized_names,
        "skipped_layers": skipped_names,
    }

    os.makedirs(eval_dir, exist_ok=True)
    eval_results_path = os.path.join(eval_dir, "eval_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {eval_results_path}")

    ############################################################################
    # END save results
    ############################################################################


if __name__ == "__main__":
    main()
