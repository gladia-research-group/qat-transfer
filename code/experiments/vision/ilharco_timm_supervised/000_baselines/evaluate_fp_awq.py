# ==============================================================================
# AWQ(FP) baseline
# ==============================================================================
# Reviewers objected that the paper's only quantization baseline is vanilla RTN
# per-channel PTQ (`apply_ptq_`) and named GPTQ, AWQ and SmoothQuant as the
# strong PTQ a current submission should be measured against. `evaluate_fp_gptq.py`
# supplies the GPTQ column; this script supplies the AWQ one:
#
#     acc(awq(FP_{S}^{D}))   vs   the recorded acc(rtn(FP_{S}^{D}))  (fp_ptq)
#                            vs   the recorded acc(gptq(FP_{S}^{D})) (fp_gptq)
#
# i.e. `evaluate_fp_ptq.py` with the quantizer swapped from `apply_ptq_` to
# `apply_awq_` (code/src/awq.py, Lin et al. MLSys 2024). Two competitor columns
# rather than one matters: GPTQ and AWQ attack the problem from opposite ends —
# GPTQ compensates rounding error after the fact with second-order information,
# AWQ rescales input channels by activation salience so that less error is made
# in the first place — so agreement between them is evidence about the regime
# rather than about one method's idiosyncrasy.
#
# Methodology notes — none is cosmetic:
#
# * AWQ runs on the project's own quantize/dequantize grid (symmetric, true
#   zero, per-tensor or per-output-row), not official AWQ's asymmetric group-128
#   grid. "3-bit per-channel" therefore means exactly the same thing in the RTN,
#   GPTQ and AWQ columns, and the columns differ only in method. See the
#   deviations section of code/src/awq.py.
# * Calibration reads the first `num_calib_batches` batches of the dataset's OWN
#   training split (labels never used) via the existing `get_dataset` registry —
#   the same choice `evaluate_fp_gptq.py` makes, for the same reason: it is the
#   standard setting and the strongest version of the baseline. Giving the
#   competitor weaker calibration data would be a strawman.
# * NO `apply_ptq_` runs after AWQ. AWQ *is* the quantizer. Its output is
#   Q(W diag(s)) diag(1/s), whose per-row absmax differs from the grid it was
#   quantized on, so a following RTN pass would re-round onto a fresh grid and
#   destroy both the activation-aware scaling and the searched clipping. The
#   result is nonetheless a genuine 3-bit weight-only model — see the
#   "Why no scale folding" section of code/src/awq.py.
# * The eval path carries an `awq=` fragment in place of the `ptq=` fragment,
#   built by `awq_path_frag` in code/src/awq.py rather than spelled here. The
#   `gptq=` fragment is hand-spelled at eight sites and has stayed consistent
#   only by luck; the `pv=` fragment already uses the helper pattern.
# * The alpha=0 self-pair cells of 009_qat_transfer_awq compute the same
#   quantity inside the transfer sweep (bit-identical calibration across its
#   compared cells); this script exists so the number also lives in the
#   000_baselines grammar every analysis and visualization script reads, and
#   agreement between the two trees is an end-to-end correctness check — the
#   same cross-check that validated GPTQ.
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
from src.awq import apply_awq_, awq_path_frag
from src.awq import (
    _CLIP_GRID_STEPS,
    _CLIP_MIN_SHRINK,
    _CLIP_SKIP_PATTERNS,
    _N_CLIP_TOKEN,
    _N_SCALE_TOKEN,
)

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
    config_name="evaluate_fp_awq",
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
    # BEGIN AWQ
    ############################################################################

    skip_modules = frozenset(cfg.awq.skip_modules)

    all_linear_names = [
        name for name, module in image_classifier.named_modules()
        if isinstance(module, nn.Linear)
    ]

    quantized_names = apply_awq_(
        model=image_classifier,
        bits=cfg.awq.bits,
        granularity=cfg.awq.granularity,
        skip_modules=skip_modules,
        calib_loader=dataset.train_loader,
        device=device,
        num_calib_batches=cfg.awq.num_calib_batches,
        n_grid=cfg.awq.n_grid,
        clip=cfg.awq.clip,
    )

    skipped_names = sorted(set(all_linear_names) - set(quantized_names))

    print(
        f"\nAWQ config: bits={cfg.awq.bits}, granularity={cfg.awq.granularity}, "
        f"skip_modules={list(cfg.awq.skip_modules)}, ncal={cfg.awq.num_calib_batches}, "
        f"n_grid={cfg.awq.n_grid}, clip={cfg.awq.clip}"
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
    # END AWQ
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

    print(f"\n    eval {cfg.split}_accuracy (AWQ): {accuracy_value}\n")

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

    eval_dir_parts = [
        evaluation_base_path,
        "vision",
        "ilharco_timm_supervised",
        "000_baselines",
        "vision",
        "fp_awq_dryrun" if is_dryrun else "fp_awq",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        mult_path_frag(cfg.epoch_mult),
        awq_path_frag(
            bits=cfg.awq.bits,
            granularity=cfg.awq.granularity,
            skip_modules=cfg.awq.skip_modules,
            num_calib_batches=cfg.awq.num_calib_batches,
            n_grid=cfg.awq.n_grid,
            clip=cfg.awq.clip,
        ),
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
        "awq_bits": cfg.awq.bits,
        "awq_granularity": cfg.awq.granularity,
        "awq_skip_modules": list(cfg.awq.skip_modules),
        "awq_num_calib_batches": cfg.awq.num_calib_batches,
        "awq_n_grid": cfg.awq.n_grid,
        "awq_clip": cfg.awq.clip,
        # Fixed in code/src/awq.py (official values), so they never enter the
        # path fragment; recorded here so a run is fully reconstructible.
        "awq_clip_grid_steps": _CLIP_GRID_STEPS,
        "awq_clip_min_shrink": _CLIP_MIN_SHRINK,
        "awq_clip_skip_patterns": list(_CLIP_SKIP_PATTERNS),
        "awq_n_scale_token": _N_SCALE_TOKEN,
        "awq_n_clip_token": _N_CLIP_TOKEN,
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
