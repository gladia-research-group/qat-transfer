"""PV(FP) baseline: what does a PV-tuned checkpoint score with no PTQ applied?

The PV analogue of `evaluate_qat.py`. It measures the saved checkpoint as it
loads, with no evaluation-time quantization.

For QAT that distinction is substantial -- `evaluate_qat.py` measures the
latent full-precision weights, `evaluate_qat_ptq.py` measures them after
rounding, and the gap between the two is the price of deployment. For PV it is
not, and deliberately so: `finetune_pv.py` settles the model onto the grid
before saving, so this script and `evaluate_pv_ptq.py` are measuring the same
weights and must report the same accuracy. Keeping both is what makes that
equality an observation instead of an assumption, and it keeps the baseline
directory grammar parallel to QAT's so the plotting scripts need no special
case.
"""

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
from src.pv_tuning import pv_path_frag
from src.vision.ilharco_timm_supervised.modeling import ImageClassifier
from src.vision.data.registry import get_dataset
from src.vision.data.common import maybe_dictionarize, DATASET_NAME_TO_EPOCHS, DATASET_NAME_TO_NUM_CLASSES
from src.vision.utils import (
    accuracy,
    random_tqdm_color,
    sanitize_timm_model_name,
    set_seed,
)

import hydra
from omegaconf import DictConfig
from rich.pretty import pprint
from tqdm import tqdm
import torch


def evaluate(
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    limit_num_batches: int = None,
):

    loader = dataset.test_loader

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
            desc="Evaluating (test)",
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
    config_name="evaluate_pv",
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

    checkpoint_dir_parts = [
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
        checkpoint_dir_parts.append(f"lnb={lnb}_lne={lne}")
    checkpoint_dir = os.path.join(*checkpoint_dir_parts)
    classifier_path = os.path.join(checkpoint_dir, f"classifier_epoch_{epochs}.pt")

    print(f"\nLoading classifier from: {classifier_path}")
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
    # BEGIN evaluation
    ############################################################################

    # No PTQ is applied here: finetune_pv.py settled the model onto the grid
    # and saved it after disable_pv_, so the weights load as plain nn.Linear
    # and this is a standard FP forward pass. Unlike the QAT case, those
    # weights are already exactly representable at cfg.pv.bits, so this number
    # is expected to equal evaluate_pv_ptq.py's.
    test_accuracy = evaluate(
        dataset=dataset,
        model=image_classifier,
        device=device,
        limit_num_batches=cfg.limit_num_batches,
    )

    print(f"\n    eval test_accuracy (PV): {test_accuracy}\n")

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
        "pv_dryrun" if is_dryrun else "pv",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        mult_path_frag(cfg.epoch_mult),
        pv_frag,
        f"seed={cfg.seed}",
    ]
    if is_dryrun:
        eval_dir_parts.append(f"lnb={lnb}_lne={lne}")
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
        "test_accuracy": test_accuracy,
        "random_chance": random_chance,
        "num_classes": num_classes,
        "classifier_path": classifier_path,
        "pv_bits": cfg.pv.bits,
        "pv_granularity": cfg.pv.granularity,
        "pv_skip_modules": list(cfg.pv.skip_modules),
        "pv_delta_decay": cfg.pv.delta_decay,
        "pv_max_code_change_per_step": cfg.pv.max_code_change_per_step,
        "pv_trust_ratio": cfg.pv.trust_ratio,
        "pv_p_every": cfg.pv.p_every,
        "pv_temperature": cfg.pv.temperature,
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
