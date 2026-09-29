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
    DATASET_NAME_TO_EPOCHS,
)
from src.vision.utils import (
    accuracy,
    random_tqdm_color,
    sanitize_timm_model_name,
    set_seed,
)
from src.quantization import apply_ptq_

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
    config_name="evaluate_pretrained_ptq",
    version_base=None,
)
def main(cfg: DictConfig):

    set_seed(cfg.seed)

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    epochs = checkpoint_epochs(
        cfg.dataset_name, DATASET_NAME_TO_EPOCHS, cfg.limit_num_epochs
    )

    ############################################################################
    # BEGIN pre-trained model instantiation
    ############################################################################

    print(f"\nInstantiating pre-trained model: {cfg.model_name}")
    image_classifier = ImageClassifier(
        model_name=cfg.model_name,
        # num_classes=DATASET_NAME_TO_NUM_CLASSES[cfg.dataset_name],
        num_classes=1
    )
    image_classifier.to(device)
    print(f"\n\nimage_classifier (pretrained, before head swap):")
    pprint(image_classifier, expand_all=True)
    print(f"\n\n")
    if cfg.log_to_file:
        log.info(f"image_classifier (pretrained):\n{image_classifier}")

    ############################################################################
    # END pre-trained model instantiation
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
    # BEGIN classification head loading (from QAT checkpoint)
    ############################################################################

    checkpoint_base_path = os.environ['CHECKPOINT_BASE_PATH']

    qat_skip_modules_sorted = sorted(cfg.qat.skip_modules)
    qat_skip_tag = "-".join(qat_skip_modules_sorted) if qat_skip_modules_sorted else "none"

    head_dir = os.path.join(
        checkpoint_base_path,
        "vision",
        "ilharco_timm_supervised",
        "qat",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        mult_path_frag(cfg.epoch_mult),
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={qat_skip_tag}",
        f"seed={cfg.seed}",
    )
    head_path = os.path.join(head_dir, f"head_epoch_{epochs}.pt")

    if os.path.exists(head_path):
        print(f"\nLoading QAT head from: {head_path}")
        finetuned_head = torch.load(head_path, map_location=device, weights_only=False)
    else:
        classifier_path = os.path.join(head_dir, f"classifier_epoch_{epochs}.pt")
        print(f"\nhead file not found, extracting head from: {classifier_path}")
        sd = torch.load(classifier_path, map_location=device, weights_only=False)
        head_weight = sd["model.head.weight"]
        head_bias = sd["model.head.bias"]
        finetuned_head = torch.nn.Linear(head_weight.shape[1], head_weight.shape[0])
        finetuned_head.weight = torch.nn.Parameter(head_weight)
        finetuned_head.bias = torch.nn.Parameter(head_bias)
        del sd

    image_classifier.model.head = finetuned_head

    print(f"\n\nimage_classifier (after head swap):")
    pprint(image_classifier, expand_all=True)
    print(f"\n\n")
    if cfg.log_to_file:
        log.info(f"image_classifier (after head swap):\n{image_classifier}")

    ############################################################################
    # END classification head loading
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

    print(f"\nPTQ config: bits={cfg.ptq.bits}, granularity={cfg.ptq.granularity}, skip_modules={list(cfg.ptq.skip_modules)}")

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
    # END PTQ
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN evaluation
    ############################################################################

    test_accuracy = evaluate(
        dataset=dataset,
        model=image_classifier,
        device=device,
        limit_num_batches=cfg.limit_num_batches,
    )

    print(f"\n    eval test_accuracy (PTQ): {test_accuracy}\n")

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

    skip_modules_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) > 0 else "none"

    eval_dir = os.path.join(
        evaluation_base_path,
        "vision",
        "ilharco_timm_supervised",
        "000_baselines",
        "vision",
        "pretrained_ptq",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        mult_path_frag(cfg.epoch_mult),
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={skip_modules_tag}",
        f"seed={cfg.seed}",
    )

    results = {
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "batch_size": cfg.batch_size,
        "seed": cfg.seed,
        "limit_num_batches": cfg.limit_num_batches,
        "device": str(device),
        "test_accuracy": test_accuracy,
        "random_chance": random_chance,
        "num_classes": num_classes,
        "head_path": head_path,
        "qat_bits": cfg.qat.bits,
        "qat_granularity": cfg.qat.granularity,
        "qat_skip_modules": list(cfg.qat.skip_modules),
        "ptq_bits": cfg.ptq.bits,
        "ptq_granularity": cfg.ptq.granularity,
        "ptq_skip_modules": list(cfg.ptq.skip_modules),
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
