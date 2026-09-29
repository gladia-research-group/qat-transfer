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
from src.vision.ilharco_hf_clip.modeling import ImageClassifier, ImageEncoder
from src.vision.ilharco_hf_clip.heads import get_classification_head
from src.vision.data.registry import get_dataset
from src.vision.data.common import maybe_dictionarize, DATASET_NAME_TO_EPOCHS
from src.vision.utils import (
    accuracy,
    random_tqdm_color,
    sanitize_hf_model_name,
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
    config_path="../../../../../config/experiments/vision/ilharco_hf_clip/000_baselines",
    config_name="evaluate_qat",
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

    qat_skip_modules_sorted = sorted(cfg.qat.skip_modules)
    qat_skip_tag = "-".join(qat_skip_modules_sorted) if qat_skip_modules_sorted else "none"

    checkpoint_dir = os.path.join(
        checkpoint_base_path,
        "vision",
        "ilharco_hf_clip",
        "qat",
        sanitize_hf_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        mult_path_frag(cfg.epoch_mult),
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={qat_skip_tag}",
        f"seed={cfg.seed}",
    )
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epochs}.pt")

    print(f"\nLoading checkpoint from: {checkpoint_path}")
    image_encoder = ImageEncoder.load(
        model_name=cfg.model_name,
        filename=checkpoint_path,
    )
    image_encoder.to(device)
    print(f"\n\nimage_encoder:")
    pprint(image_encoder, expand_all=True)
    print(f"\n\n")
    if cfg.log_to_file:
        log.info(f"image_encoder:\n{image_encoder}")

    ############################################################################
    # END checkpoint loading
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN dataset creation
    ############################################################################

    dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        preprocess_train=image_encoder.train_preprocess,
        preprocess_inference=image_encoder.val_preprocess,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ['TORCH_NUM_WORKERS']),
        seed=cfg.seed,
    )

    ############################################################################
    # END dataset creation
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN classification head creation
    ############################################################################

    head_base_path = os.environ['HEAD_BASE_PATH']

    classification_head = get_classification_head(
        model_name=cfg.model_name,
        dataset_name=cfg.dataset_name,
        save_dir=head_base_path,
        device=device,
    )

    ############################################################################
    # END classification head creation
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN image classifier creation
    ############################################################################

    image_classifier = ImageClassifier(
        image_encoder=image_encoder,
        classification_head=classification_head
    )
    image_classifier.to(device)
    print(f"\n\nimage_classifier:")
    pprint(image_classifier, expand_all=True)
    print(f"\n\n")
    if cfg.log_to_file:
        log.info(f"image_classifier:\n{image_classifier}")

    ############################################################################
    # END image classifier creation
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN evaluation
    ############################################################################

    # No PTQ is applied here: the QAT checkpoint was saved by finetune_qat.py
    # after disable_qat_, so the weights load as plain nn.Linear and we run a
    # standard FP forward pass to measure the QAT-trained model's accuracy.
    test_accuracy = evaluate(
        dataset=dataset,
        model=image_classifier,
        device=device,
        limit_num_batches=cfg.limit_num_batches,
    )

    print(f"\n    eval test_accuracy (QAT): {test_accuracy}\n")

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

    eval_dir = os.path.join(
        evaluation_base_path,
        "vision",
        "ilharco_hf_clip",
        "000_baselines",
        "vision",
        "qat",
        sanitize_hf_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        mult_path_frag(cfg.epoch_mult),
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={qat_skip_tag}",
        f"seed={cfg.seed}",
    )

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
        "checkpoint_path": checkpoint_path,
        "qat_bits": cfg.qat.bits,
        "qat_granularity": cfg.qat.granularity,
        "qat_skip_modules": list(cfg.qat.skip_modules),
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
