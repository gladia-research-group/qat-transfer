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
    config_name="evaluate_pretrained",
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
        num_classes=1 # this will be replaced with the actual head
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
    # BEGIN classification head loading (from finetuned checkpoint)
    ############################################################################

    checkpoint_base_path = os.environ['CHECKPOINT_BASE_PATH']

    head_dir = os.path.join(
        checkpoint_base_path,
        "vision",
        "ilharco_timm_supervised",
        "fp",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        mult_path_frag(cfg.epoch_mult),
        f"seed={cfg.seed}",
    )
    head_path = os.path.join(head_dir, f"head_epoch_{epochs}.pt")

    print(f"\nLoading finetuned head from: {head_path}")
    finetuned_head = torch.load(head_path, map_location=device, weights_only=False)
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
    # BEGIN evaluation
    ############################################################################

    test_accuracy = evaluate(
        dataset=dataset,
        model=image_classifier,
        device=device,
        limit_num_batches=cfg.limit_num_batches,
    )

    print(f"\n    eval test_accuracy: {test_accuracy}\n")

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
        "ilharco_timm_supervised",
        "000_baselines",
        "vision",
        "pretrained",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        mult_path_frag(cfg.epoch_mult),
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
        "head_path": head_path,
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
