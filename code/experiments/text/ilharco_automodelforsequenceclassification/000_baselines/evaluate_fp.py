import json
import logging
import os
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.duration import checkpoint_epochs, mult_path_frag
from src.text.data.common import DATASET_NAME_TO_EPOCHS
from src.text.data.registry import get_dataset
from src.vision.utils import random_tqdm_color, sanitize_hf_model_name, set_seed


OmegaConf.register_new_resolver("sanitize_hf", sanitize_hf_model_name, replace=True)


def evaluate(
    dataset,
    model: torch.nn.Module,
    tokenizer,
    max_length: int,
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

    with torch.no_grad():
        batch_bar = tqdm(
            enumerate(loader),
            total=effective_num_batches,
            desc="Evaluating (test)",
            colour=random_tqdm_color(),
            leave=False,
        )

        for i, batch in batch_bar:
            if i >= effective_num_batches:
                break

            texts, labels = batch
            encoding = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            labels = labels.to(device=device, dtype=torch.long)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

            batch_bar.set_postfix(
                batch=f"{i}/{effective_num_batches}",
                acc=f"{100.0 * correct / max(total, 1):.2f}%",
            )

    return correct / max(total, 1)


def build_model_and_tokenizer(model_name: str, num_labels: int, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device=device, dtype=torch.float32)
    return model, tokenizer


def load_split_checkpoint_(model: torch.nn.Module, backbone_path: str, head_path: str, device: torch.device):
    backbone_state = torch.load(backbone_path, map_location=device, weights_only=False)
    head_state = torch.load(head_path, map_location=device, weights_only=False)

    model.load_state_dict(backbone_state, strict=False)
    model.load_state_dict(head_state, strict=False)


@hydra.main(
    config_path="../../../../../config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines",
    config_name="evaluate_fp",
    version_base=None,
)
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    epochs = checkpoint_epochs(
        cfg.dataset_name, DATASET_NAME_TO_EPOCHS, cfg.limit_num_epochs
    )

    dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        batch_size=cfg.batch_size,
        num_workers=int(os.environ["TORCH_NUM_WORKERS"]),
        seed=cfg.seed,
    )

    model, tokenizer = build_model_and_tokenizer(
        model_name=cfg.model_name,
        num_labels=len(dataset.class_names),
        device=device,
    )

    checkpoint_base_path = os.environ["CHECKPOINT_BASE_PATH"]

    is_dryrun = cfg.limit_num_batches is not None or cfg.limit_num_epochs is not None
    checkpoint_dir_parts = [
        checkpoint_base_path,
        "text",
        "ilharco_automodelforsequenceclassification",
        "fp_dryrun" if is_dryrun else "fp",
        sanitize_hf_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}_ml={cfg.max_length}",
        mult_path_frag(cfg.epoch_mult),
        f"seed={cfg.seed}",
    ]
    if is_dryrun:
        lnb = cfg.limit_num_batches if cfg.limit_num_batches is not None else "all"
        lne = cfg.limit_num_epochs if cfg.limit_num_epochs is not None else "all"
        checkpoint_dir_parts.append(f"lnb={lnb}_lne={lne}")
    checkpoint_dir = os.path.join(*checkpoint_dir_parts)

    backbone_path = os.path.join(checkpoint_dir, f"backbone_epoch_{epochs}.pt")
    head_path = os.path.join(checkpoint_dir, f"head_epoch_{epochs}.pt")

    print(f"\nLoading backbone from: {backbone_path}")
    print(f"Loading head from: {head_path}")
    load_split_checkpoint_(
        model=model,
        backbone_path=backbone_path,
        head_path=head_path,
        device=device,
    )

    print("\n\nmodel:")
    pprint(model, expand_all=True)
    print("\n\n")
    if cfg.log_to_file:
        log.info("model:\n%s", model)

    test_accuracy = evaluate(
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        device=device,
        limit_num_batches=cfg.limit_num_batches,
    )

    print(f"\n    eval test_accuracy: {test_accuracy}\n")

    evaluation_base_path = os.environ["EVALUATION_BASE_PATH"]

    eval_dir = os.path.join(
        evaluation_base_path,
        "text",
        "ilharco_automodelforsequenceclassification",
        "000_baselines",
        "text",
        "fp",
        sanitize_hf_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}_ml={cfg.max_length}",
        mult_path_frag(cfg.epoch_mult),
        f"seed={cfg.seed}",
    )

    results = {
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "batch_size": cfg.batch_size,
        "lr": cfg.lr,
        "wd": cfg.wd,
        "ls": cfg.ls,
        "max_grad_norm": cfg.max_grad_norm,
        "max_length": cfg.max_length,
        "seed": cfg.seed,
        "limit_num_epochs": cfg.limit_num_epochs,
        "limit_num_batches": cfg.limit_num_batches,
        "epochs": epochs,
        "device": str(device),
        "test_accuracy": test_accuracy,
        "backbone_path": backbone_path,
        "head_path": head_path,
    }

    os.makedirs(eval_dir, exist_ok=True)
    eval_results_path = os.path.join(eval_dir, "eval_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {eval_results_path}")


if __name__ == "__main__":
    main()
