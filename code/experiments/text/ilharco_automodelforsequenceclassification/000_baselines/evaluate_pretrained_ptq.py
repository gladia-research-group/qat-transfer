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

from src.quantization import apply_ptq_
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


@hydra.main(
    config_path="../../../../../config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines",
    config_name="evaluate_pretrained_ptq",
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

    print(f"\nInstantiating pre-trained model: {cfg.model_name}")
    model, tokenizer = build_model_and_tokenizer(
        model_name=cfg.model_name,
        num_labels=len(dataset.class_names),
        device=device,
    )

    print("\n\nmodel (pretrained, before head swap):")
    pprint(model, expand_all=True)
    print("\n\n")
    if cfg.log_to_file:
        log.info("model (pretrained):\n%s", model)

    checkpoint_base_path = os.environ["CHECKPOINT_BASE_PATH"]

    qat_skip_modules_sorted = sorted(cfg.qat.skip_modules)
    qat_skip_tag = "-".join(qat_skip_modules_sorted) if qat_skip_modules_sorted else "none"

    head_dir = os.path.join(
        checkpoint_base_path,
        "text",
        "ilharco_automodelforsequenceclassification",
        "qat",
        sanitize_hf_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}_ml={cfg.max_length}",
        mult_path_frag(cfg.epoch_mult),
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip={qat_skip_tag}",
        f"seed={cfg.seed}",
    )
    head_path = os.path.join(head_dir, f"head_epoch_{epochs}.pt")

    print(f"\nLoading QAT head state from: {head_path}")
    head_state = torch.load(head_path, map_location=device, weights_only=False)
    model.load_state_dict(head_state, strict=False)

    print("\n\nmodel (after head swap):")
    pprint(model, expand_all=True)
    print("\n\n")
    if cfg.log_to_file:
        log.info("model (after head swap):\n%s", model)

    skip_modules = frozenset(cfg.ptq.skip_modules)

    all_linear_names = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]

    quantized_names = apply_ptq_(
        model=model,
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
        log.info("Quantized layers (%d): %s", len(quantized_names), quantized_names)
        log.info("Skipped layers (%d): %s", len(skipped_names), skipped_names)

    test_accuracy = evaluate(
        dataset=dataset,
        model=model,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        device=device,
        limit_num_batches=cfg.limit_num_batches,
    )

    print(f"\n    eval test_accuracy (PTQ): {test_accuracy}\n")

    num_classes = len(dataset.class_names)
    random_chance = 1.0 / num_classes
    print(f"    random chance baseline : {random_chance}  (1 / {num_classes} classes)\n")

    evaluation_base_path = os.environ["EVALUATION_BASE_PATH"]
    skip_modules_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) > 0 else "none"

    eval_dir = os.path.join(
        evaluation_base_path,
        "text",
        "ilharco_automodelforsequenceclassification",
        "000_baselines",
        "text",
        "pretrained_ptq",
        sanitize_hf_model_name(cfg.model_name),
        cfg.dataset_name,
        mult_path_frag(cfg.epoch_mult),
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={skip_modules_tag}",
        f"seed={cfg.seed}",
    )

    results = {
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "batch_size": cfg.batch_size,
        "max_length": cfg.max_length,
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


if __name__ == "__main__":
    main()
