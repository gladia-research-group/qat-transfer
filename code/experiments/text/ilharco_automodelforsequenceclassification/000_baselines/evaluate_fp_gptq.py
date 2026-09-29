# ==============================================================================
# GPTQ(FP) baseline, text family
# ==============================================================================
# Reviewers objected that the paper's only quantization baseline is vanilla RTN
# per-channel PTQ (`apply_ptq_`) and asked for a comparison against strong
# recent PTQ (GPTQ, VPTQ, SliM-LLM were named; GPTQ is the one that is
# architecture-agnostic and applies unchanged to encoder-style sequence
# classifiers). This script produces the GPTQ column for the text FP
# checkpoints:
#
#     acc(gptq(FP_{S}^{D}))   vs   the recorded acc(rtn(FP_{S}^{D}))  (fp_ptq)
#
# i.e. the text `evaluate_fp_ptq.py` with the quantizer swapped from
# `apply_ptq_` to `apply_gptq_` (code/src/gptq.py, Frantar et al. ICLR 2023).
# GPTQ runs on the project's own quantize/dequantize grid, so the RTN and GPTQ
# columns differ only in error compensation — "3-bit per-channel" means
# exactly the same thing in both.
#
# Methodology notes — none is cosmetic:
#
# * Calibration reads the first `num_calib_batches` batches of the dataset's
#   OWN training split (labels never used). Text loaders yield raw
#   (texts, labels) pairs and tokenize inside the consuming loop, so a custom
#   `forward_fn` closure carries the tokenizer into `apply_gptq_`'s calibration
#   forward passes with the same padding/truncation/max_length settings the
#   evaluation loop uses — calibration and evaluation see identically
#   preprocessed inputs.
# * The eval path carries a `gptq=` fragment in place of the `ptq=` fragment,
#   with bits/gran/skip + ncal/percdamp/actorder. `block_size` is deliberately
#   excluded: it is result-invariant solver batching and would split identical
#   results across paths. It is still recorded in the results JSON.
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

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.gptq import apply_gptq_
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
    config_name="evaluate_fp_gptq",
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

    skip_modules = frozenset(cfg.gptq.skip_modules)

    all_linear_names = [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]

    # Text loaders yield raw (texts, labels); tokenization happens here, with
    # the same settings as the evaluation loop, so calibration forwards see
    # identically preprocessed inputs.
    def _calib_forward(model, batch, device):
        texts, _ = batch
        encoding = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device),
        )

    quantized_names = apply_gptq_(
        model=model,
        bits=cfg.gptq.bits,
        granularity=cfg.gptq.granularity,
        skip_modules=skip_modules,
        calib_loader=dataset.train_loader,
        device=device,
        num_calib_batches=cfg.gptq.num_calib_batches,
        percdamp=cfg.gptq.percdamp,
        actorder=cfg.gptq.actorder,
        block_size=cfg.gptq.block_size,
        forward_fn=_calib_forward,
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

    print(f"\n    eval test_accuracy (GPTQ): {test_accuracy}\n")

    num_classes = len(dataset.class_names)
    random_chance = 1.0 / num_classes
    print(f"    random chance baseline : {random_chance}  (1 / {num_classes} classes)\n")

    evaluation_base_path = os.environ["EVALUATION_BASE_PATH"]
    skip_modules_tag = "-".join(sorted(cfg.gptq.skip_modules)) if len(cfg.gptq.skip_modules) > 0 else "none"

    eval_dir = os.path.join(
        evaluation_base_path,
        "text",
        "ilharco_automodelforsequenceclassification",
        "000_baselines",
        "text",
        "fp_gptq_dryrun" if is_dryrun else "fp_gptq",
        sanitize_hf_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}_ml={cfg.max_length}",
        mult_path_frag(cfg.epoch_mult),
        f"gptq=bits={cfg.gptq.bits}_gran={cfg.gptq.granularity}_skip={skip_modules_tag}"
        f"_ncal={cfg.gptq.num_calib_batches}_percdamp={cfg.gptq.percdamp}_actorder={cfg.gptq.actorder}",
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
        "random_chance": random_chance,
        "num_classes": num_classes,
        "backbone_path": backbone_path,
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


if __name__ == "__main__":
    main()
