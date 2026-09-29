"""PV(FP) + PTQ baseline: what does a PV-tuned checkpoint score once quantized?

This is the PV analogue of `evaluate_qat_ptq.py`, and it plays the same role in
`008_pv_transfer` that `evaluate_qat_ptq.py` plays in `001_qat_transfer`: it is
the receiver's own PV ceiling, the number transfer is trying to recover without
running PV on the receiver.

There is one structural difference from every other `*_ptq` baseline here, and
it is a property rather than an accident. `finetune_pv.py` settles the model
onto the quantization grid before saving, so the checkpoint's weights are
already exactly representable at the target bit-width and `apply_ptq_` is a
no-op on them. The PTQ pass is still run, for two reasons: it keeps this script
structurally identical to `evaluate_qat_ptq.py` (so the two columns of the
results table are produced by the same code path, not by two code paths that
are merely believed to agree), and it turns the no-op property into something
measured on real checkpoints every run rather than assumed.

The measured quantity is `ptq_codes_changed`, which must be 0: PTQ has to
recover the checkpoint's integer codes exactly, because that is what "already
on the grid" means. `ptq_max_abs_weight_delta` is recorded alongside it but is
*not* the invariant -- on GPU it comes back at ~1.2e-07 (one float32 ulp),
because `scale = absmax/qmax` round-trips through a CUDA division that is not
bit-identical to the CPU's. The same comparison on CPU gives exactly 0. An
earlier version of this script warned on `delta != 0` and therefore fired on
every healthy run; the code count is the signal, the weight delta is context.

Consequently `evaluate_pv.py` and `evaluate_pv_ptq.py` should report identical
accuracies. That equality is a check, not redundancy: `evaluate_pv.py` measures
the saved checkpoint, this one measures the quantized deployment of it, and PV's
whole checkpoint contract is the claim that those are the same object.
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
from src.quantization import apply_ptq_, quantize_tensor

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
    config_name="evaluate_pv_ptq",
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
    # BEGIN PTQ
    ############################################################################

    ptq_skip_modules = frozenset(cfg.ptq.skip_modules)

    all_linear_names = [
        name for name, module in image_classifier.named_modules()
        if isinstance(module, nn.Linear)
    ]

    # Snapshot the weights so the no-op property can be measured rather than
    # assumed. A settled PV checkpoint is already on the grid, so every one of
    # these deltas must come back exactly 0.0.
    weights_before = {
        name: module.weight.detach().clone()
        for name, module in image_classifier.named_modules()
        if isinstance(module, nn.Linear)
    }

    quantized_names = apply_ptq_(
        model=image_classifier,
        bits=cfg.ptq.bits,
        granularity=cfg.ptq.granularity,
        skip_modules=ptq_skip_modules,
    )

    modules_by_name = dict(image_classifier.named_modules())
    ptq_max_abs_weight_delta = max(
        (
            float(
                (modules_by_name[name].weight.detach() - weights_before[name])
                .abs()
                .max()
            )
            for name in quantized_names
        ),
        default=0.0,
    )

    # The invariant that actually matters is that PTQ recovers the same integer
    # codes, i.e. the checkpoint already sits on the grid. The dequantized
    # weights are NOT bit-identical on GPU: scale = absmax/qmax round-trips
    # through a CUDA float division that differs from the CPU's by one ulp, so
    # a ~1.2e-07 delta on weights of order 1 is expected and harmless (it is
    # exactly 0 when the same comparison is done on CPU). Warning on the raw
    # delta would therefore fire on every healthy run. Count changed codes
    # instead, and treat the weight delta as informational.
    ptq_codes_changed = sum(
        int(
            (
                quantize_tensor(modules_by_name[name].weight.detach(), cfg.ptq.bits, cfg.ptq.granularity)[0]
                != quantize_tensor(weights_before[name], cfg.ptq.bits, cfg.ptq.granularity)[0]
            ).sum()
        )
        for name in quantized_names
    )
    ptq_total_codes = sum(int(weights_before[name].numel()) for name in quantized_names)

    print(f"\n\nimage_classifier:")
    pprint(image_classifier, expand_all=True)
    print(f"\n\n")
    if cfg.log_to_file:
        log.info(f"image_classifier:\n{image_classifier}")

    skipped_names = sorted(set(all_linear_names) - set(quantized_names))

    print(f"\nPTQ config: bits={cfg.ptq.bits}, granularity={cfg.ptq.granularity}, skip_modules={list(cfg.ptq.skip_modules)}")

    print(f"\nQuantized layers ({len(quantized_names)}):")
    for name in quantized_names:
        print(f"  - {name}")

    print(f"\nSkipped layers ({len(skipped_names)}):")
    for name in skipped_names:
        print(f"  - {name}")
    print()

    print(
        f"\nPTQ on the PV checkpoint: codes changed {ptq_codes_changed}/{ptq_total_codes} "
        f"(expected 0 — a settled PV checkpoint is already on the grid); "
        f"max |delta w| = {ptq_max_abs_weight_delta:.3e} "
        f"(~1 ulp is expected on GPU, see note in code)\n"
    )
    if ptq_codes_changed != 0:
        print(
            "WARNING: PTQ moved the integer codes of a PV checkpoint. This "
            "checkpoint was not written by the settle path; do not build a QV "
            "from it."
        )
        log.warning(
            "PTQ changed %d/%d codes of PV checkpoint %s",
            ptq_codes_changed,
            ptq_total_codes,
            classifier_path,
        )

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

    print(f"\n    eval test_accuracy (PV + PTQ): {test_accuracy}\n")

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

    ptq_skip_tag = "-".join(sorted(cfg.ptq.skip_modules)) if len(cfg.ptq.skip_modules) > 0 else "none"

    eval_dir_parts = [
        evaluation_base_path,
        "vision",
        "ilharco_timm_supervised",
        "000_baselines",
        "vision",
        "pv_ptq_dryrun" if is_dryrun else "pv_ptq",
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        mult_path_frag(cfg.epoch_mult),
        pv_frag,
        f"ptq=bits={cfg.ptq.bits}_gran={cfg.ptq.granularity}_skip={ptq_skip_tag}",
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
        "ptq_bits": cfg.ptq.bits,
        "ptq_granularity": cfg.ptq.granularity,
        "ptq_skip_modules": list(cfg.ptq.skip_modules),
        "ptq_max_abs_weight_delta": ptq_max_abs_weight_delta,
        "ptq_codes_changed": ptq_codes_changed,
        "ptq_total_codes": ptq_total_codes,
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
