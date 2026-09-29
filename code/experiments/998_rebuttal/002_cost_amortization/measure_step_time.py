"""998 — Wall-clock cost of a single QAT training step.

The cost-amortization figure is expressed in training samples seen, which keeps
it hardware-independent.  To annotate it with a concrete "= N GPU-hours on
<GPU>" statement we need one measured constant: how long one QAT training step
actually takes on real hardware.  This script measures it.

Methodology (each choice matters, none is cosmetic):

  * The model, dataset, QAT wrapping, optimizer, scheduler and loss are built
    exactly as in code/src/vision/ilharco_timm_supervised/finetune_qat.py, so
    the fake-quant (QATLinear / STE) overhead is inside the measurement.
  * The first `warmup_batches` batches are discarded: CUDA context creation,
    lazy cuDNN/cuBLAS kernel selection and allocator growth make them wildly
    unrepresentative.
  * torch.cuda.synchronize() is called before and after every timed region;
    without it we would be timing asynchronous kernel *launches*, not kernels.
  * `timed_batches` batches are then timed individually and summarized by the
    MEDIAN (plus p25/p75, min/max, mean and stdev) — a mean alone is hostage to
    a single dataloader stall or a background process on the GPU.
  * Two clocks are recorded per batch:
      - "step": host-to-device copy + forward + backward + (on gradient
        accumulation boundaries) clip/optimizer/scheduler/zero_grad.  This is
        the hardware-bound number and it is what reproduces across runs.
      - "wall": iteration-to-iteration time, i.e. step time *plus* the time
        spent blocked on the dataloader.  This is what a real training job
        actually burns on a reserved GPU, so it is the honest basis for a
        GPU-hours claim, at the price of depending on num_workers and on
        whether the dataset is warm in the page cache.
    The headline `seconds_per_sample` is the wall-clock one (never understating
    the cost); `seconds_per_sample_step` is reported alongside it.

With the default batch_size == REFERENCE_BATCH_SIZE (128) the gradient
accumulation factor is 1, so every timed batch is a complete optimizer step and
the per-batch distribution is unimodal.  Smaller batch sizes still work, but
then only every accum_steps-th batch carries the optimizer update and the
distribution becomes bimodal by construction; the per-sample numbers remain
correct because they divide by the samples actually processed.

Short-running by construction: only warmup_batches + timed_batches batches are
ever touched, never a full epoch, and no checkpoint is written.

Writes step_time.json under
evaluations/998_rebuttal/002_cost_amortization/step_time/<gpu>/<model>/...,
so measurements on different GPUs or backbones never overwrite each other.
Feed the resulting file to compute_costs.py via --step-time-json.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# MUST be the first thing that runs: HF libs (transformers, huggingface_hub,
# datasets) and our own src.vision.data.common all snapshot env vars at import
# time. Loading .env after those imports has no effect.
from dotenv import load_dotenv
load_dotenv()

import json
import logging
import os
import platform
import re
import statistics
import time

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
from tqdm import tqdm

log = logging.getLogger(__name__)

IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)
LOG_EVERY = 50
REFERENCE_BATCH_SIZE = 128

from src.quantization import QATLinear, enable_qat_
from src.duration import clamped_warmup, training_budget
from src.vision.data.common import (
    DATASET_NAME_TO_EPOCHS,
    DATASET_NAME_TO_NUM_CLASSES,
    maybe_dictionarize,
)
from src.vision.data.registry import get_dataset
from src.vision.ilharco_timm_supervised.modeling import ImageClassifier
from src.vision.utils import (
    LabelSmoothing,
    cosine_lr,
    random_tqdm_color,
    sanitize_timm_model_name,
    set_seed,
)

# Repo-relative output root, resolved before Hydra changes the working dir.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[4])
EVAL_ROOT = os.path.join(
    _PROJECT_ROOT, "evaluations", "998_rebuttal", "002_cost_amortization", "step_time"
)

OmegaConf.register_new_resolver(
    "sanitize_timm", sanitize_timm_model_name, replace=True
)


def _sanitize_device_name(device_name: str) -> str:
    """Turn 'NVIDIA GeForce RTX 4090' into 'NVIDIA_GeForce_RTX_4090'."""
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", device_name)).strip("_")


def _summarize(samples: list[float]) -> dict:
    """Median-centred summary of a list of per-batch durations (seconds)."""
    ordered = sorted(samples)
    if len(ordered) > 1:
        quantiles = statistics.quantiles(ordered, n=4, method="inclusive")
        p25, p75 = quantiles[0], quantiles[2]
    else:
        p25 = p75 = ordered[0]
    return {
        "n": len(ordered),
        "median": statistics.median(ordered),
        "p25": p25,
        "p75": p75,
        "mean": statistics.fmean(ordered),
        "stdev": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "min": ordered[0],
        "max": ordered[-1],
        "total": sum(ordered),
    }


@hydra.main(
    config_path="../../../../config/experiments/998_rebuttal/002_cost_amortization",
    config_name="measure_step_time",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(dict(cfg), expand_all=True)
    set_seed(cfg.seed)

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")
    num_workers = int(os.environ["TORCH_NUM_WORKERS"])

    assert cfg.warmup_batches >= 1, "at least one warmup batch is required"
    assert cfg.timed_batches >= 2, "need >= 2 timed batches for a dispersion estimate"

    ############################################################################
    # BEGIN model construction (mirrors finetune_qat.py)
    ############################################################################

    classifier = ImageClassifier(
        model_name=cfg.model_name,
        num_classes=DATASET_NAME_TO_NUM_CLASSES[cfg.dataset_name],
    )

    dataset = get_dataset(
        dataset_name=cfg.dataset_name,
        preprocess_train=classifier.train_preprocess,
        preprocess_inference=classifier.val_preprocess,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        seed=cfg.seed,
    )

    classifier.to(device=device)
    if cfg.model_name in [
        "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        "deit3_large_patch16_224.fb_in1k"
    ]:
        classifier.model.set_grad_checkpointing(enable=True)

    skip_modules = frozenset(cfg.qat.skip_modules)
    all_linear_names_before = sorted(
        name for name, m in classifier.named_modules() if isinstance(m, nn.Linear)
    )
    enable_qat_(
        classifier,
        bits=cfg.qat.bits,
        granularity=cfg.qat.granularity,
        skip_modules=skip_modules,
    )
    qat_wrapped_parents = sorted(
        name for name, m in classifier.named_modules() if isinstance(m, QATLinear)
    )
    surviving_plain_linears = sorted(
        name
        for name, m in classifier.named_modules()
        if isinstance(m, nn.Linear) and not name.endswith(".linear")
    )
    if IS_SLURM:
        log.info(
            "QAT enabled: wrapped=%d, skipped=%d (of %d original nn.Linear)",
            len(qat_wrapped_parents),
            len(surviving_plain_linears),
            len(all_linear_names_before),
        )
    else:
        print(
            f"QAT enabled: {len(qat_wrapped_parents)} wrapped, "
            f"{len(surviving_plain_linears)} skipped "
            f"(of {len(all_linear_names_before)} original nn.Linear)"
        )

    ############################################################################
    # END model construction
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN optimizer / scheduler / loss (mirrors finetune_qat.py)
    ############################################################################

    num_batches = len(dataset.train_loader)
    assert REFERENCE_BATCH_SIZE % cfg.batch_size == 0, (
        f"batch_size={cfg.batch_size} must evenly divide {REFERENCE_BATCH_SIZE}"
    )
    accum_steps = REFERENCE_BATCH_SIZE // cfg.batch_size

    # This block mirrors finetune_qat.py and must keep mirroring it: what is
    # being timed is a QAT step as actually configured, so a divergence here
    # silently mistimes the amortization argument. Resolved through the same
    # helper the finetuners use, for exactly that reason.
    budget = training_budget(
        cfg.dataset_name, cfg.epoch_mult, num_batches, accum_steps,
        DATASET_NAME_TO_EPOCHS,
    )
    epochs = budget.loop_epochs
    max_steps = budget.max_steps

    params = [p for p in classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = cosine_lr(
        optimizer, cfg.lr, clamped_warmup(cfg.wl, max_steps), max_steps
    )
    loss_fn = LabelSmoothing(cfg.ls) if cfg.ls > 0 else torch.nn.CrossEntropyLoss()

    ############################################################################
    # END optimizer / scheduler / loss
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN timing loop
    ############################################################################

    total_batches = cfg.warmup_batches + cfg.timed_batches
    classifier.train()
    optimizer.zero_grad()

    step_times: list[float] = []
    wall_times: list[float] = []

    loader_iter = iter(dataset.train_loader)
    bar = tqdm(
        range(total_batches),
        desc="batches",
        colour=random_tqdm_color(),
        **TQDM_KW,
    )
    prev_wall_end = None
    for i in bar:
        try:
            batch = next(loader_iter)
        except StopIteration:
            # Fewer batches in the epoch than requested: restart the loader.
            # The restart itself is excluded from the wall clock below.
            loader_iter = iter(dataset.train_loader)
            batch = next(loader_iter)
            prev_wall_end = None

        step_start = time.perf_counter()

        batch = maybe_dictionarize(batch)
        images = batch["images"].to(device)
        labels = batch["labels"].to(device=device, dtype=torch.long)

        logits = classifier(images)
        loss: torch.Tensor = loss_fn(logits, labels) / accum_steps
        loss.backward()

        if (i + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()
            scheduler((i + 1) // accum_steps - 1)
            optimizer.zero_grad()

        # Without this the timer would stop at the last kernel *launch*.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_end = time.perf_counter()

        is_timed = i >= cfg.warmup_batches
        if is_timed:
            step_times.append(step_end - step_start)
            if prev_wall_end is not None:
                wall_times.append(step_end - prev_wall_end)
        prev_wall_end = step_end

        if IS_SLURM:
            if (i + 1) % LOG_EVERY == 0 or i == total_batches - 1:
                log.info(
                    "batch %d/%d (%s) step=%.4fs",
                    i + 1,
                    total_batches,
                    "timed" if is_timed else "warmup",
                    step_end - step_start,
                )
        else:
            bar.set_postfix(
                phase="timed" if is_timed else "warmup",
                step=f"{step_end - step_start:.4f}s",
            )

    assert len(step_times) == cfg.timed_batches, (
        f"collected {len(step_times)} timed batches, expected {cfg.timed_batches}"
    )
    assert wall_times, "no wall-clock intervals collected"

    ############################################################################
    # END timing loop
    ############################################################################

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ############################################################################
    # BEGIN results
    ############################################################################

    step_summary = _summarize(step_times)
    wall_summary = _summarize(wall_times)

    bs = cfg.batch_size
    seconds_per_sample = wall_summary["median"] / bs
    seconds_per_sample_step = step_summary["median"] / bs

    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    )

    results = {
        "measurement": "wall-clock time of one QAT training step (fwd + bwd + optimizer)",
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "seed": cfg.seed,
        "gpu": cfg.gpu,
        "device": str(device),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version() if device.type == "cuda" else None
        ),
        "python_version": platform.python_version(),
        "batch_size": bs,
        "reference_batch_size": REFERENCE_BATCH_SIZE,
        "accum_steps": accum_steps,
        "num_workers": num_workers,
        "lr": cfg.lr,
        "wd": cfg.wd,
        "ls": cfg.ls,
        "wl": cfg.wl,
        "max_grad_norm": cfg.max_grad_norm,
        "qat": {
            "bits": cfg.qat.bits,
            "granularity": cfg.qat.granularity,
            "skip_modules": sorted(cfg.qat.skip_modules),
            "linears_wrapped": len(qat_wrapped_parents),
            "linears_skipped": len(surviving_plain_linears),
        },
        "warmup_batches": cfg.warmup_batches,
        "timed_batches": cfg.timed_batches,
        # "step": h2d + forward + backward + optimizer, CUDA-synchronized.
        "seconds_per_batch_step": step_summary,
        # "wall": the same, plus time blocked on the dataloader.
        "seconds_per_batch_wall": wall_summary,
        # Headline derived quantities (wall clock: what a reserved GPU burns).
        "seconds_per_batch": wall_summary["median"],
        "samples_per_second": bs / wall_summary["median"],
        "seconds_per_sample": seconds_per_sample,
        # Compute-only counterparts, free of dataloader stalls.
        "seconds_per_batch_compute_only": step_summary["median"],
        "samples_per_second_step": bs / step_summary["median"],
        "seconds_per_sample_step": seconds_per_sample_step,
    }

    eval_dir = os.path.join(
        EVAL_ROOT,
        _sanitize_device_name(device_name),
        sanitize_timm_model_name(cfg.model_name),
        cfg.dataset_name,
        f"bs={bs}",
        f"qat=bits={cfg.qat.bits}_gran={cfg.qat.granularity}_skip="
        + ("-".join(sorted(cfg.qat.skip_modules)) or "none"),
        f"seed={cfg.seed}",
    )
    os.makedirs(eval_dir, exist_ok=True)
    out_path = os.path.join(eval_dir, "step_time.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    ############################################################################
    # END results
    ############################################################################

    summary = (
        f"device                 {device_name}\n"
        f"step (median)          {step_summary['median']:.4f}s "
        f"[p25 {step_summary['p25']:.4f}, p75 {step_summary['p75']:.4f}, "
        f"sd {step_summary['stdev']:.4f}]\n"
        f"wall (median)          {wall_summary['median']:.4f}s "
        f"[p25 {wall_summary['p25']:.4f}, p75 {wall_summary['p75']:.4f}, "
        f"sd {wall_summary['stdev']:.4f}]\n"
        f"samples/s              {bs / wall_summary['median']:.1f} (wall), "
        f"{bs / step_summary['median']:.1f} (compute only)\n"
        f"seconds/sample         {seconds_per_sample:.6f} (wall), "
        f"{seconds_per_sample_step:.6f} (compute only)\n"
        f"wrote {out_path}"
    )
    if IS_SLURM:
        log.info("\n%s", summary)
    else:
        print(summary)


if __name__ == "__main__":
    main()
