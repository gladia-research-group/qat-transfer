# ==============================================================================
# GPTQ checkpoint materialization (phase 006, step A)
# ==============================================================================
# Phase 006 asks whether GPTQ itself is transferrable: define
#
#   QV_gptq = GPTQ(FP^{D1}) - FP^{D1}
#
# and patch a different task's FP checkpoint with it. Unlike QAT checkpoints,
# GPTQ checkpoints do not exist on disk anywhere in this repo: every prior
# consumer of `apply_gptq_` (000_baselines/evaluate_fp_gptq, 005_qat_transfer_
# gptq) quantizes in memory at evaluation time and saves only eval_results.json.
# The QV construction, however, is a state-dict subtraction, and the transfer
# grid reads each donor's GPTQ checkpoint up to 22 times — so this script
# materializes GPTQ(FP) once per dataset, into a `gptq/` checkpoint subtree that
# parallels `fp/` and `qat/`.
#
# Methodology, and why it makes the checkpoints reusable:
# * Calibration is the dataset's own training split — the same choice as every
#   other GPTQ use in this repo. The first `num_calib_batches`
#   batches are materialized once and replayed for every layer, so the saved
#   checkpoint is a deterministic function of (FP checkpoint, calib batches,
#   gptq config).
# * The checkpoint is saved with the same key set as the FP checkpoint (GPTQ
#   replaces no modules and skips the head, whose weights therefore remain
#   bit-identical to FP). This is exactly the property that makes
#   QV_gptq = GPTQ(FP) - FP a well-defined subtraction, mirroring how QAT
#   checkpoints are saved with wrappers stripped.
# * The checkpoint path carries the full `gptq=` fragment (bits/gran/skip +
#   ncal/percdamp/actorder; `block_size` excluded — result-invariant solver
#   batching) in the slot where QAT checkpoints
#   carry `qat=`, so checkpoints from different GPTQ configurations can never
#   collide.
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

IS_SLURM = "SLURM_JOB_ID" in os.environ
TQDM_KW = dict(disable=IS_SLURM, mininterval=1.0)

from src.vision.ilharco_timm_supervised.modeling import ImageClassifier
from src.vision.data.registry import get_dataset
from src.duration import checkpoint_epochs, mult_path_frag
from src.vision.data.common import DATASET_NAME_TO_EPOCHS, DATASET_NAME_TO_NUM_CLASSES
from src.vision.utils import sanitize_timm_model_name, set_seed
from src.gptq import apply_gptq_

import hydra
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint
import torch
from torch import nn


def _fp_ckpt_dir(cfg: DictConfig, dataset_name: str) -> str:
    return os.path.join(
        os.environ['CHECKPOINT_BASE_PATH'],
        "vision",
        "ilharco_timm_supervised",
        "fp",
        sanitize_timm_model_name(cfg.model_name),
        dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        mult_path_frag(cfg.epoch_mult),
        f"seed={cfg.seed}",
    )


def _gptq_ckpt_dir(cfg: DictConfig, dataset_name: str) -> str:
    skip_modules_sorted = sorted(cfg.gptq.skip_modules)
    skip_tag = "-".join(skip_modules_sorted) if skip_modules_sorted else "none"
    return os.path.join(
        os.environ['CHECKPOINT_BASE_PATH'],
        "vision",
        "ilharco_timm_supervised",
        "gptq",
        sanitize_timm_model_name(cfg.model_name),
        dataset_name,
        f"optim=adamw_lr={cfg.lr}_wd={cfg.wd}_ls={cfg.ls}_wl={cfg.wl}_mgn={cfg.max_grad_norm}_bs={cfg.batch_size}",
        mult_path_frag(cfg.epoch_mult),
        f"gptq=bits={cfg.gptq.bits}_gran={cfg.gptq.granularity}_skip={skip_tag}"
        f"_ncal={cfg.gptq.num_calib_batches}_percdamp={cfg.gptq.percdamp}_actorder={cfg.gptq.actorder}",
        f"seed={cfg.seed}",
    )


@hydra.main(
    config_path="../../../../../config/experiments/vision/ilharco_timm_supervised/007_gptq_transfer",
    config_name="compute_gptq_checkpoints",
    version_base=None,
)
def main(cfg: DictConfig):

    if IS_SLURM:
        log.info("cfg:\n%s", dict(cfg))
    else:
        pprint(dict(cfg), expand_all=True)

    set_seed(cfg.seed)

    dataset_names = OmegaConf.to_container(cfg.dataset_names, resolve=True)

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    for di, dataset_name in enumerate(dataset_names):

        if IS_SLURM:
            log.info("=== Dataset %d/%d: %s ===", di + 1, len(dataset_names), dataset_name)
        else:
            print(f"\n{'='*60}")
            print(f"  Dataset {di + 1}/{len(dataset_names)}: {dataset_name}")
            print(f"{'='*60}")

        epochs = checkpoint_epochs(
            dataset_name, DATASET_NAME_TO_EPOCHS, cfg.limit_num_epochs
        )

        gptq_dir = _gptq_ckpt_dir(cfg, dataset_name)
        gptq_path = os.path.join(gptq_dir, f"classifier_epoch_{epochs}.pt")

        if cfg.skip_existing and os.path.exists(gptq_path):
            if IS_SLURM:
                log.info("Skipping %s: checkpoint exists: %s", dataset_name, gptq_path)
            else:
                print(f"Skipping {dataset_name}: checkpoint exists: {gptq_path}")
            continue

        fp_path = os.path.join(_fp_ckpt_dir(cfg, dataset_name), f"classifier_epoch_{epochs}.pt")
        if not os.path.exists(fp_path):
            log.warning("Skipping %s: FP checkpoint missing: %s", dataset_name, fp_path)
            continue

        ########################################################################
        # BEGIN checkpoint loading
        ########################################################################

        classifier = ImageClassifier.load(
            model_name=cfg.model_name,
            num_classes=DATASET_NAME_TO_NUM_CLASSES[dataset_name],
            filename=fp_path,
        )
        classifier.to(device)
        classifier.eval()

        ########################################################################
        # END checkpoint loading
        ########################################################################

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        ########################################################################
        # BEGIN calibration data (this dataset's own training split)
        ########################################################################

        dataset = get_dataset(
            dataset_name=dataset_name,
            preprocess_train=classifier.train_preprocess,
            preprocess_inference=classifier.val_preprocess,
            batch_size=cfg.batch_size,
            num_workers=int(os.environ['TORCH_NUM_WORKERS']),
            seed=cfg.seed,
        )

        calib_batches = []
        for batch in dataset.train_loader:
            if len(calib_batches) >= cfg.gptq.num_calib_batches:
                break
            calib_batches.append(batch)

        if IS_SLURM:
            log.info("Materialized %d calibration batches from the %s train split",
                     len(calib_batches), dataset_name)
        else:
            print(f"Materialized {len(calib_batches)} calibration batches "
                  f"from the {dataset_name} train split")

        ########################################################################
        # END calibration data
        ########################################################################

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        ########################################################################
        # BEGIN GPTQ + save
        ########################################################################

        skip_modules = frozenset(cfg.gptq.skip_modules)

        all_linear_names = [
            name for name, module in classifier.named_modules()
            if isinstance(module, nn.Linear)
        ]

        quantized_names = apply_gptq_(
            model=classifier,
            bits=cfg.gptq.bits,
            granularity=cfg.gptq.granularity,
            skip_modules=skip_modules,
            calib_loader=calib_batches,
            device=device,
            num_calib_batches=cfg.gptq.num_calib_batches,
            percdamp=cfg.gptq.percdamp,
            actorder=cfg.gptq.actorder,
            block_size=cfg.gptq.block_size,
        )

        skipped_names = sorted(set(all_linear_names) - set(quantized_names))

        if IS_SLURM:
            log.info("GPTQ: quantized %d layers, skipped %d", len(quantized_names), len(skipped_names))
        else:
            print(f"GPTQ: quantized {len(quantized_names)} layers, skipped {len(skipped_names)}")

        classifier.to("cpu")
        classifier.save(gptq_path)

        # Provenance sidecar: which FP checkpoint and gptq config produced this.
        manifest = {
            "model_name": cfg.model_name,
            "dataset_name": dataset_name,
            "seed": cfg.seed,
            "epochs": epochs,
            "fp_classifier_path": fp_path,
            "gptq": {
                "bits": cfg.gptq.bits,
                "granularity": cfg.gptq.granularity,
                "skip_modules": list(cfg.gptq.skip_modules),
                "num_calib_batches": cfg.gptq.num_calib_batches,
                "percdamp": cfg.gptq.percdamp,
                "actorder": cfg.gptq.actorder,
                "block_size": cfg.gptq.block_size,
                "calibration_split": "train",
            },
            "quantized_layers": quantized_names,
            "skipped_layers": skipped_names,
        }
        with open(os.path.join(gptq_dir, "gptq_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        if IS_SLURM:
            log.info("Saved GPTQ checkpoint: %s", gptq_path)
        else:
            print(f"Saved GPTQ checkpoint: {gptq_path}")

        ########################################################################
        # END GPTQ + save
        ########################################################################

        del classifier, dataset, calib_batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if IS_SLURM:
        log.info("All %d datasets processed. Forcing exit.", len(dataset_names))
        os._exit(0)


if __name__ == "__main__":
    main()
