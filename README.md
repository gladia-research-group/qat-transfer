# qat-transfer

Official code for **Zero-Shot Quantization via Weight-Space Arithmetic** (NeurIPS 2026).

Daniele Solombrino, Antonio Andrea Gargiulo, Alessandro Zirilli, Luca Zhou, Adrian Robert Minut, Emanuele Rodolà

[arXiv](https://arxiv.org/abs/2604.03420) · [OpenReview](https://openreview.net/forum?id=wrUEnnSgXa)

> We show that robustness to post-training quantization (PTQ) is a transferable direction in weight space. We call this direction the quantization vector: extracted from a donor task by simple weight-space arithmetic, it can be used to patch a receiver model and improve post-PTQ Top-1 accuracy by up to 60 points in a 3-bit setting, without receiver-side quantization-aware training (QAT). Because the method requires no receiver training data, it provides a zero-shot, low-cost alternative to QAT for extremely low-bit deployment. Across four ViT scales and 22 image classification tasks, donor quantization vectors often yield substantial gains even when donor and receiver tasks differ markedly. We further prove rigorously that quantization vectors are well-defined and do not suffer from reparameterization symmetries, and provide a local geometric account of their effect. Together, these results suggest that quantization robustness can be partially isolated, reused, and transferred through simple weight-space algebra.

## Method in one line

Let `FP_D` be a full-precision finetune on task `D` and `QAT_D` a quantization-aware finetune on the same task. The **quantization vector** is their difference, `QV = QAT_D - FP_D`. The central claim is that this vector is largely task-agnostic: it can be computed once on a **donor** task and added to a different **receiver** task's FP checkpoint,

```
patched = FP_receiver + lambda * QV_donor
```

recovering much of the receiver's QAT benefit under post-training quantization (PTQ) without ever running QAT on the receiver. With `lambda = 1` this needs no receiver data at all.

The repo covers vision (timm supervised ViT/DeiT/Swin, OpenCLIP, HF CLIP) and text (`AutoModelForSequenceClassification`) backbones, and includes Lean formalizations of the paper's propositions under [`proofs/`](proofs/).

## Data and checkpoints

This repository contains **code and configs only**. Datasets are downloaded through HuggingFace / torchvision into the caches configured in `.env`. Finetuned checkpoints (`storage/`) and evaluation outputs (`evaluations/`) are **not** included. The analysis (`code/experiments/998_rebuttal/`) and plotting (`code/visualizations/`, `visualizations/`) scripts read `evaluations/`, which is produced by running the finetuning, `000_baselines` and `001_qat_transfer` pipelines described below.

## Experiment phases

| Phase | Question | Families |
|---|---|---|
| `000_baselines` | FP, QAT, PTQ (and GPTQ / AWQ / PV-Tuning) accuracies on their own | all |
| `001_qat_transfer` | Does `FP_tgt + lambda * QV_src` recover `QAT_tgt` under PTQ? The core experiment. | all |
| `002_qat_transfer_reversed` / `002z_...` | Does the QV also work subtracted from a QAT checkpoint? | timm |
| `005_qat_transfer_gptq` / `009_qat_transfer_awq` | Does QV patching still help on top of GPTQ / AWQ instead of RTN PTQ? | timm |
| `007_gptq_transfer` | Transfer of a GPTQ-derived vector `GPTQ(FP) - FP` | timm |
| `008_pv_transfer` | Does a stronger quantization-aware finetuner (PV-Tuning) give a better-transferring QV? | timm |
| `998_rebuttal/*` | Cross-family analyses: zero-shot framing, cost amortization, lambda sensitivity, quantization mechanism, QV alignment | cross-family |
| `999_paper_stuff` | Paper figures and tables | open_clip, timm, text |

`config/` mirrors `code/` one-to-one: every Hydra script `code/a/b/script.py` has its YAML at `config/a/b/script.yaml`.

## Getting Started

All commands below assume the repo root as the working directory and the project `.venv` managed by `uv`. Every Hydra script resolves config search paths from `${oc.env:PWD}`, so **you must launch from the repo root** — not from inside `code/` or `config/`.

Scripts come in two families:
- **Hydra scripts** — everything under `code/src/` and `code/experiments/`. They support three launch modes: single local run, local *sequential* sweep (Hydra's basic launcher), and Slurm *parallel* sweep (submitit launcher).
- **Argparse scripts** — the plotting utilities under `code/visualizations/` and the `pick_best_alpha` helpers under `code/experiments/.../001_qat_transfer/`. Plain CLI, no Hydra.

### Prerequisites

- Python 3.11 (pinned in `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) package manager

### Setup

```
git clone https://github.com/gladia-research-group/qat-transfer.git && cd qat-transfer
uv sync
cp .env.example .env   # then fill in the values below
```

### Environment variables

Edit `.env` before running any script. Every entry below is required at runtime (loaded via `dotenv`). `SBATCH_PARTITION`, `SBATCH_ACCOUNT`, `SLURM_PROJECT_ROOT` and `TORCHINDUCTOR_CACHE_DIR` are only needed for Slurm submission and are commented out in `.env.example`.

| Variable | Purpose |
|---|---|
| `CHECKPOINT_BASE_PATH` | Root directory for finetuned model checkpoints (backbone + head `.pt` files) |
| `HEAD_BASE_PATH` | Root directory for classification head checkpoints |
| `EVALUATION_BASE_PATH` | Root directory where `eval_results.json` files are written |
| `TORCH_NUM_WORKERS` | Number of DataLoader workers |
| `HF_DATASETS_CACHE` | HuggingFace datasets cache directory |
| `HF_HUB_CACHE` | HuggingFace model hub cache directory |
| `HF_HOME` | HuggingFace home directory |
| `HF_TOKEN` | HuggingFace API token (for gated models / datasets) |
| `HUGGING_FACE_HUB_TOKEN` | Legacy HF token (some libraries still read this) |
| `OPENCLIP_CACHE_DIR` | OpenCLIP model cache directory |
| `CACHE_DIR` | General-purpose cache directory |

---

## Datasets

### Vision datasets

The full list of supported vision datasets (used in every vision sweep example below) is defined in [code/src/vision/data/common.py](code/src/vision/data/common.py#L35):

```
Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet
```

### Text datasets

The full list of supported text datasets is defined in [code/src/text/data/common.py](code/src/text/data/common.py#L15):

```
Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction
```

All text datasets train for 5 epochs by default. Split constants shared across all domains: `SPLIT_SEED=0`, `VAL_FRACTION=0.1`, `MAX_VAL_SAMPLES=5000`.

That per-dataset epoch schedule is the **1x** budget, and it is now scaled by an explicit `epoch_mult` (see `code/src/duration.py`). The schedule normalises every dataset to roughly 2,000 optimizer steps — Cars 2,030, DTD 2,052, GTSRB 2,068 — with **ImageNet the outlier at 9,971, about 4.8x the median**. ImageNet is therefore confounded as a donor: it is simultaneously the most diverse task and by far the longest-trained one. `epoch_mult` exists to separate those two explanations, by running ImageNet on the common budget (`mult=0.25`) and ordinary datasets on ImageNet's (`mult=4`). Every path states its budget explicitly; a path carrying no `mult=` predates the axis.

---

## Project structure

```
qat-transfer/
  code/
    src/              # Shared library code (models, data, quantization)
    common/           # run-id and status helpers shared by some experiments
    experiments/      # Evaluation, transfer and analysis scripts (see Experiment phases)
    test/             # Unit and smoke tests
    visualizations/   # Argparse plotting scripts
  config/             # Hydra YAML configs — mirrors code/ 1:1
  visualizations/     # Standalone plotting scripts for the 998_rebuttal QV-alignment analyses
  proofs/             # Lean formalizations of the paper's propositions
  evaluations/        # eval_results.json outputs         (created at runtime, not in git)
  storage/            # Checkpoints and heads             (created at runtime, not in git)
  plots/              # Generated figures                 (created at runtime, not in git)
  logs/               # Hydra run/sweep logs              (created at runtime, not in git)
```

### Conventions

Follow these rules when adding new experiments or model families:

- **`config/` mirrors `code/` 1:1** — every Hydra script at `code/a/b/script.py` has a matching YAML at `config/a/b/script.yaml`.
- **Numbered experiment phases** — `000_baselines`, `001_qat_transfer`, etc. New experiments get the next sequential number.
- **Model families** — each gets its own subdirectory under `code/src/{vision,text}/`, `code/experiments/{vision,text}/`, `config/...`, etc. All follow the same internal structure: finetune scripts → baselines → transfer → visualizations.
- **`skip_modules` always explicit** — no defaults in any script or config. Every call must specify which modules to skip during quantization.
- **Split constants** — `SPLIT_SEED=0`, `VAL_FRACTION=0.1`, `MAX_VAL_SAMPLES=5000` (shared across all domains; defined in `code/src/{vision,text}/data/common.py`).

### Model name sanitization

Each model family uses a different sanitizer to convert model identifiers into safe filesystem path components:

| Family | Sanitizer | Example |
|---|---|---|
| `ilharco_hf_clip`, text families | `sanitize_hf_model_name` | `openai/clip-vit-base-patch16` → `openai--clip-vit-base-patch16` |
| `ilharco_timm_supervised` | `sanitize_timm_model_name` | `vit_base_patch16_clip_224.openai_ft_in12k_in1k` (unchanged) |
| `ilharco_open_clip` | `sanitize_open_clip_model_name` | `(ViT-B-32, openai)` → `ViT-B-32__openai` |

All sanitizers live in [code/src/vision/utils.py](code/src/vision/utils.py).

### Output path layouts

**Checkpoint paths** (vision):
```
{CHECKPOINT_BASE_PATH}/vision/{family}/{fp,qat}/{sanitized_model}/{dataset}/optim=adamw_lr={lr}_wd={wd}_ls={ls}_wl={wl}_mgn={max_grad_norm}_bs={batch_size}/mult={m}/[qat=bits={bits}_gran={granularity}_skip={skip_tag}/]seed={seed}/backbone_epoch_{N}.pt
```

**Checkpoint paths** (text): same structure but the optim fragment uses `_ml={max_length}` instead of `_wl={wl}`:
```
{CHECKPOINT_BASE_PATH}/text/{family}/{fp,qat}/{sanitized_model}/{dataset}/optim=adamw_lr={lr}_wd={wd}_ls={ls}_mgn={max_grad_norm}_bs={batch_size}_ml={max_length}/mult={m}/[qat=bits={bits}_gran={granularity}_skip={skip_tag}/]seed={seed}/backbone_epoch_{N}.pt
```

**Evaluation paths**:
```
{EVALUATION_BASE_PATH}/{vision,text}/{family}/{phase}/{experiment_type}/{sanitized_model}/{dataset}/optim=.../mult={m}/{seed}/eval_results.json
```

---

## Launch modes for Hydra scripts

For **every** Hydra script listed below, three launch modes are available. They only differ in the wrapper around the command; the config overrides are identical.

### 1. Single local run (no `-m`)
Runs in the current process. Good for debugging on a workstation / login node.
```
uv run --active python <script.py> <overrides...>
```

### 2. Local sequential sweep (`-m`, no launcher override)
Hydra's default `basic` launcher. Every combination of swept values runs **sequentially in the same process**, one after the other. Use this when you want a sweep on a single local GPU without Slurm.
```
uv run --active python <script.py> -m <overrides...> key=a,b,c other=x,y
```

### 3. Submitit parallel sweep (`-m hydra/launcher=submitit_slurm`)
Every combination of swept values is dispatched as a **separate Slurm job** via `hydra-submitit-launcher`. Jobs run in parallel subject to Slurm scheduling. Also usable for a single run (no comma-separated values) to get one Slurm job instead of a local run.
```
uv run --active python <script.py> -m hydra/launcher=submitit_slurm <overrides...> key=a,b,c
```
The launcher config lives at [config/hydra/launcher/submitit_slurm.yaml](config/hydra/launcher/submitit_slurm.yaml) (GPUs, CPUs, mem, walltime; partition and account come from `SBATCH_PARTITION` / `SBATCH_ACCOUNT`). Override any field on the CLI, e.g. `hydra.launcher.timeout_min=240 hydra.launcher.mem_gb=64`.

> Modes 2 and 3 both use `-m`. The **only** difference is whether you add `hydra/launcher=submitit_slurm`. Without it, Hydra's basic launcher runs the sweep sequentially in-process; with it, submitit fans the sweep out to Slurm.

---

## Test suite

Smoke tests for data loading and model forward passes live under [code/test/](code/test/).

### Vision data loading — [code/test/vision/data/dataloading.py](code/test/vision/data/dataloading.py)
```
uv run --active python code/test/vision/data/dataloading.py --dataset-name CIFAR10 CIFAR100 --batch-size 64 --num-workers 4
```

### Text data loading — [code/test/text/data/dataloading.py](code/test/text/data/dataloading.py)
```
uv run --active python code/test/text/data/dataloading.py --dataset-name Emotion IMDB --batch-size 32 --num-workers 4 --seed 42
```

### Vision modeling — [code/test/vision/modeling.py](code/test/vision/modeling.py)
```
uv run --active python code/test/vision/modeling.py --model-name openai/clip-vit-base-patch32 --dataset-name CIFAR10 --batch-size 8 --num-workers 2 --max-batches 2 --gpu 0
```

---

# ilharco_hf_clip

## Finetuning

### FP finetuning — [code/src/vision/ilharco_hf_clip/finetune_fp.py](code/src/vision/ilharco_hf_clip/finetune_fp.py)
Config: [config/src/vision/ilharco_hf_clip/finetune_fp.yaml](config/src/vision/ilharco_hf_clip/finetune_fp.yaml)

Single local run:
```
uv run --active python code/src/vision/ilharco_hf_clip/finetune_fp.py model_name=openai/clip-vit-base-patch16 dataset_name=CIFAR10 seed=2038 gpu=0
```

Local sequential sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_hf_clip/finetune_fp.py -m model_name=openai/clip-vit-base-patch16 dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 gpu=0
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_hf_clip/finetune_fp.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 gpu=0
```

### QAT finetuning — [code/src/vision/ilharco_hf_clip/finetune_qat.py](code/src/vision/ilharco_hf_clip/finetune_qat.py)
Config: [config/src/vision/ilharco_hf_clip/finetune_qat.yaml](config/src/vision/ilharco_hf_clip/finetune_qat.yaml). Requires `qat.bits`, `qat.granularity`, `qat.skip_modules`.

Single local run:
```
uv run --active python code/src/vision/ilharco_hf_clip/finetune_qat.py model_name=openai/clip-vit-base-patch16 dataset_name=CIFAR10 seed=2038 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_hf_clip/finetune_qat.py -m model_name=openai/clip-vit-base-patch16 dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classification_head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_hf_clip/finetune_qat.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classification_head]'
```

> The user's feedback memory notes: `finetune_*` scripts are GPU-intensive and long-running — do not launch them casually.

---

## 000_baselines — evaluation scripts

All scripts live in [code/experiments/vision/ilharco_hf_clip/000_baselines/](code/experiments/vision/ilharco_hf_clip/000_baselines/) with matching configs in [config/experiments/vision/ilharco_hf_clip/000_baselines/](config/experiments/vision/ilharco_hf_clip/000_baselines/).

### evaluate_pretrained — [code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_pretrained.py](code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_pretrained.py)
Config: [config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_pretrained.yaml](config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_pretrained.yaml)

Single local run:
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_pretrained.py model_name=openai/clip-vit-base-patch16 dataset_name=CIFAR10 batch_size=128 seed=2038 gpu=0
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_pretrained.py -m model_name=openai/clip-vit-base-patch16 dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 seed=1,2,3 gpu=0
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_pretrained.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 seed=1,2,3 gpu=0
```

### evaluate_fp — [code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp.py](code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp.py)
Config: [config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp.yaml](config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp.yaml)

Single local run:
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp.py model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp.py -m model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0
```

### evaluate_fp_ptq — [code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq.py](code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq.py)
Config: [config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq.yaml](config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq.yaml). Adds a `ptq` group.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq.py model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq.py -m model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

### evaluate_fp_ptq_bias_norm_emb_from_pt — [code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq_bias_norm_emb_from_pt.py](code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq_bias_norm_emb_from_pt.py)
Config: [config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq_bias_norm_emb_from_pt.yaml](config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq_bias_norm_emb_from_pt.yaml). Same parameters as `evaluate_fp_ptq`. Before applying PTQ, swaps biases, layer-norm weights, and embedding weights back to pretrained values.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq_bias_norm_emb_from_pt.py model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq_bias_norm_emb_from_pt.py -m model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq_bias_norm_emb_from_pt.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

### evaluate_qat — [code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat.py](code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat.py)
Config: [config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat.yaml](config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat.yaml). The `qat.*` group must match the training config of the checkpoint being evaluated.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat.py model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat.py -m model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classification_head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classification_head]'
```

### evaluate_qat_ptq — [code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat_ptq.py](code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat_ptq.py)
Config: [config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat_ptq.yaml](config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat_ptq.yaml). Has both a `qat` group (to locate the checkpoint) and a `ptq` group (for the PTQ applied at eval time).

Single local run:
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat_ptq.py model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat_ptq.py -m model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat_ptq.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 pretrained=true dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

---

## 001_qat_transfer — QV transfer experiment

### qv_transfer — [code/experiments/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer.py](code/experiments/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer.py)
Config: [config/experiments/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer.yaml](config/experiments/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer.yaml)

Requires three checkpoints to already exist on disk: `FP_source`, `QAT_source`, `FP_target`. Defines a QV = `QAT_source - FP_source`, patches it into `FP_target` at scale `qv.alpha`, then applies PTQ and evaluates.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer.py model_name=openai/clip-vit-base-patch16 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 source.dataset_name=CIFAR10 source.seed=2038 target.dataset_name=CIFAR100 target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' qv.alpha=1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]' gpu=0
```

Local sequential sweep (all source × all target datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer.py -m model_name=openai/clip-vit-base-patch16 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 source.dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet source.seed=2038 target.dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet target.seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' qv.alpha=0.25,0.5,0.75,1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]' gpu=0
```

Submitit parallel sweep (all source × all target datasets):
```
uv run --active python code/experiments/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer.py -m hydra/launcher=submitit_slurm model_name=openai/clip-vit-base-patch16 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 source.dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet source.seed=2038 target.dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet target.seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' qv.alpha=0.25,0.5,0.75,1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]' gpu=0
```

---

## Visualizations (argparse, no Hydra, no Slurm)

These scripts read the JSON results produced by the experiments above and render plots. They use plain `argparse`, so Hydra sweeps and submitit do not apply — just call them one at a time (or wrap them in a shell `for` loop for a local sequential sweep).

### qv_transfer_heatmap — [code/visualizations/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer_heatmap.py](code/visualizations/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer_heatmap.py)
```
uv run --active python code/visualizations/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer_heatmap.py --model-name openai/clip-vit-base-patch16 --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules classification_head --qv-alpha 1.0
```

### qv_transfer_heatmap_best_sf — [code/visualizations/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer_heatmap_best_sf.py](code/visualizations/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer_heatmap_best_sf.py)
```
uv run --active python code/visualizations/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer_heatmap_best_sf.py --model-name openai/clip-vit-base-patch16 --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules classification_head --qv-alpha 1.0
```

Local sequential loop over `qv.alpha` values:
```
for alpha in 0.25 0.5 0.75 1.0; do uv run --active python code/visualizations/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer_heatmap.py --model-name openai/clip-vit-base-patch16 --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules classification_head --qv-alpha $alpha; done
```

---

# ilharco_timm_supervised

## Finetuning

### FP finetuning — [code/src/vision/ilharco_timm_supervised/finetune_fp.py](code/src/vision/ilharco_timm_supervised/finetune_fp.py)
Config: [config/src/vision/ilharco_timm_supervised/finetune_fp.yaml](config/src/vision/ilharco_timm_supervised/finetune_fp.yaml)

Single local run:
```
uv run --active python code/src/vision/ilharco_timm_supervised/finetune_fp.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=CIFAR10 seed=2038 gpu=0
```

Local sequential sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_timm_supervised/finetune_fp.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 gpu=0
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_timm_supervised/finetune_fp.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 gpu=0
```

### QAT finetuning — [code/src/vision/ilharco_timm_supervised/finetune_qat.py](code/src/vision/ilharco_timm_supervised/finetune_qat.py)
Config: [config/src/vision/ilharco_timm_supervised/finetune_qat.yaml](config/src/vision/ilharco_timm_supervised/finetune_qat.yaml). Requires `qat.bits`, `qat.granularity`, `qat.skip_modules`.

Single local run:
```
uv run --active python code/src/vision/ilharco_timm_supervised/finetune_qat.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=CIFAR10 seed=2038 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_timm_supervised/finetune_qat.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_timm_supervised/finetune_qat.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[head]'
```

---

## 000_baselines — evaluation scripts

All scripts live in [code/experiments/vision/ilharco_timm_supervised/000_baselines/](code/experiments/vision/ilharco_timm_supervised/000_baselines/) with matching configs in [config/experiments/vision/ilharco_timm_supervised/000_baselines/](config/experiments/vision/ilharco_timm_supervised/000_baselines/).

### evaluate_pretrained — [code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained.py](code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained.py)
Config: [config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained.yaml](config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained.yaml). Unlike the hf_clip variant, requires training hyperparameters (`lr`, `wd`, `ls`, `wl`, `max_grad_norm`) to locate the finetuned head checkpoint.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0
```

### evaluate_pretrained_ptq — [code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained_ptq.py](code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained_ptq.py)
Config: [config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained_ptq.yaml](config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained_ptq.yaml). Evaluates the pretrained backbone after applying PTQ. Has both `qat.*` (to locate the QAT-trained head checkpoint) and `ptq.*` groups.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained_ptq.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained_ptq.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[head]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained_ptq.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[head]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[head]'
```

### evaluate_fp — [code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp.py](code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp.py)
Config: [config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp.yaml](config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp.yaml)

Single local run:
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0
```

### evaluate_fp_ptq — [code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_ptq.py](code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_ptq.py)
Config: [config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_ptq.yaml](config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_ptq.yaml). Adds a `ptq` group.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_ptq.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_ptq.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_ptq.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[head]'
```

### evaluate_qat — [code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat.py](code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat.py)
Config: [config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat.yaml](config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat.yaml). The `qat.*` group must match the training config of the checkpoint being evaluated.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[head]'
```

### evaluate_qat_ptq — [code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat_ptq.py](code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat_ptq.py)
Config: [config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat_ptq.yaml](config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat_ptq.yaml). Has both a `qat` group (to locate the checkpoint) and a `ptq` group (for the PTQ applied at eval time).

Single local run:
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat_ptq.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat_ptq.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat_ptq.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[head]'
```

---

## 001_qat_transfer — QV transfer experiment

### qv_transfer — [code/experiments/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer.py](code/experiments/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer.py)
Config: [config/experiments/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer.yaml](config/experiments/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer.yaml)

Same concept as the hf_clip variant but uses `source.dataset_names` (plural — a list of source datasets iterated internally by the script) instead of `source.dataset_name`.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[CIFAR10]' source.seed=2038 target.dataset_name=CIFAR100 target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]' gpu=0
```

Local sequential sweep (all source × all target datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 target.dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet target.seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=0.25,0.5,0.75,1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]' gpu=0
```

Submitit parallel sweep (all source × all target datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 target.dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet target.seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[head]' qv.alpha=0.25,0.5,0.75,1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]' gpu=0
```

### pick_best_alpha — [code/experiments/vision/ilharco_timm_supervised/001_qat_transfer/pick_best_alpha.py](code/experiments/vision/ilharco_timm_supervised/001_qat_transfer/pick_best_alpha.py)

Reads val-split `eval_results.json` files produced by `qv_transfer.py`, finds the alpha that maximises accuracy for each `(source_dataset, target_dataset)` pair. Only considers the restricted alpha sweep: `(0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00, 1.05, 1.20, 1.35, 1.50)`.

Output modes: `table` (markdown), `json`, `commands` (local), `commands-bg` (background), `commands-sbatch` (Slurm).

```
uv run --active python code/experiments/vision/ilharco_timm_supervised/001_qat_transfer/pick_best_alpha.py --model-name vit_base_patch16_clip_224.openai_ft_in12k_in1k --seed 2038 --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules head --slurm-timeout 120 --slurm-job-name pick_best --output table
```

---

## Visualizations (argparse, no Hydra, no Slurm)

### qv_transfer_heatmap — [code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap.py](code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap.py)
```
uv run --active python code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap.py --model-name vit_base_patch16_clip_224.openai_ft_in12k_in1k --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules head --qv-alpha 1.0
```

### qv_transfer_heatmap_best_sf — [code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap_best_sf.py](code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap_best_sf.py)
```
uv run --active python code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap_best_sf.py --model-name vit_base_patch16_clip_224.openai_ft_in12k_in1k --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules head --qv-alpha 1.0
```

Local sequential loop over `qv.alpha` values:
```
for alpha in 0.25 0.5 0.75 1.0; do uv run --active python code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap.py --model-name vit_base_patch16_clip_224.openai_ft_in12k_in1k --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules head --qv-alpha $alpha; done
```

---

# ilharco_open_clip

All open_clip scripts require an extra `pretrained` parameter (e.g. `openai`) alongside `model_name`.

## Finetuning

### FP finetuning — [code/src/vision/ilharco_open_clip/finetune_fp.py](code/src/vision/ilharco_open_clip/finetune_fp.py)
Config: [config/src/vision/ilharco_open_clip/finetune_fp.yaml](config/src/vision/ilharco_open_clip/finetune_fp.yaml)

Single local run:
```
uv run --active python code/src/vision/ilharco_open_clip/finetune_fp.py model_name=ViT-B-32 pretrained=openai dataset_name=CIFAR10 seed=2038 gpu=0
```

Local sequential sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_open_clip/finetune_fp.py -m model_name=ViT-B-32 pretrained=openai dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 gpu=0
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_open_clip/finetune_fp.py -m hydra/launcher=submitit_slurm model_name=ViT-B-32 pretrained=openai dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 gpu=0
```

### QAT finetuning — [code/src/vision/ilharco_open_clip/finetune_qat.py](code/src/vision/ilharco_open_clip/finetune_qat.py)
Config: [config/src/vision/ilharco_open_clip/finetune_qat.yaml](config/src/vision/ilharco_open_clip/finetune_qat.yaml). Requires `qat.bits`, `qat.granularity`, `qat.skip_modules`.

Single local run:
```
uv run --active python code/src/vision/ilharco_open_clip/finetune_qat.py model_name=ViT-B-32 pretrained=openai dataset_name=CIFAR10 seed=2038 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_open_clip/finetune_qat.py -m model_name=ViT-B-32 pretrained=openai dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classification_head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_open_clip/finetune_qat.py -m hydra/launcher=submitit_slurm model_name=ViT-B-32 pretrained=openai dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classification_head]'
```

---

## 000_baselines — evaluation scripts

All scripts live in [code/experiments/vision/ilharco_open_clip/000_baselines/](code/experiments/vision/ilharco_open_clip/000_baselines/) with matching configs in [config/experiments/vision/ilharco_open_clip/000_baselines/](config/experiments/vision/ilharco_open_clip/000_baselines/).

### evaluate_pretrained — [code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_pretrained.py](code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_pretrained.py)
Config: [config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_pretrained.yaml](config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_pretrained.yaml)

Single local run:
```
uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_pretrained.py model_name=ViT-B-32 pretrained=openai dataset_name=CIFAR10 batch_size=128 seed=2038 gpu=0
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_pretrained.py -m model_name=ViT-B-32 pretrained=openai dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 seed=1,2,3 gpu=0
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_pretrained.py -m hydra/launcher=submitit_slurm model_name=ViT-B-32 pretrained=openai dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 seed=1,2,3 gpu=0
```

### evaluate_fp — [code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp.py](code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp.py)
Config: [config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp.yaml](config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp.yaml)

Single local run:
```
uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp.py model_name=ViT-B-32 pretrained=openai dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp.py -m model_name=ViT-B-32 pretrained=openai dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp.py -m hydra/launcher=submitit_slurm model_name=ViT-B-32 pretrained=openai dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0
```

### evaluate_fp_ptq — [code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp_ptq.py](code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp_ptq.py)
Config: [config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp_ptq.yaml](config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp_ptq.yaml). Adds a `ptq` group.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp_ptq.py model_name=ViT-B-32 pretrained=openai dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp_ptq.py -m model_name=ViT-B-32 pretrained=openai dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp_ptq.py -m hydra/launcher=submitit_slurm model_name=ViT-B-32 pretrained=openai dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

### evaluate_qat — [code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat.py](code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat.py)
Config: [config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat.yaml](config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat.yaml). The `qat.*` group must match the training config of the checkpoint being evaluated.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat.py model_name=ViT-B-32 pretrained=openai dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat.py -m model_name=ViT-B-32 pretrained=openai dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classification_head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat.py -m hydra/launcher=submitit_slurm model_name=ViT-B-32 pretrained=openai dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classification_head]'
```

### evaluate_qat_ptq — [code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat_ptq.py](code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat_ptq.py)
Config: [config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat_ptq.yaml](config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat_ptq.yaml). Has both a `qat` group (to locate the checkpoint) and a `ptq` group (for the PTQ applied at eval time).

Single local run:
```
uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat_ptq.py model_name=ViT-B-32 pretrained=openai dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat_ptq.py -m model_name=ViT-B-32 pretrained=openai dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat_ptq.py -m hydra/launcher=submitit_slurm model_name=ViT-B-32 pretrained=openai dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classification_head]'
```

---

## 001_qat_transfer — QV transfer experiment

### qv_transfer — [code/experiments/vision/ilharco_open_clip/001_qat_transfer/qv_transfer.py](code/experiments/vision/ilharco_open_clip/001_qat_transfer/qv_transfer.py)
Config: [config/experiments/vision/ilharco_open_clip/001_qat_transfer/qv_transfer.yaml](config/experiments/vision/ilharco_open_clip/001_qat_transfer/qv_transfer.yaml)

Same concept as the other vision variants. Uses `source.dataset_names` and `target.dataset_names` (plural — lists of datasets iterated internally by the script). Requires `eval_split` (`val` or `test`). Supports `qv.alpha=best` to read the best alpha per (source, target) pair from `best_alpha.json` files written by `pick_best_alpha.py --output disk`.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_open_clip/001_qat_transfer/qv_transfer.py model_name=ViT-B-32 pretrained=openai batch_size=128 eval_split=val lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[CIFAR10]' source.seed=2038 'target.dataset_names=[CIFAR100]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' qv.alpha=1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]' gpu=0
```

Local sequential sweep (all source x all target datasets):
```
uv run --active python code/experiments/vision/ilharco_open_clip/001_qat_transfer/qv_transfer.py -m model_name=ViT-B-32 pretrained=openai batch_size=128 eval_split=val lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 'target.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' target.seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' qv.alpha=0.25,0.5,0.75,1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]' gpu=0
```

Submitit parallel sweep (all source x all target datasets):
```
uv run --active python code/experiments/vision/ilharco_open_clip/001_qat_transfer/qv_transfer.py -m hydra/launcher=submitit_slurm model_name=ViT-B-32 pretrained=openai batch_size=128 eval_split=val lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 'target.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' target.seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' qv.alpha=0.25,0.5,0.75,1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]' gpu=0
```

Test-set evaluation with best alpha (requires `pick_best_alpha.py --output disk` first):
```
uv run --active python code/experiments/vision/ilharco_open_clip/001_qat_transfer/qv_transfer.py -m hydra/launcher=submitit_slurm model_name=ViT-B-32 pretrained=openai batch_size=128 eval_split=test lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 'source.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' source.seed=2038 'target.dataset_names=[Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet]' target.seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classification_head]' qv.alpha=best ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classification_head]' gpu=0
```

### pick_best_alpha — [code/experiments/vision/ilharco_open_clip/001_qat_transfer/pick_best_alpha.py](code/experiments/vision/ilharco_open_clip/001_qat_transfer/pick_best_alpha.py)

Same as the timm_supervised variant but adds `--pretrained`. Restricted alpha sweep: `(0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00, 1.05, 1.20, 1.35, 1.50)`. Output modes: `table`, `json`, `commands`, `commands-bg`, `commands-sbatch`, `disk`.

Show best alphas as a table:
```
uv run --active python code/experiments/vision/ilharco_open_clip/001_qat_transfer/pick_best_alpha.py --model-name ViT-B-32 --pretrained openai --seed 2038 --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules classification_head --slurm-timeout 120 --slurm-job-name pick_best --output table
```

Write best alphas to disk (one `best_alpha.json` per (source, target) pair, used by `qv.alpha=best`):
```
uv run --active python code/experiments/vision/ilharco_open_clip/001_qat_transfer/pick_best_alpha.py --model-name ViT-B-32 --pretrained openai --seed 2038 --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules classification_head --slurm-timeout 120 --slurm-job-name pick_best --output disk
```

---

## Visualizations (argparse, no Hydra, no Slurm)

### qv_transfer_heatmap — [code/visualizations/vision/ilharco_open_clip/001_qat_transfer/qv_transfer_heatmap.py](code/visualizations/vision/ilharco_open_clip/001_qat_transfer/qv_transfer_heatmap.py)
```
uv run --active python code/visualizations/vision/ilharco_open_clip/001_qat_transfer/qv_transfer_heatmap.py --model-name ViT-B-32 --pretrained openai --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules classification_head --qv-alpha 1.0
```

### qv_transfer_heatmap_best_sf — [code/visualizations/vision/ilharco_open_clip/001_qat_transfer/qv_transfer_heatmap_best_sf.py](code/visualizations/vision/ilharco_open_clip/001_qat_transfer/qv_transfer_heatmap_best_sf.py)
```
uv run --active python code/visualizations/vision/ilharco_open_clip/001_qat_transfer/qv_transfer_heatmap_best_sf.py --model-name ViT-B-32 --pretrained openai --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules classification_head --qv-alpha 1.0
```

Local sequential loop over `qv.alpha` values:
```
for alpha in 0.25 0.5 0.75 1.0; do uv run --active python code/visualizations/vision/ilharco_open_clip/001_qat_transfer/qv_transfer_heatmap.py --model-name ViT-B-32 --pretrained openai --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules classification_head --qv-alpha $alpha; done
```

---

# ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head

A specialised variant of `ilharco_timm_supervised` that keeps biases, patch embeddings, layer norms, and the classification head frozen during finetuning. Only FP finetuning is available (no QAT).

## Finetuning

### FP finetuning — [code/src/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/finetune_fp.py](code/src/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/finetune_fp.py)
Config: [config/src/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/finetune_fp.yaml](config/src/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/finetune_fp.yaml)

Single local run:
```
uv run --active python code/src/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/finetune_fp.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=CIFAR10 seed=2038 gpu=0
```

Local sequential sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/finetune_fp.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 gpu=0
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/src/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/finetune_fp.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet seed=1,2,3 gpu=0
```

---

## 000_baselines — evaluation scripts

Scripts live in [code/experiments/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/000_baselines/](code/experiments/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/000_baselines/) with matching configs in [config/experiments/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/000_baselines/](config/experiments/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/000_baselines/).

### evaluate_fp_ptq — [code/experiments/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/000_baselines/evaluate_fp_ptq.py](code/experiments/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/000_baselines/evaluate_fp_ptq.py)
Config: [config/experiments/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/000_baselines/evaluate_fp_ptq.yaml](config/experiments/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/000_baselines/evaluate_fp_ptq.yaml). Adds a `ptq` group.

Single local run:
```
uv run --active python code/experiments/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/000_baselines/evaluate_fp_ptq.py model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=CIFAR10 batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=2038 gpu=0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[head]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/000_baselines/evaluate_fp_ptq.py -m model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[head]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/000_baselines/evaluate_fp_ptq.py -m hydra/launcher=submitit_slurm model_name=vit_base_patch16_clip_224.openai_ft_in12k_in1k dataset_name=Cars,DTD,EuroSAT,GTSRB,MNIST,RESISC45,SUN397,SVHN,CIFAR10,CIFAR100,STL10,Food101,Flowers102,FER2013,PCAM,OxfordIIITPet,RenderedSST2,EMNIST,FashionMNIST,KMNIST,TinyImageNet,ImageNet batch_size=128 lr=1e-5 wd=0.1 ls=0.0 wl=500 max_grad_norm=1.0 seed=1,2,3 gpu=0 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[head]'
```

---

# ilharco_automodelforsequenceclassification

Text classification models using HuggingFace `AutoModelForSequenceClassification`.

Key differences from the vision families:
- `batch_size` defaults to **32** (not 128)
- `max_length=128` replaces `wl` (warmup length)
- `skip_modules` is `classifier` (not `classification_head` or `head`)
- Optim path fragment uses `_ml={max_length}` instead of `_wl={wl}`

## Finetuning

### FP finetuning — [code/src/text/ilharco_automodelforsequenceclassification/finetune_fp.py](code/src/text/ilharco_automodelforsequenceclassification/finetune_fp.py)
Config: [config/src/text/ilharco_automodelforsequenceclassification/finetune_fp.yaml](config/src/text/ilharco_automodelforsequenceclassification/finetune_fp.yaml)

Single local run:
```
uv run --active python code/src/text/ilharco_automodelforsequenceclassification/finetune_fp.py model_name=google-bert/bert-base-uncased dataset_name=Emotion seed=2038 gpu=0
```

Local sequential sweep (all datasets):
```
uv run --active python code/src/text/ilharco_automodelforsequenceclassification/finetune_fp.py -m model_name=google-bert/bert-base-uncased dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction seed=1,2,3 gpu=0
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/src/text/ilharco_automodelforsequenceclassification/finetune_fp.py -m hydra/launcher=submitit_slurm model_name=google-bert/bert-base-uncased dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction seed=1,2,3 gpu=0
```

### QAT finetuning — [code/src/text/ilharco_automodelforsequenceclassification/finetune_qat.py](code/src/text/ilharco_automodelforsequenceclassification/finetune_qat.py)
Config: [config/src/text/ilharco_automodelforsequenceclassification/finetune_qat.yaml](config/src/text/ilharco_automodelforsequenceclassification/finetune_qat.yaml). Requires `qat.bits`, `qat.granularity`, `qat.skip_modules`.

Single local run:
```
uv run --active python code/src/text/ilharco_automodelforsequenceclassification/finetune_qat.py model_name=google-bert/bert-base-uncased dataset_name=Emotion seed=2038 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classifier]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/src/text/ilharco_automodelforsequenceclassification/finetune_qat.py -m model_name=google-bert/bert-base-uncased dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classifier]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/src/text/ilharco_automodelforsequenceclassification/finetune_qat.py -m hydra/launcher=submitit_slurm model_name=google-bert/bert-base-uncased dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classifier]'
```

---

## 000_baselines — evaluation scripts

All scripts live in [code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/](code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/) with matching configs in [config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/](config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/).

### evaluate_pretrained — [code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained.py](code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained.py)
Config: [config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained.yaml](config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained.yaml). Requires training hyperparameters (`lr`, `wd`, `ls`, `max_grad_norm`) to locate the finetuned head checkpoint.

Single local run:
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained.py model_name=google-bert/bert-base-uncased dataset_name=Emotion batch_size=32 max_length=128 lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 seed=2038 gpu=0
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained.py -m model_name=google-bert/bert-base-uncased dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction batch_size=32 max_length=128 lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 seed=1,2,3 gpu=0
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained.py -m hydra/launcher=submitit_slurm model_name=google-bert/bert-base-uncased dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction batch_size=32 max_length=128 lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 seed=1,2,3 gpu=0
```

### evaluate_pretrained_ptq — [code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained_ptq.py](code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained_ptq.py)
Config: [config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained_ptq.yaml](config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained_ptq.yaml). Evaluates the pretrained backbone after applying PTQ. Has both `qat.*` (to locate the QAT-trained head checkpoint) and `ptq.*` groups.

Single local run:
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained_ptq.py model_name=google-bert/bert-base-uncased dataset_name=Emotion batch_size=32 max_length=128 lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 seed=2038 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classifier]' ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classifier]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained_ptq.py -m model_name=google-bert/bert-base-uncased dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction batch_size=32 max_length=128 lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classifier]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classifier]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained_ptq.py -m hydra/launcher=submitit_slurm model_name=google-bert/bert-base-uncased dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction batch_size=32 max_length=128 lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classifier]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classifier]'
```

### evaluate_fp — [code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp.py](code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp.py)
Config: [config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp.yaml](config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp.yaml)

Single local run:
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp.py model_name=google-bert/bert-base-uncased dataset_name=Emotion batch_size=32 lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 max_length=128 seed=2038 gpu=0
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp.py -m model_name=google-bert/bert-base-uncased dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction batch_size=32 lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 max_length=128 seed=1,2,3 gpu=0
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp.py -m hydra/launcher=submitit_slurm model_name=google-bert/bert-base-uncased dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction batch_size=32 lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 max_length=128 seed=1,2,3 gpu=0
```

### evaluate_fp_ptq — [code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp_ptq.py](code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp_ptq.py)
Config: [config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp_ptq.yaml](config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp_ptq.yaml). Adds a `ptq` group.

Single local run:
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp_ptq.py model_name=google-bert/bert-base-uncased dataset_name=Emotion batch_size=32 lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 max_length=128 seed=2038 gpu=0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classifier]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp_ptq.py -m model_name=google-bert/bert-base-uncased dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction batch_size=32 lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 max_length=128 seed=1,2,3 gpu=0 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classifier]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp_ptq.py -m hydra/launcher=submitit_slurm model_name=google-bert/bert-base-uncased dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction batch_size=32 lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 max_length=128 seed=1,2,3 gpu=0 ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classifier]'
```

### evaluate_qat — [code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat.py](code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat.py)
Config: [config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat.yaml](config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat.yaml). The `qat.*` group must match the training config of the checkpoint being evaluated.

Single local run:
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat.py model_name=google-bert/bert-base-uncased dataset_name=Emotion batch_size=32 lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 max_length=128 seed=2038 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classifier]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat.py -m model_name=google-bert/bert-base-uncased dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction batch_size=32 lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 max_length=128 seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classifier]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat.py -m hydra/launcher=submitit_slurm model_name=google-bert/bert-base-uncased dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction batch_size=32 lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 max_length=128 seed=1,2,3 gpu=0 qat.bits=2,4,8 qat.granularity=tensor,channel 'qat.skip_modules=[classifier]'
```

### evaluate_qat_ptq — [code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat_ptq.py](code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat_ptq.py)
Config: [config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat_ptq.yaml](config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat_ptq.yaml). Has both a `qat` group (to locate the checkpoint) and a `ptq` group (for the PTQ applied at eval time).

Single local run:
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat_ptq.py model_name=google-bert/bert-base-uncased dataset_name=Emotion batch_size=32 lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 max_length=128 seed=2038 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classifier]' ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classifier]'
```

Local sequential sweep (all datasets):
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat_ptq.py -m model_name=google-bert/bert-base-uncased dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction batch_size=32 lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 max_length=128 seed=1,2,3 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classifier]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classifier]'
```

Submitit parallel sweep (all datasets):
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat_ptq.py -m hydra/launcher=submitit_slurm model_name=google-bert/bert-base-uncased dataset_name=Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction batch_size=32 lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 max_length=128 seed=1,2,3 gpu=0 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classifier]' ptq.bits=2,4,8 ptq.granularity=tensor,channel 'ptq.skip_modules=[classifier]'
```

---

## 001_qat_transfer — QV transfer experiment

### qv_transfer — [code/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer.py](code/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer.py)
Config: [config/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer.yaml](config/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer.yaml)

Same concept as the vision variants. Uses `source.dataset_names` and `target.dataset_names` (plural — lists of datasets iterated internally by the script). Requires `eval_split` (`val` or `test`) and `max_length`.

Single local run:
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer.py model_name=google-bert/bert-base-uncased batch_size=32 max_length=128 eval_split=val lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 'source.dataset_names=[Emotion]' source.seed=2038 'target.dataset_names=[IMDB]' target.seed=2038 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classifier]' qv.alpha=1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classifier]' gpu=0
```

Local sequential sweep (all source x all target datasets):
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer.py -m model_name=google-bert/bert-base-uncased batch_size=32 max_length=128 eval_split=val lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 'source.dataset_names=[Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction]' source.seed=2038 'target.dataset_names=[Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction]' target.seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classifier]' qv.alpha=0.25,0.5,0.75,1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classifier]' gpu=0
```

Submitit parallel sweep (all source x all target datasets):
```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer.py -m hydra/launcher=submitit_slurm model_name=google-bert/bert-base-uncased batch_size=32 max_length=128 eval_split=val lr=1e-5 wd=0.01 ls=0.0 max_grad_norm=1.0 'source.dataset_names=[Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction]' source.seed=2038 'target.dataset_names=[Emotion,IMDB,Banking77,AmazonReviewsClassification,AmazonCounterfactual,MassiveIntent,MassiveScenario,MTOPDomain,MTOPIntent,ToxicConversations,TweetSentimentExtraction]' target.seed=1,2,3 qat.bits=3 qat.granularity=channel 'qat.skip_modules=[classifier]' qv.alpha=0.25,0.5,0.75,1.0 ptq.bits=3 ptq.granularity=channel 'ptq.skip_modules=[classifier]' gpu=0
```

### pick_best_alpha — [code/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/pick_best_alpha.py](code/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/pick_best_alpha.py)

Same as the vision variants but uses `--max-length` instead of `--wl`. Restricted alpha sweep: `(0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00, 1.05, 1.20, 1.35, 1.50)`. Output modes: `table`, `json`, `commands`, `commands-bg`, `commands-sbatch`.

```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/pick_best_alpha.py --model-name google-bert/bert-base-uncased --seed 2038 --lr 1e-5 --wd 0.01 --ls 0.0 --max-grad-norm 1.0 --batch-size 32 --max-length 128 --bits 4 --granularity channel --skip-modules classifier --slurm-timeout 120 --slurm-job-name pick_best --output table
```

---

## Visualizations (argparse, no Hydra, no Slurm)

### qv_transfer_heatmap — [code/visualizations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer_heatmap.py](code/visualizations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer_heatmap.py)
```
uv run --active python code/visualizations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer_heatmap.py --model-name google-bert/bert-base-uncased --seed 2038 --optim adamw --lr 1e-5 --wd 0.01 --ls 0.0 --max-grad-norm 1.0 --batch-size 32 --max-length 128 --bits 4 --granularity channel --skip-modules classifier --qv-alpha 1.0
```

### qv_transfer_heatmap_best_sf — [code/visualizations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer_heatmap_best_sf.py](code/visualizations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer_heatmap_best_sf.py)
```
uv run --active python code/visualizations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer_heatmap_best_sf.py --model-name google-bert/bert-base-uncased --seed 2038 --optim adamw --lr 1e-5 --wd 0.01 --ls 0.0 --max-grad-norm 1.0 --batch-size 32 --max-length 128 --bits 4 --granularity channel --skip-modules classifier --qv-alpha 1.0
```

Local sequential loop over `qv.alpha` values:
```
for alpha in 0.25 0.5 0.75 1.0; do uv run --active python code/visualizations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer_heatmap.py --model-name google-bert/bert-base-uncased --seed 2038 --optim adamw --lr 1e-5 --wd 0.01 --ls 0.0 --max-grad-norm 1.0 --batch-size 32 --max-length 128 --bits 4 --granularity channel --skip-modules classifier --qv-alpha $alpha; done
```

---

# Notes

- **Always launch from the repo root.** Hydra config search paths use `${oc.env:PWD}` and will not resolve otherwise.
- **List-valued overrides** (e.g. `skip_modules`) must be quoted to protect brackets from the shell: `'qat.skip_modules=[classification_head]'`.
- **Tuning the Slurm job** without editing YAML: append `hydra.launcher.timeout_min=...`, `hydra.launcher.mem_gb=...`, `hydra.launcher.cpus_per_task=...`, etc.
- **Output layout**: single runs land under `hydra.run.dir`; sweeps (both local and submitit) land under `hydra.sweep.dir` with per-combo `subdir`s defined in each experiment's YAML. Submitit additionally writes per-job metadata under `${hydra.sweep.dir}/.submitit/%j`.
- **Environment variables**: see the [Getting Started](#getting-started) section for the full list and their purposes.
- **Text vs vision parameter differences**: text scripts use `max_length` (no `wl`), default `batch_size=32`, and `skip_modules=classifier`. Vision scripts use `wl` (warmup length), default `batch_size=128`, and `skip_modules` varies by family (`classification_head` for hf_clip/open_clip, `head` for timm).

---

# Citation

If you use this code, please cite:

```bibtex
@inproceedings{solombrino2026zeroshot,
  title     = {Zero-Shot Quantization via Weight-Space Arithmetic},
  author    = {Solombrino, Daniele and Gargiulo, Antonio Andrea and Zirilli, Alessandro and Zhou, Luca and Minut, Adrian Robert and Rodol{\`a}, Emanuele},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/abs/2604.03420}
}
```

# License

Released under the [MIT License](LICENSE).
