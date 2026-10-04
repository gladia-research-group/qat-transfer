# qat-transfer

Official code for **Zero-Shot Quantization via Weight-Space Arithmetic** (NeurIPS 2026).

Daniele Solombrino, Antonio Andrea Gargiulo, Alessandro Zirilli, Luca Zhou, Adrian Robert Minut, Emanuele Rodolà

[arXiv](https://arxiv.org/abs/2604.03420) · [OpenReview](https://openreview.net/forum?id=wrUEnnSgXa) · [Research graph](https://flywheel.paradigma.inc/node/de486ca0-8e83-4041-bbf6-0b0be565435b)

> We show that robustness to post-training quantization (PTQ) is a transferable direction in weight space. We call this direction the quantization vector: extracted from a donor task by simple weight-space arithmetic, it can be used to patch a receiver model and improve post-PTQ Top-1 accuracy by up to 60 points in a 3-bit setting, without receiver-side quantization-aware training (QAT). Because the method requires no receiver training data, it provides a zero-shot, low-cost alternative to QAT for extremely low-bit deployment. Across four ViT scales and 22 image classification tasks, donor quantization vectors often yield substantial gains even when donor and receiver tasks differ markedly. We further prove rigorously that quantization vectors are well-defined and do not suffer from reparameterization symmetries, and provide a local geometric account of their effect. Together, these results suggest that quantization robustness can be partially isolated, reused, and transferred through simple weight-space algebra.

## Method in one line

Let `FP_D` be a full-precision finetune on task `D` and `QAT_D` a quantization-aware finetune on the same task. The **quantization vector** is their difference, `QV = QAT_D - FP_D`. The central claim is that this vector is largely task-agnostic: it can be computed once on a **donor** task and added to a different **receiver** task's FP checkpoint,

```
patched = FP_receiver + lambda * QV_donor
```

recovering much of the receiver's QAT benefit under post-training quantization (PTQ) without ever running QAT on the receiver. With `lambda = 1` this needs no receiver data at all.

The repo covers vision (timm supervised ViT/DeiT/Swin, OpenCLIP, HF CLIP) and text (`AutoModelForSequenceClassification`) backbones, and includes Lean formalizations of the paper's propositions under [`proofs/`](proofs/).

## Data and checkpoints

Datasets are downloaded through HuggingFace / torchvision into the caches configured in `.env`. Evaluation outputs (`evaluations/`) are not included; they are produced by the finetuning, `000_baselines` and `001_qat_transfer` pipelines described in [`code/experiments/README.md`](code/experiments/README.md), and are what the analysis (`code/experiments/998_rebuttal/`) and plotting (`code/visualizations/`, `visualizations/`) scripts read.

### Pretrained checkpoints

The FP and 3-bit QAT finetunes behind the paper's results are on the HuggingFace Hub, one repository per model family:

| Repository | Backbones | Size |
|---|---|---|
| [`gladia/qat-transfer-timm`](https://huggingface.co/gladia/qat-transfer-timm) | DeiT-III B/L, Swin B/L, ViT B/L/H (`orig_in21k`) on 22 vision tasks, plus 5 of the 22 PV-Tuning donors for ViT-B (see below) | 302 GB |
| [`gladia/qat-transfer-open_clip`](https://huggingface.co/gladia/qat-transfer-open_clip) | OpenCLIP ViT-B/16, L/14, H/14 (LAION-2B) on 22 vision tasks, with their zero-shot heads | 180 GB |
| [`gladia/qat-transfer-text`](https://huggingface.co/gladia/qat-transfer-text) | BERT-base/large, EmbeddingGemma-300M, Qwen3-Embedding-0.6B on 11 text tasks | 118 GB |

Each repository mirrors the `storage/` layout the code expects, so downloading into `storage/` is enough to run every transfer and evaluation script without retraining:

```
uv run hf download gladia/qat-transfer-timm --local-dir storage
```

Add `--include "*/vit_base_patch16_224_orig_in21k/*"` (or any other sanitized model name) to fetch a single backbone. Then set `CHECKPOINT_BASE_PATH=storage/checkpoints` and `HEAD_BASE_PATH=storage/heads` in `.env`. Each repository's model card documents its contents, configuration and license; the text repository's EmbeddingGemma finetunes are distributed under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).

The PV-Tuning donors of `008_pv_transfer` are incomplete: 17 of the 22 checkpoints were lost to a storage failure after the experiment ran, and only Cars, EuroSAT, Flowers102, RESISC45 and STL10 are released. The FP and QAT checkpoints are complete, and the missing PV donors can be regenerated with [`finetune_pv.py`](code/src/vision/ilharco_timm_supervised/finetune_pv.py) using the configuration in [`config/src/vision/ilharco_timm_supervised/finetune_pv.yaml`](config/src/vision/ilharco_timm_supervised/finetune_pv.yaml) (`delta=0.0`, `tau=0.01`, seed 2038).

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

## Research graph

The questions behind each experiment phase, how they depend on one another and on the paper's propositions, and the scripts that answer them are laid out as a public [Flywheel graph](https://flywheel.paradigma.inc/node/de486ca0-8e83-4041-bbf6-0b0be565435b). Each node names its question, its method and the code under `code/` that implements it.

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

The full list of supported vision datasets (used in every vision sweep example in [`code/experiments/README.md`](code/experiments/README.md)) is defined in [code/src/vision/data/common.py](code/src/vision/data/common.py#L35):

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
    experiments/      # Evaluation, transfer and analysis scripts (command reference: code/experiments/README.md)
    test/             # Unit and smoke tests
    visualizations/   # Argparse plotting scripts
  config/             # Hydra YAML configs — mirrors code/ 1:1
  visualizations/     # Standalone plotting scripts for the 998_rebuttal QV-alignment analyses
  proofs/             # Lean formalizations of the paper's propositions
  evaluations/        # eval_results.json outputs         (created at runtime, not in git)
  storage/            # Checkpoints and heads             (download from the Hub or create by training; not in git)
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
{CHECKPOINT_BASE_PATH}/vision/{family}/{fp,qat}/{sanitized_model}/{dataset}/optim=adamw_lr={lr}_wd={wd}_ls={ls}_wl={wl}_mgn={max_grad_norm}_bs={batch_size}/mult={m}/[qat=bits={bits}_gran={granularity}_skip={skip_tag}/]seed={seed}/{classifier_epoch_{N}.pt, head_epoch_{N}.pt}
```

The timm families save the full classifier (`classifier_epoch_{N}.pt`) plus its head; the CLIP families (`ilharco_open_clip`, `ilharco_hf_clip`) save the image encoder alone as `epoch_{N}.pt`, with the zero-shot head under `{HEAD_BASE_PATH}/vision/{family}/{sanitized_model}/head_{dataset}.pt`.

**Checkpoint paths** (text): same structure but the optim fragment uses `_ml={max_length}` instead of `_wl={wl}`:
```
{CHECKPOINT_BASE_PATH}/text/{family}/{fp,qat}/{sanitized_model}/{dataset}/optim=adamw_lr={lr}_wd={wd}_ls={ls}_mgn={max_grad_norm}_bs={batch_size}_ml={max_length}/mult={m}/[qat=bits={bits}_gran={granularity}_skip={skip_tag}/]seed={seed}/{backbone_epoch_{N}.pt, head_epoch_{N}.pt}
```

**Evaluation paths**:
```
{EVALUATION_BASE_PATH}/{vision,text}/{family}/{phase}/{experiment_type}/{sanitized_model}/{dataset}/optim=.../mult={m}/{seed}/eval_results.json
```

---

## Running the experiments

The full command reference (finetuning, baselines, QV transfer and plotting, for every model family, with single-run, local-sweep and Slurm variants) lives in [`code/experiments/README.md`](code/experiments/README.md). A minimal end-to-end run on one donor/receiver pair with the timm family:

```
uv run --active python code/src/vision/ilharco_timm_supervised/finetune_fp.py model_name=vit_base_patch16_224.orig_in21k dataset_name=CIFAR10 seed=2038 gpu=0
```

then the matching `finetune_qat.py` on the donor, `finetune_fp.py` on the receiver, and `001_qat_transfer/qv_transfer.py` on the pair, exactly as listed there.

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
# Citation

If you use this code, please cite:

```bibtex
@inproceedings{solombrino2026zeroshot,
  title     = {Zero-Shot Quantization via Weight-Space Arithmetic},
  author    = {Solombrino, Daniele and Gargiulo, Antonio Andrea and Zirilli, Alessandro and Zhou, Luca and Minut, Adrian Robert and Rodol{\`a}, Emanuele},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=wrUEnnSgXa}
}
```

# License

Released under the [MIT License](LICENSE).
