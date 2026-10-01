# Running the experiments

Per-script command reference for every pipeline in the paper: finetuning, `000_baselines` evaluation, `001_qat_transfer` and the plotting scripts, organized by model family. For the method, setup, environment variables, datasets and output path layouts, see the [main README](../../README.md).

All commands assume the **repo root** as the working directory (not this directory): Hydra scripts resolve their config search paths from `${oc.env:PWD}`. Every path below is written relative to the repo root.

Model families:

- [`ilharco_hf_clip`](#ilharco_hf_clip) — HF CLIP vision encoders
- [`ilharco_timm_supervised`](#ilharco_timm_supervised) — timm supervised ViT / DeiT / Swin
- [`ilharco_open_clip`](#ilharco_open_clip) — OpenCLIP vision encoders
- [`ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head`](#ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head) — ablation of the timm family
- [`ilharco_automodelforsequenceclassification`](#ilharco_automodelforsequenceclassification) — text classifiers

Cross-family analyses live in [`998_rebuttal/`](998_rebuttal/); they read the `evaluations/` tree produced by the pipelines below.

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
The launcher config lives at [config/hydra/launcher/submitit_slurm.yaml](../../config/hydra/launcher/submitit_slurm.yaml) (GPUs, CPUs, mem, walltime; partition and account come from `SBATCH_PARTITION` / `SBATCH_ACCOUNT`). Override any field on the CLI, e.g. `hydra.launcher.timeout_min=240 hydra.launcher.mem_gb=64`.

> Modes 2 and 3 both use `-m`. The **only** difference is whether you add `hydra/launcher=submitit_slurm`. Without it, Hydra's basic launcher runs the sweep sequentially in-process; with it, submitit fans the sweep out to Slurm.

---

# ilharco_hf_clip

## Finetuning

### FP finetuning — [code/src/vision/ilharco_hf_clip/finetune_fp.py](../../code/src/vision/ilharco_hf_clip/finetune_fp.py)
Config: [config/src/vision/ilharco_hf_clip/finetune_fp.yaml](../../config/src/vision/ilharco_hf_clip/finetune_fp.yaml)

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

### QAT finetuning — [code/src/vision/ilharco_hf_clip/finetune_qat.py](../../code/src/vision/ilharco_hf_clip/finetune_qat.py)
Config: [config/src/vision/ilharco_hf_clip/finetune_qat.yaml](../../config/src/vision/ilharco_hf_clip/finetune_qat.yaml). Requires `qat.bits`, `qat.granularity`, `qat.skip_modules`.

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

> `finetune_*` scripts are GPU-intensive and long-running: each run needs a GPU and the full dataset, and a full sweep costs many GPU-hours.

---

## 000_baselines — evaluation scripts

All scripts live in [code/experiments/vision/ilharco_hf_clip/000_baselines/](../../code/experiments/vision/ilharco_hf_clip/000_baselines/) with matching configs in [config/experiments/vision/ilharco_hf_clip/000_baselines/](../../config/experiments/vision/ilharco_hf_clip/000_baselines/).

### evaluate_pretrained — [code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_pretrained.py](../../code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_pretrained.py)
Config: [config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_pretrained.yaml](../../config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_pretrained.yaml)

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

### evaluate_fp — [code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp.py](../../code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp.py)
Config: [config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp.yaml](../../config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp.yaml)

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

### evaluate_fp_ptq — [code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq.py](../../code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq.py)
Config: [config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq.yaml](../../config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq.yaml). Adds a `ptq` group.

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

### evaluate_fp_ptq_bias_norm_emb_from_pt — [code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq_bias_norm_emb_from_pt.py](../../code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq_bias_norm_emb_from_pt.py)
Config: [config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq_bias_norm_emb_from_pt.yaml](../../config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_fp_ptq_bias_norm_emb_from_pt.yaml). Same parameters as `evaluate_fp_ptq`. Before applying PTQ, swaps biases, layer-norm weights, and embedding weights back to pretrained values.

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

### evaluate_qat — [code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat.py](../../code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat.py)
Config: [config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat.yaml](../../config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat.yaml). The `qat.*` group must match the training config of the checkpoint being evaluated.

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

### evaluate_qat_ptq — [code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat_ptq.py](../../code/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat_ptq.py)
Config: [config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat_ptq.yaml](../../config/experiments/vision/ilharco_hf_clip/000_baselines/evaluate_qat_ptq.yaml). Has both a `qat` group (to locate the checkpoint) and a `ptq` group (for the PTQ applied at eval time).

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

### qv_transfer — [code/experiments/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer.py](../../code/experiments/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer.py)
Config: [config/experiments/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer.yaml](../../config/experiments/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer.yaml)

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

### qv_transfer_heatmap — [code/visualizations/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer_heatmap.py](../../code/visualizations/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer_heatmap.py)
```
uv run --active python code/visualizations/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer_heatmap.py --model-name openai/clip-vit-base-patch16 --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules classification_head --qv-alpha 1.0
```

### qv_transfer_heatmap_best_sf — [code/visualizations/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer_heatmap_best_sf.py](../../code/visualizations/vision/ilharco_hf_clip/001_qat_transfer/qv_transfer_heatmap_best_sf.py)
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

### FP finetuning — [code/src/vision/ilharco_timm_supervised/finetune_fp.py](../../code/src/vision/ilharco_timm_supervised/finetune_fp.py)
Config: [config/src/vision/ilharco_timm_supervised/finetune_fp.yaml](../../config/src/vision/ilharco_timm_supervised/finetune_fp.yaml)

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

### QAT finetuning — [code/src/vision/ilharco_timm_supervised/finetune_qat.py](../../code/src/vision/ilharco_timm_supervised/finetune_qat.py)
Config: [config/src/vision/ilharco_timm_supervised/finetune_qat.yaml](../../config/src/vision/ilharco_timm_supervised/finetune_qat.yaml). Requires `qat.bits`, `qat.granularity`, `qat.skip_modules`.

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

All scripts live in [code/experiments/vision/ilharco_timm_supervised/000_baselines/](../../code/experiments/vision/ilharco_timm_supervised/000_baselines/) with matching configs in [config/experiments/vision/ilharco_timm_supervised/000_baselines/](../../config/experiments/vision/ilharco_timm_supervised/000_baselines/).

### evaluate_pretrained — [code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained.py](../../code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained.py)
Config: [config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained.yaml](../../config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained.yaml). Unlike the hf_clip variant, requires training hyperparameters (`lr`, `wd`, `ls`, `wl`, `max_grad_norm`) to locate the finetuned head checkpoint.

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

### evaluate_pretrained_ptq — [code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained_ptq.py](../../code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained_ptq.py)
Config: [config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained_ptq.yaml](../../config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_pretrained_ptq.yaml). Evaluates the pretrained backbone after applying PTQ. Has both `qat.*` (to locate the QAT-trained head checkpoint) and `ptq.*` groups.

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

### evaluate_fp — [code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp.py](../../code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp.py)
Config: [config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp.yaml](../../config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp.yaml)

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

### evaluate_fp_ptq — [code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_ptq.py](../../code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_ptq.py)
Config: [config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_ptq.yaml](../../config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_fp_ptq.yaml). Adds a `ptq` group.

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

### evaluate_qat — [code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat.py](../../code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat.py)
Config: [config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat.yaml](../../config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat.yaml). The `qat.*` group must match the training config of the checkpoint being evaluated.

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

### evaluate_qat_ptq — [code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat_ptq.py](../../code/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat_ptq.py)
Config: [config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat_ptq.yaml](../../config/experiments/vision/ilharco_timm_supervised/000_baselines/evaluate_qat_ptq.yaml). Has both a `qat` group (to locate the checkpoint) and a `ptq` group (for the PTQ applied at eval time).

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

### qv_transfer — [code/experiments/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer.py](../../code/experiments/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer.py)
Config: [config/experiments/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer.yaml](../../config/experiments/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer.yaml)

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

### pick_best_alpha — [code/experiments/vision/ilharco_timm_supervised/001_qat_transfer/pick_best_alpha.py](../../code/experiments/vision/ilharco_timm_supervised/001_qat_transfer/pick_best_alpha.py)

Reads val-split `eval_results.json` files produced by `qv_transfer.py`, finds the alpha that maximises accuracy for each `(source_dataset, target_dataset)` pair. Only considers the restricted alpha sweep: `(0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00, 1.05, 1.20, 1.35, 1.50)`.

Output modes: `table` (markdown), `json`, `commands` (local), `commands-bg` (background), `commands-sbatch` (Slurm).

```
uv run --active python code/experiments/vision/ilharco_timm_supervised/001_qat_transfer/pick_best_alpha.py --model-name vit_base_patch16_clip_224.openai_ft_in12k_in1k --seed 2038 --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules head --slurm-timeout 120 --slurm-job-name pick_best --output table
```

---

## Visualizations (argparse, no Hydra, no Slurm)

### qv_transfer_heatmap — [code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap.py](../../code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap.py)
```
uv run --active python code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap.py --model-name vit_base_patch16_clip_224.openai_ft_in12k_in1k --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules head --qv-alpha 1.0
```

### qv_transfer_heatmap_best_sf — [code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap_best_sf.py](../../code/visualizations/vision/ilharco_timm_supervised/001_qat_transfer/qv_transfer_heatmap_best_sf.py)
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

### FP finetuning — [code/src/vision/ilharco_open_clip/finetune_fp.py](../../code/src/vision/ilharco_open_clip/finetune_fp.py)
Config: [config/src/vision/ilharco_open_clip/finetune_fp.yaml](../../config/src/vision/ilharco_open_clip/finetune_fp.yaml)

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

### QAT finetuning — [code/src/vision/ilharco_open_clip/finetune_qat.py](../../code/src/vision/ilharco_open_clip/finetune_qat.py)
Config: [config/src/vision/ilharco_open_clip/finetune_qat.yaml](../../config/src/vision/ilharco_open_clip/finetune_qat.yaml). Requires `qat.bits`, `qat.granularity`, `qat.skip_modules`.

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

All scripts live in [code/experiments/vision/ilharco_open_clip/000_baselines/](../../code/experiments/vision/ilharco_open_clip/000_baselines/) with matching configs in [config/experiments/vision/ilharco_open_clip/000_baselines/](../../config/experiments/vision/ilharco_open_clip/000_baselines/).

### evaluate_pretrained — [code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_pretrained.py](../../code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_pretrained.py)
Config: [config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_pretrained.yaml](../../config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_pretrained.yaml)

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

### evaluate_fp — [code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp.py](../../code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp.py)
Config: [config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp.yaml](../../config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp.yaml)

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

### evaluate_fp_ptq — [code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp_ptq.py](../../code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp_ptq.py)
Config: [config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp_ptq.yaml](../../config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_fp_ptq.yaml). Adds a `ptq` group.

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

### evaluate_qat — [code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat.py](../../code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat.py)
Config: [config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat.yaml](../../config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat.yaml). The `qat.*` group must match the training config of the checkpoint being evaluated.

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

### evaluate_qat_ptq — [code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat_ptq.py](../../code/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat_ptq.py)
Config: [config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat_ptq.yaml](../../config/experiments/vision/ilharco_open_clip/000_baselines/evaluate_qat_ptq.yaml). Has both a `qat` group (to locate the checkpoint) and a `ptq` group (for the PTQ applied at eval time).

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

### qv_transfer — [code/experiments/vision/ilharco_open_clip/001_qat_transfer/qv_transfer.py](../../code/experiments/vision/ilharco_open_clip/001_qat_transfer/qv_transfer.py)
Config: [config/experiments/vision/ilharco_open_clip/001_qat_transfer/qv_transfer.yaml](../../config/experiments/vision/ilharco_open_clip/001_qat_transfer/qv_transfer.yaml)

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

### pick_best_alpha — [code/experiments/vision/ilharco_open_clip/001_qat_transfer/pick_best_alpha.py](../../code/experiments/vision/ilharco_open_clip/001_qat_transfer/pick_best_alpha.py)

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

### qv_transfer_heatmap — [code/visualizations/vision/ilharco_open_clip/001_qat_transfer/qv_transfer_heatmap.py](../../code/visualizations/vision/ilharco_open_clip/001_qat_transfer/qv_transfer_heatmap.py)
```
uv run --active python code/visualizations/vision/ilharco_open_clip/001_qat_transfer/qv_transfer_heatmap.py --model-name ViT-B-32 --pretrained openai --seed 2038 --optim adamw --lr 1e-5 --wd 0.1 --ls 0.0 --wl 500 --max-grad-norm 1.0 --batch-size 128 --bits 4 --granularity channel --skip-modules classification_head --qv-alpha 1.0
```

### qv_transfer_heatmap_best_sf — [code/visualizations/vision/ilharco_open_clip/001_qat_transfer/qv_transfer_heatmap_best_sf.py](../../code/visualizations/vision/ilharco_open_clip/001_qat_transfer/qv_transfer_heatmap_best_sf.py)
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

### FP finetuning — [code/src/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/finetune_fp.py](../../code/src/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/finetune_fp.py)
Config: [config/src/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/finetune_fp.yaml](../../config/src/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/finetune_fp.yaml)

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

Scripts live in [code/experiments/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/000_baselines/](../../code/experiments/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/000_baselines/) with matching configs in [config/experiments/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/000_baselines/](../../config/experiments/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/000_baselines/).

### evaluate_fp_ptq — [code/experiments/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/000_baselines/evaluate_fp_ptq.py](../../code/experiments/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/000_baselines/evaluate_fp_ptq.py)
Config: [config/experiments/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/000_baselines/evaluate_fp_ptq.yaml](../../config/experiments/vision/ilharco_timm_supervised_frozen_biases_patch_embeddings_norms_cls_head/000_baselines/evaluate_fp_ptq.yaml). Adds a `ptq` group.

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

### FP finetuning — [code/src/text/ilharco_automodelforsequenceclassification/finetune_fp.py](../../code/src/text/ilharco_automodelforsequenceclassification/finetune_fp.py)
Config: [config/src/text/ilharco_automodelforsequenceclassification/finetune_fp.yaml](../../config/src/text/ilharco_automodelforsequenceclassification/finetune_fp.yaml)

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

### QAT finetuning — [code/src/text/ilharco_automodelforsequenceclassification/finetune_qat.py](../../code/src/text/ilharco_automodelforsequenceclassification/finetune_qat.py)
Config: [config/src/text/ilharco_automodelforsequenceclassification/finetune_qat.yaml](../../config/src/text/ilharco_automodelforsequenceclassification/finetune_qat.yaml). Requires `qat.bits`, `qat.granularity`, `qat.skip_modules`.

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

All scripts live in [code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/](../../code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/) with matching configs in [config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/](../../config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/).

### evaluate_pretrained — [code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained.py](../../code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained.py)
Config: [config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained.yaml](../../config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained.yaml). Requires training hyperparameters (`lr`, `wd`, `ls`, `max_grad_norm`) to locate the finetuned head checkpoint.

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

### evaluate_pretrained_ptq — [code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained_ptq.py](../../code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained_ptq.py)
Config: [config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained_ptq.yaml](../../config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_pretrained_ptq.yaml). Evaluates the pretrained backbone after applying PTQ. Has both `qat.*` (to locate the QAT-trained head checkpoint) and `ptq.*` groups.

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

### evaluate_fp — [code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp.py](../../code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp.py)
Config: [config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp.yaml](../../config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp.yaml)

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

### evaluate_fp_ptq — [code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp_ptq.py](../../code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp_ptq.py)
Config: [config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp_ptq.yaml](../../config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_fp_ptq.yaml). Adds a `ptq` group.

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

### evaluate_qat — [code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat.py](../../code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat.py)
Config: [config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat.yaml](../../config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat.yaml). The `qat.*` group must match the training config of the checkpoint being evaluated.

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

### evaluate_qat_ptq — [code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat_ptq.py](../../code/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat_ptq.py)
Config: [config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat_ptq.yaml](../../config/experiments/text/ilharco_automodelforsequenceclassification/000_baselines/evaluate_qat_ptq.yaml). Has both a `qat` group (to locate the checkpoint) and a `ptq` group (for the PTQ applied at eval time).

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

### qv_transfer — [code/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer.py](../../code/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer.py)
Config: [config/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer.yaml](../../config/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer.yaml)

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

### pick_best_alpha — [code/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/pick_best_alpha.py](../../code/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/pick_best_alpha.py)

Same as the vision variants but uses `--max-length` instead of `--wl`. Restricted alpha sweep: `(0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.00, 1.05, 1.20, 1.35, 1.50)`. Output modes: `table`, `json`, `commands`, `commands-bg`, `commands-sbatch`.

```
uv run --active python code/experiments/text/ilharco_automodelforsequenceclassification/001_qat_transfer/pick_best_alpha.py --model-name google-bert/bert-base-uncased --seed 2038 --lr 1e-5 --wd 0.01 --ls 0.0 --max-grad-norm 1.0 --batch-size 32 --max-length 128 --bits 4 --granularity channel --skip-modules classifier --slurm-timeout 120 --slurm-job-name pick_best --output table
```

---

## Visualizations (argparse, no Hydra, no Slurm)

### qv_transfer_heatmap — [code/visualizations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer_heatmap.py](../../code/visualizations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer_heatmap.py)
```
uv run --active python code/visualizations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer_heatmap.py --model-name google-bert/bert-base-uncased --seed 2038 --optim adamw --lr 1e-5 --wd 0.01 --ls 0.0 --max-grad-norm 1.0 --batch-size 32 --max-length 128 --bits 4 --granularity channel --skip-modules classifier --qv-alpha 1.0
```

### qv_transfer_heatmap_best_sf — [code/visualizations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer_heatmap_best_sf.py](../../code/visualizations/text/ilharco_automodelforsequenceclassification/001_qat_transfer/qv_transfer_heatmap_best_sf.py)
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
- **Environment variables**: see the [Getting Started](../../README.md#getting-started) section for the full list and their purposes.
- **Text vs vision parameter differences**: text scripts use `max_length` (no `wl`), default `batch_size=32`, and `skip_modules=classifier`. Vision scripts use `wl` (warmup length), default `batch_size=128`, and `skip_modules` varies by family (`classification_head` for hf_clip/open_clip, `head` for timm).
