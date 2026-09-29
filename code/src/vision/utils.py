import os
import random

import numpy as np
import torch

SPLIT_SEED = 0
VAL_FRACTION = 0.1
MAX_VAL_SAMPLES = 5000


def random_tqdm_color():
    return f'#{random.randint(0, 0xFFFFFF):06x}'


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0).cpu())
        for k in topk
    ]


def sanitize_hf_model_name(model_name: str) -> str:
    """Make a HuggingFace model id safe for use as a single path component.

    Replaces the org/model separator "/" with "_" so e.g.
    "openai/clip-vit-base-patch32" becomes "openai_clip-vit-base-patch32".
    """
    return model_name.replace("/", "_").replace("-", "_")


def sanitize_open_clip_model_name(model_name: str, pretrained: str) -> str:
    """Make an open_clip (model_name, pretrained) pair safe for use as a single path component.

    Combines model_name and pretrained with "__" separator and replaces
    "/", "-", and "." with "_", so e.g. model_name="ViT-B-32" and
    pretrained="openai" becomes "ViT_B_32__openai".
    """
    combined = f"{model_name}__{pretrained}"
    return combined.replace("/", "_").replace("-", "_").replace(".", "_")


def sanitize_timm_model_name(model_name: str) -> str:
    """Make a timm model id safe for use as a single path component.

    Replaces "/", "-", and "." with "_" so e.g.
    "vit_base_patch16_clip_224.openai" becomes "vit_base_patch16_clip_224_openai".
    """
    return model_name.replace("/", "_").replace("-", "_").replace(".", "_")


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch RNGs, and enable deterministic cuDNN.

    This is the per-run seed used for model init, shuffling, worker state, etc.
    It is NOT the dataset splitting seed — SPLIT_SEED controls that and is kept
    independent on purpose.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn that seeds numpy and python random per worker."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    """Cosine LR schedule with linear warmup. Ported from task_vectors."""
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            _assign_learning_rate(param_group, lr)

    return _lr_adjuster


def linear_lr(optimizer, base_lrs, steps):
    """Linear LR schedule: decays linearly from base_lr to 0 over `steps`."""
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            lr = base_lr * (1 - step / steps)
            _assign_learning_rate(param_group, lr)

    return _lr_adjuster


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
