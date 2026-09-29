import os

import torch
from tqdm import tqdm
from transformers import CLIPTokenizer

from ..data.registry import get_dataset
from ..data.templates import get_templates
from .modeling import ClassificationHead, ImageEncoder
from ..utils import SPLIT_SEED, sanitize_hf_model_name


def build_classification_head(model, tokenizer, dataset_name, device):
    template = get_templates(dataset_name)

    logit_scale = model.logit_scale
    # NOTE: head construction is fully deterministic — the head weights are the
    # CLIP text-encoder embeddings of `template(classname)`, averaged across
    # templates and L2-normalized (see the loop below). We only call
    # get_dataset() here to read `class_names`; the `seed=SPLIT_SEED` is
    # required by the get_dataset signature but has no effect on the resulting
    # head. In particular, the per-run seed (`cfg.seed`) is intentionally NOT
    # threaded in here: a single cached head per (model_name, dataset_name) is
    # correct for *every* run seed of finetune_fp / evaluate_*, and
    # regenerating it per seed would produce a bit-identical file.
    dataset = get_dataset(
        dataset_name,
        preprocess_train=None,
        preprocess_inference=None,
        batch_size=1,
        num_workers=0,
        seed=SPLIT_SEED,
    )
    model.eval()
    model.to(device)

    print("Building classification head.")
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(dataset.class_names):
            texts = [t(classname) for t in template]
            tokens = tokenizer(texts, padding=True, return_tensors="pt").to(device)
            embeddings = model.get_text_features(**tokens)  # embed with text encoder
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

            embeddings = embeddings.mean(dim=0, keepdim=True)
            embeddings /= embeddings.norm()

            zeroshot_weights.append(embeddings)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(device)
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)

        zeroshot_weights *= logit_scale.exp()

        zeroshot_weights = zeroshot_weights.squeeze().float()
        zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

    classification_head = ClassificationHead(normalize=True, weights=zeroshot_weights)

    return classification_head


def get_classification_head(model_name: str, dataset_name: str, save_dir: str, device):
    filename = os.path.join(save_dir, "vision", "ilharco_hf_clip", sanitize_hf_model_name(model_name), f"head_{dataset_name}.pt")
    if os.path.exists(filename):
        print(f"Classification head for {model_name} on {dataset_name} exists at {filename}")
        return ClassificationHead.load(filename)

    print(
        f"Did not find classification head for {model_name} on {dataset_name} at {filename}, "
        f"building one from scratch."
    )
    encoder = ImageEncoder(model_name=model_name, keep_lang=True)
    model = encoder.model
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    classification_head = build_classification_head(model, tokenizer, dataset_name, device)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    classification_head.save(filename)
    return classification_head
