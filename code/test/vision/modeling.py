import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

import torch
from tqdm import tqdm

from src.vision.data.registry import registry, get_dataset
from src.vision.ilharco_hf_clip.modeling import ClassificationHead, ImageClassifier, ImageEncoder
from src.vision.utils import SPLIT_SEED, random_tqdm_color


def test_init(model_name: str, num_classes: int):
    print(f"\n{'=' * 50}")
    print(f"Init test: {model_name}")
    print(f"{'-' * 50}")

    encoder = ImageEncoder(model_name=model_name, keep_lang=False)
    print(f"  ImageEncoder loaded ({type(encoder.model).__name__})")
    assert not hasattr(encoder.model, "text_model"), "text_model should be dropped"
    assert not hasattr(encoder.model, "text_projection"), "text_projection should be dropped"
    print("  keep_lang=False dropped text_model and text_projection")

    embed_dim = encoder.model.config.projection_dim
    weights = torch.zeros(num_classes, embed_dim)
    head = ClassificationHead(normalize=True, weights=weights)
    print(f"  ClassificationHead built (in={embed_dim}, out={num_classes})")

    classifier = ImageClassifier(encoder, head)
    assert classifier.train_preprocess is encoder.train_preprocess
    assert classifier.val_preprocess is encoder.val_preprocess
    print("  ImageClassifier composed; preprocess inherited from encoder")
    print(f"{'=' * 50}")

    return classifier


def test_dataset_forward(
    classifier: ImageClassifier,
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    max_batches: int,
    gpu: int,
):
    assert dataset_name in registry, (
        f"Unsupported dataset: {dataset_name}. Supported: {sorted(registry.keys())}"
    )

    print(f"\n{'=' * 50}")
    print(f"Dataset forward test: {dataset_name}")
    print(f"{'-' * 50}")

    dataset = get_dataset(
        dataset_name=dataset_name,
        preprocess_train=classifier.train_preprocess,
        preprocess_inference=classifier.val_preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=SPLIT_SEED,
    )

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    classifier.eval()

    num_classes = classifier.classification_head.weight.shape[0]
    colour = random_tqdm_color()
    seen = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataset.test_loader, desc=dataset_name, colour=colour)):
            if max_batches > 0 and i >= max_batches:
                break
            images = batch[0] if isinstance(batch, (list, tuple)) else batch["image"]
            images = images.to(device)
            logits = classifier(images)
            assert logits.shape == (images.shape[0], num_classes), (
                f"unexpected logits shape: {tuple(logits.shape)}"
            )
            seen += images.shape[0]

    print(f"  Forward pass OK on {seen} samples ({device})")
    print(f"  Logits shape per batch: (B, {num_classes})")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Smoke-test ImageClassifier init and dataset forward.")

    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32",
                        help="HuggingFace CLIP model id.")
    parser.add_argument("--dataset-name", type=str, nargs="+", required=True,
                        help=f"Dataset class name(s). Options: {sorted(registry.keys())}")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=2,
                        help="Max number of batches per dataset (0 = all).")
    parser.add_argument("--gpu", type=int, required=True,
                        help="CUDA device index.")

    args = parser.parse_args()

    for dataset_name in args.dataset_name:
        # Probe num_classes from the dataset itself so the head matches.
        probe = get_dataset(
            dataset_name=dataset_name,
            preprocess_train=None,
            preprocess_inference=None,
            batch_size=1,
            num_workers=0,
            seed=SPLIT_SEED,
        )
        num_classes = len(probe.class_names)

        classifier = test_init(model_name=args.model_name, num_classes=num_classes)
        test_dataset_forward(
            classifier=classifier,
            dataset_name=dataset_name,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_batches=args.max_batches,
            gpu=args.gpu,
        )


if __name__ == "__main__":
    main()
