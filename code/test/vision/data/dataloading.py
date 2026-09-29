import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dotenv import load_dotenv
load_dotenv()

from torchvision import transforms
from tqdm import tqdm

from src.vision.data.registry import registry, get_dataset
from src.vision.utils import random_tqdm_color, VAL_FRACTION, MAX_VAL_SAMPLES


def iterate_loader(loader, split_name):
    colour = random_tqdm_color()
    num_batches = 0
    for _ in tqdm(loader, desc=split_name, colour=colour):
        num_batches += 1
    return num_batches


def test_dataset(
    dataset_name: str, 
    preprocess_train,
    preprocess_inference,
    batch_size: int,
    num_workers: int,
):
    assert dataset_name in registry, (
        f"Unsupported dataset: {dataset_name}. "
        f"Supported: {sorted(registry.keys())}"
    )

    print(f"\nLoading dataset: {dataset_name}")
    dataset = get_dataset(
        dataset_name=dataset_name,
        preprocess_train=preprocess_train,
        preprocess_inference=preprocess_inference,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    splits = [
        ("train", dataset.train_loader),
        ("val", dataset.val_loader),
        ("test", dataset.test_loader),
    ]

    summary = {}
    for split_name, loader in splits:
        if loader is None:
            summary[split_name] = (0, 0)
            continue
        num_batches = iterate_loader(loader, f"{dataset_name} [{split_name}]")
        num_samples = len(loader.dataset)
        summary[split_name] = (num_samples, num_batches)

    print(f"\n{'=' * 50}")
    print(f"Dataset: {dataset_name}")
    print(f"Batch size: {batch_size} | Workers: {num_workers}")
    print(f"{'-' * 50}")
    for split_name, (num_samples, num_batches) in summary.items():
        if num_samples > 0:
            print(f"  {split_name:>5}: {num_samples:>7} samples, {num_batches:>5} batches")
        else:
            print(f"  {split_name:>5}: (not available)")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser(description="Smoke-test dataloading for a vision dataset.")

    parser.add_argument("--dataset-name", type=str, nargs="+", required=True,
                        help=f"Dataset class name(s). Options: {sorted(registry.keys())}")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    
    args = parser.parse_args()

    preprocess_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    preprocess_inference = preprocess_train

    for dataset_name in args.dataset_name:
        test_dataset(
            dataset_name=dataset_name, 
            preprocess_train=preprocess_train,
            preprocess_inference=preprocess_inference,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )


if __name__ == "__main__":
    main()
