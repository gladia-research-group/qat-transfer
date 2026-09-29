"""998 — Training-set sizes for the cost amortization figure

Resolves, for every vision dataset in the suite, the number of examples the QAT
run actually iterates over: the HuggingFace train split minus the validation
slice carved out by make_val_split.  No model is built and no image is decoded —
split sizes come from the dataset metadata whenever the builder exposes them,
so this is cheap even for ImageNet.

Writes dataset_sizes.json into
evaluations/998_rebuttal/002_cost_amortization/, which compute_costs.py reads.

Run with --offline to forbid any network access and rely purely on the local
HuggingFace cache.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is the working directory and on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_CODE_DIR = _PROJECT_ROOT / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import os

os.chdir(_PROJECT_ROOT)

# dotenv must run before any HF import: the HF libraries snapshot the cache
# environment variables at import time.
from dotenv import load_dotenv

load_dotenv()

import argparse
import json

from datasets import load_dataset, load_dataset_builder

from src.vision.data.common import DATASET_NAME_TO_EPOCHS
from src.vision.utils import MAX_VAL_SAMPLES, VAL_FRACTION


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_ROOT_OUT = "evaluations/998_rebuttal/002_cost_amortization"

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASETS_CACHE = os.environ.get("HF_DATASETS_CACHE")

# (repository, config name or None, train split name), mirroring the loaders in
# code/src/vision/data/.  Kept as an explicit table rather than introspected
# from the loader classes so that reading a size never triggers a dataset build.
DATASET_NAME_TO_HF = {
    "Cars":          ("tanganke/stanford_cars",          None,             "train"),
    "CIFAR10":       ("uoft-cs/cifar10",                  None,             "train"),
    "CIFAR100":      ("uoft-cs/cifar100",                 None,             "train"),
    "DTD":           ("tanganke/dtd",                     None,             "train"),
    "EMNIST":        ("tanganke/emnist_letters",          None,             "train"),
    "EuroSAT":       ("tanganke/eurosat",                 None,             "train"),
    "FashionMNIST":  ("zalando-datasets/fashion_mnist",   None,             "train"),
    "FER2013":       ("clip-benchmark/wds_fer2013",       None,             "train"),
    "Flowers102":    ("dpdl-benchmark/oxford_flowers102", None,             "train"),
    "Food101":       ("ethz/food101",                     None,             "train"),
    "GTSRB":         ("tanganke/gtsrb",                   None,             "train"),
    "ImageNet":      ("ILSVRC/imagenet-1k",               None,             "train"),
    "KMNIST":        ("tanganke/kmnist",                  None,             "train"),
    "MNIST":         ("ylecun/mnist",                     None,             "train"),
    "OxfordIIITPet": ("timm/oxford-iiit-pet",             None,             "train"),
    "PCAM":          ("1aurent/PatchCamelyon",            None,             "train"),
    "RESISC45":      ("tanganke/resisc45",                None,             "train"),
    "RenderedSST2":  ("nateraw/rendered-sst2",            None,             "train"),
    "STL10":         ("tanganke/stl10",                   None,             "train"),
    "SUN397":        ("tanganke/sun397",                  None,             "train"),
    "SVHN":          ("ufldl-stanford/svhn",              "cropped_digits", "train"),
    "TinyImageNet":  ("zh-plus/tiny-imagenet",            None,             "train"),
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Subset of dataset names to resolve (default: all).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Forbid network access; resolve sizes from the local HF cache only.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Size resolution
# ---------------------------------------------------------------------------
def _split_num_examples(repo, config, split):
    """Number of examples in `split`, preferring metadata over materialization."""
    try:
        builder = load_dataset_builder(
            repo, config, token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE
        )
        split_info = builder.info.splits.get(split) if builder.info.splits else None
        if split_info is not None and split_info.num_examples:
            return int(split_info.num_examples), "metadata"
    except Exception:
        pass

    # Fallback: memory-mapped load, which does not decode any image.
    dataset = load_dataset(
        repo, config, split=split, token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE
    )
    return int(len(dataset)), "materialized"


def _train_size_after_val_split(total_size):
    """Replicates make_val_split's arithmetic (code/src/vision/data/common.py)."""
    val_size = min(int(total_size * VAL_FRACTION), MAX_VAL_SAMPLES)
    return total_size - val_size, val_size


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    dataset_names = args.datasets if args.datasets is not None else sorted(DATASET_NAME_TO_HF)

    unknown = [d for d in dataset_names if d not in DATASET_NAME_TO_HF]
    assert not unknown, f"Unknown datasets: {unknown}. Known: {sorted(DATASET_NAME_TO_HF)}"

    sizes = {}
    failures = {}

    for dataset_name in dataset_names:
        repo, config, split = DATASET_NAME_TO_HF[dataset_name]

        try:
            total_size, source = _split_num_examples(repo, config, split)
        except Exception as exc:
            failures[dataset_name] = f"{type(exc).__name__}: {exc}"
            print(f"[FAIL] {dataset_name:<14} {failures[dataset_name]}")
            continue

        train_size, val_size = _train_size_after_val_split(total_size)

        sizes[dataset_name] = {
            "hf_repo": repo,
            "hf_config": config,
            "hf_split": split,
            "hf_split_size": total_size,
            "val_size": val_size,
            "train_size": train_size,
            "epochs": DATASET_NAME_TO_EPOCHS.get(dataset_name),
            "resolved_from": source,
        }

        print(
            f"[ok]   {dataset_name:<14} hf={total_size:>9,}  "
            f"train={train_size:>9,}  epochs={DATASET_NAME_TO_EPOCHS.get(dataset_name)}"
        )

    results = {
        "val_fraction": VAL_FRACTION,
        "max_val_samples": MAX_VAL_SAMPLES,
        "datasets": sizes,
        "failures": failures,
    }

    os.makedirs(EVAL_ROOT_OUT, exist_ok=True)
    out_path = os.path.join(EVAL_ROOT_OUT, "dataset_sizes.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nresolved {len(sizes)}/{len(dataset_names)} datasets -> {out_path}")
    if failures:
        print(f"unresolved: {sorted(failures)}")


if __name__ == "__main__":
    main()
