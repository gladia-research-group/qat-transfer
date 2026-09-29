import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

SPLIT_SEED = 0
VAL_FRACTION = 0.1
MAX_VAL_SAMPLES = 5000

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASETS_CACHE = os.environ.get("HF_DATASETS_CACHE")

DATASET_NAME_TO_EPOCHS: dict[str, int] = {
    "Emotion": 5,
    "IMDB": 5,
    "Banking77": 5,
    "AmazonReviewsClassification": 5,
    "AmazonCounterfactual": 5,
    # "AmazonPolarity": 5,
    "MassiveIntent": 5,
    "MassiveScenario": 5,
    "MTOPDomain": 5,
    "MTOPIntent": 5,
    "ToxicConversations": 5,
    "TweetSentimentExtraction": 5,
}


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn that seeds numpy and python random per worker."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def text_collate_fn(batch):
    """Custom collate that keeps texts as a list of strings."""
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels)


class HFTextDataset(Dataset):
    """PyTorch Dataset wrapper for HuggingFace text classification datasets."""

    def __init__(self, hf_dataset, text_col="text", label_col="label", label_map=None):
        self.dataset = hf_dataset
        self.text_col = text_col
        self.label_col = label_col
        self.label_map = label_map

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        text = sample[self.text_col]
        label = sample[self.label_col]
        if self.label_map is not None:
            label = self.label_map[label]
        return text, label


def make_val_split(train_dataset):
    """Split train_dataset into train/val using a fixed seed.

    The generator is seeded with SPLIT_SEED so that the split is identical
    across runs regardless of the per-run training seed.
    """
    total_size = len(train_dataset)
    val_size = min(int(total_size * VAL_FRACTION), MAX_VAL_SAMPLES)
    train_size = total_size - val_size

    return random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SPLIT_SEED)
    )


def make_seeded_loader(dataset, shuffle, batch_size, num_workers, seed):
    """Build a DataLoader whose shuffling and worker RNGs are seeded with `seed`.

    `seed` here is the per-run training seed, NOT SPLIT_SEED.
    SPLIT_SEED is reserved exclusively for deciding the train/val
    index split inside make_val_split and must never reach this function.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        collate_fn=text_collate_fn,
    )


def get_class_names(hf_dataset, label_col="label"):
    """Return class name list from ClassLabel feature, or sorted unique values as fallback."""
    feat = hf_dataset.features[label_col]
    if hasattr(feat, "names"):
        return feat.names
    return sorted(set(str(v) for v in hf_dataset[label_col]))


def encode_string_labels(hf_dataset, label_col="label"):
    """Build a deterministic string-to-int label mapping from sorted unique values."""
    unique_labels = sorted(set(hf_dataset[label_col]))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    return label_to_id, unique_labels
