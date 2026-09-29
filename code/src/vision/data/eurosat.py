import torch
from datasets import load_dataset
from .common import HFVisionDataset, HF_TOKEN, HF_DATASETS_CACHE, make_val_split, make_seeded_loader


eurosat_classnames = [
    'annual crop land',
    'forest',
    'brushland or shrubland',
    'highway or road',
    'industrial buildings or commercial buildings',
    'pasture land',
    'permanent crop land',
    'residential buildings or homes or apartments',
    'river',
    'lake or sea',
]


class EuroSAT:
    def __init__(
        self,
        preprocess_train,
        preprocess_inference,
        batch_size,
        num_workers,
        seed
    ):

        hf_train = load_dataset("tanganke/eurosat", split="train", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)

        self.train_dataset, self.val_dataset = make_val_split(
            train_dataset=HFVisionDataset(
                hf_dataset=hf_train,
                transform=None
            ),
            train_transform=preprocess_train,
            val_transform=preprocess_inference,
        )

        self.train_loader = make_seeded_loader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )
        self.val_loader = make_seeded_loader(
            dataset=self.val_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )

        hf_test = load_dataset("tanganke/eurosat", split="test", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)
        self.test_dataset = HFVisionDataset(hf_test, transform=preprocess_inference)

        self.test_loader = make_seeded_loader(
            dataset=self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )

        self.class_names = eurosat_classnames
