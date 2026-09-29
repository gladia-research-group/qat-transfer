import torch
from datasets import load_dataset
from .common import HFVisionDataset, HF_TOKEN, HF_DATASETS_CACHE, make_val_split, make_seeded_loader


class CIFAR100:
    def __init__(
        self,
        preprocess_train,
        preprocess_inference,
        batch_size,
        num_workers,
        seed
    ):

        hf_train = load_dataset("uoft-cs/cifar100", split="train", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)

        self.train_dataset, self.val_dataset = make_val_split(
            train_dataset=HFVisionDataset(
                hf_dataset=hf_train,
                transform=None,
                image_col="img",
                label_col="fine_label"
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

        hf_test = load_dataset("uoft-cs/cifar100", split="test", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)
        self.test_dataset = HFVisionDataset(
            hf_test, transform=preprocess_inference, image_col="img", label_col="fine_label"
        )

        self.test_loader = make_seeded_loader(
            dataset=self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )

        self.class_names = hf_test.features['fine_label'].names
