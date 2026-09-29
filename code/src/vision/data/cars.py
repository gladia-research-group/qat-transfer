import torch
from datasets import load_dataset
from .common import HFVisionDataset, HF_TOKEN, HF_DATASETS_CACHE, make_val_split, make_seeded_loader


class Cars:
    def __init__(
        self,
        preprocess_train,
        preprocess_inference,
        batch_size,
        num_workers,
        seed
    ):

        # get HF train dataset
        hf_train = load_dataset("tanganke/stanford_cars", split="train", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)
        
        # split HF train dataset into train and validation 
        # do NOT pass transforms here, since train and inference (i.e. val) have different transforms
        self.train_dataset, self.val_dataset = make_val_split(
            train_dataset=HFVisionDataset(
                hf_dataset=hf_train,
                transform=None
            ),
            train_transform=preprocess_train,
            val_transform=preprocess_inference,
        )

        # create dataloaders after dataset splitting
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

        # get HF test dataset
        hf_test = load_dataset("tanganke/stanford_cars", split="test", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)
        self.test_dataset = HFVisionDataset(hf_test, transform=preprocess_inference)

        # create dataloaders after dataset splitting
        self.test_loader = make_seeded_loader(
            dataset=self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )

        # get class names and process them
        class_names = hf_train.features['label'].names
        self.class_names = [c.replace('_', ' ') for c in class_names]
