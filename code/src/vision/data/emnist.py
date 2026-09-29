import torch
import torchvision
from datasets import load_dataset
from .common import HFVisionDataset, HF_TOKEN, HF_DATASETS_CACHE, make_val_split, make_seeded_loader


def rotate_img(img):
    return torchvision.transforms.functional.rotate(img, -90)


def flip_img(img):
    return torchvision.transforms.functional.hflip(img)


def emnist_preprocess():
    return torchvision.transforms.Compose(
        [
            rotate_img,
            flip_img,
        ]
    )


class EMNIST:
    def __init__(
        self,
        preprocess_train,
        preprocess_inference,
        batch_size,
        num_workers,
        seed
    ):
        emnist_correction = emnist_preprocess()
        preprocess_train = torchvision.transforms.Compose(
            [
                preprocess_train,
                emnist_correction,
            ]
        )
        preprocess_inference = torchvision.transforms.Compose(
            [
                preprocess_inference,
                emnist_correction,
            ]
        )

        hf_train = load_dataset("tanganke/emnist_letters", split="train", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)

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

        hf_test = load_dataset("tanganke/emnist_letters", split="test", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)
        self.test_dataset = HFVisionDataset(hf_test, transform=preprocess_inference)

        self.test_loader = make_seeded_loader(
            dataset=self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )

        self.class_names = hf_train.features['label'].names
