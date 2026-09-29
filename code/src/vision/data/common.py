import os
import torch
import glob
import collections

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataset import random_split

from ..utils import SPLIT_SEED, VAL_FRACTION, MAX_VAL_SAMPLES, seed_worker


def make_seeded_loader(dataset, shuffle, batch_size, num_workers, seed):
    """Build a DataLoader whose shuffling and worker RNGs are seeded with `seed`.

    `seed` here is the per-run training seed (cfg.seed in finetune_fp.py), NOT
    SPLIT_SEED. SPLIT_SEED is reserved exclusively for deciding the train/val
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
    )

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_DATASETS_CACHE = os.environ.get("HF_DATASETS_CACHE")

DATASET_NAME_TO_EPOCHS = {
    "Cars": 35,
    "DTD": 76,
    "EuroSAT": 12,
    "GTSRB": 11,
    "MNIST": 5,
    "RESISC45": 15,
    "SUN397": 14,
    "SVHN": 4,
    "CIFAR10": 6,
    "CIFAR100": 6,
    "STL10": 60,
    "Food101": 4,
    "Flowers102": 147,
    "FER2013": 10,
    "PCAM": 1,
    "OxfordIIITPet": 82,
    "RenderedSST2": 39,
    "EMNIST": 2,
    "FashionMNIST": 5,
    "KMNIST": 5,
    "TinyImageNet": 4,
    "ImageNet": 1
}

DATASET_NAME_TO_NUM_CLASSES = {
    "Cars": 196,
    "DTD": 47,
    "EuroSAT": 10,
    "GTSRB": 43,
    "MNIST": 10,
    "RESISC45": 45,
    "SUN397": 397,
    "SVHN": 10,
    "CIFAR10": 10,
    "CIFAR100": 100,
    "STL10": 10,
    "Food101": 101,
    "Flowers102": 102,
    "FER2013": 7,
    "PCAM": 2,
    "OxfordIIITPet": 37,
    "RenderedSST2": 2,
    "EMNIST": 26,
    "FashionMNIST": 10,
    "KMNIST": 10,
    "TinyImageNet": 200,
    "ImageNet": 1000,
}

class TransformedSubset(Dataset):
    """Wraps a Subset with a transform applied after fetching items."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def make_val_split(train_dataset, train_transform=None, val_transform=None):
    total_size = len(train_dataset)
    val_size = min(int(total_size * VAL_FRACTION), MAX_VAL_SAMPLES)
    train_size = total_size - val_size

    trainset, valset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SPLIT_SEED)
    )

    return TransformedSubset(
        trainset, transform=train_transform
    ), TransformedSubset(valset, transform=val_transform)


class HFVisionDataset(Dataset):
    """PyTorch Dataset wrapper for HuggingFace datasets."""

    def __init__(self, hf_dataset, transform=None, image_col="image", label_col="label"):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample[self.image_col]

        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        label = sample[self.label_col]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class SubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


def maybe_dictionarize(batch):
    if isinstance(batch, dict):
        return batch

    if len(batch) == 2:
        batch = {'images': batch[0], 'labels': batch[1]}
    elif len(batch) == 3:
        batch = {'images': batch[0], 'labels': batch[1], 'metadata': batch[2]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}')

    return batch


def get_features_helper(image_encoder, dataloader, device):
    all_data = collections.defaultdict(list)

    image_encoder = image_encoder.to(device)
    image_encoder.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = maybe_dictionarize(batch)
            features = image_encoder(batch['images'].to(device))

            all_data['features'].append(features.cpu())

            for key, val in batch.items():
                if key == 'images':
                    continue
                if hasattr(val, 'cpu'):
                    val = val.cpu()
                    all_data[key].append(val)
                else:
                    all_data[key].extend(val)

    for key, val in all_data.items():
        if torch.is_tensor(val[0]):
            all_data[key] = torch.cat(val).numpy()

    return all_data


def get_features(is_train, image_encoder, dataset, device):
    split = 'train' if is_train else 'val'
    dname = type(dataset).__name__
    if image_encoder.cache_dir is not None:
        cache_dir = f'{image_encoder.cache_dir}/{dname}/{split}'
        cached_files = glob.glob(f'{cache_dir}/*')
    if image_encoder.cache_dir is not None and len(cached_files) > 0:
        print(f'Getting features from {cache_dir}')
        data = {}
        for cached_file in cached_files:
            name = os.path.splitext(os.path.basename(cached_file))[0]
            data[name] = torch.load(cached_file)
    else:
        print(f'Did not find cached features at {cache_dir}. Building from scratch.')
        loader = dataset.train_loader if is_train else dataset.test_loader
        data = get_features_helper(image_encoder, loader, device)
        if image_encoder.cache_dir is None:
            print('Not caching because no cache directory was passed.')
        else:
            os.makedirs(cache_dir, exist_ok=True)
            print(f'Caching data at {cache_dir}')
            for name, val in data.items():
                torch.save(val, f'{cache_dir}/{name}.pt')
    return data


class FeatureDataset(Dataset):
    def __init__(self, is_train, image_encoder, dataset, device):
        self.data = get_features(is_train, image_encoder, dataset, device)

    def __len__(self):
        return len(self.data['features'])

    def __getitem__(self, idx):
        data = {k: v[idx] for k, v in self.data.items()}
        data['features'] = torch.from_numpy(data['features']).float()
        return data


def get_dataloader(dataset, is_train, args, image_encoder=None):
    if image_encoder is not None:
        feature_dataset = FeatureDataset(is_train, image_encoder, dataset, args.device)
        dataloader = DataLoader(feature_dataset, batch_size=args.batch_size, shuffle=is_train)
    else:
        dataloader = dataset.train_loader if is_train else dataset.test_loader
    return dataloader
