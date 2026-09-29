import torch
from datasets import load_dataset
from .common import HFVisionDataset, HF_TOKEN, HF_DATASETS_CACHE, make_val_split, make_seeded_loader


resisc45_classes = [
    "airplane", "airport", "baseball_diamond", "basketball_court", "beach",
    "bridge", "chaparral", "church", "circular_farmland", "cloud",
    "commercial_area", "dense_residential", "desert", "forest", "freeway",
    "golf_course", "ground_track_field", "harbor", "industrial_area",
    "intersection", "island", "lake", "meadow", "medium_residential",
    "mobile_home_park", "mountain", "overpass", "palace", "parking_lot",
    "railway", "railway_station", "rectangular_farmland", "river", "roundabout",
    "runway", "sea_ice", "ship", "snowberg", "sparse_residential", "stadium",
    "storage_tank", "tennis_court", "terrace", "thermal_power_station", "wetland",
]


class RESISC45:
    def __init__(
        self,
        preprocess_train,
        preprocess_inference,
        batch_size,
        num_workers,
        seed
    ):

        hf_train = load_dataset("tanganke/resisc45", split="train", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)

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

        hf_test = load_dataset("tanganke/resisc45", split="test", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)
        self.test_dataset = HFVisionDataset(hf_test, transform=preprocess_inference)

        self.test_loader = make_seeded_loader(
            dataset=self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )

        self.class_names = [' '.join(c.split('_')) for c in resisc45_classes]
