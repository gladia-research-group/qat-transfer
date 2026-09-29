from datasets import load_dataset, concatenate_datasets
from .common import HFTextDataset, HF_TOKEN, HF_DATASETS_CACHE, make_val_split, make_seeded_loader, encode_string_labels


class Emotion:
    def __init__(self, batch_size, num_workers, seed):

        hf_train = load_dataset("mteb/emotion", split="train", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)
        hf_test = load_dataset("mteb/emotion", split="test", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)

        label_to_id, class_names = encode_string_labels(concatenate_datasets([hf_train, hf_test]))

        self.train_dataset, self.val_dataset = make_val_split(
            HFTextDataset(hf_dataset=hf_train, label_map=label_to_id),
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

        self.test_dataset = HFTextDataset(hf_test, label_map=label_to_id)

        self.test_loader = make_seeded_loader(
            dataset=self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )

        self.class_names = class_names
