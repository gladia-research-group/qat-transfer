from datasets import load_dataset
from .common import HFTextDataset, HF_TOKEN, HF_DATASETS_CACHE, make_val_split, make_seeded_loader, get_class_names


class ToxicConversations:
    def __init__(self, batch_size, num_workers, seed):

        hf_train = load_dataset("mteb/toxic_conversations_50k", split="train", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)

        self.train_dataset, self.val_dataset = make_val_split(
            HFTextDataset(hf_dataset=hf_train),
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

        hf_test = load_dataset("mteb/toxic_conversations_50k", split="test", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)
        self.test_dataset = HFTextDataset(hf_test)

        self.test_loader = make_seeded_loader(
            dataset=self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )

        self.class_names = get_class_names(hf_test)
