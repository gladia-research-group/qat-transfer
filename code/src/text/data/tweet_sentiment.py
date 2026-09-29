from datasets import load_dataset
from .common import HFTextDataset, HF_TOKEN, HF_DATASETS_CACHE, make_val_split, make_seeded_loader, get_class_names


class TweetSentimentExtraction:
    def __init__(self, batch_size, num_workers, seed):

        hf_train = load_dataset("mteb/tweet_sentiment_extraction", split="train", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)

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

        hf_test = load_dataset("mteb/tweet_sentiment_extraction", split="test", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)
        self.test_dataset = HFTextDataset(hf_test)

        self.test_loader = make_seeded_loader(
            dataset=self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )

        self.class_names = get_class_names(hf_test)
