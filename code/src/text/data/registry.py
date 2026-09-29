import sys
import inspect

from .amazon_counterfactual import AmazonCounterfactual
from .amazon_polarity import AmazonPolarity
from .amazon_reviews import AmazonReviewsClassification
from .banking77 import Banking77
from .emotion import Emotion
from .imdb import IMDB
from .massive_intent import MassiveIntent
from .massive_scenario import MassiveScenario
from .mtop_domain import MTOPDomain
from .mtop_intent import MTOPIntent
from .toxic_conversations import ToxicConversations
from .tweet_sentiment import TweetSentimentExtraction

registry = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}


class GenericDataset(object):
    def __init__(self):

        self.train_dataset = None
        self.train_loader = None

        self.test_dataset = None
        self.test_loader = None

        self.val_dataset = None
        self.val_loader = None

        self.class_names = None


def get_dataset(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    seed: int,
):

    assert dataset_name in registry, f'Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}'

    dataset_class = registry[dataset_name]

    return dataset_class(
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
