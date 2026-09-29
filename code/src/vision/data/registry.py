import sys
import inspect

from .cars import Cars
from .cifar10 import CIFAR10
from .cifar100 import CIFAR100
from .dtd import DTD
from .emnist import EMNIST
from .eurosat import EuroSAT
from .fashionmnist import FashionMNIST
from .fer2013 import FER2013
from .flowers102 import Flowers102
from .food101 import Food101
from .gtsrb import GTSRB
from .imagenet import ImageNet
from .kmnist import KMNIST
from .mnist import MNIST
from .oxfordpets import OxfordIIITPet
from .pcam import PCAM
from .resisc45 import RESISC45
from .sst2 import RenderedSST2
from .stl10 import STL10
from .sun397 import SUN397
from .svhn import SVHN
from .tinyimagenet import TinyImageNet

registry = {
    name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)
}


class GenericDataset(object):
    def __init__(self):

        self.train_preprocess = None
        self.inference_preprocess = None

        self.train_dataset = None
        self.train_loader = None

        self.test_dataset = None
        self.test_loader = None
        
        self.val_dataset = None
        self.val_loader = None
        
        self.class_names = None


def get_dataset(
    dataset_name: str,
    preprocess_train,
    preprocess_inference,
    batch_size: int,
    num_workers: int,
    seed: int,
):

    assert dataset_name in registry, f'Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}'

    dataset_class = registry[dataset_name]

    return dataset_class(
        preprocess_train=preprocess_train,
        preprocess_inference=preprocess_inference,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
    )
