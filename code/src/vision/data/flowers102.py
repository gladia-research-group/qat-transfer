import torch
from datasets import load_dataset
from .common import HFVisionDataset, HF_TOKEN, HF_DATASETS_CACHE, make_val_split, make_seeded_loader


class Flowers102:
    def __init__(
        self,
        preprocess_train,
        preprocess_inference,
        batch_size,
        num_workers,
        seed
    ):

        hf_train = load_dataset("dpdl-benchmark/oxford_flowers102", split="train", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)

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

        hf_test = load_dataset("dpdl-benchmark/oxford_flowers102", split="test", token=HF_TOKEN, cache_dir=HF_DATASETS_CACHE)
        self.test_dataset = HFVisionDataset(hf_test, transform=preprocess_inference)

        self.test_loader = make_seeded_loader(
            dataset=self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )

        self.class_names = [
            "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
            "sweet pea", "english marigold", "tiger lily", "moon orchid",
            "bird of paradise", "monkshood", "globe thistle", "snapdragon",
            "colt's foot", "king protea", "spear thistle", "yellow iris",
            "globe flower", "purple coneflower", "peruvian lily",
            "balloon flower", "giant white arum lily", "fire lily",
            "pincushion flower", "fritillary", "red ginger", "grape hyacinth",
            "corn poppy", "prince of wales feathers", "stemless gentian",
            "artichoke", "sweet william", "carnation", "garden phlox",
            "love in the mist", "mexican aster", "alpine sea holly",
            "ruby-lipped cattleya", "cape flower", "great masterwort",
            "siam tulip", "lenten rose", "barbeton daisy", "daffodil",
            "sword lily", "poinsettia", "bolero deep blue", "wallflower",
            "marigold", "buttercup", "oxeye daisy", "common dandelion",
            "petunia", "wild pansy", "primula", "sunflower", "pelargonium",
            "bishop of llandaff", "gaura", "geranium", "orange dahlia",
            "pink and yellow dahlia", "cautleya spicata", "japanese anemone",
            "black-eyed susan", "silverbush", "californian poppy",
            "osteospermum", "spring crocus", "bearded iris", "windflower",
            "tree poppy", "gazania", "azalea", "water lily", "rose",
            "thorn apple", "morning glory", "passion flower", "lotus",
            "toad lily", "anthurium", "frangipani", "clematis", "hibiscus",
            "columbine", "desert-rose", "tree mallow", "magnolia", "cyclamen",
            "watercress", "canna lily", "hippeastrum", "bee balm", "air plant",
            "foxglove", "bougainvillea", "camellia", "mallow",
            "mexican petunia", "bromelia", "blanket flower",
            "trumpet creeper", "blackberry lily",
        ]
