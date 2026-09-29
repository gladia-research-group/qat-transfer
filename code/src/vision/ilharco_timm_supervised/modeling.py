import os
from typing import Optional

import timm
import timm.data
import torch


class ImageClassifier(torch.nn.Module):
    def __init__(self, model_name: str, num_classes: int):
        super().__init__()

        print(f"Loading {model_name} pre-trained weights.")
        self.model_name = model_name

        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

        data_config = timm.data.resolve_data_config(self.model.pretrained_cfg)
        self.train_preprocess = timm.data.create_transform(**data_config, is_training=True)
        self.val_preprocess = timm.data.create_transform(**data_config, is_training=False)

    def forward(self, images):
        return self.model(images)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image classifier to {filename}")
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.state_dict(), filename)

    @classmethod
    def load(
        cls,
        model_name: str,
        num_classes: int,
        filename: str,
        map_location: str = "cpu",
    ):
        print(f"Loading image classifier from {filename}")
        obj = cls(model_name=model_name, num_classes=num_classes)
        state_dict = torch.load(filename, map_location=map_location)
        obj.load_state_dict(state_dict)
        return obj







if __name__ == "__main__":

    from rich.pretty import pprint

    model = ImageClassifier(
        model_name="timm/vit_base_patch16_clip_224.openai_ft_in12k_in1k",
        num_classes=23
    )
    
    pprint(model, expand_all=True)