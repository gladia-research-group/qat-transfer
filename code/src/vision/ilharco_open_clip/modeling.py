import os
from typing import Optional

import open_clip
import torch


class ImageEncoder(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        cache_dir: Optional[str] = None,
        keep_lang: bool = False,
    ):
        super().__init__()

        print(f"Loading open_clip {model_name} (pretrained={pretrained}).")
        self.model_name = model_name
        self.pretrained = pretrained
        self.cache_dir = cache_dir

        model, train_preprocess, val_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, cache_dir=cache_dir,
        )
        self.model = model
        self.train_preprocess = train_preprocess
        self.val_preprocess = val_preprocess

        if not keep_lang:
            # CustomTextCLIP-style (has self.text as a module)
            if hasattr(self.model, 'text'):
                delattr(self.model, 'text')
            # CLIP-style (has transformer, token_embedding, etc.)
            for attr in ('transformer', 'token_embedding', 'positional_embedding', 'ln_final'):
                if hasattr(self.model, attr):
                    delattr(self.model, attr)
            # Both styles may have text_projection
            if hasattr(self.model, 'text_projection'):
                delattr(self.model, 'text_projection')

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving image encoder to {filename}")
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.state_dict(), filename)

    @classmethod
    def load(
        cls,
        model_name: str,
        pretrained: str,
        filename: str,
        cache_dir: Optional[str] = None,
        keep_lang: bool = False,
        map_location: str = "cpu",
    ):
        print(f"Loading image encoder from {filename}")
        obj = cls(model_name=model_name, pretrained=pretrained, cache_dir=cache_dir, keep_lang=keep_lang)
        state_dict = torch.load(filename, map_location=map_location)
        obj.load_state_dict(state_dict)
        return obj


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize: bool, weights: torch.Tensor, biases: Optional[torch.Tensor] = None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        # Stored as a buffer so it round-trips through state_dict.
        self.register_buffer("normalize_flag", torch.tensor(bool(normalize)))
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if bool(self.normalize_flag):
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        print(f"Saving classification head to {filename}")
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.state_dict(), filename)

    @classmethod
    def load(cls, filename: str, map_location: str = "cpu"):
        print(f"Loading classification head from {filename}")
        state_dict = torch.load(filename, map_location=map_location)
        weights = state_dict["weight"]
        biases = state_dict["bias"]
        normalize = bool(state_dict.get("normalize_flag", torch.tensor(True)))
        obj = cls(normalize=normalize, weights=weights, biases=biases)
        obj.load_state_dict(state_dict)
        return obj


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head
        if self.image_encoder is not None:
            self.train_preprocess = self.image_encoder.train_preprocess
            self.val_preprocess = self.image_encoder.val_preprocess

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def forward(self, inputs):
        features = self.image_encoder(inputs)
        outputs = self.classification_head(features)
        return outputs

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
        pretrained: str,
        filename: str,
        cache_dir: Optional[str] = None,
        keep_lang: bool = False,
        map_location: str = "cpu",
    ):
        print(f"Loading image classifier from {filename}")
        state_dict = torch.load(filename, map_location=map_location)

        head_weight = state_dict["classification_head.weight"]
        head_bias = state_dict["classification_head.bias"]
        head_normalize = bool(state_dict.get("classification_head.normalize_flag", torch.tensor(True)))

        image_encoder = ImageEncoder(
            model_name=model_name, pretrained=pretrained, cache_dir=cache_dir, keep_lang=keep_lang
        )
        classification_head = ClassificationHead(
            normalize=head_normalize, weights=head_weight, biases=head_bias
        )
        obj = cls(image_encoder, classification_head)
        obj.load_state_dict(state_dict)
        return obj
