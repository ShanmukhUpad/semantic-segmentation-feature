"""Paired augmentation and normalization for an image and its segmentation mask.

Geometric operations are applied to both the image and the mask so they stay aligned. Color
jitter touches the image only. Normalization always runs. For any split other than train the
augmentation block is skipped and only normalization is applied so validation is repeatable.

Rotation uses ninety degree multiples on purpose. For aerial and satellite imagery a rotation
by zero, ninety, one hundred eighty or two hundred seventy degrees needs no interpolation and
introduces no border fill, so the mask stays valid with no ignore pixels added at the edges.
"""
import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class PairedTransform:
    """Joint transform built from the augmentation section of the config."""

    def __init__(self, cfg, split):
        aug = cfg.augmentation
        self.split = split
        self.enabled = bool(aug.get_path("enabled", True)) and split == "train"
        self.random_hflip = bool(aug.get_path("random_hflip", True))
        self.random_vflip = bool(aug.get_path("random_vflip", True))
        self.random_rot90 = bool(aug.get_path("random_rot90", True))
        self.use_jitter = bool(aug.get_path("color_jitter", True))

        amount = float(aug.get_path("jitter", 0.2))
        hue = min(amount, 0.5)
        self.jitter = T.ColorJitter(
            brightness=amount, contrast=amount, saturation=amount, hue=hue
        )
        self.mean = list(aug.get_path("mean", [0.485, 0.456, 0.406]))
        self.std = list(aug.get_path("std", [0.229, 0.224, 0.225]))

    def __call__(self, image, mask):
        """Apply the transform to a float image tensor and a long mask tensor."""
        if self.enabled:
            if self.random_hflip and random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if self.random_vflip and random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if self.random_rot90:
                k = random.randint(0, 3)
                if k:
                    image = torch.rot90(image, k, dims=(-2, -1))
                    mask = torch.rot90(mask, k, dims=(-2, -1))
            if self.use_jitter:
                image = self.jitter(image)

        image = TF.normalize(image, self.mean, self.std)
        return image.contiguous(), mask.contiguous().long()
