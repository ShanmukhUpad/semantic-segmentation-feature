"""LoveDA land cover dataset loader.

LoveDA stores RGB tiles as PNG images and single channel label PNGs. The label value zero
marks no data and is mapped to the ignore index. The seven land cover classes use label
values one through seven and are remapped to zero through six so they line up with the network
output channels.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.base import BaseSegmentationDataset
from data.registry import register_dataset

LOVEDA_CLASSES = [
    "background",
    "building",
    "road",
    "water",
    "barren",
    "forest",
    "agriculture",
]

_MISSING_DATA = (
    "LoveDA data was not found under {root}. Download LoveDA from the official release and "
    "arrange it so that {root} contains Train, Val and Test folders, each with Rural and "
    "Urban subfolders that hold images_png and masks_png. Then set dataset.root in your "
    "config to that path. To verify the pipeline without real data set dataset.name to dummy."
)


@register_dataset("loveda")
class LoveDADataset(BaseSegmentationDataset):
    """Loader for the LoveDA Rural and Urban land cover tiles."""

    class_names = LOVEDA_CLASSES
    split_dirs = {"train": "Train", "val": "Val", "test": "Test"}

    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        root = Path(cfg.dataset.root)
        split_dir = root / self.split_dirs.get(split, split.capitalize())
        if not split_dir.exists():
            raise FileNotFoundError(_MISSING_DATA.format(root=root))

        self.images = []
        self.masks = []
        for region in ["Rural", "Urban"]:
            image_dir = split_dir / region / "images_png"
            mask_dir = split_dir / region / "masks_png"
            if not image_dir.exists():
                continue
            for image_path in sorted(image_dir.glob("*.png")):
                mask_path = mask_dir / image_path.name
                if mask_path.exists():
                    self.images.append(image_path)
                    self.masks.append(mask_path)

        if not self.images:
            raise FileNotFoundError(_MISSING_DATA.format(root=root))

    def __len__(self):
        return len(self.images)

    def read_image(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array.transpose(2, 0, 1).copy()).float()

    def read_mask(self, idx):
        raw = np.asarray(Image.open(self.masks[idx]), dtype=np.int64)
        mask = np.full(raw.shape, self.ignore_index, dtype=np.int64)
        labeled = raw >= 1
        mask[labeled] = raw[labeled] - 1
        return torch.from_numpy(mask)
