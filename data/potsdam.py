"""ISPRS Potsdam land cover dataset loader.

Potsdam ships large orthophoto tiles with RGB color coded labels for six classes. This loader
expects the tiles to be cut into smaller patches ahead of time and placed in an images folder
and a labels folder per split. The color coded labels are decoded into class index masks using
the standard Potsdam palette.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from data.base import BaseSegmentationDataset
from data.registry import register_dataset

POTSDAM_CLASSES = [
    "impervious_surface",
    "building",
    "low_vegetation",
    "tree",
    "car",
    "clutter",
]

# Standard Potsdam label colors in the same order as the class list above.
POTSDAM_COLORS = [
    (255, 255, 255),
    (0, 0, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 255, 0),
    (255, 0, 0),
]

_MISSING_DATA = (
    "ISPRS Potsdam data was not found under {root}. Cut the orthophotos and labels into "
    "patches ahead of time and place them so that {root} contains a folder per split, each "
    "with an images folder and a labels folder using matching file names. Labels are read as "
    "RGB color coded PNG tiles. Then set dataset.root in your config to that path. To verify "
    "the pipeline without real data set dataset.name to dummy."
)


@register_dataset("potsdam")
class PotsdamDataset(BaseSegmentationDataset):
    """Loader for pre tiled ISPRS Potsdam patches with color coded labels."""

    class_names = POTSDAM_CLASSES

    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        root = Path(cfg.dataset.root)
        image_dir = root / split / "images"
        mask_dir = root / split / "labels"
        if not image_dir.exists() or not mask_dir.exists():
            raise FileNotFoundError(_MISSING_DATA.format(root=root))

        candidates = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.tif"))
        self.images = []
        self.masks = []
        for image_path in candidates:
            mask_path = mask_dir / image_path.name
            if mask_path.exists():
                self.images.append(image_path)
                self.masks.append(mask_path)

        if not self.images:
            raise FileNotFoundError(_MISSING_DATA.format(root=root))

        self._color_index = {
            (color[0] << 16) | (color[1] << 8) | color[2]: idx
            for idx, color in enumerate(POTSDAM_COLORS)
        }

    def __len__(self):
        return len(self.images)

    def read_image(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array.transpose(2, 0, 1).copy()).float()

    def read_mask(self, idx):
        rgb = np.asarray(Image.open(self.masks[idx]).convert("RGB"), dtype=np.int64)
        packed = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
        mask = np.full(packed.shape, self.ignore_index, dtype=np.int64)
        for code, class_index in self._color_index.items():
            mask[packed == code] = class_index
        return torch.from_numpy(mask)
