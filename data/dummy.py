"""Synthetic land cover style dataset for verifying the pipeline before real data exists.

Each sample is built from a low resolution label grid sampled from a deliberately skewed
class distribution and then upsampled with nearest neighbor so the mask has contiguous
regions with real boundaries. The image is produced from a fixed per class color palette plus
Gaussian noise, so the image carries signal that matches the mask and a model can actually
reduce its loss during a smoke test. Every sample is reproducible because it is generated from
a seed derived from the base seed, a split offset and the sample index.
"""
import numpy as np
import torch

from data.base import BaseSegmentationDataset
from data.registry import register_dataset


@register_dataset("dummy")
class DummySegmentationDataset(BaseSegmentationDataset):
    """Random but structured dataset that needs no download."""

    def __init__(self, cfg, split):
        super().__init__(cfg, split)
        params = cfg.dataset.get_path("dummy", {}) or {}
        self.num_train = int(params.get("num_train", 32))
        self.num_val = int(params.get("num_val", 8))
        self.num_test = int(params.get("num_test", 8))
        # Distinct seed offsets per split so train, val and test are disjoint sample sets.
        sizes = {"train": self.num_train, "val": self.num_val, "test": self.num_test}
        offsets = {"train": 0, "val": 100000, "test": 200000}
        self.length = sizes.get(split, self.num_val)
        self.seed_offset = offsets.get(split, 100000)

        self.base_seed = int(cfg.get_path("seed", 42))
        self.grid = int(params.get("grid", 16))
        self.noise = float(params.get("noise", 0.08))

        # Skewed class frequencies so the later rare class analysis has something to find.
        decay = float(params.get("decay", 0.6))
        weights = np.array([decay ** k for k in range(self.num_classes)], dtype=np.float64)
        self.class_probs = weights / weights.sum()

        # Fixed palette shared by every sample so each class keeps one color.
        palette_rng = np.random.default_rng(self.base_seed)
        self.palette = palette_rng.uniform(
            0.15, 0.95, size=(self.num_classes, 3)
        ).astype(np.float32)

        self.class_names = [f"class_{i}" for i in range(self.num_classes)]
        self._cache_idx = None
        self._cache = None

    def __len__(self):
        return self.length

    def _generate(self, idx):
        rng = np.random.default_rng(self.base_seed + self.seed_offset + idx)
        grid = self.grid
        low = rng.choice(self.num_classes, size=(grid, grid), p=self.class_probs)

        height, width = self.image_size
        row_index = (np.arange(height) * grid) // height
        col_index = (np.arange(width) * grid) // width
        mask = low[row_index][:, col_index]

        color = self.palette[mask]
        noise = rng.normal(0.0, self.noise, size=color.shape).astype(np.float32)
        image = np.clip(color + noise, 0.0, 1.0)

        image_tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).float()
        mask_tensor = torch.from_numpy(mask.astype(np.int64))
        return image_tensor, mask_tensor

    def _get(self, idx):
        if self._cache_idx != idx:
            self._cache = self._generate(idx)
            self._cache_idx = idx
        return self._cache

    def read_image(self, idx):
        return self._get(idx)[0].clone()

    def read_mask(self, idx):
        return self._get(idx)[1].clone()
