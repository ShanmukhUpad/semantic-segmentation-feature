"""Shared base class for land cover segmentation datasets.

Subclasses implement three things. The number of samples through __len__, a single image
read through read_image returning a float tensor of shape C by H by W with values in the
range zero to one, and a single mask read through read_mask returning a long tensor of shape
H by W holding class indices with the ignore index for unlabeled pixels. The base class takes
care of resizing to the configured size and the shared augmentation and normalization step.
"""
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from data.transforms import PairedTransform


class BaseSegmentationDataset(Dataset):
    """Common behavior for every dataset in this project."""

    class_names = None  # subclasses may set a list of human readable class names

    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        ds = cfg.dataset
        self.num_classes = int(ds.num_classes)
        self.ignore_index = int(ds.get_path("ignore_index", 255))
        size = ds.image_size
        self.image_size = (int(size[0]), int(size[1]))
        self.transform = PairedTransform(cfg, split)

    def __len__(self):
        raise NotImplementedError

    def read_image(self, idx):
        """Return the image at idx as a float tensor of shape C by H by W in zero to one."""
        raise NotImplementedError

    def read_mask(self, idx):
        """Return the mask at idx as a long tensor of shape H by W of class indices."""
        raise NotImplementedError

    def _resize(self, image, mask):
        """Resize image and mask to the configured size, bilinear for the image and nearest
        for the mask so no new label values are invented."""
        height, width = self.image_size
        if tuple(image.shape[-2:]) != (height, width):
            image = TF.resize(
                image,
                [height, width],
                interpolation=TF.InterpolationMode.BILINEAR,
                antialias=True,
            )
        if tuple(mask.shape[-2:]) != (height, width):
            mask = TF.resize(
                mask.unsqueeze(0),
                [height, width],
                interpolation=TF.InterpolationMode.NEAREST,
            ).squeeze(0)
        return image, mask

    def __getitem__(self, idx):
        image = self.read_image(idx)
        mask = self.read_mask(idx)
        image, mask = self._resize(image, mask)
        image, mask = self.transform(image, mask)
        return image, mask
