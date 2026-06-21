"""Boundary error analysis.

Segmentation errors tend to cluster along the edges between classes. This module splits each
image into a boundary band, the pixels within a configurable number of pixels of any class
boundary, and the interior, everything else. It accumulates a separate confusion matrix for
each region so the mean IoU on boundaries can be compared with the mean IoU in interiors. A
large gap means the model is mostly failing at edges rather than inside regions.
"""
import numpy as np
from scipy.ndimage import binary_dilation

from utils.metrics import ConfusionMatrix


def _to_numpy_2d(array):
    """Return a two dimensional numpy int64 array from a tensor or array."""
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array).astype(np.int64)


def boundary_band(gt, width):
    """Return a boolean mask of pixels within width pixels of a class boundary.

    A boundary pixel is one where the ground truth label changes between neighbors. The one
    pixel boundary line is then dilated by width so the band has the requested thickness.
    """
    edge = np.zeros(gt.shape, dtype=bool)
    horizontal = gt[:, 1:] != gt[:, :-1]
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    vertical = gt[1:, :] != gt[:-1, :]
    edge[1:, :] |= vertical
    edge[:-1, :] |= vertical
    if width and width > 0:
        return binary_dilation(edge, iterations=int(width))
    return edge


class BoundaryAnalyzer:
    """Accumulate confusion matrices for boundary band pixels and interior pixels."""

    def __init__(self, num_classes, ignore_index, width):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.width = width
        self.boundary = ConfusionMatrix(num_classes, ignore_index)
        self.interior = ConfusionMatrix(num_classes, ignore_index)

    def update(self, gt, pred):
        """Add one image of ground truth and prediction, split into the two regions."""
        gt = _to_numpy_2d(gt)
        pred = _to_numpy_2d(pred)
        band = boundary_band(gt, self.width)
        valid = gt != self.ignore_index
        in_band = band & valid
        in_interior = (~band) & valid
        self.boundary.update(
            np.where(in_band, gt, self.ignore_index),
            np.where(in_band, pred, self.ignore_index),
        )
        self.interior.update(
            np.where(in_interior, gt, self.ignore_index),
            np.where(in_interior, pred, self.ignore_index),
        )

    def results(self):
        """Return mean IoU and pixel accuracy for each region and the IoU gap."""
        boundary_miou = self.boundary.mean_iou()
        interior_miou = self.interior.mean_iou()
        return {
            "boundary_width": self.width,
            "boundary": {
                "mean_iou": boundary_miou,
                "pixel_accuracy": self.boundary.pixel_accuracy(),
                "pixels": int(self.boundary.mat.sum()),
            },
            "interior": {
                "mean_iou": interior_miou,
                "pixel_accuracy": self.interior.pixel_accuracy(),
                "pixels": int(self.interior.mat.sum()),
            },
            "interior_minus_boundary_miou": interior_miou - boundary_miou,
        }
