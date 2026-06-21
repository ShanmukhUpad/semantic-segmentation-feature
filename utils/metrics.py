"""Confusion matrix based segmentation metrics.

A single ConfusionMatrix accumulates predictions over a dataset and then yields mean IoU, pixel
accuracy and per class IoU, precision, recall and F1. The same object is reused by the training
loop for validation logging and by the evaluation and failure analysis stages, so all stages
report numbers computed the same way. Pixels equal to the ignore index are excluded.
"""
import math

import numpy as np


def _to_numpy(array):
    """Return a flat numpy int64 view of a torch tensor or numpy array."""
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array).reshape(-1).astype(np.int64)


def _safe_divide(numerator, denominator):
    """Divide elementwise and return nan where the denominator is zero."""
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


class ConfusionMatrix:
    """Accumulate a class by class confusion matrix and derive metrics from it.

    Rows index the ground truth class and columns index the predicted class, so the entry at
    row i column j counts pixels whose true class is i and predicted class is j.
    """

    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def reset(self):
        self.mat[:] = 0

    def update(self, target, pred):
        """Add one batch of ground truth and predicted labels to the matrix."""
        target = _to_numpy(target)
        pred = _to_numpy(pred)
        valid = target != self.ignore_index
        target = target[valid]
        pred = pred[valid]
        index = target * self.num_classes + pred
        counts = np.bincount(index, minlength=self.num_classes ** 2)
        self.mat += counts.reshape(self.num_classes, self.num_classes)

    def iou_per_class(self):
        """Return the IoU for each class, nan where a class never appears."""
        true_positive = np.diag(self.mat).astype(np.float64)
        false_positive = self.mat.sum(axis=0) - true_positive
        false_negative = self.mat.sum(axis=1) - true_positive
        denom = true_positive + false_positive + false_negative
        return _safe_divide(true_positive, denom)

    def mean_iou(self):
        """Return the mean IoU across classes that appear at least once."""
        iou = self.iou_per_class()
        return float(np.nanmean(iou)) if np.isfinite(iou).any() else 0.0

    def pixel_accuracy(self):
        """Return the fraction of valid pixels predicted correctly."""
        total = self.mat.sum()
        return float(np.diag(self.mat).sum() / total) if total > 0 else 0.0

    def precision_recall_f1(self):
        """Return per class precision, recall and F1 as three numpy arrays."""
        true_positive = np.diag(self.mat).astype(np.float64)
        false_positive = self.mat.sum(axis=0) - true_positive
        false_negative = self.mat.sum(axis=1) - true_positive
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        denom = precision + recall
        f1 = _safe_divide(2 * precision * recall, denom)
        return precision, recall, f1


def jsonable(value):
    """Convert a numpy scalar or a not a number value into a plain JSON friendly value."""
    if value is None:
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) else number
    return value


def _nanmean_or_none(values):
    """Return the mean of the finite entries, or None when none are finite."""
    values = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(values)) if np.isfinite(values).any() else None


def summarize(confusion, class_names):
    """Return an overall metrics dict and a per class list from a confusion matrix.

    Both evaluate and analyze use this so every stage reports identical numbers. Per class
    accuracy is the recall, that is the fraction of that class ground truth predicted correctly.
    """
    iou = confusion.iou_per_class()
    precision, recall, f1 = confusion.precision_recall_f1()
    support = confusion.mat.sum(axis=1)

    per_class = []
    for index in range(confusion.num_classes):
        per_class.append(
            {
                "class_id": index,
                "class_name": class_names[index],
                "support": int(support[index]),
                "iou": jsonable(iou[index]),
                "precision": jsonable(precision[index]),
                "recall": jsonable(recall[index]),
                "f1": jsonable(f1[index]),
                "accuracy": jsonable(recall[index]),
            }
        )

    overall = {
        "mean_iou": jsonable(confusion.mean_iou()),
        "pixel_accuracy": jsonable(confusion.pixel_accuracy()),
        "mean_f1": _nanmean_or_none(f1),
        "macro_precision": _nanmean_or_none(precision),
        "macro_recall": _nanmean_or_none(recall),
    }
    return overall, per_class
