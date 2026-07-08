"""Out of distribution novelty scoring against the training distribution.

The pixel uncertainty maps catch local ambiguity, but a model can be confidently and
uniformly wrong on a scene unlike anything it trained on. This module scores that case.
Each image is represented by its global average pooled backbone feature vector. A
Gaussian with shrinkage is fit to a sample of training images once, and a new image is
scored by its Mahalanobis distance to that Gaussian. Raw distances are normalized
against the stored training distances, so a score near zero sits inside the training
distribution and a score near one sits at or beyond the training tail. Higher means
more out of distribution.
"""
from pathlib import Path

import numpy as np
import torch


def _to_numpy_2d(features):
    """Return features as a two dimensional float64 numpy array."""
    if hasattr(features, "detach"):
        features = features.detach().cpu().numpy()
    array = np.asarray(features, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    return array


@torch.no_grad()
def extract_features(model, images):
    """Return one global average pooled backbone feature vector per image.

    Works with the torchvision DeepLabV3 family, where model.backbone returns a
    dictionary holding the final feature map under the key out. Any model exposing a
    backbone attribute with the same contract works as well. The input batch must
    already be normalized the same way as during training.
    """
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise AttributeError("model has no backbone attribute to extract features from")
    features = backbone(images)
    if isinstance(features, dict):
        features = features["out"]
    return features.mean(dim=(2, 3))


def fit_reference(features, rgb_mean=None, rgb_std=None, shrinkage=0.1):
    """Fit the Gaussian reference and return it as a plain dictionary.

    features is an N by D array of training feature vectors. The covariance is shrunk
    toward its own diagonal by the shrinkage fraction and given a small ridge, so the
    inverse stays stable even when N is far below D. The training distances are stored
    sorted so a new distance can be turned into a percentile style novelty score. The
    optional rgb mean and std of the training tiles ride along as a cheap extra
    fingerprint of the training imagery.
    """
    array = _to_numpy_2d(features)
    if array.shape[0] < 2:
        raise ValueError("fit_reference needs at least two feature vectors")
    mean = array.mean(axis=0)
    covariance = np.cov(array, rowvar=False)
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * np.diag(np.diag(covariance))
    ridge = 1e-6 * float(np.trace(covariance)) / covariance.shape[0]
    covariance += ridge * np.eye(covariance.shape[0])
    precision = np.linalg.pinv(covariance)
    delta = array - mean
    distances = np.sqrt(np.maximum(np.einsum("nd,de,ne->n", delta, precision, delta), 0.0))
    reference = {
        "feature_mean": mean,
        "precision": precision,
        "train_distances": np.sort(distances),
        "shrinkage": shrinkage,
        "num_images": int(array.shape[0]),
    }
    if rgb_mean is not None:
        reference["rgb_mean"] = np.asarray(rgb_mean, dtype=np.float64)
    if rgb_std is not None:
        reference["rgb_std"] = np.asarray(rgb_std, dtype=np.float64)
    return reference


def mahalanobis(features, reference):
    """Return the Mahalanobis distance of each feature vector to the reference Gaussian."""
    array = _to_numpy_2d(features)
    delta = array - np.asarray(reference["feature_mean"], dtype=np.float64)
    precision = np.asarray(reference["precision"], dtype=np.float64)
    return np.sqrt(np.maximum(np.einsum("nd,de,ne->n", delta, precision, delta), 0.0))


def normalize_novelty(distances, reference):
    """Return raw distances as zero to one novelty scores against the training set.

    The score is the fraction of stored training distances that fall below the given
    distance, so 0.5 sits at the training median and 1.0 sits at or beyond the largest
    training distance. Higher means more out of distribution.
    """
    train = np.asarray(reference["train_distances"], dtype=np.float64)
    distances = np.atleast_1d(np.asarray(distances, dtype=np.float64))
    ranks = np.searchsorted(train, distances, side="right")
    return ranks / float(len(train))


def verdict(distance, reference):
    """Return low, medium or high novelty for one raw Mahalanobis distance.

    Low means the distance sits inside the bulk of the training distances, below their
    95th percentile. Medium means it sits in the training tail or a little beyond, up
    to 1.25 times the largest training distance. High means it is far beyond anything
    seen during training, where predictions deserve very little trust.
    """
    train = np.asarray(reference["train_distances"], dtype=np.float64)
    if distance <= float(np.percentile(train, 95)):
        return "low"
    if distance <= 1.25 * float(train[-1]):
        return "medium"
    return "high"


def score_images(model, images, reference):
    """Return raw distances and normalized novelty scores for one normalized batch."""
    features = extract_features(model, images)
    distances = mahalanobis(features, reference)
    return distances, normalize_novelty(distances, reference)


class BackboneCatcher:
    """Capture the backbone feature map during the main forward pass.

    Scoring novelty alongside a prediction would otherwise run the backbone twice. The
    catcher hooks the backbone once, records its output during the normal forward, and
    hands back the pooled features. When the model exposes no backbone attribute the
    catcher stays inert and features returns None.
    """

    def __init__(self, model):
        self.value = None
        self.handle = None
        backbone = getattr(model, "backbone", None)
        if backbone is not None:
            self.handle = backbone.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.value = output

    def features(self):
        """Return global average pooled features from the last forward, or None."""
        if self.value is None:
            return None
        value = self.value
        if isinstance(value, dict):
            value = value["out"]
        return value.mean(dim=(2, 3))

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def save_reference(reference, path):
    """Write the reference dictionary to disk, creating parent folders when needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(reference, path)


def load_reference(path):
    """Load a saved reference, or return None when the file does not exist."""
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _smoke_check():
    """Fit a reference on random features and check the scores behave sensibly."""
    rng = np.random.default_rng(0)
    train = rng.normal(0.0, 1.0, size=(60, 12))
    reference = fit_reference(train, rgb_mean=[0.4, 0.4, 0.4], rgb_std=[0.2, 0.2, 0.2])
    inside = rng.normal(0.0, 1.0, size=(4, 12))
    far = np.full((1, 12), 25.0)
    inside_novelty = normalize_novelty(mahalanobis(inside, reference), reference)
    far_distance = mahalanobis(far, reference)
    far_novelty = normalize_novelty(far_distance, reference)
    assert inside_novelty.shape == (4,)
    assert float(far_novelty[0]) == 1.0
    assert verdict(float(far_distance[0]), reference) == "high"
    assert verdict(float(np.median(reference["train_distances"])), reference) == "low"
    print("ood smoke check OK")


if __name__ == "__main__":
    _smoke_check()
