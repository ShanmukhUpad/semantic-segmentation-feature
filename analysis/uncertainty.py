"""Label free pixel uncertainty maps computed from model logits.

Every function takes raw logits of shape B by C by H by W and returns a per pixel map of
shape B by H by W with values in the zero to one range. No ground truth is needed, so the
maps work on imagery from anywhere in the world. Except for max_softmax_confidence, where
higher means more confident, higher always means more failure risk. The upload app, the
batch scanner and the signal validation script all share these implementations so the
logic lives in one place.
"""
import math

import torch


def max_softmax_confidence(logits):
    """Return the max softmax probability per pixel, higher means more confident."""
    return torch.softmax(logits, dim=1).max(dim=1).values


def predictive_entropy(logits):
    """Return the Shannon entropy of the softmax per pixel, normalized to zero to one.

    The entropy is divided by the log of the class count, so a uniform distribution
    scores one and a one hot distribution scores zero. High entropy means the model
    spreads its probability over many classes, which flags pixels it is unsure about.
    """
    log_probs = torch.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=1)
    return (entropy / math.log(max(logits.shape[1], 2))).clamp(0.0, 1.0)


def margin(logits):
    """Return one minus the gap between the top two softmax probabilities per pixel.

    A small gap means two classes are competing for the same pixel, a classic ambiguity
    signal at class boundaries and on look alike textures. The gap is flipped to one
    minus the gap so higher means more failure risk, in line with the other maps.
    """
    top2 = torch.softmax(logits, dim=1).topk(2, dim=1).values
    return (1.0 - (top2[:, 0] - top2[:, 1])).clamp(0.0, 1.0)


def failure_score(logits, weights=(0.4, 0.4, 0.2)):
    """Return a combined per pixel failure heatmap in the zero to one range.

    The score blends the normalized entropy, the margin risk and one minus the max
    softmax confidence, with weights in that order. The defaults are 0.4 entropy, 0.4
    margin risk and 0.2 confidence shortfall. Weights are renormalized to sum to one so
    any override still lands in the zero to one range. Higher means more likely wrong.
    """
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("failure_score weights must sum to a positive value")
    w_entropy, w_margin, w_confidence = (float(w) / total for w in weights)
    blended = (
        w_entropy * predictive_entropy(logits)
        + w_margin * margin(logits)
        + w_confidence * (1.0 - max_softmax_confidence(logits))
    )
    return blended.clamp(0.0, 1.0)


def _smoke_check():
    """Run every map on random logits and assert output shapes and value ranges."""
    torch.manual_seed(0)
    logits = torch.randn(2, 7, 16, 16)
    maps = {
        "confidence": max_softmax_confidence(logits),
        "entropy": predictive_entropy(logits),
        "margin risk": margin(logits),
        "failure score": failure_score(logits),
    }
    for name, value in maps.items():
        assert value.shape == (2, 16, 16), f"{name} has shape {tuple(value.shape)}"
        low, high = float(value.min()), float(value.max())
        assert 0.0 <= low and high <= 1.0, f"{name} leaves the zero to one range"
    # A near one hot distribution must read as confident and low risk everywhere.
    sure = torch.full((1, 7, 4, 4), -20.0)
    sure[:, 3] = 20.0
    assert float(predictive_entropy(sure).max()) < 0.01
    assert float(margin(sure).max()) < 0.01
    assert float(failure_score(sure).max()) < 0.01
    assert float(max_softmax_confidence(sure).min()) > 0.99
    print("uncertainty smoke check OK")


if __name__ == "__main__":
    _smoke_check()
