"""Standard segmentation losses.

Cross entropy is the default objective. It optionally accepts per class weights through
train.class_weights in the config, which is useful when rare classes are underlearned. New
research losses are added in their own module and registered the same way.
"""
import torch
import torch.nn as nn

from losses.registry import register_loss


@register_loss("cross_entropy")
def build_cross_entropy(cfg, ignore_index):
    """Build a cross entropy loss with optional per class weights and a masked ignore index."""
    weights = cfg.train.get_path("class_weights", None)
    weight = torch.tensor(weights, dtype=torch.float32) if weights else None
    return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
