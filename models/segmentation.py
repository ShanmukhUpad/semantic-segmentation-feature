"""Segmentation model builders backed by torchvision.

The baseline is DeepLabV3 with a ResNet50 backbone. The class count comes from the config so
the head matches the dataset. When pretrained is true the backbone is initialized with
ImageNet weights and the segmentation head starts random, which is the usual transfer learning
setup for a new dataset. Auxiliary loss is enabled so the training loop can use the aux head.

Adding another architecture means writing a builder like the one below and decorating it with
register_model. The training, evaluation and analysis code does not need to change.
"""
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50

from models.registry import register_model


@register_model("deeplabv3_resnet50")
def build_deeplabv3_resnet50(cfg):
    """Build a DeepLabV3 ResNet50 with a head sized to the dataset class count."""
    num_classes = int(cfg.dataset.num_classes)
    pretrained = bool(cfg.model.get_path("pretrained", False))
    weights_backbone = ResNet50_Weights.DEFAULT if pretrained else None
    model = deeplabv3_resnet50(
        weights=None,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
        aux_loss=True,
    )
    return model
