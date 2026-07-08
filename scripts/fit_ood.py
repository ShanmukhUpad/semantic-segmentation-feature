"""Fit the out of distribution novelty reference for a trained checkpoint.

Run from the project root.

    python scripts/fit_ood.py --checkpoint results/loveda_deeplabv3/checkpoints/best.pth

This runs a sample of the training split through the model backbone, fits the Gaussian
reference described in analysis/ood.py and saves it as ood_reference.pt next to the
checkpoint. The upload app, the failure scanner and the signal validation script pick it
up from there automatically. The config is taken from the checkpoint by default, so only
the checkpoint path is required. For a quick smoke test point it at a dummy checkpoint or
pass --set dataset.subset=16.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from analysis import ood  # noqa: E402
from data.registry import build_dataset  # noqa: E402
from models.registry import build_model  # noqa: E402
from utils.config import Config, load_config  # noqa: E402
from utils.device import get_device  # noqa: E402
from utils.seed import set_seed  # noqa: E402


def load_checkpoint_config(checkpoint, config_path, overrides):
    """Return the config, preferring an explicit path and falling back to the checkpoint."""
    if config_path is not None:
        return load_config(config_path, overrides)
    cfg = Config(checkpoint["config"])
    import yaml

    for item in overrides:
        key, _, value = item.partition("=")
        cfg.set_path(key.strip(), yaml.safe_load(value))
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Fit the feature space novelty reference for a checkpoint"
    )
    parser.add_argument("--checkpoint", required=True, help="path to a saved checkpoint")
    parser.add_argument("--config", default=None, help="optional config path override")
    parser.add_argument("--set", nargs="*", default=[], metavar="key=value")
    parser.add_argument("--split", default="train", help="split the reference is fit on")
    parser.add_argument(
        "--num-images",
        type=int,
        default=400,
        help="cap on the number of images used to fit the reference",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output path, defaults to ood_reference.pt next to the checkpoint",
    )
    parser.add_argument("--shrinkage", type=float, default=0.1, help="covariance shrinkage")
    parser.add_argument("--batch-size", type=int, default=None, help="loader batch override")
    parser.add_argument("--device", default=None, help="auto, cpu or cuda")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = load_checkpoint_config(checkpoint, args.config, args.set)
    # Augmentation is switched off so the reference features are deterministic.
    cfg.set_path("augmentation.enabled", False)

    seed = int(cfg.get_path("seed", 42))
    set_seed(seed)
    device = get_device(args.device or str(cfg.get_path("device", "auto")))

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = build_dataset(cfg, args.split)
    keep = len(dataset)
    if args.num_images and args.num_images < keep:
        keep = args.num_images
    # A seeded random sample keeps the reference representative even when the dataset
    # lists its images grouped by region.
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(dataset), size=keep, replace=False).tolist())
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=int(args.batch_size or cfg.dataloader.get_path("batch_size", 4)),
        shuffle=False,
        num_workers=int(cfg.dataloader.get_path("num_workers", 0)),
        pin_memory=bool(cfg.dataloader.get_path("pin_memory", False)),
    )

    print(
        f"Fitting the novelty reference on {keep} images from the {cfg.dataset.name} "
        f"{args.split} split on device {device}"
    )

    mean_cfg = torch.tensor(
        cfg.augmentation.get_path("mean", [0.485, 0.456, 0.406])
    ).view(1, 3, 1, 1)
    std_cfg = torch.tensor(
        cfg.augmentation.get_path("std", [0.229, 0.224, 0.225])
    ).view(1, 3, 1, 1)

    features = []
    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_squares = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0
    for images, _ in tqdm(loader, desc="fit ood", leave=False):
        features.append(ood.extract_features(model, images.to(device)).cpu())
        # Undo the normalization to record the raw zero to one channel statistics.
        raw = (images * std_cfg + mean_cfg).clamp(0.0, 1.0).double()
        channel_sum += raw.sum(dim=(0, 2, 3))
        channel_squares += raw.pow(2).sum(dim=(0, 2, 3))
        pixel_count += raw.shape[0] * raw.shape[2] * raw.shape[3]

    features = torch.cat(features, dim=0)
    rgb_mean = (channel_sum / pixel_count).numpy()
    rgb_std = torch.sqrt(
        (channel_squares / pixel_count - (channel_sum / pixel_count).pow(2)).clamp(min=0.0)
    ).numpy()

    reference = ood.fit_reference(
        features, rgb_mean=rgb_mean, rgb_std=rgb_std, shrinkage=args.shrinkage
    )

    output = Path(args.output) if args.output else Path(args.checkpoint).parent / "ood_reference.pt"
    ood.save_reference(reference, output)

    train_distances = reference["train_distances"]
    print(
        f"feature dimension {len(reference['feature_mean'])}  "
        f"shrinkage {reference['shrinkage']}"
    )
    print(
        f"training Mahalanobis distances  median {np.median(train_distances):.2f}  "
        f"p95 {np.percentile(train_distances, 95):.2f}  max {train_distances[-1]:.2f}"
    )
    print(f"Saved the novelty reference to {output}")


if __name__ == "__main__":
    main()
