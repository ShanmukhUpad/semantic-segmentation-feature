"""Quick data exploration utility.

This prints the per class pixel distribution, prints the image and mask tensor shapes and
dtypes, and saves a side by side grid of sample images with their masks. Run it as a module
from the project root.

    python -m data.explore --config configs/dummy.yaml

Augmentation is turned off here so the printed distribution and the saved figure show the raw
data rather than a randomly flipped or jittered view.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before this import)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from data.registry import build_dataset  # noqa: E402
from utils.config import load_config  # noqa: E402


def denormalize(image, mean, std):
    """Undo normalization so an image can be shown with its original colors."""
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


def class_distribution(dataset, num_classes, ignore_index, max_samples=None):
    """Count pixels per class across the dataset and report ignored pixels.

    Returns the per class counts, the number of ignored pixels and the number of samples used.
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    ignored = 0
    total = len(dataset)
    used = total if max_samples is None else min(max_samples, total)
    for i in range(used):
        _, mask = dataset[i]
        flat = mask.numpy().ravel()
        ignored += int((flat == ignore_index).sum())
        valid = flat[flat != ignore_index]
        if valid.size:
            counts += np.bincount(valid, minlength=num_classes)[:num_classes]
    return counts, ignored, used


def save_sample_grid(dataset, num_samples, num_classes, mean, std, out_path):
    """Save a figure with sample images next to their masks."""
    count = min(num_samples, len(dataset))
    fig, axes = plt.subplots(count, 2, figsize=(6, 3 * count))
    if count == 1:
        axes = axes.reshape(1, 2)
    cmap = plt.get_cmap("tab20", num_classes)
    for row in range(count):
        image, mask = dataset[row]
        display = denormalize(image, mean, std).permute(1, 2, 0).numpy()
        axes[row, 0].imshow(display)
        axes[row, 0].set_title(f"image {row}")
        axes[row, 0].axis("off")
        axes[row, 1].imshow(mask.numpy(), cmap=cmap, vmin=0, vmax=num_classes - 1)
        axes[row, 1].set_title(f"mask {row}")
        axes[row, 1].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Explore a segmentation dataset")
    parser.add_argument("--config", required=True, help="path to a YAML config file")
    parser.add_argument("--split", default="train", help="dataset split to explore")
    parser.add_argument("--num-samples", type=int, default=4, help="samples in the figure")
    parser.add_argument(
        "--max-dist-samples",
        type=int,
        default=None,
        help="cap on samples used for the distribution count",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.set_path("augmentation.enabled", False)

    num_classes = int(cfg.dataset.num_classes)
    ignore_index = int(cfg.dataset.get_path("ignore_index", 255))
    mean = list(cfg.augmentation.get_path("mean", [0.485, 0.456, 0.406]))
    std = list(cfg.augmentation.get_path("std", [0.229, 0.224, 0.225]))

    dataset = build_dataset(cfg, args.split)
    print(f"Dataset {cfg.dataset.name} split {args.split} has {len(dataset)} samples")

    image0, mask0 = dataset[0]
    print(f"Image tensor shape {tuple(image0.shape)} dtype {image0.dtype}")
    print(f"Mask tensor shape {tuple(mask0.shape)} dtype {mask0.dtype}")

    counts, ignored, used = class_distribution(
        dataset, num_classes, ignore_index, args.max_dist_samples
    )
    names = getattr(dataset, "class_names", None) or [
        f"class_{i}" for i in range(num_classes)
    ]
    total = int(counts.sum())
    table = pd.DataFrame(
        {
            "class_id": list(range(num_classes)),
            "class_name": names,
            "pixels": counts,
            "percent": (counts / max(total, 1) * 100).round(3),
        }
    )
    print(f"\nClass distribution over {used} samples (ignored pixels {ignored})")
    print(table.to_string(index=False))

    out_dir = Path(cfg.get_path("output_dir", "results/run")) / "exploration"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "class_distribution.csv"
    table.to_csv(csv_path, index=False)

    fig_path = out_dir / "samples.png"
    save_sample_grid(dataset, args.num_samples, num_classes, mean, std, fig_path)

    print(f"\nSaved class distribution to {csv_path}")
    print(f"Saved sample grid to {fig_path}")


if __name__ == "__main__":
    main()
