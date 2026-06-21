"""Evaluation entry point.

Run from the project root.

    python scripts/evaluate.py --checkpoint results/dummy_run/checkpoints/best.pth

This loads a trained checkpoint and reports far more than a single mean IoU. It writes overall
mean IoU and pixel accuracy, per class IoU, precision, recall, F1 and accuracy, a confusion
matrix as both PNG and CSV, and per class accuracy and IoU bar charts. All numbers are saved as
CSV and JSON in a structured layout so the failure analysis stage can read them
programmatically. The config is taken from the checkpoint by default, so only the checkpoint
path is required.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from data.registry import build_dataset  # noqa: E402
from models.registry import build_model  # noqa: E402
from utils.config import Config, load_config  # noqa: E402
from utils.device import get_device  # noqa: E402
from utils.metrics import ConfusionMatrix, summarize  # noqa: E402
from utils.plots import save_bar_chart, save_confusion_heatmap  # noqa: E402


def get_class_names(dataset, num_classes):
    """Return human readable class names, unwrapping a Subset when needed."""
    target = getattr(dataset, "dataset", dataset)
    names = getattr(target, "class_names", None)
    if not names:
        names = [f"class_{i}" for i in range(num_classes)]
    return list(names)[:num_classes]


def build_eval_loader(cfg, split):
    """Build a dataset and a non shuffled loader for one split."""
    dataset = build_dataset(cfg, split)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.dataloader.get_path("batch_size", 4)),
        shuffle=False,
        num_workers=int(cfg.dataloader.get_path("num_workers", 0)),
        pin_memory=bool(cfg.dataloader.get_path("pin_memory", False)),
    )
    return dataset, loader


@torch.no_grad()
def run_inference(model, loader, device, num_classes, ignore_index):
    """Run the model over the loader and return a filled confusion matrix."""
    model.eval()
    confusion = ConfusionMatrix(num_classes, ignore_index)
    for images, masks in tqdm(loader, desc="evaluate", leave=False):
        images = images.to(device)
        output = model(images)
        logits = output["out"] if isinstance(output, dict) else output
        preds = logits.argmax(dim=1).cpu()
        confusion.update(masks, preds)
    return confusion


def save_confusion_matrix(confusion, class_names, out_dir):
    """Save the confusion matrix as a raw count CSV and a row normalized heatmap PNG."""
    frame = pd.DataFrame(confusion.mat, index=class_names, columns=class_names)
    frame.to_csv(out_dir / "confusion_matrix.csv")
    save_confusion_heatmap(confusion.mat, class_names, out_dir / "confusion_matrix.png")


def build_results(confusion, class_names, checkpoint, split):
    """Assemble the structured metrics dictionary and a per class table."""
    overall, per_class = summarize(confusion, class_names)
    return {
        "checkpoint": str(checkpoint),
        "split": split,
        "num_classes": confusion.num_classes,
        "overall": overall,
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a segmentation checkpoint")
    parser.add_argument("--checkpoint", required=True, help="path to a saved checkpoint")
    parser.add_argument(
        "--config",
        default=None,
        help="optional config path, the checkpoint config is used when omitted",
    )
    parser.add_argument("--set", nargs="*", default=[], metavar="key=value")
    parser.add_argument("--split", default="val", help="dataset split to evaluate")
    parser.add_argument("--output-dir", default=None, help="where to write the report")
    parser.add_argument("--device", default=None, help="auto, cpu or cuda")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.config is not None:
        cfg = load_config(args.config, args.set)
    else:
        cfg = Config(checkpoint["config"])
        for item in args.set:
            key, _, value = item.partition("=")
            import yaml

            cfg.set_path(key.strip(), yaml.safe_load(value))

    device = get_device(args.device or str(cfg.get_path("device", "auto")))
    num_classes = int(cfg.dataset.num_classes)
    ignore_index = int(cfg.dataset.get_path("ignore_index", 255))

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])

    dataset, loader = build_eval_loader(cfg, args.split)
    class_names = get_class_names(dataset, num_classes)
    print(f"Evaluating {args.checkpoint} on split {args.split} with {len(dataset)} samples")

    confusion = run_inference(model, loader, device, num_classes, ignore_index)

    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(cfg.output_dir) / "evaluation" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    results = build_results(confusion, class_names, args.checkpoint, args.split)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    table = pd.DataFrame(results["per_class"])
    table.to_csv(out_dir / "per_class_metrics.csv", index=False)

    save_confusion_matrix(confusion, class_names, out_dir)
    save_bar_chart(
        [row["accuracy"] for row in results["per_class"]],
        class_names,
        "Per class accuracy",
        "accuracy (recall)",
        out_dir / "per_class_accuracy.png",
    )
    save_bar_chart(
        [row["iou"] for row in results["per_class"]],
        class_names,
        "Per class IoU",
        "IoU",
        out_dir / "per_class_iou.png",
    )

    overall = results["overall"]
    print(
        f"mean IoU {overall['mean_iou']:.4f}  pixel accuracy {overall['pixel_accuracy']:.4f}  "
        f"mean F1 {overall['mean_f1']:.4f}"
    )
    print(table.to_string(index=False))
    print(f"\nSaved evaluation report to {out_dir}")


if __name__ == "__main__":
    main()
