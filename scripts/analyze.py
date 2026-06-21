"""Failure analysis entry point.

Run from the project root.

    python scripts/analyze.py --checkpoint results/dummy_run/checkpoints/best.pth

This takes a trained checkpoint and produces a structured failure analysis report. In a single
pass over the chosen split it builds the overall and per class metrics, the class size analysis,
the boundary versus interior analysis, the confidence versus correctness analysis and the error
maps for the worst scoring images. It then writes a markdown report along with all figures, a
machine readable analysis.json and the per class and confusion matrix CSV files. The config is
taken from the checkpoint by default, so only the checkpoint path is required.
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

from analysis.boundary import BoundaryAnalyzer  # noqa: E402
from analysis.class_size import analyze_class_size, save_class_size_figure  # noqa: E402
from analysis.confidence import ConfidenceAnalyzer  # noqa: E402
from analysis.error_maps import WorstImageCollector, get_palette, render_error_map  # noqa: E402
from analysis.report import write_markdown_report  # noqa: E402
from data.registry import build_dataset  # noqa: E402
from models.registry import build_model  # noqa: E402
from utils.config import Config, load_config  # noqa: E402
from utils.device import get_device  # noqa: E402
from utils.metrics import ConfusionMatrix, summarize  # noqa: E402
from utils.plots import save_confusion_heatmap  # noqa: E402


def denormalize(image, mean, std):
    """Undo normalization so an image can be saved with its original colors."""
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


def get_class_names(dataset, num_classes):
    """Return human readable class names, unwrapping a Subset when needed."""
    target = getattr(dataset, "dataset", dataset)
    names = getattr(target, "class_names", None)
    if not names:
        names = [f"class_{i}" for i in range(num_classes)]
    return list(names)[:num_classes]


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


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze segmentation failure modes")
    parser.add_argument("--checkpoint", required=True, help="path to a saved checkpoint")
    parser.add_argument("--config", default=None, help="optional config path override")
    parser.add_argument("--set", nargs="*", default=[], metavar="key=value")
    parser.add_argument("--split", default="val", help="dataset split to analyze")
    parser.add_argument("--output-dir", default=None, help="where to write the report")
    parser.add_argument("--device", default=None, help="auto, cpu or cuda")
    parser.add_argument("--boundary-width", type=int, default=None)
    parser.add_argument("--num-error-images", type=int, default=None)
    parser.add_argument("--confidence-bins", type=int, default=None)
    parser.add_argument("--size-bins", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = load_checkpoint_config(checkpoint, args.config, args.set)

    device = get_device(args.device or str(cfg.get_path("device", "auto")))
    num_classes = int(cfg.dataset.num_classes)
    ignore_index = int(cfg.dataset.get_path("ignore_index", 255))
    mean = list(cfg.augmentation.get_path("mean", [0.485, 0.456, 0.406]))
    std = list(cfg.augmentation.get_path("std", [0.229, 0.224, 0.225]))

    boundary_width = (
        args.boundary_width
        if args.boundary_width is not None
        else int(cfg.get_path("analysis.boundary_width", 3))
    )
    num_error_images = (
        args.num_error_images
        if args.num_error_images is not None
        else int(cfg.get_path("analysis.num_error_images", 6))
    )
    confidence_bins = (
        args.confidence_bins
        if args.confidence_bins is not None
        else int(cfg.get_path("analysis.confidence_bins", 20))
    )
    size_bins = (
        args.size_bins
        if args.size_bins is not None
        else int(cfg.get_path("analysis.size_bins", 3))
    )

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = build_dataset(cfg, args.split)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.dataloader.get_path("batch_size", 4)),
        shuffle=False,
        num_workers=int(cfg.dataloader.get_path("num_workers", 0)),
        pin_memory=bool(cfg.dataloader.get_path("pin_memory", False)),
    )
    class_names = get_class_names(dataset, num_classes)
    palette = get_palette(num_classes)

    overall_cm = ConfusionMatrix(num_classes, ignore_index)
    boundary = BoundaryAnalyzer(num_classes, ignore_index, boundary_width)
    confidence = ConfidenceAnalyzer(confidence_bins, ignore_index)
    collector = WorstImageCollector(num_error_images)

    print(
        f"Analyzing {args.checkpoint} on split {args.split} with {len(dataset)} samples"
    )

    running_index = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="analyze", leave=False):
            output = model(images.to(device))
            logits = output["out"] if isinstance(output, dict) else output
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            pred = pred.cpu()
            conf = conf.cpu()
            for b in range(images.size(0)):
                gt = masks[b].numpy()
                pr = pred[b].numpy()
                cf = conf[b].numpy()
                overall_cm.update(gt, pr)
                boundary.update(gt, pr)
                confidence.update(gt, pr, cf)
                per_image = ConfusionMatrix(num_classes, ignore_index)
                per_image.update(gt, pr)
                score = per_image.mean_iou()
                image_hwc = denormalize(images[b], mean, std).permute(1, 2, 0).numpy()
                collector.consider(score, running_index, image_hwc, gt, pr)
                running_index += 1

    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(cfg.output_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    error_dir = out_dir / "error_maps"
    error_dir.mkdir(parents=True, exist_ok=True)

    overall, per_class = summarize(overall_cm, class_names)
    pd.DataFrame(per_class).to_csv(out_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(overall_cm.mat, index=class_names, columns=class_names).to_csv(
        out_dir / "confusion_matrix.csv"
    )
    save_confusion_heatmap(overall_cm.mat, class_names, out_dir / "confusion_matrix.png")

    class_size = analyze_class_size(overall_cm, class_names, size_bins)
    save_class_size_figure(class_size, out_dir / "class_size_bins.png")

    boundary_result = boundary.results()
    confidence_result = confidence.results()
    confidence.save_figure(out_dir / "confidence_histogram.png")

    worst_images = []
    for rank, entry in enumerate(collector.sorted_items(), start=1):
        file_name = f"worst_{rank:02d}_idx{entry['index']}.png"
        render_error_map(entry, num_classes, ignore_index, palette, error_dir / file_name)
        worst_images.append(
            {
                "index": entry["index"],
                "miou": entry["score"],
                "figure": f"error_maps/{file_name}",
            }
        )

    results = {
        "model": cfg.model.name,
        "dataset": cfg.dataset.name,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "num_samples": running_index,
        "num_classes": num_classes,
        "overall": overall,
        "per_class": per_class,
        "class_size": class_size,
        "boundary": boundary_result,
        "confidence": confidence_result,
        "worst_images": worst_images,
    }
    with open(out_dir / "analysis.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    figures = {
        "confusion_matrix": "confusion_matrix.png",
        "class_size": "class_size_bins.png",
        "confidence": "confidence_histogram.png",
    }
    report_path = write_markdown_report(out_dir, results, figures)

    print(
        f"mean IoU {overall['mean_iou']:.4f}  "
        f"boundary mIoU {boundary_result['boundary']['mean_iou']:.4f}  "
        f"interior mIoU {boundary_result['interior']['mean_iou']:.4f}"
    )
    print(
        f"mean confidence correct {confidence_result['mean_confidence_correct']}  "
        f"incorrect {confidence_result['mean_confidence_incorrect']}"
    )
    print(f"Wrote report to {report_path}")
    print(f"Wrote analysis artifacts to {out_dir}")


if __name__ == "__main__":
    main()
