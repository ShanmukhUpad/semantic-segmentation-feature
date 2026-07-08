"""Label free failure scan over arbitrary aerial imagery.

Run from the project root.

    python scripts/scan_failures.py --checkpoint results/loveda_deeplabv3/checkpoints/best.pth --images path/to/tiles

This is the worldwide tool. It takes a folder of aerial images, or a single image, with
no ground truth anywhere, and produces per image failure visuals plus a ranked report of
where the model looks least trustworthy. Two label free signal families drive it. The
pixel uncertainty maps from analysis/uncertainty.py flag ambiguous pixels, and the
novelty score from analysis/ood.py flags scenes unlike the training distribution, where
the model can be confidently wrong.

The model was trained at 0.3 m per pixel, so scale matters. Instead of squashing a large
image down to the network input size, the scanner slides a window at the native
resolution and stitches the results back together, which preserves the ground sample
distance. Imagery at a very different resolution still degrades and that degradation is
exactly what the failure maps surface.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from analysis import ood  # noqa: E402
from analysis.error_maps import colorize, get_palette  # noqa: E402
from analysis.uncertainty import (  # noqa: E402
    failure_score,
    margin,
    max_softmax_confidence,
    predictive_entropy,
)
from models.registry import build_model  # noqa: E402
from utils.config import Config, load_config  # noqa: E402
from utils.device import get_device  # noqa: E402
from utils.seed import set_seed  # noqa: E402

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

Image.MAX_IMAGE_PIXELS = None  # orthophotos are large on purpose


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


def collect_images(images_arg):
    """Return the list of image paths named by a file or a directory argument."""
    path = Path(images_arg)
    if path.is_file():
        return [path]
    if path.is_dir():
        found = sorted(
            p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not found:
            raise FileNotFoundError(f"no images with a known extension under {path}")
        return found
    raise FileNotFoundError(f"{path} is neither a file nor a directory")


def window_positions(length, tile, stride):
    """Return window start offsets covering one dimension, always touching the far edge."""
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def _md_table(headers, rows):
    """Build a github flavored markdown table from headers and a list of row lists."""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def scan_image(image_path, model, device, mean, std, tile, overlap, batch_size, reference):
    """Scan one image with a sliding window and return stitched maps and window scores.

    Returns the image as a float array, the stitched class prediction, a dictionary of
    stitched risk maps, the per window novelty heatmap and the list of per window
    novelty scores. Novelty entries are None when no reference is available.
    """
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    original_h, original_w = image.shape[:2]

    # Small images are padded up to the window size with edge replication and the
    # padding is cropped away from every output at the end.
    pad_h = max(tile - original_h, 0)
    pad_w = max(tile - original_w, 0)
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    height, width = image.shape[:2]

    stride = max(tile - overlap, 1)
    positions = [
        (y, x)
        for y in window_positions(height, tile, stride)
        for x in window_positions(width, tile, stride)
    ]

    num_classes = None
    probs_sum = None
    counts = torch.zeros((height, width), dtype=torch.int32)
    novelty_sum = torch.zeros((height, width), dtype=torch.float32)
    window_novelties = []

    catcher = ood.BackboneCatcher(model) if reference is not None else None
    tensor = torch.from_numpy(image).permute(2, 0, 1)

    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        batch = torch.stack(
            [tensor[:, y : y + tile, x : x + tile] for y, x in chunk]
        )
        batch = TF.normalize(batch, mean, std).to(device)
        with torch.no_grad():
            output = model(batch)
            logits = output["out"] if isinstance(output, dict) else output
            probs = torch.softmax(logits, dim=1).cpu()
            if catcher is not None:
                features = catcher.features()
        if probs_sum is None:
            num_classes = probs.shape[1]
            estimated = num_classes * height * width * 4
            dtype = torch.float32 if estimated < 1_500_000_000 else torch.float16
            probs_sum = torch.zeros((num_classes, height, width), dtype=dtype)

        chunk_novelty = None
        if catcher is not None and features is not None:
            distances = ood.mahalanobis(features, reference)
            chunk_novelty = ood.normalize_novelty(distances, reference)
            window_novelties.extend(float(v) for v in chunk_novelty)

        for k, (y, x) in enumerate(chunk):
            probs_sum[:, y : y + tile, x : x + tile] += probs[k].to(probs_sum.dtype)
            counts[y : y + tile, x : x + tile] += 1
            if chunk_novelty is not None:
                novelty_sum[y : y + tile, x : x + tile] += float(chunk_novelty[k])

    if catcher is not None:
        catcher.close()

    weight = counts.clamp(min=1)
    probs_mean = probs_sum.float() / weight
    prediction = probs_mean.argmax(dim=0).numpy().astype(np.int64)
    novelty_map = (novelty_sum / weight).numpy() if reference is not None else None

    # The stitched averaged probabilities act as the distribution for the risk maps.
    # log of a probability vector is a valid logit vector because softmax undoes it.
    maps = {name: np.zeros((height, width), dtype=np.float32) for name in
            ("confidence", "entropy", "margin risk", "failure score")}
    chunk_rows = 1024
    for y0 in range(0, height, chunk_rows):
        y1 = min(y0 + chunk_rows, height)
        pseudo_logits = torch.log(probs_mean[:, y0:y1, :].unsqueeze(0) + 1e-8)
        maps["confidence"][y0:y1] = max_softmax_confidence(pseudo_logits)[0].numpy()
        maps["entropy"][y0:y1] = predictive_entropy(pseudo_logits)[0].numpy()
        maps["margin risk"][y0:y1] = margin(pseudo_logits)[0].numpy()
        maps["failure score"][y0:y1] = failure_score(pseudo_logits)[0].numpy()

    crop = (slice(0, original_h), slice(0, original_w))
    image = image[crop]
    prediction = prediction[crop]
    maps = {name: value[crop] for name, value in maps.items()}
    if novelty_map is not None:
        novelty_map = novelty_map[crop]

    return image, prediction, maps, novelty_map, window_novelties, len(positions)


def save_panels(out_path, image, prediction, maps, novelty_map, num_classes, ignore_index, palette):
    """Save the five panel figure for one scanned image."""
    prediction_color = colorize(prediction, num_classes, palette, ignore_index)
    overlay = (0.5 * image * 255 + 0.5 * prediction_color).astype(np.uint8)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.4))
    axes[0].imshow(np.clip(image, 0.0, 1.0))
    axes[0].set_title("input")
    axes[1].imshow(overlay)
    axes[1].set_title("prediction overlay")
    axes[2].imshow(maps["failure score"], cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("failure score, bright is likely wrong")
    axes[3].imshow(maps["entropy"], cmap="magma", vmin=0.0, vmax=1.0)
    axes[3].set_title("entropy, bright is uncertain")
    if novelty_map is not None:
        axes[4].imshow(novelty_map, cmap="magma", vmin=0.0, vmax=1.0)
        axes[4].set_title("novelty per window, bright is unfamiliar")
    else:
        axes[4].imshow(np.zeros_like(maps["entropy"]), cmap="magma", vmin=0.0, vmax=1.0)
        axes[4].set_title("novelty unavailable, run fit_ood.py")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_report(out_dir, rows, args_summary, top):
    """Write report.md summarizing the scan and the worst images."""
    lines = []
    lines.append("# Label free failure scan")
    lines.append("")
    lines.append(args_summary)
    lines.append("")
    lines.append(
        "No ground truth was used anywhere in this scan. The failure score blends "
        "normalized entropy, margin risk and confidence shortfall, so bright regions "
        "mark pixels where the model is least trustworthy. Entropy is high where the "
        "model spreads probability over many classes. Margin risk is high where the top "
        "two classes compete. Novelty compares each window against the training "
        "distribution in feature space, so a high value means the scene looks unlike "
        "anything the model trained on and even confident predictions deserve doubt."
    )
    lines.append("")
    lines.append(
        "The model assumes imagery near 0.3 m per pixel, the LoveDA ground sample "
        "distance. The scanner preserves scale by sliding a window at native resolution "
        "instead of resizing the whole image. Imagery at a very different resolution "
        "will read as degraded, which is a real failure mode rather than an artifact."
    )
    lines.append("")

    def table(sorted_rows):
        return _md_table(
            ["rank", "image", "windows", "mean failure", "p95 failure", "novelty mean", "novelty max", "panels"],
            [
                [
                    rank,
                    row["image"],
                    row["windows"],
                    f"{row['mean_failure_score']:.4f}",
                    f"{row['p95_failure_score']:.4f}",
                    "n a" if row["novelty_mean"] is None else f"{row['novelty_mean']:.3f}",
                    "n a" if row["novelty_max"] is None else f"{row['novelty_max']:.3f}",
                    row["panels"],
                ]
                for rank, row in enumerate(sorted_rows[:top], start=1)
            ],
        )

    by_failure = sorted(rows, key=lambda r: r["mean_failure_score"], reverse=True)
    lines.append("## Worst images by mean failure score")
    lines.append("")
    lines.append(table(by_failure))
    lines.append("")

    if any(row["novelty_mean"] is not None for row in rows):
        by_novelty = sorted(
            rows,
            key=lambda r: (r["novelty_mean"] is not None, r["novelty_mean"] or 0.0),
            reverse=True,
        )
        lines.append("## Most novel images against the training distribution")
        lines.append("")
        lines.append(table(by_novelty))
        lines.append("")

    report_path = Path(out_dir) / "report.md"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return report_path


def run_scan(model, cfg, device, image_paths, out_dir, tile, overlap, batch_size, reference, top):
    """Scan every image, write panels, rankings and the report, and return the rows."""
    num_classes = int(cfg.dataset.num_classes)
    ignore_index = int(cfg.dataset.get_path("ignore_index", 255))
    mean = list(cfg.augmentation.get_path("mean", [0.485, 0.456, 0.406]))
    std = list(cfg.augmentation.get_path("std", [0.229, 0.224, 0.225]))
    palette = get_palette(num_classes)

    out_dir = Path(out_dir)
    panels_dir = out_dir / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in tqdm(image_paths, desc="scan", leave=False):
        image, prediction, maps, novelty_map, window_novelties, num_windows = scan_image(
            image_path, model, device, mean, std, tile, overlap, batch_size, reference
        )
        panel_name = f"{image_path.stem}_scan.png"
        save_panels(
            panels_dir / panel_name,
            image,
            prediction,
            maps,
            novelty_map,
            num_classes,
            ignore_index,
            palette,
        )
        failure = maps["failure score"]
        rows.append(
            {
                "image": image_path.name,
                "path": str(image_path),
                "height": int(image.shape[0]),
                "width": int(image.shape[1]),
                "windows": int(num_windows),
                "mean_failure_score": float(failure.mean()),
                "p95_failure_score": float(np.percentile(failure, 95)),
                "mean_entropy": float(maps["entropy"].mean()),
                "mean_margin_risk": float(maps["margin risk"].mean()),
                "mean_confidence": float(maps["confidence"].mean()),
                "novelty_mean": float(np.mean(window_novelties)) if window_novelties else None,
                "novelty_max": float(np.max(window_novelties)) if window_novelties else None,
                "panels": f"panels/{panel_name}",
            }
        )

    by_failure = sorted(rows, key=lambda r: r["mean_failure_score"], reverse=True)
    pd.DataFrame(by_failure).to_csv(out_dir / "rankings.csv", index=False)
    by_novelty = sorted(
        rows,
        key=lambda r: (r["novelty_mean"] is not None, r["novelty_mean"] or 0.0),
        reverse=True,
    )
    with open(out_dir / "rankings.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"by_failure_score": by_failure, "by_novelty": by_novelty},
            handle,
            indent=2,
        )
    return rows


def run_selftest():
    """Run the whole pipeline on random noise images with a random model and exit.

    This needs no checkpoint, no dataset and no GPU, so it verifies the tiling, the map
    stitching, the novelty path and the report writing anywhere.
    """
    set_seed(0)
    cfg = Config(
        {
            "seed": 0,
            "output_dir": "results/scan_selftest",
            "device": "cpu",
            "dataset": {
                "name": "dummy",
                "num_classes": 7,
                "ignore_index": 255,
                "image_size": [96, 96],
            },
            "augmentation": {
                "enabled": False,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "dataloader": {"batch_size": 2, "num_workers": 0, "pin_memory": False},
            "model": {"name": "deeplabv3_resnet50", "pretrained": False},
        }
    )
    device = torch.device("cpu")
    model = build_model(cfg).to(device)
    model.eval()

    out_dir = Path(cfg.output_dir)
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name, size in (("noise_small.png", (80, 70)), ("noise_wide.png", (120, 260))):
        array = rng.integers(0, 255, size=(size[0], size[1], 3), dtype=np.uint8)
        Image.fromarray(array).save(inputs_dir / name)

    # A tiny reference fit on random windows exercises the novelty path end to end.
    mean = list(cfg.augmentation.mean)
    std = list(cfg.augmentation.std)
    with torch.no_grad():
        noise = torch.rand(6, 3, 96, 96)
        features = ood.extract_features(model, TF.normalize(noise, mean, std))
    reference = ood.fit_reference(features, shrinkage=1.0)

    rows = run_scan(
        model,
        cfg,
        device,
        collect_images(inputs_dir),
        out_dir,
        tile=96,
        overlap=32,
        batch_size=2,
        reference=reference,
        top=5,
    )
    report = write_report(out_dir, rows, "Selftest run on random noise images.", top=5)
    assert (out_dir / "rankings.csv").exists()
    assert (out_dir / "rankings.json").exists()
    for row in rows:
        assert (out_dir / row["panels"]).exists()
        assert 0.0 <= row["mean_failure_score"] <= 1.0
        assert row["novelty_mean"] is not None
        print(
            f" scanned {row['image']}  windows {row['windows']}  "
            f"mean failure {row['mean_failure_score']:.3f}  novelty {row['novelty_mean']:.3f}"
        )
    print(f" report at {report}")
    print("selftest OK")


def main():
    parser = argparse.ArgumentParser(
        description="Label free failure scan over a folder of aerial images",
        epilog=(
            "The model assumes imagery near 0.3 m per pixel. Large images are scanned "
            "with a sliding window at native resolution so the scale is preserved."
        ),
    )
    parser.add_argument("--checkpoint", default=None, help="path to a saved checkpoint")
    parser.add_argument("--config", default=None, help="optional config path override")
    parser.add_argument("--set", nargs="*", default=[], metavar="key=value")
    parser.add_argument("--images", default=None, help="an image file or a directory of images")
    parser.add_argument("--output-dir", default=None, help="where to write panels and rankings")
    parser.add_argument(
        "--ood-reference",
        default=None,
        help="novelty reference path, defaults to ood_reference.pt next to the checkpoint",
    )
    parser.add_argument(
        "--tile-size", type=int, default=None, help="window size, defaults to the model input size"
    )
    parser.add_argument("--overlap", type=int, default=64, help="window overlap in pixels")
    parser.add_argument("--batch-size", type=int, default=4, help="windows per forward pass")
    parser.add_argument("--top", type=int, default=10, help="rows shown in the report tables")
    parser.add_argument("--device", default=None, help="auto, cpu or cuda")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run on generated noise images with a random model and exit",
    )
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
        return

    if args.checkpoint is None or args.images is None:
        raise SystemExit("--checkpoint and --images are required unless --selftest is used")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = load_checkpoint_config(checkpoint, args.config, args.set)
    set_seed(int(cfg.get_path("seed", 42)))
    device = get_device(args.device or str(cfg.get_path("device", "auto")))

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    reference_path = (
        Path(args.ood_reference)
        if args.ood_reference
        else Path(args.checkpoint).parent / "ood_reference.pt"
    )
    reference = ood.load_reference(reference_path)
    if reference is None:
        print(
            f"No novelty reference at {reference_path}, novelty scoring is skipped. "
            "Build it once with scripts/fit_ood.py."
        )

    size = cfg.dataset.image_size
    tile = int(args.tile_size or int(size[0]))
    out_dir = Path(args.output_dir) if args.output_dir else Path(cfg.output_dir) / "scan"

    image_paths = collect_images(args.images)
    print(
        f"Scanning {len(image_paths)} images with window {tile} and overlap {args.overlap} "
        f"on device {device}"
    )

    rows = run_scan(
        model,
        cfg,
        device,
        image_paths,
        out_dir,
        tile,
        int(args.overlap),
        int(args.batch_size),
        reference,
        int(args.top),
    )
    summary = (
        f"Checkpoint {args.checkpoint} scanned {len(rows)} images with window size {tile} "
        f"and overlap {args.overlap}."
    )
    report = write_report(out_dir, rows, summary, int(args.top))

    worst = max(rows, key=lambda r: r["mean_failure_score"])
    print(
        f"worst image by failure score is {worst['image']} at {worst['mean_failure_score']:.3f}"
    )
    print(f"Wrote rankings and report to {out_dir}")
    print(f"Report at {report}")


if __name__ == "__main__":
    main()
