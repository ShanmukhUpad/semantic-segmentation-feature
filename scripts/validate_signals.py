"""Validate that the label free failure signals predict true error.

Run from the project root.

    python scripts/validate_signals.py --checkpoint results/loveda_deeplabv3/checkpoints/best.pth

The scanner and the app trust entropy, margin risk, confidence and novelty to point at
failures without any labels. This script checks that trust on a labeled split. Every
signal is scored on how well it ranks wrong pixels above correct ones (AUROC, AUPR and
the area under the risk coverage curve), and on how well the per image mean signal
tracks the per image error rate (Spearman rank correlation). When a baseline csv from
an in domain run is given, it also measures how cleanly the novelty score separates the
two datasets.

The intended experiment has two runs. First run on the LoveDA val split for the in
domain check. Then run on ISPRS Potsdam patches with --label-map potsdam_to_loveda and
--baseline-csv pointing at the per_image.csv of the first run. Potsdam is German urban
imagery at a different ground resolution, so a model trained only on China has never
seen anything like it, which makes it a real geographic domain shift. If the ranking
metrics stay clearly above chance there, the label free signals are validated for use
on unlabeled imagery from anywhere.
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
from scipy.stats import spearmanr  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from analysis import ood  # noqa: E402
from analysis.uncertainty import (  # noqa: E402
    failure_score,
    margin,
    max_softmax_confidence,
    predictive_entropy,
)
from analysis.validation import (  # noqa: E402
    LABEL_MAPS,
    error_aupr,
    error_auroc,
    pixel_error_mask,
    remap_mask,
    risk_coverage,
)
from data.registry import build_dataset  # noqa: E402
from models.registry import build_model  # noqa: E402
from utils.config import Config, load_config  # noqa: E402
from utils.device import get_device  # noqa: E402
from utils.metrics import ConfusionMatrix  # noqa: E402
from utils.plots import save_bar_chart  # noqa: E402
from utils.seed import set_seed  # noqa: E402

SIGNALS = ["confidence_shortfall", "entropy", "margin_risk", "failure_score"]


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


def image_name(dataset, idx):
    """Return a stable name for one sample, using the source file when available."""
    target = getattr(dataset, "dataset", dataset)
    images = getattr(target, "images", None)
    if images is not None and idx < len(images):
        return Path(images[idx]).name
    return f"index_{idx}"


def _spearman(a, b):
    """Return the Spearman rank correlation, or None when it is undefined."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    value = float(spearmanr(a, b).statistic)
    return None if np.isnan(value) else value


def _fmt(value, places=4):
    """Format a number for a table, showing n a when the value is missing."""
    if value is None:
        return "n a"
    return f"{value:.{places}f}"


def _md_table(headers, rows):
    """Build a github flavored markdown table from headers and a list of row lists."""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def collect_signal_data(model, loader, dataset, device, cfg, mapping, reference, max_pixels, seed):
    """Run the labeled split once and return pixel samples and per image rows.

    For every image the four uncertainty signals and the prediction are computed, the
    ground truth is optionally remapped through the class mapping, and up to max_pixels
    valid pixels are sampled for the pixel level ranking metrics. The novelty score is
    recorded per image when a reference is available.
    """
    num_classes = int(cfg.dataset.num_classes)
    ignore_index = int(cfg.dataset.get_path("ignore_index", 255))
    rng = np.random.default_rng(seed)

    pixel_scores = {name: [] for name in SIGNALS}
    pixel_errors = []
    rows = []
    skipped = 0
    running_index = 0

    catcher = ood.BackboneCatcher(model) if reference is not None else None
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="validate", leave=False):
            output = model(images.to(device))
            logits = output["out"] if isinstance(output, dict) else output
            maps = {
                "confidence_shortfall": (1.0 - max_softmax_confidence(logits)).cpu().numpy(),
                "entropy": predictive_entropy(logits).cpu().numpy(),
                "margin_risk": margin(logits).cpu().numpy(),
                "failure_score": failure_score(logits).cpu().numpy(),
            }
            preds = logits.argmax(dim=1).cpu().numpy()

            distances = novelties = None
            if catcher is not None:
                features = catcher.features()
                if features is not None:
                    distances = ood.mahalanobis(features, reference)
                    novelties = ood.normalize_novelty(distances, reference)

            for b in range(preds.shape[0]):
                gt = masks[b].numpy()
                if mapping is not None:
                    gt = remap_mask(gt, mapping, ignore_index)
                error, valid = pixel_error_mask(preds[b], gt, ignore_index)
                n_valid = int(valid.sum())
                if n_valid == 0:
                    skipped += 1
                    running_index += 1
                    continue

                flat_valid = np.flatnonzero(valid.ravel())
                if len(flat_valid) > max_pixels:
                    flat_valid = rng.choice(flat_valid, size=max_pixels, replace=False)
                pixel_errors.append(error.ravel()[flat_valid])
                for name in SIGNALS:
                    pixel_scores[name].append(maps[name][b].ravel()[flat_valid])

                per_image = ConfusionMatrix(num_classes, ignore_index)
                per_image.update(gt, preds[b])
                row = {
                    "image": image_name(dataset, running_index),
                    "error_rate": float(error.sum()) / n_valid,
                    "miou": per_image.mean_iou(),
                }
                for name in SIGNALS:
                    row["mean_" + name] = float(maps[name][b][valid].mean())
                row["novelty"] = float(novelties[b]) if novelties is not None else None
                row["novelty_distance"] = float(distances[b]) if distances is not None else None
                rows.append(row)
                running_index += 1
    if catcher is not None:
        catcher.close()

    pixel_errors = np.concatenate(pixel_errors) if pixel_errors else np.zeros(0, dtype=bool)
    pixel_scores = {
        name: (np.concatenate(chunks) if chunks else np.zeros(0))
        for name, chunks in pixel_scores.items()
    }
    return pixel_scores, pixel_errors, rows, skipped


def save_figures(out_dir, pixel_scores, pixel_errors, curves, base_error, frame, baseline_frame):
    """Save the risk coverage curves, the decile bars, the scatter and the histogram."""
    figures = {}

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (coverages, risks) in curves.items():
        ax.plot(coverages, risks, label=name.replace("_", " "))
    ax.axhline(base_error, color="gray", linestyle="--", label="chance level")
    ax.set_xlabel("coverage, fraction of pixels kept")
    ax.set_ylabel("error rate among kept pixels")
    ax.set_title("Risk coverage curves, lower is better")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "risk_coverage.png", dpi=120)
    plt.close(fig)
    figures["risk_coverage"] = "risk_coverage.png"

    scores = pixel_scores["failure_score"]
    if len(scores):
        order = np.argsort(scores, kind="stable")
        chunks = np.array_split(pixel_errors[order].astype(np.float64), 10)
        decile_rates = [float(chunk.mean()) if len(chunk) else 0.0 for chunk in chunks]
        save_bar_chart(
            decile_rates,
            [f"d{i}" for i in range(1, 11)],
            "True error rate by failure score decile",
            "error rate",
            out_dir / "failure_deciles.png",
        )
        figures["failure_deciles"] = "failure_deciles.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(frame["mean_failure_score"], frame["error_rate"], s=12, alpha=0.5)
    ax.set_xlabel("mean failure score per image")
    ax.set_ylabel("true error rate per image")
    ax.set_title("Per image failure score against true error")
    fig.tight_layout()
    fig.savefig(out_dir / "failure_vs_error.png", dpi=120)
    plt.close(fig)
    figures["failure_vs_error"] = "failure_vs_error.png"

    if baseline_frame is not None and "novelty_distance" in baseline_frame:
        current = frame["novelty_distance"].dropna()
        baseline = baseline_frame["novelty_distance"].dropna()
        if len(current) and len(baseline):
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(baseline, bins=30, alpha=0.6, label="baseline run", density=True)
            ax.hist(current, bins=30, alpha=0.6, label="this run", density=True)
            ax.set_xlabel("Mahalanobis distance to the training distribution")
            ax.set_ylabel("density")
            ax.set_title("Novelty separation between the two runs")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "novelty_hist.png", dpi=120)
            plt.close(fig)
            figures["novelty_hist"] = "novelty_hist.png"

    return figures


def write_report(out_dir, results, figures, label_map_name):
    """Write report.md interpreting the validation numbers in plain language."""
    lines = []
    lines.append("# Label free signal validation")
    lines.append("")
    lines.append(
        f"Checkpoint {results['checkpoint']} on dataset {results['dataset']} split "
        f"{results['split']}, {results['num_images']} images scored."
    )
    lines.append("")
    lines.append(
        "The failure signals are computed without labels, then judged against the real "
        "labels of this split. AUROC is the chance that a random wrong pixel outscores "
        "a random correct pixel, so 0.5 is chance and 1.0 is perfect ranking. AUPR "
        "rewards putting errors first and its chance level equals the base error rate. "
        "The risk coverage area (AURC) is the average error rate when only the lowest "
        "scoring pixels are kept, so lower is better."
    )
    lines.append("")
    lines.append(f"Base pixel error rate on this split is {_fmt(results['base_error_rate'])}.")
    lines.append("")

    lines.append("## Pixel level ranking quality")
    lines.append("")
    lines.append(
        _md_table(
            ["signal", "AUROC", "AUPR", "AURC"],
            [
                [
                    name.replace("_", " "),
                    _fmt(results["pixel"][name]["auroc"]),
                    _fmt(results["pixel"][name]["aupr"]),
                    _fmt(results["pixel"][name]["aurc"]),
                ]
                for name in SIGNALS
            ],
        )
    )
    lines.append("")
    if "risk_coverage" in figures:
        lines.append(f"![risk coverage]({figures['risk_coverage']})")
        lines.append("")
    if "failure_deciles" in figures:
        lines.append(f"![failure deciles]({figures['failure_deciles']})")
        lines.append("")

    lines.append("## Image level ranking quality")
    lines.append("")
    lines.append(
        "Spearman rank correlation between the mean signal of an image and its true "
        "error rate. A high value means the signal can rank whole images from safest "
        "to most broken without any labels."
    )
    lines.append("")
    image_rows = [
        [name.replace("_", " "), _fmt(results["image_spearman"][name])] for name in SIGNALS
    ]
    if results["novelty"]["spearman"] is not None:
        image_rows.append(["novelty", _fmt(results["novelty"]["spearman"])])
    lines.append(_md_table(["signal", "Spearman"], image_rows))
    lines.append("")
    if "failure_vs_error" in figures:
        lines.append(f"![failure versus error]({figures['failure_vs_error']})")
        lines.append("")

    if results["novelty"]["domain_auroc"] is not None:
        lines.append("## Novelty as a domain detector")
        lines.append("")
        lines.append(
            f"Comparing against the baseline run, the novelty score separates the two "
            f"datasets with AUROC {_fmt(results['novelty']['domain_auroc'])}. A value "
            "near 1.0 means the model itself can tell this imagery is unlike its "
            "training data before a single label is checked."
        )
        lines.append("")
        if "novelty_hist" in figures:
            lines.append(f"![novelty histogram]({figures['novelty_hist']})")
            lines.append("")

    if label_map_name and label_map_name != "none":
        lines.append("## Label mapping caveat")
        lines.append("")
        lines.append(
            f"Ground truth was remapped with the {label_map_name.replace('_', ' ')} "
            "mapping. The mapping is coarse, unmappable classes were ignored, and the "
            "absolute error rates are approximate. The ranking metrics above only need "
            "the error mask to be roughly right, so they remain meaningful."
        )
        lines.append("")

    report_path = Path(out_dir) / "report.md"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return report_path


def run_validation(model, cfg, device, loader, dataset, out_dir, mapping, label_map_name,
                   reference, max_pixels, baseline_csv, seed, checkpoint_label, split):
    """Run the full validation and write metrics, tables, figures and the report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pixel_scores, pixel_errors, rows, skipped = collect_signal_data(
        model, loader, dataset, device, cfg, mapping, reference, max_pixels, seed
    )
    if not rows:
        raise RuntimeError("no image produced any valid labeled pixel to validate against")

    base_error = float(pixel_errors.mean()) if len(pixel_errors) else 0.0
    pixel_metrics = {}
    curves = {}
    for name in SIGNALS:
        scores = pixel_scores[name]
        coverages, risks, aurc = risk_coverage(scores, pixel_errors)
        pixel_metrics[name] = {
            "auroc": error_auroc(scores, pixel_errors),
            "aupr": error_aupr(scores, pixel_errors),
            "aurc": aurc,
        }
        curves[name] = (coverages, risks)

    frame = pd.DataFrame(rows)
    image_spearman = {
        name: _spearman(frame["mean_" + name], frame["error_rate"]) for name in SIGNALS
    }

    novelty_spearman = None
    domain_auroc = None
    baseline_frame = None
    if frame["novelty"].notna().any():
        novelty_spearman = _spearman(
            frame["novelty_distance"].fillna(0.0), frame["error_rate"]
        )
    if baseline_csv:
        baseline_frame = pd.read_csv(baseline_csv)
        if "novelty_distance" in baseline_frame and frame["novelty_distance"].notna().any():
            current = frame["novelty_distance"].dropna().to_numpy()
            baseline = baseline_frame["novelty_distance"].dropna().to_numpy()
            if len(current) and len(baseline):
                scores = np.concatenate([baseline, current])
                labels = np.concatenate(
                    [np.zeros(len(baseline), dtype=bool), np.ones(len(current), dtype=bool)]
                )
                domain_auroc = error_auroc(scores, labels)

    results = {
        "checkpoint": checkpoint_label,
        "dataset": cfg.dataset.name,
        "split": split,
        "label_map": label_map_name,
        "num_images": len(rows),
        "skipped_images": skipped,
        "sampled_pixels": int(len(pixel_errors)),
        "base_error_rate": base_error,
        "pixel": pixel_metrics,
        "image_spearman": image_spearman,
        "novelty": {
            "spearman": novelty_spearman,
            "domain_auroc": domain_auroc,
            "baseline_csv": str(baseline_csv) if baseline_csv else None,
        },
    }

    frame.to_csv(out_dir / "per_image.csv", index=False)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    figures = save_figures(
        out_dir, pixel_scores, pixel_errors, curves, base_error, frame, baseline_frame
    )
    report_path = write_report(out_dir, results, figures, label_map_name)

    print(f"validated {len(rows)} images, base error rate {base_error:.4f}")
    for name in SIGNALS:
        print(
            f" {name.replace('_', ' '):22s} AUROC {_fmt(pixel_metrics[name]['auroc'])}  "
            f"AUPR {_fmt(pixel_metrics[name]['aupr'])}  AURC {_fmt(pixel_metrics[name]['aurc'])}  "
            f"Spearman {_fmt(image_spearman[name])}"
        )
    if novelty_spearman is not None:
        print(f" novelty Spearman {_fmt(novelty_spearman)}")
    if domain_auroc is not None:
        print(f" novelty domain AUROC {_fmt(domain_auroc)}")
    print(f"Wrote validation artifacts to {out_dir}")
    print(f"Report at {report_path}")
    return results


def run_selftest():
    """Run the validation end to end on the dummy dataset with a random model."""
    set_seed(0)
    cfg = Config(
        {
            "seed": 0,
            "output_dir": "results/validate_selftest",
            "device": "cpu",
            "dataset": {
                "name": "dummy",
                "num_classes": 7,
                "ignore_index": 255,
                "image_size": [64, 64],
                "dummy": {"num_train": 8, "num_val": 6, "num_test": 4, "grid": 8},
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
    dataset = build_dataset(cfg, "val")
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    results = run_validation(
        model,
        cfg,
        device,
        loader,
        dataset,
        Path(cfg.output_dir),
        mapping=None,
        label_map_name="none",
        reference=None,
        max_pixels=1500,
        baseline_csv=None,
        seed=0,
        checkpoint_label="selftest random model",
        split="val",
    )
    out_dir = Path(cfg.output_dir)
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "per_image.csv").exists()
    assert (out_dir / "report.md").exists()
    for name in SIGNALS:
        assert np.isfinite(results["pixel"][name]["aurc"])
    print("selftest OK")


def main():
    parser = argparse.ArgumentParser(
        description="Check that the label free failure signals predict true error"
    )
    parser.add_argument("--checkpoint", default=None, help="path to a saved checkpoint")
    parser.add_argument("--config", default=None, help="optional config path override")
    parser.add_argument("--set", nargs="*", default=[], metavar="key=value")
    parser.add_argument("--split", default="val", help="labeled split to validate on")
    parser.add_argument(
        "--label-map",
        default="none",
        choices=["none"] + sorted(LABEL_MAPS),
        help="remap the ground truth classes, use potsdam_to_loveda for Potsdam patches",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap the number of images")
    parser.add_argument(
        "--max-pixels-per-image",
        type=int,
        default=20000,
        help="valid pixels sampled per image for the pixel level metrics",
    )
    parser.add_argument(
        "--ood-reference",
        default=None,
        help="novelty reference path, defaults to ood_reference.pt next to the checkpoint",
    )
    parser.add_argument(
        "--baseline-csv",
        default=None,
        help="per_image.csv of an in domain run, enables the novelty domain AUROC",
    )
    parser.add_argument("--output-dir", default=None, help="where to write the results")
    parser.add_argument("--device", default=None, help="auto, cpu or cuda")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run on the dummy dataset with a random model and exit",
    )
    args = parser.parse_args()

    if args.selftest:
        run_selftest()
        return

    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --selftest is used")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = load_checkpoint_config(checkpoint, args.config, args.set)
    seed = int(cfg.get_path("seed", 42))
    set_seed(seed)
    device = get_device(args.device or str(cfg.get_path("device", "auto")))

    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = build_dataset(cfg, args.split)
    if args.limit and args.limit < len(dataset):
        dataset = Subset(dataset, list(range(args.limit)))
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.dataloader.get_path("batch_size", 4)),
        shuffle=False,
        num_workers=int(cfg.dataloader.get_path("num_workers", 0)),
        pin_memory=bool(cfg.dataloader.get_path("pin_memory", False)),
    )

    reference_path = (
        Path(args.ood_reference)
        if args.ood_reference
        else Path(args.checkpoint).parent / "ood_reference.pt"
    )
    reference = ood.load_reference(reference_path)
    if reference is None:
        print(
            f"No novelty reference at {reference_path}, novelty checks are skipped. "
            "Build it once with scripts/fit_ood.py."
        )

    mapping = LABEL_MAPS.get(args.label_map)
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(cfg.output_dir) / "validation" / f"{cfg.dataset.name}_{args.split}"

    print(
        f"Validating {args.checkpoint} on {cfg.dataset.name} split {args.split} "
        f"with {len(dataset)} images on device {device}"
    )
    run_validation(
        model,
        cfg,
        device,
        loader,
        dataset,
        out_dir,
        mapping,
        args.label_map,
        reference,
        int(args.max_pixels_per_image),
        args.baseline_csv,
        seed,
        args.checkpoint,
        args.split,
    )


if __name__ == "__main__":
    main()
