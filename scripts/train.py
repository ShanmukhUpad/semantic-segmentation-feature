"""Training entry point.

Run from the project root.

    python scripts/train.py --config configs/dummy.yaml

Cross entropy is the default loss but the loss, model and dataset all come from registries so
each can be swapped from the config. The loop logs train and validation loss along with mean
IoU and pixel accuracy to TensorBoard and to a metrics CSV, saves the best and last
checkpoints, and seeds python, numpy and torch for reproducibility. It runs on CPU now and on
a GPU later with no code change. Mixed precision, gradient accumulation and a dataset subset
knob are wired so scaling up is a config change rather than a rewrite.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from analysis.error_maps import colorize, get_palette  # noqa: E402
from data.loader import build_dataloaders  # noqa: E402
from losses.registry import build_loss  # noqa: E402
from models.registry import build_model  # noqa: E402
from utils.config import load_config, save_config  # noqa: E402
from utils.device import get_device  # noqa: E402
from utils.metrics import ConfusionMatrix  # noqa: E402
from utils.seed import set_seed  # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter

    _HAS_TENSORBOARD = True
except Exception:
    _HAS_TENSORBOARD = False


def build_optimizer(cfg, parameters):
    """Build the optimizer named in the config."""
    name = str(cfg.train.get_path("optimizer", "adam")).lower()
    lr = float(cfg.train.lr)
    weight_decay = float(cfg.train.get_path("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(cfg.train.get_path("momentum", 0.9))
        return torch.optim.SGD(
            parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    raise ValueError(f"Unknown optimizer '{name}'")


def build_scheduler(cfg, optimizer):
    """Build an optional learning rate scheduler, or return None."""
    name = str(cfg.train.get_path("scheduler", "none")).lower()
    if name in ("none", ""):
        return None
    if name == "step":
        step_size = int(cfg.train.get_path("step_size", 10))
        gamma = float(cfg.train.get_path("gamma", 0.1))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "cosine":
        epochs = int(cfg.train.epochs)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"Unknown scheduler '{name}'")


def forward_loss(model, images, masks, criterion, aux_weight):
    """Run the model and return the loss and the main output logits.

    DeepLabV3 returns a dictionary with a main output and an auxiliary output. When the
    auxiliary head is present its loss is added with a small weight, which is standard.
    """
    output = model(images)
    if isinstance(output, dict):
        logits = output["out"]
        loss = criterion(logits, masks)
        if output.get("aux") is not None:
            loss = loss + aux_weight * criterion(output["aux"], masks)
    else:
        logits = output
        loss = criterion(logits, masks)
    return loss, logits


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg, epoch):
    """Run one training epoch and return the average loss per sample."""
    model.train()
    accum = max(int(cfg.train.get_path("grad_accum_steps", 1)), 1)
    amp = bool(cfg.train.get_path("amp", False)) and device.type == "cuda"
    aux_weight = float(cfg.train.get_path("aux_loss_weight", 0.4))

    running = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for step, (images, masks) in enumerate(progress):
        images = images.to(device)
        masks = masks.to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            loss, _ = forward_loss(model, images, masks, criterion, aux_weight)
        scaler.scale(loss / accum).backward()
        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        running += loss.item() * images.size(0)
        count += images.size(0)
        progress.set_postfix(loss=f"{loss.item():.4f}")
    return running / max(count, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes, ignore_index):
    """Run validation and return the average loss and a filled confusion matrix."""
    model.eval()
    confusion = ConfusionMatrix(num_classes, ignore_index)
    running = 0.0
    count = 0
    for images, masks in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        output = model(images)
        logits = output["out"] if isinstance(output, dict) else output
        loss = criterion(logits, masks)
        running += loss.item() * images.size(0)
        count += images.size(0)
        confusion.update(masks, logits.argmax(dim=1))
    return running / max(count, 1), confusion


def denormalize(image, mean, std):
    """Undo normalization so a logged image shows its original colors."""
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


@torch.no_grad()
def log_sample_predictions(
    writer, model, images, masks, device, epoch, mean, std, palette, num_classes, ignore_index
):
    """Log a fixed set of input, ground truth and prediction strips to TensorBoard.

    The same images are logged every epoch so the Images tab in the web UI shows the
    prediction improving over time next to the input and the ground truth.
    """
    model.eval()
    output = model(images.to(device))
    logits = output["out"] if isinstance(output, dict) else output
    preds = logits.argmax(dim=1).cpu()
    for i in range(images.size(0)):
        inp = denormalize(images[i], mean, std).permute(1, 2, 0).numpy()
        gt_rgb = colorize(masks[i].numpy(), num_classes, palette, ignore_index) / 255.0
        pred_rgb = colorize(preds[i].numpy(), num_classes, palette, ignore_index) / 255.0
        strip = np.concatenate([inp, gt_rgb, pred_rgb], axis=1)
        writer.add_image(f"val_samples/{i}", strip, epoch, dataformats="HWC")
    model.train()


def save_checkpoint(path, model, optimizer, epoch, best_miou, cfg):
    """Save a checkpoint holding everything needed to resume or to evaluate later."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_miou": best_miou,
            "config": cfg.to_dict(),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a segmentation model")
    parser.add_argument("--config", required=True, help="path to a YAML config file")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="key=value",
        help="optional dotted config overrides, for example train.epochs=1",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    set_seed(int(cfg.get_path("seed", 42)))
    device = get_device(str(cfg.get_path("device", "auto")))
    if device.type == "cuda":
        # Input size is fixed for a run, so cudnn autotune picks the fastest kernels.
        torch.backends.cudnn.benchmark = True

    out_dir = Path(cfg.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    writer = None
    if _HAS_TENSORBOARD:
        writer = SummaryWriter(str(out_dir / "tensorboard"))
    else:
        print("TensorBoard is not installed, logging to console and metrics.csv only")

    num_classes = int(cfg.dataset.num_classes)
    ignore_index = int(cfg.dataset.get_path("ignore_index", 255))

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    criterion = build_loss(cfg, ignore_index).to(device)
    optimizer = build_optimizer(cfg, model.parameters())
    scheduler = build_scheduler(cfg, optimizer)

    amp = bool(cfg.train.get_path("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    epochs = int(cfg.train.epochs)
    save_every = int(cfg.train.get_path("save_every", 0))

    # Prepare a fixed batch of validation samples for TensorBoard image logging.
    log_images = int(cfg.train.get_path("log_images", 0))
    fixed_images = fixed_masks = None
    palette = None
    if writer is not None and log_images > 0:
        batch_images, batch_masks = next(iter(val_loader))
        keep = min(log_images, batch_images.size(0))
        fixed_images = batch_images[:keep]
        fixed_masks = batch_masks[:keep]
        palette = get_palette(num_classes)

    print(
        f"Training {cfg.model.name} on {cfg.dataset.name} for {epochs} epochs "
        f"on device {device}"
    )

    metrics_path = out_dir / "metrics.csv"
    history = []
    best_miou = -1.0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, cfg, epoch
        )
        val_loss, confusion = validate(
            model, val_loader, criterion, device, num_classes, ignore_index
        )
        miou = confusion.mean_iou()
        pixel_acc = confusion.pixel_accuracy()
        lr = optimizer.param_groups[0]["lr"]

        if scheduler is not None:
            scheduler.step()

        print(
            f"epoch {epoch:03d}  train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
            f"mIoU {miou:.4f}  pixel_acc {pixel_acc:.4f}  lr {lr:.6f}"
        )

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("metric/mIoU", miou, epoch)
            writer.add_scalar("metric/pixel_acc", pixel_acc, epoch)
            writer.add_scalar("train/lr", lr, epoch)
            if fixed_images is not None:
                log_sample_predictions(
                    writer,
                    model,
                    fixed_images,
                    fixed_masks,
                    device,
                    epoch,
                    cfg.augmentation.get_path("mean", [0.485, 0.456, 0.406]),
                    cfg.augmentation.get_path("std", [0.229, 0.224, 0.225]),
                    palette,
                    num_classes,
                    ignore_index,
                )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "mIoU": miou,
                "pixel_acc": pixel_acc,
                "lr": lr,
            }
        )
        with open(metrics_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = ["epoch", "train_loss", "val_loss", "mIoU", "pixel_acc", "lr"]
            writer_csv = csv.DictWriter(handle, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(history)

        save_checkpoint(ckpt_dir / "last.pth", model, optimizer, epoch, best_miou, cfg)
        if miou > best_miou:
            best_miou = miou
            save_checkpoint(ckpt_dir / "best.pth", model, optimizer, epoch, best_miou, cfg)
        if save_every and epoch % save_every == 0:
            save_checkpoint(
                ckpt_dir / f"epoch_{epoch:03d}.pth", model, optimizer, epoch, best_miou, cfg
            )

    if writer is not None:
        writer.close()
    print(f"Best mIoU {best_miou:.4f}. Checkpoints saved under {ckpt_dir}")


if __name__ == "__main__":
    main()
