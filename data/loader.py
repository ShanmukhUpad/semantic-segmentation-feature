"""Dataloader construction from a config.

build_dataloaders returns a train loader and a validation loader honoring the batch size,
worker count and pin memory settings. The train loader is shuffled and seeded so the order is
repeatable across runs.
"""
import torch
from torch.utils.data import DataLoader, Subset

from data.registry import build_dataset
from utils.seed import seed_worker


def build_dataloaders(cfg):
    """Build the train and validation dataloaders described by the config.

    When dataset.subset is set to a positive integer both splits are capped to that many
    samples, which gives a fast development loop before a full scale run on real data.
    """
    train_dataset = build_dataset(cfg, "train")
    val_dataset = build_dataset(cfg, "val")

    subset = cfg.dataset.get_path("subset", None)
    if subset:
        train_dataset = Subset(train_dataset, list(range(min(int(subset), len(train_dataset)))))
        val_dataset = Subset(val_dataset, list(range(min(int(subset), len(val_dataset)))))

    loader_cfg = cfg.dataloader
    common = dict(
        batch_size=int(loader_cfg.get_path("batch_size", 4)),
        num_workers=int(loader_cfg.get_path("num_workers", 0)),
        pin_memory=bool(loader_cfg.get_path("pin_memory", False)),
    )

    generator = torch.Generator()
    generator.manual_seed(int(cfg.get_path("seed", 42)))

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader
