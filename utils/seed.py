"""Seeding helpers for reproducible runs.

A single call to set_seed fixes the random state for python, numpy and torch. The
seed_worker helper is passed to the dataloader so each worker derives a repeatable seed.
"""
import os
import random

import numpy as np
import torch


def set_seed(seed, deterministic=True):
    """Seed python, numpy and torch and optionally request deterministic algorithms."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Seed a dataloader worker so its augmentation and shuffling are repeatable."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
