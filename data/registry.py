"""Dataset registry so new datasets are added with a single decorator.

Each dataset class registers itself under a short name. The build_dataset function maps the
name in the config to the right class. Adding a dataset means writing one class and adding
the register_dataset decorator, with no change needed here or in the config loader.
"""

DATASETS = {}


def register_dataset(name):
    """Class decorator that records a dataset class under a short name."""

    def decorator(cls):
        if name in DATASETS and DATASETS[name] is not cls:
            raise KeyError(f"Dataset name '{name}' is already registered")
        DATASETS[name] = cls
        return cls

    return decorator


def build_dataset(cfg, split):
    """Build the dataset named in the config for the requested split.

    The concrete dataset modules are imported here so their register_dataset decorators run
    before the lookup. This keeps the package import light and avoids import cycles.
    """
    from data import dummy, loveda, potsdam  # noqa: F401  (import triggers registration)

    name = cfg.dataset.name
    if name not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{name}'. Registered datasets are {sorted(DATASETS)}."
        )
    return DATASETS[name](cfg, split)
