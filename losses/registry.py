"""Loss registry so the training objective is swappable from the config.

The eventual research contribution is expected to change the loss function, so losses are kept
behind a registry exactly like datasets and models. Adding a new loss means writing one builder
and decorating it with register_loss, then setting train.loss in the config to its name.
"""

LOSSES = {}


def register_loss(name):
    """Decorator that records a loss builder function under a short name."""

    def decorator(builder):
        if name in LOSSES and LOSSES[name] is not builder:
            raise KeyError(f"Loss name '{name}' is already registered")
        LOSSES[name] = builder
        return builder

    return decorator


def build_loss(cfg, ignore_index):
    """Build the loss named in the config, passing the ignore index for masked pixels."""
    from losses import standard  # noqa: F401  (import triggers registration)

    name = cfg.train.get_path("loss", "cross_entropy")
    if name not in LOSSES:
        raise KeyError(
            f"Unknown loss '{name}'. Registered losses are {sorted(LOSSES)}."
        )
    return LOSSES[name](cfg, ignore_index)
