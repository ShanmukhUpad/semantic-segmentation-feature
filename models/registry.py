"""Model registry so new architectures are added with a single decorator.

Each architecture registers a builder function under a short name. The build_model function
maps the name in the config to the right builder. Adding DeepLabV3+, U-Net or any other model
means writing one builder and adding the register_model decorator, with no change needed here.
"""

MODELS = {}


def register_model(name):
    """Decorator that records a model builder function under a short name."""

    def decorator(builder):
        if name in MODELS and MODELS[name] is not builder:
            raise KeyError(f"Model name '{name}' is already registered")
        MODELS[name] = builder
        return builder

    return decorator


def build_model(cfg):
    """Build the model named in the config."""
    from models import segmentation  # noqa: F401  (import triggers registration)

    name = cfg.model.name
    if name not in MODELS:
        raise KeyError(
            f"Unknown model '{name}'. Registered models are {sorted(MODELS)}."
        )
    return MODELS[name](cfg)
