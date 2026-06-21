"""Lightweight YAML configuration with attribute and dotted path access.

A single YAML file drives the dataset, augmentation, dataloader, model, training and
output settings. The Config class wraps a plain dictionary so values can be read either
as attributes or through dotted paths, and nested dictionaries are wrapped on demand.
"""
from pathlib import Path

import yaml


class Config(dict):
    """Dictionary that also supports attribute access and dotted paths.

    Reading an attribute returns the stored value. Nested dictionaries are wrapped as
    Config the first time they are read so dotted attribute access works at any depth.
    """

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name, value):
        self[name] = value

    def get_path(self, dotted, default=None):
        """Return the value at a dotted path or the default when any part is missing."""
        node = self
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set_path(self, dotted, value):
        """Set the value at a dotted path, creating intermediate dictionaries."""
        parts = dotted.split(".")
        node = self
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value

    def to_dict(self):
        """Return a plain nested dictionary copy of this config."""
        out = {}
        for key, value in self.items():
            if isinstance(value, dict):
                out[key] = Config(value).to_dict() if not isinstance(value, Config) else value.to_dict()
            else:
                out[key] = value
        return out


def load_config(path, overrides=None):
    """Load a YAML file into a Config and apply optional dotted overrides.

    Each override is a string of the form path.to.key=value where the value is parsed as
    YAML so numbers, booleans and lists are handled correctly.
    """
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    cfg = Config(raw or {})
    for item in overrides or []:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override '{item}' must use the form key=value")
        cfg.set_path(key.strip(), yaml.safe_load(value))
    return cfg


def save_config(cfg, path):
    """Write a config out as YAML, used to record the exact settings of a run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cfg.to_dict() if isinstance(cfg, Config) else dict(cfg)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
