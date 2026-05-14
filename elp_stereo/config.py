"""Load the YAML configuration file."""

from pathlib import Path

import yaml

_DEFAULT = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_config(path=None):
    """Return the config dict from ``path`` (defaults to config/default.yaml)."""
    cfg_path = Path(path) if path else _DEFAULT
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def project_root():
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent
