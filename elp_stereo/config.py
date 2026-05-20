"""Load the YAML configuration file."""

from pathlib import Path

import yaml

_DEFAULT = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
_SAVED = Path(__file__).resolve().parent.parent / "config" / "saved_params.yaml"


def _deep_merge(base, overlay):
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=None):
    """Return config from ``path`` or default plus saved parameter overrides."""
    cfg_path = Path(path) if path else _DEFAULT
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    if path is None and _SAVED.is_file():
        with open(_SAVED, "r") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def save_config(cfg, path=None):
    """Write ``cfg`` to ``path`` (defaults to config/saved_params.yaml)."""
    cfg_path = Path(path) if path else _SAVED
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path


def project_root():
    """Absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent
