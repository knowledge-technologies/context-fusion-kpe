"""Configuration loading utilities."""

from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return the project root directory.

    Assumes the project structure has src/keyword_pecker/utils/config.py
    and returns the directory containing src/.
    """
    return Path(__file__).parent.parent.parent.parent


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML file. Can be absolute or relative
            to the project root.

    Returns:
        Dictionary containing the parsed YAML configuration.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    path = Path(config_path)

    # If relative path, resolve from project root
    if not path.is_absolute():
        path = get_project_root() / path

    with open(path) as f:
        return yaml.safe_load(f)
