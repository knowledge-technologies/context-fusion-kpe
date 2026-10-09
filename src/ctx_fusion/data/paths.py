"""Dataset path resolution utilities.

This module provides functions for resolving dataset paths using the
DATASET_FOLDER environment variable. Processed data is stored in a
'derived' subdirectory alongside raw data.

Directory structure:
    $DATASET_FOLDER/
    └── {dataset_name}/
        ├── {dataset_name}_train.json    # raw data
        ├── {dataset_name}_valid.json
        ├── {dataset_name}_test.json
        └── derived/                      # processed data
            ├── train.parquet
            ├── valid.parquet
            ├── test.parquet
            └── metadata.json
"""

import os
from pathlib import Path

from dotenv import load_dotenv


def get_dataset_folder() -> Path:
    """Get the dataset folder from environment variable.

    Loads the DATASET_FOLDER environment variable from .env file.

    Returns:
        Path to the dataset folder.

    Raises:
        ValueError: If DATASET_FOLDER environment variable is not set.
    """
    load_dotenv()
    dataset_folder = os.getenv("DATASET_FOLDER")
    if not dataset_folder:
        raise ValueError(
            "DATASET_FOLDER environment variable not set. "
            "Please set it in your .env file."
        )
    return Path(dataset_folder)


def get_dataset_path(dataset_name: str) -> Path:
    """Get path to raw dataset directory.

    Args:
        dataset_name: Name of dataset (e.g., 'kp20k').

    Returns:
        Path to dataset directory.

    Example:
        >>> get_dataset_path('kp20k')
        PosixPath('/path/to/data/kp20k')
    """
    return get_dataset_folder() / dataset_name


def get_derived_path(dataset_name: str, split: str | None = None) -> Path:
    """Get path to derived (processed) data.

    Args:
        dataset_name: Name of dataset.
        split: Optional split name (e.g., 'train', 'valid', 'test').
            If provided, returns path to the .parquet file.
            If None, returns path to derived/ directory.

    Returns:
        Path to derived directory or specific parquet file.

    Examples:
        >>> get_derived_path('kp20k')
        PosixPath('/path/to/data/kp20k/derived')

        >>> get_derived_path('kp20k', 'train')
        PosixPath('/path/to/data/kp20k/derived/train.parquet')
    """
    derived_dir = get_dataset_path(dataset_name) / "derived"
    if split is None:
        return derived_dir
    return derived_dir / f"{split}.parquet"


def ensure_derived_dir(dataset_name: str) -> Path:
    """Ensure derived/ directory exists for dataset.

    Creates the directory if it doesn't exist.

    Args:
        dataset_name: Name of dataset.

    Returns:
        Path to derived directory.
    """
    derived_dir = get_derived_path(dataset_name)
    derived_dir.mkdir(parents=True, exist_ok=True)
    return derived_dir


def get_metadata_path(dataset_name: str) -> Path:
    """Get path to metadata.json for dataset.

    Args:
        dataset_name: Name of dataset.

    Returns:
        Path to metadata.json file.
    """
    return get_derived_path(dataset_name) / "metadata.json"


def list_dataset_files(dataset_name: str, pattern: str = "*.json") -> list[Path]:
    """List all files in dataset directory matching pattern.

    Args:
        dataset_name: Name of dataset.
        pattern: Glob pattern for files (default: "*.json").

    Returns:
        List of file paths sorted by name.
    """
    dataset_path = get_dataset_path(dataset_name)
    if not dataset_path.exists():
        return []
    return sorted(dataset_path.glob(pattern))
