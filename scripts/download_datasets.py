#!/usr/bin/env python
"""Download dataset files to the local dataset folder.

Usage:
    uv run python scripts/download_datasets.py                    # Download all datasets
    uv run python scripts/download_datasets.py kp20k              # Download specific dataset
    uv run python scripts/download_datasets.py --list             # List available datasets
    uv run python scripts/download_datasets.py kp20k --splits train valid  # Download specific splits
"""

import argparse
import os
import urllib.request
from pathlib import Path
from urllib.parse import unquote, urlparse

from dotenv import load_dotenv

from keyword_pecker.utils import load_config


def get_dataset_folder() -> Path:
    """Get the dataset folder from environment variable."""
    load_dotenv()
    dataset_folder = os.getenv("DATASET_FOLDER")
    if not dataset_folder:
        raise ValueError(
            "DATASET_FOLDER environment variable not set. "
            "Please set it in your .env file."
        )
    return Path(dataset_folder)


def download_file(url: str, dest: Path) -> None:
    """Download a file from URL to destination path.

    Args:
        url: URL to download from.
        dest: Local destination path.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        print(f"  Skipping {dest.name} (already exists)")
        return

    print(f"  Downloading {dest.name}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved to {dest}")
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        raise


def download_dataset(
    name: str,
    config: dict,
    dataset_folder: Path,
    splits: list[str] | None = None,
) -> None:
    """Download a single dataset.

    Args:
        name: Dataset name.
        config: Dataset configuration dict with 'splits' and 'description'.
        dataset_folder: Base folder for datasets.
        splits: Optional list of specific splits to download. If None, downloads all.
    """
    print(f"\nDataset: {name}")
    print(f"  Description: {config.get('description', 'No description')}")

    dataset_dir = dataset_folder / name
    available_splits = config.get("splits", {})

    if not available_splits:
        print(f"  Warning: No splits defined for {name}")
        return

    for split_name, url in available_splits.items():
        # Parse URL to extract clean filename (without query params)
        parsed = urlparse(url)
        filename = unquote(Path(parsed.path).name)
        dest = dataset_dir / filename
        download_file(url, dest)


def list_datasets(datasets_config: dict) -> None:
    """Print available datasets and their splits."""
    print("Available datasets:\n")
    for name, config in datasets_config.items():
        print(f"  {name}:")
        print(f"    Description: {config.get('description', 'No description')}")
        splits = config.get("splits", {})
        if splits:
            print(f"    Splits: {', '.join(splits.keys())}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download datasets for keyword-pecker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset names to download. If not specified, downloads all.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit.",
    )
    parser.add_argument(
        "--config",
        default="configs/datasets.yaml",
        help="Path to datasets config file (default: configs/datasets.yaml).",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    datasets_config = config.get("datasets", {})

    if not datasets_config:
        print("No datasets found in configuration.")
        raise SystemExit(1)

    # List mode
    if args.list:
        list_datasets(datasets_config)
        raise SystemExit(0)

    # Download mode
    dataset_folder = get_dataset_folder()
    print(f"Dataset folder: {dataset_folder}")

    # Determine which datasets to download
    if args.datasets:
        datasets_to_download = args.datasets
        # Validate dataset names
        unknown = set(datasets_to_download) - set(datasets_config.keys())
        if unknown:
            print(f"Error: Unknown datasets: {unknown}")
            print(f"Available: {list(datasets_config.keys())}")
            raise SystemExit(1)
    else:
        datasets_to_download = list(datasets_config.keys())

    # Download each dataset
    for name in datasets_to_download:
        download_dataset(
            name=name,
            config=datasets_config[name],
            dataset_folder=dataset_folder,
        )


if __name__ == "__main__":
    main()
