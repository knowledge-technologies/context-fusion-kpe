"""Split a validation parquet into training and validation subsets.

Creates an 80/20 document-level split of the validation set for
validation-set training experiments (as described in the paper's
"validation set training scenario").

The split is performed at the document level to preserve the document
structure required by the two-level batching strategy.

Usage:
    # Create default 80/20 split
    uv run python scripts/split_validation.py kptimes

    # Custom split ratio
    uv run python scripts/split_validation.py kptimes --train-ratio 0.7

    # Force overwrite existing splits
    uv run python scripts/split_validation.py kptimes --overwrite

    # Custom random seed
    uv run python scripts/split_validation.py kptimes --seed 123
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import polars as pl

from ctx_fusion.data import get_derived_path, get_metadata_path
from ctx_fusion.utils import get_logger

logger = get_logger("split_validation")


def split_validation_dataset(
    dataset: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Split validation parquet into train and val subsets.

    Args:
        dataset: Dataset name (e.g., "kptimes").
        train_ratio: Fraction of documents for training (remainder for validation).
        seed: Random seed for reproducibility.
        overwrite: Whether to overwrite existing split files.

    Returns:
        Tuple of (train_80_path, val_20_path).
    """
    VALID_SUFFIX = "valid"
    valid_path = get_derived_path(dataset, VALID_SUFFIX)
    if not valid_path.exists():
        raise FileNotFoundError(f"Validation parquet not found: {valid_path}")

    train_out = get_derived_path(dataset_name=dataset, split="valid_80")
    val_out = get_derived_path(dataset_name=dataset, split="valid_20")
    meta_out = train_out.parent / "split_metadata.json"

    if train_out.exists() and val_out.exists() and meta_out.exists() and not overwrite:
        logger.info(
            f"Split files exist: {train_out}, {val_out} (use --overwrite to recreate)"
        )
        return train_out, val_out

    logger.info(f"Loading validation data from {valid_path}")
    df = pl.read_parquet(valid_path)

    # Get unique documents
    doc_df = df.select(["doc_id", "doc_row"]).unique(maintain_order=True)
    n_docs = len(doc_df)
    n_train = int(n_docs * train_ratio)

    logger.info(
        f"Splitting {n_docs} documents (train: {n_train}, val: {n_docs - n_train})"
    )

    # Shuffle and split at document level
    doc_df = doc_df.sample(fraction=1.0, seed=seed)
    train_docs = doc_df.slice(0, n_train)

    # Filter original dataframe by document membership
    train_df = df.join(train_docs, on=["doc_id", "doc_row"], how="semi")
    val_df = df.join(train_docs, on=["doc_id", "doc_row"], how="anti")

    # Write splits
    train_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.write_parquet(train_out, compression="snappy")
    val_df.write_parquet(val_out, compression="snappy")

    logger.info(
        f"Wrote train split: {train_out} ({len(train_df)} rows, {n_train} docs)"
    )
    logger.info(
        f"Wrote val split: {val_out} ({len(val_df)} rows, {n_docs - n_train} docs)"
    )

    # Write split metadata
    source_meta = None
    metadata_path = get_metadata_path(dataset)
    if metadata_path.exists():
        with open(metadata_path) as f:
            source_meta = json.load(f)

    split_meta = {
        "split_at": datetime.now().astimezone().isoformat(),  # local timezone
        "source_file": valid_path.name,
        "source_processed_at": source_meta.get("processed_at") if source_meta else None,
        "source_metadata": {
            "tokenizer_name": source_meta.get("tokenizer_name"),
            "scheme": source_meta.get("scheme"),
            "match_mode": source_meta.get("match_mode"),
        }
        if source_meta
        else None,
        "train_ratio": train_ratio,
        "seed": seed,
        "stats": {
            "total_documents": n_docs,
            "train_documents": n_train,
            "val_documents": n_docs - n_train,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
        },
    }

    with open(meta_out, "w") as f:
        json.dump(split_meta, f, indent=2)
    logger.info(f"Wrote split metadata to {meta_out}")

    return train_out, val_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split validation set into training and validation subsets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dataset",
        help="Dataset name (e.g., kptimes)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of documents for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split files",
    )

    args = parser.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1 (exclusive)")

    train_path, val_path = split_validation_dataset(
        dataset=args.dataset,
        train_ratio=args.train_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    logger.info(f"Split complete: {train_path} + {val_path}")


if __name__ == "__main__":
    main()
