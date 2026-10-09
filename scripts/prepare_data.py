#!/usr/bin/env python
"""Preprocess raw JSON datasets into training-ready Parquet files.

This script processes raw dataset files through the full preprocessing pipeline:
1. Sentence segmentation (spaCy)
2. Tokenization (HuggingFace)
3. Keyword matching and alignment
4. Output to Parquet format (in {DATASET_FOLDER}/derived as specified in .env)

Usage:
    # Process all files in kp20k dataset
    uv run python scripts/prepare_data.py kp20k

    # Process specific splits
    uv run python scripts/prepare_data.py kp20k --splits train valid

    # Quick test with limited rows
    uv run python scripts/prepare_data.py kp20k --limit 20

    # Force reprocessing existing files
    uv run python scripts/prepare_data.py kp20k --overwrite

    # Use custom config
    uv run python scripts/prepare_data.py kp20k --config configs/my_preprocessing.yaml
"""

import argparse
import json
from datetime import datetime

from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
from transformers import AutoTokenizer

from ctx_fusion.data import (
    ensure_derived_dir,
    get_dataset_path,
    get_derived_path,
    get_metadata_path,
    list_dataset_files,
    load_spacy_model,
    process_dataset,
    write_parquet,
)
from ctx_fusion.utils import get_logger, load_config

logger = get_logger("prepare_data")


def extract_split_name(filepath) -> str:
    """Extract split name from filepath.

    Assumes filenames like 'kp20k_train.json' -> 'train'
    or 'train.json' -> 'train'.
    """
    stem = filepath.stem
    # Try to extract split from {dataset}_{split} pattern
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[-1]
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess datasets for keyphrase extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "dataset",
        help="Dataset name to process (e.g., kp20k)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        help="Specific splits to process (e.g., train valid). "
        "If not provided, processes all JSON files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of rows per file (for testing)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed files",
    )
    parser.add_argument(
        "--config",
        default="configs/preprocessing.yaml",
        help="Path to preprocessing config file (default: configs/preprocessing.yaml)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    model_config = load_config("configs/models.yaml")

    # Initialize models (expensive, do once)
    logger.info("Loading models...")
    nlp = load_spacy_model(
        model_name=config["segmentation"]["spacy_model"],
        exclude=config["segmentation"]["exclude"],
    )
    tokenizer_cfg = model_config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg["model_name"])
    stemmer = PorterStemmer()
    device = config["processing"]["device"]
    match_mode = config.get("matching", {}).get("match_mode", "stem_equality")

    logger.info(f"spaCy model: {config['segmentation']['spacy_model']}")
    logger.info(f"Tokenizer: {tokenizer_cfg['model_name']}")
    logger.info(f"Device: {device}")
    logger.info(f"Match mode: {match_mode}")

    # Get dataset paths
    dataset_path = get_dataset_path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        raise SystemExit(1)

    derived_dir = ensure_derived_dir(args.dataset)

    # Find files to process
    all_files = list_dataset_files(args.dataset, pattern="*.json")

    if not all_files:
        logger.error(f"No JSON files found in {dataset_path}")
        raise SystemExit(1)

    # Filter by splits if specified
    if args.splits:
        files_to_process = [
            f for f in all_files if any(split in f.stem for split in args.splits)
        ]
        if not files_to_process:
            logger.error(f"No files found matching splits: {args.splits}")
            logger.error(f"Available files: {[f.name for f in all_files]}")
            raise SystemExit(1)
    else:
        files_to_process = all_files

    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Input path: {dataset_path}")
    logger.info(f"Output path: {derived_dir}")
    logger.info(f"Files to process: {len(files_to_process)}")
    if args.limit:
        logger.info(f"Row limit: {args.limit}")

    # Process each file
    total_sentences = 0
    total_documents = 0
    processing_stats = {}

    for filepath in tqdm(files_to_process, desc="Processing files"):
        # Determine output path
        split_name = extract_split_name(filepath)
        output_path = get_derived_path(args.dataset, split_name)

        # Skip if exists and not overwriting
        if output_path.exists() and not args.overwrite:
            logger.info(f"Skipping {filepath.name} (output exists, use --overwrite)")
            continue

        # Determine labeling scheme
        scheme = config.get("preprocessing", {}).get("scheme", "IOB")

        # Process file
        logger.info(f"Preprocessing (segmentation, tokenizing...) {filepath.name}...")
        processed_data = process_dataset(
            filepath=filepath,
            nlp=nlp,
            spacy_batch_size=config["segmentation"]["batch_size"],
            spacy_n_process=config["segmentation"]["n_process"],
            tokenizer=tokenizer,
            stemmer=stemmer,
            scheme=scheme,
            limit=args.limit,
            device=device,
            show_progress=True,
            match_mode=match_mode,
            truncation=tokenizer_cfg.get("truncation", True),
            padding=tokenizer_cfg.get("padding", True),
        )

        # Write to Parquet
        num_rows = write_parquet(processed_data, output_path)

        # Track stats
        doc_rows = set(row["doc_row"] for row in processed_data)
        num_docs = len(doc_rows)
        total_sentences += num_rows
        total_documents += num_docs

        processing_stats[split_name] = {
            "source_file": filepath.name,
            "output_file": output_path.name,
            "num_documents": num_docs,
            "num_sentences": num_rows,
        }

        logger.info(f"Wrote {num_rows} sentences from {num_docs} documents")

    # Write metadata
    if config["output"]["write_metadata"] and processing_stats:
        num_classes = 3 if scheme == "IOB" else 2

        metadata = {
            "dataset": args.dataset,
            "processed_at": datetime.now().isoformat(),
            "config": config,
            "scheme": scheme,
            "num_classes": num_classes,
            "match_mode": match_mode,
            "tokenizer_name": tokenizer_cfg["model_name"],
            "limit": args.limit,
            "stats": {
                "total_documents": total_documents,
                "total_sentences": total_sentences,
            },
            "splits": processing_stats,
        }

        metadata_path = get_metadata_path(args.dataset)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Wrote metadata to {metadata_path}")

    logger.info("Processing complete!")
    logger.info(f"Total documents: {total_documents}")
    logger.info(f"Total sentences: {total_sentences}")
    logger.info(f"Output directory: {derived_dir}")


if __name__ == "__main__":
    main()
