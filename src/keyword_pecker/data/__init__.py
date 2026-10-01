"""Data processing module for keyphrase extraction.

This module provides functions for preprocessing raw JSON datasets into
training-ready Parquet files with token-level keyword labels.

Main components:
- paths: Dataset path resolution using DATASET_FOLDER environment variable
- preprocessing: Text processing (sentence segmentation, keyword matching,
  tokenization, and Parquet I/O)

Example usage:
    from keyword_pecker.data import (
        get_dataset_path,
        get_derived_path,
        load_spacy_model,
        process_file,
        write_parquet,
    )

    # Initialize models
    nlp = load_spacy_model()
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    stemmer = PorterStemmer()

    # Process a file
    data = process_file(filepath, nlp, tokenizer, stemmer)
    write_parquet(data, output_path)
"""

from pathlib import Path

from keyword_pecker.data.paths import (
    ensure_derived_dir,
    get_dataset_folder,
    get_dataset_path,
    get_derived_path,
    get_metadata_path,
    list_dataset_files,
)
from keyword_pecker.data.preprocessing import (
    SENTENCE_SCHEMA,
    build_keyword_regex,
    collect_sentences,
    create_keyword_mask,
    extract_keywords,
    extract_word_level_data,
    load_spacy_model,
    pretokenize_sentence,
    process_document,
    process_dataset,
    resolve_keyword_spans,
    resolve_keyword_spans_stem,
    segment_sentences,
    tokenize_sentences,
    word_ranges_to_token_ranges,
    write_parquet,
)
from keyword_pecker.data.dataset import (
    compute_class_weights,
    create_keyphrase_dataloader,
)
from keyword_pecker.data.external import (
    BENCHMARKS_EVALS_PATH,
    EXTERNAL_DIR,
    SEKE_TRAINSET_PATH,
    SEKE_VALSET_PATH,
)

RUNS_MAPPING_PATH = Path(__file__).parent / "runs.json"

KP20K_TRAINING_SET_EVAL_FILE = "eval_kp20k_test_20260812T122055Z.json"

__all__ = [
    # Path utilities
    "ensure_derived_dir",
    "get_dataset_folder",
    "get_dataset_path",
    "get_derived_path",
    "get_metadata_path",
    "list_dataset_files",
    # Runs mapping
    "RUNS_MAPPING_PATH",
    # Manuscript constants
    "KP20K_TRAINING_SET_EVAL_FILE",
    # External data paths
    "BENCHMARKS_EVALS_PATH",
    "EXTERNAL_DIR",
    "SEKE_TRAINSET_PATH",
    "SEKE_VALSET_PATH",
    # Preprocessing functions
    "SENTENCE_SCHEMA",
    "build_keyword_regex",
    "collect_sentences",
    "create_keyword_mask",
    "extract_keywords",
    "extract_word_level_data",
    "load_spacy_model",
    "pretokenize_sentence",
    "process_document",
    "process_dataset",
    "resolve_keyword_spans",
    "resolve_keyword_spans_stem",
    "segment_sentences",
    "tokenize_sentences",
    "word_ranges_to_token_ranges",
    "write_parquet",
    # Dataloader
    "compute_class_weights",
    "create_keyphrase_dataloader",
]
