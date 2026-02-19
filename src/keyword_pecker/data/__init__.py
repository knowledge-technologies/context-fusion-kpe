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

from keyword_pecker.data.paths import (
    ensure_derived_dir,
    get_dataset_folder,
    get_dataset_path,
    get_derived_path,
    list_dataset_files,
)
from keyword_pecker.data.preprocessing import (
    SENTENCE_SCHEMA,
    build_keyword_regex,
    collect_sentences,
    create_keyword_mask,
    extract_keywords,
    load_spacy_model,
    process_document,
    process_dataset,
    resolve_keyword_spans,
    segment_sentences,
    tokenize_sentences,
    write_parquet,
)
from keyword_pecker.data.dataset import create_keyphrase_dataloader

__all__ = [
    # Path utilities
    "ensure_derived_dir",
    "get_dataset_folder",
    "get_dataset_path",
    "get_derived_path",
    "list_dataset_files",
    # Preprocessing functions
    "SENTENCE_SCHEMA",
    "build_keyword_regex",
    "collect_sentences",
    "create_keyword_mask",
    "extract_keywords",
    "load_spacy_model",
    "process_document",
    "process_dataset",
    "resolve_keyword_spans",
    "segment_sentences",
    "tokenize_sentences",
    "write_parquet",
    # Dataloader
    "create_keyphrase_dataloader",
]
