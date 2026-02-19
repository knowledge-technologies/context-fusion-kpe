# Keyword-pecker

Transformer-based keyphrase extraction model.

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)

## Installation

Install dependencies using `uv`:

```bash
uv sync
```

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and set the `DATASET_FOLDER` variable to the directory where your datasets are stored.
   ```bash
   DATASET_FOLDER=/absolute/path/to/your/datasets
   ```

## Usage

### Data Preparation

The raw `.json` data should be placed in `DATASET_FOLDER/<dataset_name>`

To prepare your data for training, use the `prepare_data.py` script. You can run it via `uv run`:

```bash
uv run scripts/prepare_data.py --help
```

Example usage:
```bash
uv run scripts/prepare_data.py kp20k --splits train valid
```

This will create a `derived` subfolder under your `DATASET_FOLDER` and and `train.parquet`, `valid.parquet` etc. files there.
### Running Tests

Run the test suite using `pytest`:

```bash
uv run pytest
```
