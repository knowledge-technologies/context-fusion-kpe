# Keyword-pecker

Transformer-based keyphrase extraction model.

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)
- [Apptainer/Singularity](https://apptainer.org/) (for container workflow)

## Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd keyword-pecker
uv sync
```

## Container

For cluster environments, build a Singularity/Apptainer image:

1. Copy the example definition file and adjust proxy settings:
   ```bash
   cp container.def.example container.def
   ```

2. Build the image:
   ```bash
   apptainer build --force --fakeroot out/container.sif container.def
   ```

   Or via Makefile:
   ```bash
   make out/container.sif
   ```

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` and set the required variables. Key settings:

   ```
   USERNAME=your-username
   GPU_ID=0
   DATASET_FOLDER=/path/to/your/dataset/location
   EXPERIMENT_RUNS_FOLDER=/path/to/your/runs/location
   WANDB_API_KEY=
   WANDB_PROJECT=keyword-pecker
   SLURM_EMAIL=your-email@example.com
   ```

   See `.env.example` for the full list of variables including proxy and server hostname settings.

## Config Files

YAML configs in `configs/` control preprocessing, model architecture, datasets, and training hyperparameters:

- `preprocessing.yaml` — sentence segmentation, tokenization, keyword matching settings
- `models.yaml` — model architecture and encoder choices
- `datasets.yaml` — dataset-specific parameters
- `training.yaml` — learning rates, batch sizes, early stopping, evaluation settings, wandb config

## Scripts

### Preprocessing

Convert raw JSON datasets into Parquet files for training:

```bash
uv run scripts/prepare_data.py kp20k --splits train valid
```

Place raw `.json` files in `DATASET_FOLDER/<dataset_name>`. The script creates a `derived` subfolder with `train.parquet`, `valid.parquet`, etc.

### Training

Prerequisites: `prepare_data.py` must have been run first (it creates the parquet files in `{DATASET_FOLDER}/derived/` that this script reads from), and `DATASET_FOLDER` must be set in `.env`.

```bash
uv run scripts/train.py --dataset kp20k --device cuda
uv run scripts/train.py --from_config --wandb
```

### Evaluation

Prerequisites: `prepare_data.py` must have been run, `DATASET_FOLDER` set in `.env`, and a trained model checkpoint available.

```bash
uv run scripts/evaluate.py --checkpoint path/to/model.pt --dataset kp20k --split test
```

## Makefile

The Makefile provides targets for container management, remote execution, and development workflow.

### Container

| Target | Description |
|---|---|
| `out/container.sif` | Build the Apptainer image |
| `shell` | Interactive shell inside the container with GPU access |

### Remote execution

Requires SSH + configured `.env`.

| Target | Description |
|---|---|
| `remote-train` | Start training on remote server (tmux or SLURM) |
| `remote-evaluate` | Run evaluation on remote server |
| `remote-preprocess-data` | Preprocess data on remote server |
| `attach-train` / `attach-evaluate` | Attach to running remote sessions |

### Data

| Target | Description |
|---|---|
| `sync-raw-data` | Sync raw JSON datasets to remote |
| `copy-run-folder RUN_NAME=...` | Copy a run folder from remote |
| `copy-run-model RUN_NAME=...` | Copy model checkpoints from remote |

### Development

| Target | Description |
|---|---|
| `sync` | Rsync project to remote server |

### SLURM management

| Target | Description |
|---|---|
| `slurm-setup` | Create project directories on cluster |
| `slurm-status` | Check job queue |
| `slurm-logs JOB_ID=...` | Tail SLURM job output |
| `slurm-cancel JOB_ID=...` | Cancel a job |

## SLURM Scripts

The `slurm/` directory contains batch scripts for cluster job submission. They run commands inside the Apptainer container with GPU access. Example: `slurm/train.sbatch` requests 1 GPU, 64G RAM, and runs `scripts/train.py` with args passed via the `TRAIN_ARGS` environment variable.

Submit via `make remote-train SERVER=kt-gpu4` or directly with `sbatch`.

## Running Tests

Run the test suite using `pytest`:

```bash
uv run pytest
```
