SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c
.DELETE_ON_ERROR:
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules
ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

# Configuration (set in .env)
USERNAME ?=
GPU_ID ?=
FOLDER_NAME = keyword-pecker

# Server selection: kt-gpu2 (default, tmux) or kt-gpu4 (slurm)
SERVER ?= kt-gpu2

ifeq ($(SERVER),kt-gpu4)
  REMOTE_HOST = $(KT_GPU4_HOST)
  REMOTE_DATA_DIR = /ceph/home/$(USERNAME)/data/$(FOLDER_NAME)
  REMOTE_PROJECT_DIR = /ceph/home/$(USERNAME)/$(FOLDER_NAME)
else
  REMOTE_HOST = $(KT_GPU2_HOST)
  REMOTE_DATA_DIR = ~/data/keyword-pecker
  REMOTE_PROJECT_DIR = ~/$(FOLDER_NAME)
endif

# Load local environment variables from .env (provides DATASET_FOLDER etc.)
ifneq (,$(wildcard .env))
  include .env
  export
endif

# ============================================================================
# CONTAINER TARGETS
# ============================================================================

# Build the Apptainer image from container.def
out/container.sif: container.def
> mkdir -p out
> apptainer build --force --fakeroot out/container.sif container.def

# Launch an interactive shell inside the container with GPU access
shell:
> apptainer shell --nv --env-file .env \
>     --env CUDA_VISIBLE_DEVICES=${GPU_ID} \
>     --bind ${PWD}:/workspace out/container.sif
.PHONY: shell

# Launch a detached training session inside the container.
# KT-GPU2: tmux session. KT-GPU4: SLURM job.
# Usage: make remote-train [SERVER=kt-gpu4] [TRAIN_ARGS="--wandb"]
remote-train:
ifeq ($(SERVER),kt-gpu4)
> ssh ${USERNAME}@${REMOTE_HOST} \
>     "sbatch --mail-user=${SLURM_EMAIL} --export=ALL,PROJECT_DIR=${REMOTE_PROJECT_DIR},DATA_DIR=${REMOTE_DATA_DIR},TRAIN_ARGS='${TRAIN_ARGS}' \
>      ${REMOTE_PROJECT_DIR}/slurm/train.sbatch"
else
> ssh ${USERNAME}@${REMOTE_HOST} \
>     "tmux new-session -d -s remote-train \
>     'apptainer exec --nv --env-file ${REMOTE_PROJECT_DIR}/.env \
>         --env CUDA_VISIBLE_DEVICES=${GPU_ID} \
>         --bind ${REMOTE_PROJECT_DIR}:/workspace \
>         ${REMOTE_PROJECT_DIR}/out/container.sif \
>         env UV_PROJECT_ENVIRONMENT=/opt/venv \
>         uv run /workspace/scripts/train.py ${TRAIN_ARGS} 2>&1 | tee ${REMOTE_DATA_DIR}/runs/train.log'"
endif
.PHONY: remote-train

attach-train:
> ssh -t ${USERNAME}@${REMOTE_HOST} "tmux attach-session -t remote-train"
.PHONY: attach-train

# Launch validation-set training on kt-gpu4 (SLURM only).
# Usage: make remote-train-valset SERVER=kt-gpu4 DATASET=kp20k
remote-train-valset:
ifndef DATASET
> $(error DATASET is required. Usage: make remote-train-valset SERVER=kt-gpu4 DATASET=kp20k)
endif
ifeq ($(SERVER),kt-gpu4)
> ssh ${USERNAME}@${REMOTE_HOST} \
>     "sbatch --mail-user=${SLURM_EMAIL} --export=ALL,PROJECT_DIR=${REMOTE_PROJECT_DIR},DATA_DIR=${REMOTE_DATA_DIR},DATASET=${DATASET} \
>      ${REMOTE_PROJECT_DIR}/slurm/train_valset.sbatch"
else
> $(error remote-train-valset is only supported on kt-gpu4)
endif
.PHONY: remote-train-valset

# Launch a detached evaluation session inside the container.
# KT-GPU2: tmux session. KT-GPU4: SLURM job.
# Usage: make remote-evaluate [SERVER=kt-gpu4] [EVAL_ARGS="--checkpoint path/to/model.pt --dataset kp20k --split test --device cuda"]
remote-evaluate:
ifeq ($(SERVER),kt-gpu4)
> ssh ${USERNAME}@${REMOTE_HOST} \
>     "sbatch --mail-user=${SLURM_EMAIL} --export=ALL,PROJECT_DIR=${REMOTE_PROJECT_DIR},DATA_DIR=${REMOTE_DATA_DIR},EVAL_ARGS='${EVAL_ARGS}' \
>      ${REMOTE_PROJECT_DIR}/slurm/evaluate.sbatch"
else
> ssh ${USERNAME}@${REMOTE_HOST} \
>     "tmux new-session -d -s remote-evaluate \
>     'apptainer exec --nv --env-file ${REMOTE_PROJECT_DIR}/.env \
>         --env CUDA_VISIBLE_DEVICES=${GPU_ID} \
>         --bind ${REMOTE_PROJECT_DIR}:/workspace \
>         --bind ${REMOTE_DATA_DIR}:/workspace/data \
>         ${REMOTE_PROJECT_DIR}/out/container.sif \
>         env UV_PROJECT_ENVIRONMENT=/opt/venv \
>         uv run /workspace/scripts/evaluate.py ${EVAL_ARGS} 2>&1 | tee ${REMOTE_DATA_DIR}/runs/evaluate.log'"
endif
.PHONY: remote-evaluate

attach-evaluate:
> ssh -t ${USERNAME}@${REMOTE_HOST} "tmux attach-session -t remote-evaluate"
.PHONY: attach-evaluate

# Batch evaluation on multiple datasets.
# Usage: make remote-evaluate-all RUN_ID=worthy-firefly-11 [EVAL_ARGS="--device cuda --datasets inspec,kp20k"]
remote-evaluate-all:
ifeq ($(SERVER),kt-gpu4)
> ssh ${USERNAME}@${REMOTE_HOST} \
>     "sbatch --mail-user=${SLURM_EMAIL} --export=ALL,PROJECT_DIR=${REMOTE_PROJECT_DIR},DATA_DIR=${REMOTE_DATA_DIR},EVAL_ARGS='${EVAL_ARGS}' \
>      ${REMOTE_PROJECT_DIR}/slurm/evaluate_all.sbatch"
else
> ssh ${USERNAME}@${REMOTE_HOST} \
>     "tmux new-session -d -s remote-eval-all \
>     'apptainer exec --nv --env-file ${REMOTE_PROJECT_DIR}/.env \
>         --env CUDA_VISIBLE_DEVICES=${GPU_ID} \
>         --bind ${REMOTE_PROJECT_DIR}:/workspace \
>         --bind ${REMOTE_DATA_DIR}:/workspace/data \
>         ${REMOTE_PROJECT_DIR}/out/container.sif \
>         env UV_PROJECT_ENVIRONMENT=/opt/venv \
>         uv run /workspace/scripts/evaluate_all.py ${EVAL_ARGS} 2>&1 | tee ${REMOTE_DATA_DIR}/runs/evaluate_all.log'"
endif
.PHONY: remote-evaluate-all

attach-evaluate-all:
> ssh -t ${USERNAME}@${REMOTE_HOST} "tmux attach-session -t remote-eval-all"
.PHONY: attach-evaluate-all

# ============================================================================
# DEVELOPMENT WORKFLOW TARGETS
# ============================================================================

# Interactive shell in container on remote cluster
shell-remote:
> ssh -t ${USERNAME}@${REMOTE_HOST} "cd ${REMOTE_PROJECT_DIR} && make shell"
.PHONY: shell-remote

# Interactive shell in container on remote cluster with data directory mounted
shell-remote-data:
> ssh -t ${USERNAME}@${REMOTE_HOST} \
>     "apptainer shell --nv --env-file ${REMOTE_PROJECT_DIR}/.env \
>         --env CUDA_VISIBLE_DEVICES=${GPU_ID} \
>         --bind ${REMOTE_PROJECT_DIR}:/workspace \
>         --bind ${REMOTE_DATA_DIR}:/workspace/data \
>         ${REMOTE_PROJECT_DIR}/out/container.sif"
.PHONY: shell-remote-data

# Quick sync (useful during development iterations)
sync:
> @echo "# Auto-generated by make sync — do not edit" > src/keyword_pecker/_version.py
> @echo "GIT_COMMIT = \"$(shell git rev-parse HEAD)\"" >> src/keyword_pecker/_version.py
> rsync -avzP \
>     --exclude=.git \
>     --exclude=.env \
>     --exclude=tmp \
>     --exclude=.venv \
>     --exclude=__pycache__ \
>     --exclude=.ruff_cache \
>     --exclude=.pytest_cache \
>     --exclude=.agents \
>     --exclude='*.pyc' \
>     --exclude=out \
>     --exclude=wandb \
>     --exclude=_build \
>     --exclude=notebooks \
>     --exclude=doc \
>     ${PWD}/ ${USERNAME}@${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/
.PHONY: sync

# Sync only raw JSON files from local DATASET_FOLDER to remote.
# Excludes derived/ (created by remote-preprocess-data) and runs/ (populated during training).
sync-raw-data:
> rsync -avzP \
>     --ignore-existing \
>     --exclude='derived/' \
>     --exclude='runs/' \
>     ${DATASET_FOLDER}/ ${USERNAME}@${REMOTE_HOST}:${REMOTE_DATA_DIR}/
.PHONY: sync-raw-data

# Copy a specific run folder from remote to local (no model checkpoints)
# Usage: make copy-run-folder RUN_NAME=my-run-name
copy-run-folder:
> rsync -avzP \
>     --exclude="*.pt" \
>     ${USERNAME}@${REMOTE_HOST}:${REMOTE_DATA_DIR}/runs/${RUN_NAME}/ \
>     ~/data/keyword-pecker-data/runs/${RUN_NAME}/
.PHONY: copy-run-folder

# Copy model checkpoints from a specific run folder on remote to local.
# Usage: make copy-run-model RUN_NAME=my-run-name
copy-run-model:
> rsync -avzP \
>     --include="model_best.pt" \
>     --include="model.pt" \
>     --exclude="*" \
>     ${USERNAME}@${REMOTE_HOST}:${REMOTE_DATA_DIR}/runs/${RUN_NAME}/ \
>     ~/data/keyword-pecker-data/runs/${RUN_NAME}/
.PHONY: copy-run-model

# Run prepare_data.py on the remote server inside the Apptainer container (CPU-only, no --nv).
# KT-GPU2: tmux session. KT-GPU4: SLURM job.
# Usage: make remote-preprocess-data [SERVER=kt-gpu4] [DATASET=kp20k] [OPT_ARGS="--arg1 --arg2"]
DATASET ?= kp20k
OPT_ARGS ?=
remote-preprocess-data:
ifeq ($(SERVER),kt-gpu4)
> ssh ${USERNAME}@${REMOTE_HOST} \
>     "sbatch --mail-user=${SLURM_EMAIL} --export=ALL,PROJECT_DIR=${REMOTE_PROJECT_DIR},DATA_DIR=${REMOTE_DATA_DIR},DATASET=${DATASET},OPT_ARGS='${OPT_ARGS}' \
>      ${REMOTE_PROJECT_DIR}/slurm/preprocess.sbatch"
else
> ssh ${USERNAME}@${REMOTE_HOST} \
>     "tmux new-session -d -s preprocess-${DATASET} \
>     'apptainer exec --env-file ${REMOTE_PROJECT_DIR}/.env \
>         --bind ${REMOTE_PROJECT_DIR}:/workspace \
>         --bind ${REMOTE_DATA_DIR}:/workspace/data \
>         ${REMOTE_PROJECT_DIR}/out/container.sif \
>         env UV_PROJECT_ENVIRONMENT=/opt/venv \
>         uv run /workspace/scripts/prepare_data.py $(DATASET) ${OPT_ARGS}'"
endif
.PHONY: remote-preprocess-data

# Attach to the running preprocess-data tmux session on the remote host to monitor progress
attach-preprocess:
> ssh -t ${USERNAME}@${REMOTE_HOST} "tmux attach-session -t preprocess-${DATASET}"
.PHONY: attach-preprocess

# Run split_validation.py on the remote server inside the Apptainer container (CPU-only, no --nv).
# KT-GPU2: tmux session. KT-GPU4: SLURM job.
# Usage: make remote-split-validation [SERVER=kt-gpu4] [DATASET=kptimes] [OPT_ARGS="--train-ratio 0.8 --seed 42"]
remote-split-validation:
ifeq ($(SERVER),kt-gpu4)
> ssh ${USERNAME}@${REMOTE_HOST} \
>     "sbatch --mail-user=${SLURM_EMAIL} --export=ALL,PROJECT_DIR=${REMOTE_PROJECT_DIR},DATA_DIR=${REMOTE_DATA_DIR},DATASET=${DATASET},OPT_ARGS='${OPT_ARGS}' \
>      ${REMOTE_PROJECT_DIR}/slurm/split_validation.sbatch"
else
> ssh ${USERNAME}@${REMOTE_HOST} \
>     "tmux new-session -d -s split-valid-${DATASET} \
>     'apptainer exec --env-file ${REMOTE_PROJECT_DIR}/.env \
>         --bind ${REMOTE_PROJECT_DIR}:/workspace \
>         --bind ${REMOTE_DATA_DIR}:/workspace/data \
>         ${REMOTE_PROJECT_DIR}/out/container.sif \
>         env UV_PROJECT_ENVIRONMENT=/opt/venv \
>         uv run /workspace/scripts/split_validation.py $(DATASET) ${OPT_ARGS}'"
endif
.PHONY: remote-split-validation

# Attach to the running split-validation tmux session on the remote host to monitor progress
attach-split-validation:
> ssh -t ${USERNAME}@${REMOTE_HOST} "tmux attach-session -t split-valid-${DATASET}"
.PHONY: attach-split-validation

clean:
> rm -rf out
.PHONY: clean

# ============================================================================
# REMOTE CONTAINER BUILD
# ============================================================================

# Build the Apptainer container on the remote server.
# Usage: make remote-build-container [SERVER=kt-gpu4]
remote-build-container:
ifeq ($(SERVER),kt-gpu4)
> ssh ${USERNAME}@${REMOTE_HOST} \
>     "cd ${REMOTE_PROJECT_DIR} && \
>      mkdir -p out && \
>      apptainer build --force --fakeroot out/container.sif container.def"
else
> ssh ${USERNAME}@${REMOTE_HOST} \
>     "cd ${REMOTE_PROJECT_DIR} && make out/container.sif"
endif
.PHONY: remote-build-container

# ============================================================================
# SLURM MANAGEMENT (KT-GPU4)
# ============================================================================

# One-time setup for KT-GPU4: create project directories on /cache
slurm-setup:
> ssh ${USERNAME}@${REMOTE_HOST} "mkdir -p /ceph/home/${USERNAME}/${FOLDER_NAME}/{logs,out,data}"
.PHONY: slurm-setup

# Check SLURM job queue
slurm-status:
> ssh ${USERNAME}@${REMOTE_HOST} "squeue -u ${USERNAME}"
.PHONY: slurm-status

# View SLURM job output log (requires JOB_ID)
slurm-logs:
ifndef JOB_ID
> $(error JOB_ID is required. Usage: make slurm-logs JOB_ID=12345)
endif
> ssh ${USERNAME}@${REMOTE_HOST} "tail -f /ceph/home/${USERNAME}/${FOLDER_NAME}/logs/train_${JOB_ID}.out"
.PHONY: slurm-logs

# Cancel a SLURM job (requires JOB_ID)
slurm-cancel:
ifndef JOB_ID
> $(error JOB_ID is required. Usage: make slurm-cancel JOB_ID=12345)
endif
> ssh ${USERNAME}@${REMOTE_HOST} "scancel ${JOB_ID}"
.PHONY: slurm-cancel
