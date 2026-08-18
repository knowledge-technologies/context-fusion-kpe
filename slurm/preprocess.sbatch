#!/bin/bash
#SBATCH --job-name=kp-preprocess
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --mail-type=BEGIN,END,FAIL



cd ${PROJECT_DIR}

apptainer exec --env-file ${PROJECT_DIR}/.env \
    --bind ${PROJECT_DIR}:/workspace \
    --bind ${DATA_DIR}:/workspace/data \
    ${PROJECT_DIR}/out/container.sif \
    /opt/venv/bin/python /workspace/scripts/prepare_data.py ${DATASET} ${OPT_ARGS}
