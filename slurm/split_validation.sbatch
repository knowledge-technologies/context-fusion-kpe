#!/bin/bash
#SBATCH --job-name=kp-split-valid
#SBATCH --output=logs/split_valid_%j.out
#SBATCH --error=logs/split_valid_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL



cd ${PROJECT_DIR}

apptainer exec --env-file ${PROJECT_DIR}/.env \
    --bind ${PROJECT_DIR}:/workspace \
    --bind ${DATA_DIR}:/workspace/data \
    ${PROJECT_DIR}/out/container.sif \
    /opt/venv/bin/python /workspace/scripts/split_validation.py ${DATASET} ${OPT_ARGS}
