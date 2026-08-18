#!/bin/bash
#SBATCH --job-name=kp-evaluate-all
#SBATCH --output=logs/evaluate_all_%j.out
#SBATCH --error=logs/evaluate_all_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=25G
#SBATCH --time=02:00:00
#SBATCH --mail-type=BEGIN,END,FAIL


cd ${PROJECT_DIR}

apptainer exec --nv --env-file ${PROJECT_DIR}/.env \
    --bind ${PROJECT_DIR}:/workspace \
    --bind ${DATA_DIR}:/workspace/data \
    ${PROJECT_DIR}/out/container.sif \
    /opt/venv/bin/python /workspace/scripts/evaluate_all.py ${EVAL_ARGS}
