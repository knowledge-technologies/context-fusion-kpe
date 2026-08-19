#!/bin/bash
#SBATCH --job-name=kp-train-valset
#SBATCH --output=logs/train_valset_%j.out
#SBATCH --error=logs/train_valset_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL


cd ${PROJECT_DIR}

apptainer exec --nv --env-file ${PROJECT_DIR}/.env \
    --bind ${PROJECT_DIR}:/workspace \
    --bind ${DATA_DIR}:/workspace/data \
    ${PROJECT_DIR}/out/container.sif \
    /opt/venv/bin/python /workspace/scripts/train.py \
        --dataset ${DATASET} \
        --train-data data/${DATASET}/derived/valid_80.parquet \
        --val-data data/${DATASET}/derived/valid_20.parquet \
        --epochs 20 \
        --outer-batch-size 100 \
        --inner-batch-size 50 \
        --val-batch-size 200 \
        --no-scheduler \
        --device cuda \
        --early-stopping-patience 5 \
        --early-stopping-delta 0.005 \
        --early-stopping-metric f1@10 \
        --early-stopping-mode max \
        --save-weights \
        --wandb
