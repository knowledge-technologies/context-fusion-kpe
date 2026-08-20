#!/bin/bash
#SBATCH --job-name=kp-test-data
#SBATCH --output=logs/test_data_%j.out
#SBATCH --error=logs/test_data_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00


cd ${PROJECT_DIR}

apptainer exec --env-file ${PROJECT_DIR}/.env \
    --bind ${PROJECT_DIR}:/workspace \
    --bind ${DATA_DIR}:/workspace/data \
    ${PROJECT_DIR}/out/container.sif \
    bash -c '/opt/venv/bin/python -c "
import os
dataset = os.environ.get(\"DATASET\", \"kp20k\")
data_path = f\"/workspace/data/{dataset}/derived/valid_80.parquet\"
print(f\"Checking: {data_path}\")
print(f\"Exists: {os.path.exists(data_path)}\")
if os.path.exists(data_path):
    print(\"SUCCESS: data bind and path resolution work\")
else:
    print(\"FAILED: file not found\")
    print(f\"DATASET={os.environ.get(\"DATASET\", \"not set\")}\")
    print(f\"DATA_DIR={os.environ.get(\"DATA_DIR\", \"not set\")}\")
    print(\"Contents of /workspace/data/:\")
    for item in os.listdir(\"/workspace/data\"):
        print(f\"  {item}\")
"'
