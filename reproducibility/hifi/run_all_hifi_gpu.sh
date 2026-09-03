#!/usr/bin/env bash
# Runs one GPU's shard of the full HiFI+ MOMFBD batch across all datasets.
# Usage: run_all_hifi_gpu.sh <gpu_id> <num_shards>
set -uo pipefail

GPU_ID="$1"
NUM_SHARDS="$2"
DATA_ROOT="/dat/andreuva/data/hifiplus/level1"
HIFI_DIR="/dat/andreuva/gpu/torchmfbd/reproducibility/hifi"

source /dat/andreuva/gpu/miniconda/etc/profile.d/conda.sh
conda activate torchmfbd
cd "$HIFI_DIR"

for dataset in 20260714 20260715 20260717; do
    echo "=== [GPU $GPU_ID] Starting dataset $dataset ($(date)) ==="
    python hifi_momfbd_batch_gpu.py \
        --input_dir "$DATA_ROOT/$dataset/" \
        --output_dir "results_momfbd/$dataset" \
        --gpu "$GPU_ID" \
        --num_shards "$NUM_SHARDS" \
        --shard_id "$GPU_ID" \
        --simultaneous_seq 200
    echo "=== [GPU $GPU_ID] Finished dataset $dataset ($(date)) ==="
done

echo "=== [GPU $GPU_ID] All datasets complete ==="
