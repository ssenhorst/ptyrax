#!/usr/bin/env bash
set -euo pipefail

GPU_ID=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
  | awk '{print NR-1, $1}' \
  | sort -k2 -nr \
  | head -n1 \
  | cut -d' ' -f1)
export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "Using GPU $CUDA_VISIBLE_DEVICES"

export XLA_PYTHON_CLIENT_MEM_FRACTION=.3
exec "$@"