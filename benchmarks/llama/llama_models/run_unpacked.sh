#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export DEVICE="cpu"
export DATA_SAMPLE="${1:-0}"
export BLOCK="${2:-31}"
NGPUS=1
CHECKPOINT_DIR="model_weights/Llama3.1-8B"
MASTER_PORT=10808 \
DEVICE="$DEVICE" PYTHONPATH=$(git rev-parse --show-toplevel) \
  torchrun --master_port 10808 --nproc_per_node=$NGPUS \
  -m unpacked $CHECKPOINT_DIR \
  --world_size $NGPUS \
  --data_sample "$DATA_SAMPLE" \
  

