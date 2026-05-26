#!/bin/bash

export DEVICE="cpu"
export LLAMA_PARAMS_BLOCK=$1
NGPUS=1
CHECKPOINT_DIR="model_weights/Llama3.1-8B"
MASTER_PORT="${MASTER_PORT:-10809}"
DEVICE="cpu" PYTHONPATH=$(git rev-parse --show-toplevel) \
  torchrun --master_port $MASTER_PORT --nproc_per_node=$NGPUS \
  -m run_params_direct $CHECKPOINT_DIR \
  --world_size $NGPUS \
  
