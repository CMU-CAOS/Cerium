#!/bin/bash

export DEVICE="${DEVICE:-cpu}"
NGPUS=1
CHECKPOINT_DIR="model_weights/Llama3.1-8B"
PYTHONPATH=$(git rev-parse --show-toplevel) \
  torchrun --standalone --nproc_per_node=$NGPUS \
  -m completion_packed_inputs "$CHECKPOINT_DIR" \
  --world_size $NGPUS "$@"
