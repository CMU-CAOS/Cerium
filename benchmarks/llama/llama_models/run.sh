#!/bin/bash

NGPUS=1
CHECKPOINT_DIR=~/.llama/checkpoints/Llama3.1-8B
PYTHONPATH=$(git rev-parse --show-toplevel) \
  torchrun --nproc_per_node=$NGPUS \
  -m mmlu $CHECKPOINT_DIR \
  --world_size $NGPUS

