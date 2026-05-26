#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Generate packed prompt embeddings and attention masks for benchmark prompts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=container_utils.sh
source "${SCRIPT_DIR}/container_utils.sh"

NUM_SAMPLES="${LLAMA_NUM_SAMPLES:-5}"
[[ "${NUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]] && (( NUM_SAMPLES <= 1000 )) || \
    llama_fail "LLAMA_NUM_SAMPLES must be an integer from 1 to 1000; got: ${NUM_SAMPLES}"

llama_ensure_container_running
llama_require_python
llama_require_checkpoint

for ((sample = 0; sample < NUM_SAMPLES; sample++)); do
    echo "Generating packed inputs for prompt ${sample}..."
    docker exec -w "${LLAMA_MODELS_DIR}" \
        -e "PATH=${LLAMA_PYTHON_BIN_DIR}:${PATH}" \
        "${LLAMA_CONTAINER_NAME}" bash run_packed_inputs.sh --data_sample "${sample}"
done

echo "Packed prompt inputs are available in ${LLAMA_MODELS_DIR}/llama_inputs/."
