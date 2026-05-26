#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Generate packed plaintext parameters for the 32 Llama transformer blocks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=container_utils.sh
source "${SCRIPT_DIR}/container_utils.sh"

MAX_JOBS="${MAX_JOBS:-1}"
[[ "${MAX_JOBS}" =~ ^[1-9][0-9]*$ ]] || \
    llama_fail "MAX_JOBS must be a positive integer; got: ${MAX_JOBS}"

llama_ensure_container_running
llama_require_python
llama_require_checkpoint

echo "Generating packed plaintext weights with MAX_JOBS=${MAX_JOBS}..."
docker exec -w "${LLAMA_MODELS_DIR}" \
    -e "MAX_JOBS=${MAX_JOBS}" \
    -e "PATH=${LLAMA_PYTHON_BIN_DIR}:${PATH}" \
    "${LLAMA_CONTAINER_NAME}" bash create_params.sh
