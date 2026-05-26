#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Create the Llama remappable plaintext inputs in a fresh Cerium container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=container_utils.sh
source "${SCRIPT_DIR}/container_utils.sh"

BATCH_SIZE="${BATCH_SIZE:-4}"
TOTAL_BLOCKS="${TOTAL_BLOCKS:-32}"
MAX_JOBS="${MAX_JOBS:-8}"

for variable in BATCH_SIZE TOTAL_BLOCKS MAX_JOBS; do
    value="${!variable}"
    [[ "${value}" =~ ^[1-9][0-9]*$ ]] || \
        llama_fail "${variable} must be a positive integer; got: ${value}"
done

llama_require_checkpoint

command -v docker >/dev/null 2>&1 || \
    llama_fail "Docker is not installed or is not on PATH."
docker info >/dev/null 2>&1 || \
    llama_fail "Docker is unavailable to the current user. Check the daemon and Docker permissions."
[[ -x "${LLAMA_PYTHON}" ]] || \
    llama_fail "Cerium virtual environment not found at ${LLAMA_PYTHON}. Run ./activate_cerium_venv.sh from the repository root first."

echo "Generating remappable plaintexts for ${TOTAL_BLOCKS} blocks (batch size: ${BATCH_SIZE}, max jobs: ${MAX_JOBS})..."
docker run --rm \
    --user "${LLAMA_USER_ID}" \
    -v "${LLAMA_REPO_ROOT}:${LLAMA_REPO_ROOT}" \
    -w "${LLAMA_SCRIPT_DIR}" \
    -e "BATCH_SIZE=${BATCH_SIZE}" \
    -e "TOTAL_BLOCKS=${TOTAL_BLOCKS}" \
    -e "MAX_JOBS=${MAX_JOBS}" \
    -e "PYTHON_BIN=${LLAMA_PYTHON}" \
    "${LLAMA_IMAGE}" bash create_remapables.sh

echo "Remappable plaintexts are available in ${LLAMA_SCRIPT_DIR}/llama_plaintexts/."
