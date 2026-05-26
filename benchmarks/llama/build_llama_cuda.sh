#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the generated Llama CUDA sources in the Cerium CUDA container.
#
# CUDA_ARCHITECTURES is a comma-separated list of NVIDIA compute capabilities.
# For example:
#   CUDA_ARCHITECTURES=80,90 bash build_llama_cuda.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=container_utils.sh
source "${SCRIPT_DIR}/container_utils.sh"

CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-80,90,100}"
# The Llama Makefile consumes a whitespace-separated architecture list.
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES//,/ }"

llama_ensure_container_running

echo "Building Llama CUDA files for architectures: ${CUDA_ARCHITECTURES}"
docker exec -w "${LLAMA_SCRIPT_DIR}" \
    -e "CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}" \
    "${LLAMA_CONTAINER_NAME}" \
    bash make_llama.sh
