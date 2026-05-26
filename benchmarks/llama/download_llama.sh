#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Download the gated native Llama 3.1-8B checkpoint inside the Cerium container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=container_utils.sh
source "${SCRIPT_DIR}/container_utils.sh"

[[ -n "${HF_TOKEN:-}" ]] || \
    llama_fail "HF_TOKEN is required. Accept the Meta Llama license on Hugging Face, then export HF_TOKEN before running this script."

llama_ensure_container_running
llama_require_python

echo "Downloading and verifying Llama 3.1-8B checkpoint..."
docker exec -w "${LLAMA_MODELS_DIR}" \
    -e HF_TOKEN \
    "${LLAMA_CONTAINER_NAME}" "${LLAMA_PYTHON}" download_llama.py
