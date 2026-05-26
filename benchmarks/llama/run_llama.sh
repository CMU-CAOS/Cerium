#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the encrypted Llama runner in an ephemeral GPU-enabled Cerium container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=container_utils.sh
source "${SCRIPT_DIR}/container_utils.sh"

usage() {
    cat <<'EOF'
Usage: run_llama_runner.sh --num_blocks N --block_start N --data_sample N [runner options]

Runs llama_runner.py in a temporary Cerium container with all host GPUs exposed.
The three named arguments are required; other llama_runner.py options, such as
--gpus and --cudagraphs, are forwarded unchanged.
All runner output is written to a timestamped file in logs/. Set LLAMA_LOG_FILE
to use a specific log-file path.

Example:
  bash run_llama_runner.sh --num_blocks 4 --block_start 0 --data_sample 0 --gpus 1
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

for required_arg in --num_blocks --block_start --data_sample; do
    found=false
    for arg in "$@"; do
        if [[ "${arg}" == "${required_arg}" || "${arg}" == "${required_arg}"=* ]]; then
            found=true
            break
        fi
    done
    "${found}" || llama_fail "Missing required argument: ${required_arg}. Run with --help for usage."
done

LOG_DIR="${LLAMA_LOG_DIR:-${SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LLAMA_LOG_FILE:-${LOG_DIR}/llama_runner_$(date -u +%Y%m%dT%H%M%SZ).log}"

command -v docker >/dev/null 2>&1 || \
    llama_fail "Docker is not installed or is not on PATH."
docker info >/dev/null 2>&1 || \
    llama_fail "Docker is unavailable to the current user. Check the daemon and Docker permissions."

[[ -x "${LLAMA_PYTHON}" ]] || \
    llama_fail "Cerium virtual environment not found at ${LLAMA_PYTHON}. Run ./activate_cerium_venv.sh from the repository root first."

exec docker run --rm --gpus all \
    --user "${LLAMA_USER_ID}" \
    -v "${LLAMA_REPO_ROOT}:${LLAMA_REPO_ROOT}" \
    -w "${LLAMA_SCRIPT_DIR}" \
    "${LLAMA_IMAGE}" \
    "${LLAMA_PYTHON}" llama_runner.py "$@" >"${LOG_FILE}" 2>&1
