#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run the native (unpacked) Llama model through block 31 for one data sample.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=container_utils.sh
source "${SCRIPT_DIR}/container_utils.sh"

usage() {
    cat <<'EOF'
Usage: get_token.sh --data_sample N

Runs llama_models/run_unpacked.sh inside the Cerium container for data sample N.
The run always checks block 31 against fhe_outputs/sampleN_block31.pkl.

Example:
  bash run_token.sh --data_sample 0
EOF
}

DATA_SAMPLE=""
while (( $# > 0 )); do
    case "$1" in
        --data_sample)
            (( $# >= 2 )) || llama_fail "--data_sample requires an integer value."
            DATA_SAMPLE="$2"
            shift 2
            ;;
        --data_sample=*)
            DATA_SAMPLE="${1#*=}"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            llama_fail "Unknown argument: $1. Run with --help for usage."
            ;;
    esac
done

[[ -n "${DATA_SAMPLE}" ]] || llama_fail "Missing required argument: --data_sample. Run with --help for usage."
[[ "${DATA_SAMPLE}" =~ ^[0-9]+$ ]] || \
    llama_fail "--data_sample must be a non-negative integer; got: ${DATA_SAMPLE}"

llama_ensure_container_running
llama_require_python
llama_require_checkpoint

echo "Getting predicted token for data sample ${DATA_SAMPLE} ..."
docker exec -w "${LLAMA_MODELS_DIR}" \
    -e "PATH=${LLAMA_PYTHON_BIN_DIR}:${PATH}" \
    "${LLAMA_CONTAINER_NAME}" \
    bash run_unpacked.sh "${DATA_SAMPLE}" 31
