#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run one compiled tutorial on the requested or all visible GPU counts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${CERIUM_IMAGE:-cerium/build-cu13.3:v1}"
PYTHON="${REPO_ROOT}/venv_cerium/bin/python3"
example="${1:?Usage: run_example.sh EXAMPLE [1|2|4|8|all]}"
selection="${2:-all}"

if [[ ! "${example}" =~ ^[a-z][a-z0-9_]*$ || ! -f "${SCRIPT_DIR}/${example}/runner.py" ]]; then
    echo "Unknown example: ${example}" >&2
    exit 2
fi
case "${selection}" in
    all) gpu_counts=(1 2 4 8) ;;
    1|2|4|8) gpu_counts=("${selection}") ;;
    *) echo "GPU count must be 1, 2, 4, 8, or all" >&2; exit 2 ;;
esac

visible="$(docker run --rm --gpus all "${IMAGE_NAME}" nvidia-smi -L)"
gpu_count="$(awk '/^GPU [0-9]+:/{ count++ } END { print count + 0 }' <<<"${visible}")"
if (( gpu_count == 0 )); then
    echo "No NVIDIA GPUs are visible in the container." >&2
    exit 1
fi

mkdir -p "${SCRIPT_DIR}/${example}/logs"
for gpus in "${gpu_counts[@]}"; do
    if (( gpus > gpu_count )); then
        if [[ "${selection}" == "all" ]]; then
            echo "Skipping ${gpus}-GPU variant: ${gpu_count} GPU(s) visible."
            continue
        fi
        echo "Requested ${gpus} GPUs, but only ${gpu_count} are visible." >&2
        exit 1
    fi
    log_file="${SCRIPT_DIR}/${example}/logs/${gpus}gpus.log"
    echo "Running ${example} on ${gpus} GPU(s); log: ${log_file}"
    docker run --rm --gpus all --user "$(id -u):$(id -g)" \
        -v "${REPO_ROOT}:${REPO_ROOT}" -w "${SCRIPT_DIR}" \
        -e "PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/build/cerium/python" \
        -e "LD_LIBRARY_PATH=${REPO_ROOT}/build/cerium/runtime" \
        "${IMAGE_NAME}" "${PYTHON}" -m "examples.${example}.runner" --gpus "${gpus}" \
        2>&1 | tee "${log_file}"
done
