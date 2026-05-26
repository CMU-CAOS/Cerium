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

#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
IMAGE_NAME="${CERIUM_IMAGE:-cerium/build-cu13.3:v1}"
LOG_DIR="${SCRIPT_DIR}/logs"
PYTHONPATH_DIR="${REPO_ROOT}/build/cerium/python"
RUNTIME_LIB_DIR="${REPO_ROOT}/build/cerium/runtime"
PYTHON="${REPO_ROOT}/venv_cerium/bin/python3"
NUM_SAMPLES="${RESNET_NUM_SAMPLES:-5}"

mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/resnet_benchmarks_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${RUN_LOG}") 2>&1

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

if ! command -v docker >/dev/null 2>&1; then
    fail "Docker is not installed or is not on PATH."
fi

if ! docker info >/dev/null 2>&1; then
    fail "Docker is not available to the current user. Check that the daemon is running and that you have permission to use it."
fi

if [[ ! -x "${PYTHON}" ]]; then
    fail "Python virtual environment not found at ${PYTHON}. Run the project setup before running benchmarks."
fi

if [[ ! "${NUM_SAMPLES}" =~ ^[1-9][0-9]*$ ]] || (( NUM_SAMPLES > 1000 )); then
    fail "RESNET_NUM_SAMPLES must be an integer from 1 to 1000 (received ${NUM_SAMPLES})."
fi

for input_file in testFile/test_values_25.txt testFile/test_label.txt; do
    if [[ ! -f "${SCRIPT_DIR}/${input_file}" ]]; then
        fail "Required ResNet input file is missing: ${SCRIPT_DIR}/${input_file}."
    fi
done

if [[ ! -f "${SCRIPT_DIR}/resnet20_params.pkl" ]]; then
    fail "Generated ResNet parameters are missing: ${SCRIPT_DIR}/resnet20_params.pkl. Run ${SCRIPT_DIR}/compile.sh first."
fi

if [[ ! -d "${SCRIPT_DIR}/outputs/compiled" ]]; then
    fail "Compiled ResNet artifacts are missing. Run ${SCRIPT_DIR}/compile.sh first."
fi

if ! gpu_list="$(docker run --rm --gpus all "${IMAGE_NAME}" nvidia-smi -L)"; then
    fail "Could not query GPUs in ${IMAGE_NAME}. Check the image and NVIDIA Container Toolkit installation."
fi

GPU_COUNT="$(awk '/^GPU [0-9]+:/{ count++ } END { print count + 0 }' <<<"${gpu_list}")"
echo "Detected ${GPU_COUNT} GPU(s). Running ${NUM_SAMPLES} sample(s) per configuration."

if (( GPU_COUNT == 0 )); then
    fail "No GPUs are available in the container."
fi

gpu_runs=()
average_times=()
failed_runs=0

for gpus in 1 2 4 8; do
    if (( gpus > GPU_COUNT )); then
        echo "Skipping ${gpus}-GPU benchmark: only ${GPU_COUNT} GPU(s) available."
        continue
    fi

    log_file="${LOG_DIR}/resnet_${gpus}gpus.log"
    echo "Running ${gpus}-GPU ResNet benchmark."
    if ! docker run --rm --gpus all --ipc=host \
        --user "$(id -u):$(id -g)" \
        -v "${REPO_ROOT}:${REPO_ROOT}" \
        -w "${SCRIPT_DIR}" \
        -e "PYTHONPATH=${PYTHONPATH_DIR}" \
        -e "LD_LIBRARY_PATH=${RUNTIME_LIB_DIR}" \
        "${IMAGE_NAME}" \
        "${PYTHON}" resnet_runner.py --gpus "${gpus}" --num_samples "${NUM_SAMPLES}" >"${log_file}" 2>&1; then
        echo "${gpus}-GPU ResNet benchmark failed; see ${log_file}." >&2
        gpu_runs+=("${gpus}")
        average_times+=("failed")
        ((failed_runs += 1))
        continue
    fi

    average="$(awk -F ' = | milliseconds' '/Average Execution Time =/ { value=$2 } END { print value }' "${log_file}")"
    if [[ -z "${average}" ]]; then
        echo "${gpus}-GPU ResNet benchmark completed without an execution-time result; see ${log_file}." >&2
        average="not found"
    fi

    if [[ "${average}" == "not found" ]]; then
        ((failed_runs += 1))
    fi

    gpu_runs+=("${gpus}")
    average_times+=("${average}")
done

printf '\n+-----------+------+------------------------+\n'
printf '| %-9s | %4s | %-22s |\n' 'Benchmark' 'GPUs' 'Execution time'
printf '+-----------+------+------------------------+\n'
for index in "${!gpu_runs[@]}"; do
    if [[ "${average_times[index]}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        last_time="${average_times[index]} ms"
    else
        last_time="${average_times[index]}"
    fi
    printf '| %-9s | %4s | %22s |\n' \
        'resnet20' "${gpu_runs[index]}" "${last_time}"
done
printf '+-----------+------+------------------------+\n'

if (( failed_runs > 0 )); then
    fail "${failed_runs} benchmark run(s) did not complete successfully."
fi
