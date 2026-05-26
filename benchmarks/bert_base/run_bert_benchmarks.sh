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

mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/bert_benchmarks_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${RUN_LOG}") 2>&1

fail() { echo "ERROR: $*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || fail "Docker is not installed or is not on PATH."
docker info >/dev/null 2>&1 || fail "Docker is not available to the current user."
[[ -x "${PYTHON}" ]] || fail "Python virtual environment not found at ${PYTHON}. Run the project setup first."
for required_path in inputs/labels.npy packed_params/params_rte_block12.pkl bert_outputs/compiled; do
    [[ -e "${SCRIPT_DIR}/${required_path}" ]] || fail "Required BERT artifact is missing: ${SCRIPT_DIR}/${required_path}. Run ${SCRIPT_DIR}/compile.sh and generate benchmark inputs first."
done

if ! gpu_list="$(docker run --rm --gpus all "${IMAGE_NAME}" nvidia-smi -L)"; then
    fail "Could not query GPUs in ${IMAGE_NAME}. Check NVIDIA Container Toolkit installation."
fi
GPU_COUNT="$(awk '/^GPU [0-9]+:/{ count++ } END { print count + 0 }' <<<"${gpu_list}")"
(( GPU_COUNT > 0 )) || fail "No GPUs are available in the container."
echo "Detected ${GPU_COUNT} GPU(s)."

gpu_runs=()
average_times=()
failed_runs=0
for gpus in 1 2 4 8; do
    if (( gpus > GPU_COUNT )); then
        echo "Skipping ${gpus}-GPU benchmark: only ${GPU_COUNT} GPU(s) available."
        continue
    fi
    log_file="${LOG_DIR}/bert_${gpus}gpus.log"
    echo "Running ${gpus}-GPU BERT benchmark."
    if ! docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" \
        -v "${REPO_ROOT}:${REPO_ROOT}" -w "${SCRIPT_DIR}" \
        -e "PYTHONPATH=${PYTHONPATH_DIR}" -e "LD_LIBRARY_PATH=${RUNTIME_LIB_DIR}" \
        "${IMAGE_NAME}" "${PYTHON}" bert_runner.py --gpus "${gpus}" >"${log_file}" 2>&1; then
        echo "${gpus}-GPU BERT benchmark failed; see ${log_file}." >&2
        gpu_runs+=("${gpus}"); average_times+=("failed"); ((failed_runs += 1)); continue
    fi
    average="$(awk '/Bert Execution time:/ { sum += $4; count++ } END { if (count) printf "%.6f", sum / count }' "${log_file}")"
    if [[ -z "${average}" ]]; then
        echo "${gpus}-GPU BERT benchmark completed without execution-time results; see ${log_file}." >&2
        average="not found"; ((failed_runs += 1))
    fi
    gpu_runs+=("${gpus}"); average_times+=("${average}")
done

printf '\n+-----------+------+------------------------+\n'
printf '| %-9s | %4s | %-22s |\n' 'Benchmark' 'GPUs' 'Average execution time'
printf '+-----------+------+------------------------+\n'
for index in "${!gpu_runs[@]}"; do
    [[ "${average_times[index]}" =~ ^[0-9]+([.][0-9]+)?$ ]] && execution_time="${average_times[index]} s" || execution_time="${average_times[index]}"
    printf '| %-9s | %4s | %22s |\n' 'BERT-Base' "${gpu_runs[index]}" "${execution_time}"
done
printf '+-----------+------+------------------------+\n'
(( failed_runs == 0 )) || fail "${failed_runs} benchmark run(s) did not complete successfully."
