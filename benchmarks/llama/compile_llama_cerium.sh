#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Generate the requested Llama Cerium program configurations in the build
# container.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=container_utils.sh
source "${SCRIPT_DIR}/container_utils.sh"

# Number of independent Cerium code-generation processes to run at once.
# Shared global-function generation remains serialized below.
CERIUM_COMPILE_JOBS="${CERIUM_COMPILE_JOBS:-1}"
if ! [[ "${CERIUM_COMPILE_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "CERIUM_COMPILE_JOBS must be a positive integer" >&2
    exit 2
fi

compile_configuration() {
    local gpu_count="$1"
    local block_count="$2"
    local skip_global="$3"
    local skip_gpu_local="$4"
    local function_name="llama_${gpu_count}gpu_${block_count}blocks"
    local output_dir="${LLAMA_SCRIPT_DIR}/outputs/${function_name}"
    local log_dir="${LLAMA_SCRIPT_DIR}/logs"
    local log_file="${log_dir}/${function_name}.log"

    if [[ "${LLAMA_FORCE_REGENERATE:-0}" != "1" && -s "${output_dir}/compile_config" ]]; then
        echo "${gpu_count}-GPU, ${block_count}-block Llama program already generated; skipping. Set LLAMA_FORCE_REGENERATE=1 to regenerate it."
        return
    fi

    echo "Generating ${gpu_count}-GPU, ${block_count}-block Llama program..."
    mkdir -p "${log_dir}"
    if [[ "${CERIUM_IN_CONTAINER:-0}" == "1" ]]; then
        LLAMA_SKIP_GLOBAL_FUNCTIONS="${skip_global}" \
            LLAMA_SKIP_GPU_LOCAL_FUNCTIONS="${skip_gpu_local}" \
            "${LLAMA_PYTHON}" llama_cerium.py --gpus "${gpu_count}" --num_blocks "${block_count}" \
            >"${log_file}" 2>&1
    else
        docker exec -w "${LLAMA_SCRIPT_DIR}" \
            -e "LLAMA_SKIP_GLOBAL_FUNCTIONS=${skip_global}" \
            -e "LLAMA_SKIP_GPU_LOCAL_FUNCTIONS=${skip_gpu_local}" \
            "${LLAMA_CONTAINER_NAME}" "${LLAMA_PYTHON}" llama_cerium.py \
            --gpus "${gpu_count}" --num_blocks "${block_count}" >"${log_file}" 2>&1
    fi || {
            echo "Generation failed; see ${log_file}." >&2
            tail -n 80 "${log_file}" >&2
            return 1
        }
    echo "Generated ${function_name}; log: ${log_file}"
}

# Each job is gpu_count:block_count:skip_global:skip_gpu_local.  All jobs
# passed here write distinct program outputs and skip shared global functions.
run_parallel_configurations() {
    local -a active_pids=()
    local job gpu_count block_count skip_global skip_gpu_local
    local result=0

    for job in "$@"; do
        IFS=: read -r gpu_count block_count skip_global skip_gpu_local <<<"${job}"
        compile_configuration "${gpu_count}" "${block_count}" \
            "${skip_global}" "${skip_gpu_local}" &
        active_pids+=("$!")

        if (( ${#active_pids[@]} >= CERIUM_COMPILE_JOBS )); then
            if ! wait "${active_pids[0]}"; then
                result=1
            fi
            active_pids=("${active_pids[@]:1}")
        fi
    done

    for job in "${active_pids[@]}"; do
        if ! wait "${job}"; then
            result=1
        fi
    done
    return "${result}"
}

if [[ "${CERIUM_IN_CONTAINER:-0}" == "1" ]]; then
    [[ -x "${LLAMA_PYTHON}" ]] || llama_fail "Cerium virtual environment not found at ${LLAMA_PYTHON}."
else
    llama_ensure_container_running
    llama_require_python
fi

# Generate shared global functions and the one-GPU-local functions once.
compile_configuration 1 1 0 0

# Generate GPU-local functions for each multi-GPU topology.
gpu_local_jobs=()
for gpu_count in 2 4 8; do
    gpu_local_jobs+=("${gpu_count}:1:1:0")
done
run_parallel_configurations "${gpu_local_jobs[@]}"

# Generate the main program for every requested block-count/GPU topology.
program_jobs=()
for gpu_count in 1 2 4 8; do
    for block_count in 4 8 12 16 32; do
        program_jobs+=("${gpu_count}:${block_count}:1:1")
    done
done
run_parallel_configurations "${program_jobs[@]}"
