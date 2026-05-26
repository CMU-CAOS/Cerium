#!/bin/bash

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


set -euo pipefail

CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-80,90,100}"
# The Makefile consumes a whitespace-separated architecture list.
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES//,/ }"
export CUDA_ARCHITECTURES

echo "Building Llama CUDA files for architectures: ${CUDA_ARCHITECTURES}"

compile_all() {
    local function_name="$1"
    local cerium_dir="$2"
    local compiled_dir="$3"
    echo "${function_name} ${cerium_dir} ${compiled_dir}"
    mkdir -p "${compiled_dir}/${function_name}"
    make -j CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        FNAME="${function_name}" DIRNAME="${cerium_dir}" all
    cp -r "${cerium_dir}/${function_name}/output/." "${compiled_dir}/${function_name}/"
}

CERIUM_DIR="outputs"
COMPILED_DIR="${CERIUM_DIR}/compiled"

# Build the standalone functions generated for the Llama workload.
for function_name in \
    bootstrap_32K_35lvl_t1_vec \
    bootstrap_32K_33lvl_t1_vec \
    matmul_ct128x128_ct128x128 \
    matmul_ct128x128_ct128x128_transpose \
    softmax_128x128_vec \
    silu; do
    compile_all "${function_name}" "${CERIUM_DIR}" "${COMPILED_DIR}"
done

for gpu_count in 1 2 4 8; do
    for function_name in \
        attention_matmul_qkv \
        attention_matmul_o \
        rmsnorm_att \
        rmsnorm_ffn \
        ffn_matmul_vw \
        ffn_matmul_o; do
        compile_all "${function_name}_${gpu_count}gpu" \
            "${CERIUM_DIR}" "${COMPILED_DIR}"
    done
done

for gpu_count in 2 4 8; do
    for function_name in ffn_silu attention_score_softmax; do
        compile_all "${function_name}_${gpu_count}gpu" \
            "${CERIUM_DIR}" "${COMPILED_DIR}"
    done
done

for gpu_count in 1 2 4 8; do
    compile_all "llama_${gpu_count}gpu_1blocks" "${CERIUM_DIR}" "${COMPILED_DIR}"
done

for gpu_count in 1 2 4 8; do
    for block_count in 4 8 16 32; do
        compile_all "llama_${gpu_count}gpu_${block_count}blocks" \
            "${CERIUM_DIR}" "${COMPILED_DIR}"
    done
done
