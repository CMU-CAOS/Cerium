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

compile_all() {
    local function_name="$1"
    local cerium_dir="$2"
    local compiled_dir="$3"
    echo "${function_name} ${cerium_dir} ${compiled_dir}"
    mkdir -p "${compiled_dir}/${function_name}"
    make all -j FNAME="${function_name}" DIRNAME="${cerium_dir}"
    cp -r "${cerium_dir}/${function_name}/output/." "${compiled_dir}/${function_name}/"
}

CERIUM_DIR="bert_outputs"
COMPILED_DIR="${CERIUM_DIR}/compiled"

for function_name in \
    bootstrap_32K_33lvl_t1 \
    bootstrap_32K_35lvl_t1_vec_1 \
    matmul_ct128x128_ct128x128 \
    matmul_ct128x128_ct128x128_transpose \
    gelu_vec \
    pool_classify_layer \
    softmax_128x128_vec \
    layernorm_att_1gpu \
    layernorm_ffn_1gpu \
    bootstrap_32K_35lvl_t1_vec_2  \
    softmax_128x128_vec2 \
    ; do
    compile_all "${function_name}" "${CERIUM_DIR}" "${COMPILED_DIR}"
done


compile_all  bootstrap_32K_35lvl_t1_vec_2 "${CERIUM_DIR}" "${COMPILED_DIR}"
compile_all  softmax_128x128_vec2 "${CERIUM_DIR}" "${COMPILED_DIR}"
for gpus in 1 2 4 8; do
    for function_name in attention_qkv attention_o ffn_up ffn_down; do
        compile_all "${function_name}_${gpus}gpu" "${CERIUM_DIR}" "${COMPILED_DIR}"
    done
done

compile_all "main_1gpu_12" "${CERIUM_DIR}" "${COMPILED_DIR}"
compile_all "main_2gpu_12" "${CERIUM_DIR}" "${COMPILED_DIR}"
compile_all "main_4gpu_12" "${CERIUM_DIR}" "${COMPILED_DIR}"
compile_all "main_8gpu_12" "${CERIUM_DIR}" "${COMPILED_DIR}"