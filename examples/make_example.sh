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

# Build generated CUDA kernels and stage runtime files for one example/GPU count.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
example="${1:?Usage: make_example.sh EXAMPLE GPUS}"
gpus="${2:?Usage: make_example.sh EXAMPLE GPUS}"
source_dir="${SCRIPT_DIR}/${example}/outputs/${example}"
compiled_dir="${source_dir}/compiled"

# Compile every generated function for this GPU count. Function calls may
# generate a helper in addition to main_<N>gpus.
shopt -s nullglob
function_dirs=("${source_dir}"/*_"${gpus}"gpus)
if (( ${#function_dirs[@]} == 0 )); then
    echo "No generated functions found for ${example} on ${gpus} GPU(s)." >&2
    exit 1
fi

for function_dir in "${function_dirs[@]}"; do
    function="$(basename "${function_dir}")"
    make -C "${SCRIPT_DIR}" -j all "FNAME=${function}" "DIRNAME=${example}/outputs/${example}"
    mkdir -p "${compiled_dir}/${function}"
    cp -a "${source_dir}/${function}/output/." "${compiled_dir}/${function}/"
done
