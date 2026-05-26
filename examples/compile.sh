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

# Compile one tutorial for the selected GPU counts, then build generated CUDA.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="${CERIUM_IMAGE:-cerium/build-cu13.3:v1}"
PYTHON="${REPO_ROOT}/venv_cerium/bin/python3"
example="${1:?Usage: compile.sh EXAMPLE [1|2|4|8|all]}"
selection="${2:-all}"

if [[ ! "${example}" =~ ^[a-z][a-z0-9_]*$ || ! -f "${SCRIPT_DIR}/${example}/program.py" ]]; then
    echo "Unknown example: ${example}" >&2
    exit 2
fi
case "${selection}" in
    all) gpu_counts=(1 2 4 8) ;;
    1|2|4|8) gpu_counts=("${selection}") ;;
    *) echo "GPU count must be 1, 2, 4, 8, or all" >&2; exit 2 ;;
esac

for gpus in "${gpu_counts[@]}"; do
    echo "Compiling ${example} for ${gpus} GPU(s)..."
    docker run --rm --user "$(id -u):$(id -g)" \
        -v "${REPO_ROOT}:${REPO_ROOT}" -w "${SCRIPT_DIR}" \
        -e "PYTHONPATH=${REPO_ROOT}:${REPO_ROOT}/build/cerium/python" \
        -e "LD_LIBRARY_PATH=${REPO_ROOT}/build/cerium/runtime" \
        "${IMAGE_NAME}" "${PYTHON}" -m "examples.${example}.program" --gpus "${gpus}"
    docker run --rm --user "$(id -u):$(id -g)" \
        -v "${REPO_ROOT}:${REPO_ROOT}" -w "${SCRIPT_DIR}" \
        "${IMAGE_NAME}" bash "${SCRIPT_DIR}/make_example.sh" "${example}" "${gpus}"
done
