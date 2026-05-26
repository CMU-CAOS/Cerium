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

REPO_ROOT="$(pwd)"
REPO_ROOT_NAME="$(basename "${REPO_ROOT}")"
CUDA_ARCHITECTURES="${1:-80;86;89;90;100;120}"
BUILD_TYPE="${2:-Release}"
DEBUG_EXEC="${3:-OFF}"
DEBUG_COMPILER="${CERIUM_DEBUG_COMPILER:-${4:-OFF}}"
BUILD_DIR="${5:-build}"
IMAGE_NAME="cerium/build-cu13.3:v1"
USER="$(whoami)"
USER_ID="$(id -u):$(id -g)"
CONTAINER_NAME="CeriumBuild_cu13.3_${USER}_${REPO_ROOT_NAME}"

ensure_container_running() {
    if docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" >/dev/null 2>&1; then
        if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" == "true" ]]; then
            return
        fi

        docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi

    echo "Launching ${CONTAINER_NAME}..."
    docker run --user "${USER_ID}" -v "${REPO_ROOT}:${REPO_ROOT}" --name "${CONTAINER_NAME}" --rm -d "${IMAGE_NAME}" tail -f /dev/null >/dev/null
}

ensure_container_running

echo "BUILD_DIR: ${BUILD_DIR}"
echo "BUILD_TYPE: ${BUILD_TYPE}"
echo "DEBUG_EXEC: ${DEBUG_EXEC}"
echo "DEBUG_COMPILER: ${DEBUG_COMPILER}"
echo "CUDA_ARCHITECTURES: ${CUDA_ARCHITECTURES}"

docker exec "${CONTAINER_NAME}" cmake \
    -S "${REPO_ROOT}" \
    -B "${REPO_ROOT}/${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DPython3_EXECUTABLE=/usr/bin/python3.10 \
    -DPython3_LIBRARY=/usr/lib/python3.10 \
    -DCERIUM_DEBUG_EXEC="${DEBUG_EXEC}" \
    -DCERIUM_DEBUG_COMPILER="${DEBUG_COMPILER}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
&& docker exec "${CONTAINER_NAME}" cmake --build "${REPO_ROOT}/${BUILD_DIR}" --target all --verbose 2>err \
&& docker exec "${CONTAINER_NAME}" cmake \
    -S "${REPO_ROOT}" \
    -B "${REPO_ROOT}/${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DPython3_EXECUTABLE=/usr/bin/python3.12 \
    -DPython3_LIBRARY=/usr/lib/python3.12 \
    -DCERIUM_DEBUG_EXEC="${DEBUG_EXEC}" \
    -DCERIUM_DEBUG_COMPILER="${DEBUG_COMPILER}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
&& docker exec "${CONTAINER_NAME}" cmake --build "${REPO_ROOT}/${BUILD_DIR}" --target all --verbose 2>err \
