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
REPO_ROOT_NAME="$(basename "${REPO_ROOT}")"
IMAGE_NAME="${CERIUM_IMAGE:-cerium/build-cu13.3:v1}"
USER_NAME="$(whoami)"
USER_ID="$(id -u):$(id -g)"
CONTAINER_NAME="CeriumBuild_cu13.3_${USER_NAME}_${REPO_ROOT_NAME}"
PYTHON="${REPO_ROOT}/venv_cerium/bin/python3"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/resnet_compile_$(date +%Y%m%d_%H%M%S).log"
exec 3>&1
printf 'ResNet compile started. Log: %s\n' "${RUN_LOG}" >&3
exec >"${RUN_LOG}" 2>&1

print_summary() {
    local status=$?
    if (( status == 0 )); then
        printf 'ResNet compile completed. Log: %s\n' "${RUN_LOG}" >&3
    else
        printf 'ResNet compile failed (exit %d). Log: %s\n' "${status}" "${RUN_LOG}" >&3
    fi
}
trap print_summary EXIT

ensure_container_running() {
    if docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" >/dev/null 2>&1; then
        if [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" == "true" ]]; then
            return
        fi

        docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi

    echo "Launching ${CONTAINER_NAME}..."
    docker run --user "${USER_ID}" -v "${REPO_ROOT}:${REPO_ROOT}" \
        --name "${CONTAINER_NAME}" --rm -d "${IMAGE_NAME}" tail -f /dev/null >/dev/null
}

ensure_container_running

if [[ ! -f "${SCRIPT_DIR}/resnet20_params.pkl" ]]; then
    docker exec -w "${SCRIPT_DIR}" "${CONTAINER_NAME}" "${PYTHON}" params.py
fi

for gpus in 1 2 4 8; do
    docker exec -w "${SCRIPT_DIR}" "${CONTAINER_NAME}" \
        "${PYTHON}" resnet.py --gpus "${gpus}"
done

docker exec -w "${SCRIPT_DIR}" "${CONTAINER_NAME}" bash make_resnet.sh
