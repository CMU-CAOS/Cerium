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
MODEL_PATH="${SCRIPT_DIR}/bert_torch_rte.pt"
MODEL_URL="https://drive.usercontent.google.com/download?id=1dOFhA1YR_hiqVEFP_Q4DVo3Dx6oESAKZ&export=download&authuser=0&confirm=t"
MODEL_SHA256="eeefe342a8476810c8a13014abfcaa115909bfd2cc7082015b16fc19400539c9"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/bert_compile_$(date +%Y%m%d_%H%M%S).log"
exec 3>&1
printf 'BERT compile started. Log: %s\n' "${RUN_LOG}" >&3
exec >"${RUN_LOG}" 2>&1

print_summary() {
    local status=$?
    if (( status == 0 )); then
        printf 'BERT compile completed. Log: %s\n' "${RUN_LOG}" >&3
    else
        printf 'BERT compile failed (exit %d). Log: %s\n' "${status}" "${RUN_LOG}" >&3
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

if [[ ! -x "${PYTHON}" ]]; then
    echo "Python virtual environment not found at ${PYTHON}. Run the project setup first." >&2
    exit 1
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "Downloading BERT model checkpoint to ${MODEL_PATH}..."
    curl --fail --location --show-error --output "${MODEL_PATH}" "${MODEL_URL}"
    if ! printf "%s  %s\n" "${MODEL_SHA256}" "${MODEL_PATH}" | sha256sum --check --status -; then
        rm -f "${MODEL_PATH}"
        echo "Downloaded BERT model checkpoint failed SHA-256 verification." >&2
        exit 1
    fi
fi

ensure_container_running
if [[ ! -f "${SCRIPT_DIR}/packed_params/params_rte_block12.pkl" ]]; then
    docker exec -w "${SCRIPT_DIR}" "${CONTAINER_NAME}" \
        "${PYTHON}" bert_params_torch.py --output-dir packed_params --task-name rte
fi

for gpus in 1 2 4 8; do
    docker exec -w "${SCRIPT_DIR}" "${CONTAINER_NAME}" \
        "${PYTHON}" bert_cerium.py --gpus "${gpus}"
done

docker exec -w "${SCRIPT_DIR}" "${CONTAINER_NAME}" bash make_bert.sh
