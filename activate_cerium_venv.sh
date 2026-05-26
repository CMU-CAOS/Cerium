#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
REPO_ROOT_NAME="$(basename "${REPO_ROOT}")"
IMAGE_NAME="${CERIUM_IMAGE:-cerium/build-cu13.3:v1}"

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

if [[ "${CERIUM_IN_CONTAINER:-}" != "1" ]]; then
    USER_NAME="$(id -un 2>/dev/null || id -u)"
    USER_ID="$(id -u):$(id -g)"
    CONTAINER_NAME="CeriumBuild_cu13.3_${USER_NAME}_${REPO_ROOT_NAME}"
    ensure_container_running
    docker exec -w "${REPO_ROOT}" -e CERIUM_IN_CONTAINER=1 \
        "${CONTAINER_NAME}" bash ./activate_cerium_venv.sh
    if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
        return
    fi
    exit
fi

cd "${REPO_ROOT}"

if [[ ! -d venv_cerium ]]; then
    python3 -m venv venv_cerium
    source venv_cerium/bin/activate
    python3 -m pip install -r requirements.txt
else
    source venv_cerium/bin/activate
fi
