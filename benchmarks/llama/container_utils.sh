#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026 Siddharth Jayashankar. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared helpers for the containerized Llama preparation entry points.
# This file is intended to be sourced, not run directly.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This helper must be sourced by a Llama preparation script." >&2
    exit 1
fi

LLAMA_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_REPO_ROOT="$(cd "${LLAMA_SCRIPT_DIR}/../.." && pwd)"
LLAMA_REPO_ROOT_NAME="$(basename "${LLAMA_REPO_ROOT}")"
LLAMA_IMAGE="${CERIUM_IMAGE:-cerium/build-cu13.3:v1}"
LLAMA_USER_NAME="$(id -un 2>/dev/null || id -u)"
LLAMA_USER_ID="$(id -u):$(id -g)"
LLAMA_CONTAINER_NAME="CeriumBuild_cu13.3_${LLAMA_USER_NAME}_${LLAMA_REPO_ROOT_NAME}"
LLAMA_PYTHON="${LLAMA_REPO_ROOT}/venv_cerium/bin/python3"
LLAMA_PYTHON_BIN_DIR="$(dirname "${LLAMA_PYTHON}")"
LLAMA_MODELS_DIR="${LLAMA_SCRIPT_DIR}/llama_models"

llama_fail() {
    echo "ERROR: $*" >&2
    exit 1
}

llama_ensure_container_running() {
    command -v docker >/dev/null 2>&1 || \
        llama_fail "Docker is not installed or is not on PATH."
    docker info >/dev/null 2>&1 || \
        llama_fail "Docker is unavailable to the current user. Check the daemon and Docker permissions."

    if docker inspect -f '{{.State.Running}}' "${LLAMA_CONTAINER_NAME}" >/dev/null 2>&1; then
        if [[ "$(docker inspect -f '{{.State.Running}}' "${LLAMA_CONTAINER_NAME}")" == "true" ]]; then
            return
        fi
        docker rm "${LLAMA_CONTAINER_NAME}" >/dev/null 2>&1 || true
    fi

    echo "Launching ${LLAMA_CONTAINER_NAME}..."
    docker run --user "${LLAMA_USER_ID}" \
        -v "${LLAMA_REPO_ROOT}:${LLAMA_REPO_ROOT}" \
        --name "${LLAMA_CONTAINER_NAME}" --rm -d "${LLAMA_IMAGE}" \
        tail -f /dev/null >/dev/null
}

llama_require_python() {
    if ! docker exec "${LLAMA_CONTAINER_NAME}" test -x "${LLAMA_PYTHON}"; then
        llama_fail "Cerium virtual environment not found at ${LLAMA_PYTHON}. Run ./activate_cerium_venv.sh from the repository root first."
    fi
}

llama_require_checkpoint() {
    local checkpoint_dir="${LLAMA_MODELS_DIR}/model_weights/Llama3.1-8B"
    local file
    for file in consolidated.00.pth params.json tokenizer.model; do
        [[ -s "${checkpoint_dir}/${file}" ]] || \
            llama_fail "Missing checkpoint file ${checkpoint_dir}/${file}. Run download_llama.sh first."
    done
}
