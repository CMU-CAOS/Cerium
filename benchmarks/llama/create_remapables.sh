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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TOTAL_BLOCKS="${TOTAL_BLOCKS:-32}"
MAX_JOBS="${MAX_JOBS:-8}"

if ! [[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE must be a positive integer; got: $BATCH_SIZE" >&2
    exit 1
fi
if ! [[ "$TOTAL_BLOCKS" =~ ^[1-9][0-9]*$ ]]; then
    echo "TOTAL_BLOCKS must be a positive integer; got: $TOTAL_BLOCKS" >&2
    exit 1
fi
if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_JOBS must be a positive integer; got: $MAX_JOBS" >&2
    exit 1
fi

cd "$SCRIPT_DIR"

declare -a pids=()
declare -a ranges=()

wait_for_batch() {
    local status=0
    local index
    for index in "${!pids[@]}"; do
        if ! wait "${pids[index]}"; then
            echo "Remappable generation failed for blocks ${ranges[index]}." >&2
            status=1
        fi
    done
    pids=()
    ranges=()
    return "$status"
}

for ((block_start = 0; block_start < TOTAL_BLOCKS; block_start += BATCH_SIZE)); do
    block_count=$((TOTAL_BLOCKS - block_start))
    if (( block_count > BATCH_SIZE )); then
        block_count=$BATCH_SIZE
    fi

    block_end=$((block_start + block_count - 1))
    echo "Generating remappable inputs for blocks ${block_start}-${block_end}."
    "$PYTHON_BIN" generate_remap_weights.py \
        --block_start "$block_start" \
        --num_blocks "$block_count" &
    pids+=("$!")
    ranges+=("${block_start}-${block_end}")

    if (( ${#pids[@]} >= MAX_JOBS )); then
        wait_for_batch
    fi
done

if (( ${#pids[@]} > 0 )); then
    wait_for_batch
fi
