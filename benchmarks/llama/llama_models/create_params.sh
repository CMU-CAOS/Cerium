#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARAMS_DIR="${LLAMA_PARAMS_DIR:-$SCRIPT_DIR/params}"
MAX_JOBS="${MAX_JOBS:-8}"

if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_JOBS must be a positive integer; got: $MAX_JOBS" >&2
    exit 1
fi

mkdir -p "$PARAMS_DIR"
cd "$SCRIPT_DIR"

declare -a pids=()
declare -a blocks=()

wait_for_one() {
    local pid="${pids[0]}"
    local block="${blocks[0]}"
    if ! wait "$pid"; then
        echo "Block ${block}: generation failed." >&2
        exit 1
    fi
    pids=("${pids[@]:1}")
    blocks=("${blocks[@]:1}")
}

for block in {0..31}; do
    params_file="$PARAMS_DIR/params_block${block}.pkl"
    if [[ -s "$params_file" ]]; then
        echo "Block ${block}: already exists; skipping."
        continue
    fi

    while (( ${#pids[@]} >= MAX_JOBS )); do
        wait_for_one
    done

    echo "Block ${block}: generating parameters."
    LLAMA_PARAMS_DIR="$PARAMS_DIR" MASTER_PORT="$((10809 + block))" \
        bash "$SCRIPT_DIR/run_params.sh" "$block" &
    pids+=("$!")
    blocks+=("$block")
done

while (( ${#pids[@]} > 0 )); do
    wait_for_one
done
