#!/usr/bin/env bash
# Internal helper: prepare a cold model cache and start one persistent server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
MODEL_FILE="${MODEL_FILE:?MODEL_FILE is required}"
LLAMA_SERVER="${LLAMA_SERVER:?LLAMA_SERVER is required}"
MEMORY_MAX="${MEMORY_MAX:?MEMORY_MAX is required}"
PORT="${PORT:?PORT is required}"
NUM_THREADS="${NUM_THREADS:-8}"
CTX_SIZE="${CTX_SIZE:-2048}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"

mkdir -p "$RUN_DIR"
python3 "$SCRIPT_DIR/capture_cgroup_snapshot.py" \
    --output "$RUN_DIR/cgroup_before_prepare.json" \
    --stage before_prepare \
    --expected-memory-max "$MEMORY_MAX"
python3 "$SCRIPT_DIR/prepare_model_cache.py" --model "$MODEL_FILE" --mode cold \
    > "$RUN_DIR/cache_preparation.json"
python3 "$SCRIPT_DIR/capture_cgroup_snapshot.py" \
    --output "$RUN_DIR/cgroup_before_server.json" \
    --stage before_server \
    --expected-memory-max "$MEMORY_MAX"

exec "$LLAMA_SERVER" \
    -m "$MODEL_FILE" \
    --host 127.0.0.1 \
    --port "$PORT" \
    -t "$NUM_THREADS" \
    -c "$CTX_SIZE" \
    -b "$BATCH_SIZE" \
    -ub "$UBATCH_SIZE" \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --flash-attn auto \
    --parallel 1 \
    --metrics \
    --no-warmup \
    > "$RUN_DIR/server_stdout.log" 2>"$RUN_DIR/server_stderr.log"
