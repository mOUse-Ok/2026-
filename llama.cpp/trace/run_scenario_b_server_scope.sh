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
KV_UNIFIED="${KV_UNIFIED:-0}"
KV_CACHE_RAM_MB="${KV_CACHE_RAM_MB:-0}"
KV_CACHE_IDLE_SLOTS="${KV_CACHE_IDLE_SLOTS:-0}"

if [ "$KV_UNIFIED" != "0" ] && [ "$KV_UNIFIED" != "1" ]; then
    echo "ERROR: KV_UNIFIED must be 0 or 1" >&2
    exit 1
fi
if ! [[ "$KV_CACHE_RAM_MB" =~ ^-?[0-9]+$ ]]; then
    echo "ERROR: KV_CACHE_RAM_MB must be an integer MiB value" >&2
    exit 1
fi
if [ "$KV_CACHE_IDLE_SLOTS" != "0" ] && [ "$KV_CACHE_IDLE_SLOTS" != "1" ]; then
    echo "ERROR: KV_CACHE_IDLE_SLOTS must be 0 or 1" >&2
    exit 1
fi
if [ "$KV_CACHE_IDLE_SLOTS" = "1" ] && { [ "$KV_UNIFIED" != "1" ] || [ "$KV_CACHE_RAM_MB" = "0" ]; }; then
    echo "ERROR: KV_CACHE_IDLE_SLOTS=1 requires KV_UNIFIED=1 and nonzero KV_CACHE_RAM_MB" >&2
    exit 1
fi

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

server_args=(
    -m "$MODEL_FILE"
    --host 127.0.0.1
    --port "$PORT"
    -t "$NUM_THREADS"
    -c "$CTX_SIZE"
    -b "$BATCH_SIZE"
    -ub "$UBATCH_SIZE"
    --cache-type-k f16
    --cache-type-v f16
    --flash-attn auto
    --parallel 1
    --metrics
    --no-warmup
)
if [ "$KV_UNIFIED" = "1" ]; then
    server_args+=(--kv-unified)
fi
if [ "$KV_CACHE_RAM_MB" != "0" ]; then
    server_args+=(--cache-ram "$KV_CACHE_RAM_MB")
fi
if [ "$KV_CACHE_IDLE_SLOTS" = "1" ]; then
    server_args+=(--cache-idle-slots)
else
    # llama-server defaults this option to on when unified KV and a prompt
    # cache are present. Pass the negative form so this helper's default is
    # stable and does not alter the existing Scenario B comparison.
    server_args+=(--no-cache-idle-slots)
fi

exec "$LLAMA_SERVER" "${server_args[@]}" \
    > "$RUN_DIR/server_stdout.log" 2>"$RUN_DIR/server_stderr.log"
