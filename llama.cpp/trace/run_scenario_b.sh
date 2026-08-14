#!/usr/bin/env bash
#
# Scenario B: compare controller-off and performance-preload in persistent
# llama-server processes.  Every group gets a fresh cgroup scope and cold file
# cache; every request in a group shares the same server process.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRACE_BASE_DIR="${TRACE_BASE_DIR:-$PROJECT_DIR/trace_output/scenario_b}"
RUN_PREFIX="${RUN_PREFIX:-scenario_b}"
MODEL_FILE="${MODEL_FILE:-$PROJECT_DIR/../models/Qwen3.5-35B-A3B-Q3_K_M.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-$PROJECT_DIR/build/bin/llama-server}"
MEMORY_MAX="${MEMORY_MAX:-20G}"
MEMORY_SWAP_MAX="${MEMORY_SWAP_MAX:-0}"
PORT_BASE="${PORT_BASE:-18080}"
NUM_THREADS="${NUM_THREADS:-8}"
CTX_SIZE="${CTX_SIZE:-2048}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
REQUEST_COUNT="${REQUEST_COUNT:-4}"
N_PREDICT="${N_PREDICT:-16}"
HTTP_TIMEOUT_S="${HTTP_TIMEOUT_S:-600}"
ALLOW_DIRTY_REPO="${ALLOW_DIRTY_REPO:-0}"
RUN_SCENARIO_B_ALLOW_HOST_OVERCOMMIT="${RUN_SCENARIO_B_ALLOW_HOST_OVERCOMMIT:-0}"
KV_RESERVE_MB="${KV_RESERVE_MB:-256}"
BUFFER_RESERVE_MB="${BUFFER_RESERVE_MB:-512}"

require_file() {
    if [ ! -f "$1" ]; then
        echo "ERROR: required file is missing: $1" >&2
        exit 1
    fi
}
require_file "$MODEL_FILE"
require_file "$LLAMA_SERVER"
if [ ! -x "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server is not executable: $LLAMA_SERVER" >&2
    exit 1
fi
if [ "$ALLOW_DIRTY_REPO" != "1" ] && [ -n "$(git -C "$PROJECT_DIR" status --porcelain)" ]; then
    echo "ERROR: working tree is dirty; commit the tested revision or set ALLOW_DIRTY_REPO=1" >&2
    exit 1
fi
if ! command -v systemd-run >/dev/null 2>&1; then
    echo "ERROR: systemd-run is required for isolated MemoryMax evidence" >&2
    exit 1
fi
if ! [[ "$REQUEST_COUNT" =~ ^[0-9]+$ ]] || [ "$REQUEST_COUNT" -lt 2 ]; then
    echo "ERROR: REQUEST_COUNT must be at least 2" >&2
    exit 1
fi

TRACE_BASE_DIR="$(realpath -m "$TRACE_BASE_DIR")"
for group in baseline performance report; do
    if [ -e "$TRACE_BASE_DIR/${RUN_PREFIX}_${group}" ]; then
        echo "ERROR: refusing to overwrite existing result: $TRACE_BASE_DIR/${RUN_PREFIX}_${group}" >&2
        exit 1
    fi
done
mkdir -p "$TRACE_BASE_DIR"

# POPULATE_READ can fault the full Expert mapping.  A cgroup limit alone is
# insufficient when the host itself has less physical memory than that mapping.
model_bytes="$(stat -c %s "$MODEL_FILE")"
host_available_bytes="$(( $(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo) * 1024 ))"
reserve_bytes=$(( (KV_RESERVE_MB + BUFFER_RESERVE_MB) * 1024 * 1024 ))
if [ "$RUN_SCENARIO_B_ALLOW_HOST_OVERCOMMIT" != "1" ] && \
        [ "$host_available_bytes" -lt $((model_bytes + reserve_bytes)) ]; then
    skip_dir="$TRACE_BASE_DIR/${RUN_PREFIX}_report"
    mkdir -p "$skip_dir"
    printf '%s\n' \
        "场景 B 未执行：主机 MemAvailable 小于完整 Expert 预加载所需的模型大小加 KV/compute 预留。" \
        "model_bytes=$model_bytes" \
        "host_available_bytes=$host_available_bytes" \
        "reserve_bytes=$reserve_bytes" \
        "请在物理可用内存至少覆盖 model + reserve 的机器上运行；不要使用 overcommit 伪造结果。" \
        > "$skip_dir/SKIPPED.md"
    echo "SKIPPED: insufficient physical host memory; see $skip_dir/SKIPPED.md"
    exit 0
fi

server_scope_pid=""
server_pid=""
cleanup_server() {
    if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid" 2>/dev/null || true
    fi
    if [ -n "$server_scope_pid" ]; then
        wait "$server_scope_pid" 2>/dev/null || true
    fi
    server_scope_pid=""
    server_pid=""
}
trap cleanup_server EXIT INT TERM

find_server_pid() {
    local port="$1"
    local deadline=$((SECONDS + HTTP_TIMEOUT_S))
    while [ "$SECONDS" -lt "$deadline" ]; do
        local candidates
        candidates="$(pgrep -f "llama-server.*--port ${port}" || true)"
        local count
        count="$(printf '%s\n' "$candidates" | sed '/^$/d' | wc -l)"
        if [ "$count" -eq 1 ]; then
            printf '%s\n' "$candidates"
            return 0
        fi
        sleep 0.2
    done
    return 1
}

write_startup_json() {
    local run_dir="$1"
    local start_ns="$2"
    local end_ns="$3"
    local pid="$4"
    python3 -c 'import json, sys; from pathlib import Path; out=Path(sys.argv[1]); start=int(sys.argv[2]); end=int(sys.argv[3]); out.write_text(json.dumps({"startup_total_s": (end-start)/1e9, "server_pid": int(sys.argv[4])}, indent=2)+"\n", encoding="utf-8")' \
        "$run_dir/startup.json" "$start_ns" "$end_ns" "$pid"
}

run_group() {
    local group="$1"
    local port="$2"
    local preload="$3"
    local run_dir="$TRACE_BASE_DIR/${RUN_PREFIX}_${group}"
    local start_ns end_ns
    local endpoint="http://127.0.0.1:${port}/completion"
    mkdir -p "$run_dir"
    start_ns="$(date +%s%N)"

    local env_values=(
        "RUN_DIR=$run_dir"
        "MODEL_FILE=$MODEL_FILE"
        "LLAMA_SERVER=$LLAMA_SERVER"
        "MEMORY_MAX=$MEMORY_MAX"
        "PORT=$port"
        "NUM_THREADS=$NUM_THREADS"
        "CTX_SIZE=$CTX_SIZE"
        "BATCH_SIZE=$BATCH_SIZE"
        "UBATCH_SIZE=$UBATCH_SIZE"
        "LLM_MEM_TRACE=1"
        "LLM_MEM_TRACE_DIR=$run_dir"
        "LLM_MEM_TRACE_RUN_ID=${RUN_PREFIX}_${group}"
        "TRACE_PROFILE=benchmark"
        "LLM_MEM_TRACE_TENSOR=0"
        "LLM_MEM_TRACE_KV=0"
        "LLM_MEM_TRACE_EXPERT=1"
        "LLM_MEM_TRACE_MEMORY=1"
        "LLM_MEM_TRACE_RESIDENCY=0"
        "LLM_MEM_TRACE_RESIDENCY_ATTRIBUTION=0"
        "LLM_MEM_TRACE_SMAPS=0"
        "LLM_MEM_TRACE_EXPERT_TASK_MODE=summary"
        "LLM_MEM_TRACE_CONTROL_ONLY=0"
        "LLM_MEM_TRACE_OPT_EXPERT_PROFILE=performance"
        "LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=off"
        "LLM_MEM_TRACE_OS_HINTS=0"
        "LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=0"
        "LLM_MEM_TRACE_OPT_EXPERT_MADV_COLD_RECLAIM=0"
        "LLM_MEM_TRACE_OPT_EXPERT_MADV_DONTNEED_RECLAIM=0"
        "LLM_MEM_TRACE_OPT_EXPERT_PRELOAD=$preload"
        "LLM_MEM_TRACE_OPT_EXPERT_PRELOAD_KV_MB=$KV_RESERVE_MB"
        "LLM_MEM_TRACE_OPT_EXPERT_PRELOAD_BUFFER_MB=$BUFFER_RESERVE_MB"
        "LLM_MEM_TRACE_OPT_EXPERT_PRELOAD_SAFETY_PCT=80"
    )
    echo "[RUN] $group: MemoryMax=$MEMORY_MAX, port=$port, preload=$preload"
    systemd-run --user --scope --quiet \
        -p "MemoryMax=$MEMORY_MAX" \
        -p "MemorySwapMax=$MEMORY_SWAP_MAX" \
        -p "OOMPolicy=continue" \
        -- env "${env_values[@]}" bash "$SCRIPT_DIR/run_scenario_b_server_scope.sh" &
    server_scope_pid=$!
    if ! python3 "$SCRIPT_DIR/wait_http_ready.py" \
            --url "http://127.0.0.1:${port}/health" \
            --timeout-s "$HTTP_TIMEOUT_S" \
            --output "$run_dir/health_ready.json"; then
        echo "ERROR: $group server failed to become healthy" >&2
        return 1
    fi
    server_pid="$(find_server_pid "$port")" || {
        echo "ERROR: cannot identify persistent llama-server PID for port $port" >&2
        return 1
    }
    end_ns="$(date +%s%N)"
    write_startup_json "$run_dir" "$start_ns" "$end_ns" "$server_pid"
    python3 "$SCRIPT_DIR/scenario_b_client.py" \
        --endpoint "$endpoint" \
        --request-count "$REQUEST_COUNT" \
        --n-predict "$N_PREDICT" \
        --timeout-s "$HTTP_TIMEOUT_S" \
        --output "$run_dir/request_metrics.json" \
        --metrics-output "$run_dir/server_metrics.prom"
    python3 "$SCRIPT_DIR/capture_cgroup_snapshot.py" \
        --pid "$server_pid" \
        --output "$run_dir/cgroup_after_requests.json" \
        --stage after_requests \
        --expected-memory-max "$MEMORY_MAX"
    cleanup_server
}

run_group baseline "$PORT_BASE" 0
run_group performance "$((PORT_BASE + 1))" 1

REPORT_DIR="$TRACE_BASE_DIR/${RUN_PREFIX}_report"
python3 "$SCRIPT_DIR/summarize_scenario_b.py" \
    --base-dir "$TRACE_BASE_DIR" \
    --run-prefix "$RUN_PREFIX" \
    --output-dir "$REPORT_DIR"
echo "Report: $REPORT_DIR/SCENARIO_B_REPORT.md"
