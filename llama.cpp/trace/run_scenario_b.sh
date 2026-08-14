#!/usr/bin/env bash
#
# Scenario B: a bounded-memory, persistent-server comparison.  The model can
# run at 12 GiB but cannot retain the whole 15 GiB model working set.  Compare
# controller-off with one Router-selected Expert hint per layer; do not perform
# a late bulk preload after model load.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRACE_BASE_DIR="${TRACE_BASE_DIR:-$PROJECT_DIR/trace_output/scenario_b}"
RUN_PREFIX="${RUN_PREFIX:-scenario_b}"
MODEL_FILE="${MODEL_FILE:-$PROJECT_DIR/../models/Qwen3.5-35B-A3B-Q3_K_M.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-$PROJECT_DIR/build/bin/llama-server}"
MEMORY_MAX="${MEMORY_MAX:-12G}"
MEMORY_SWAP_MAX="${MEMORY_SWAP_MAX:-0}"
PORT_BASE="${PORT_BASE:-18080}"
NUM_THREADS="${NUM_THREADS:-8}"
CTX_SIZE="${CTX_SIZE:-2048}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
REQUEST_COUNT="${REQUEST_COUNT:-3}"
N_PREDICT="${N_PREDICT:-32}"
HTTP_TIMEOUT_S="${HTTP_TIMEOUT_S:-600}"
ALLOW_DIRTY_REPO="${ALLOW_DIRTY_REPO:-0}"
PREFETCH_BUDGET_MB="${PREFETCH_BUDGET_MB:-64}"
PREFETCH_TOPK="${PREFETCH_TOPK:-1}"

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

memory_limit_bytes() {
    local value="$1"
    case "$value" in
        *G|*g) echo $(( ${value%?} * 1024 * 1024 * 1024 )) ;;
        *M|*m) echo $(( ${value%?} * 1024 * 1024 )) ;;
        *K|*k) echo $(( ${value%?} * 1024 )) ;;
        *) echo "$value" ;;
    esac
}

# This scenario deliberately runs below the model's full working-set size.
# The host still needs enough immediately available memory to host the bounded
# cgroup plus a small operating-system reserve.
memory_limit_bytes="$(memory_limit_bytes "$MEMORY_MAX")"
host_available_bytes="$(( $(awk '/^MemAvailable:/ { print $2; exit }' /proc/meminfo) * 1024 ))"
host_reserve_bytes=$((512 * 1024 * 1024))
if [ "$host_available_bytes" -lt $((memory_limit_bytes + host_reserve_bytes)) ]; then
    skip_dir="$TRACE_BASE_DIR/${RUN_PREFIX}_report"
    mkdir -p "$skip_dir"
    printf '%s\n' \
        "场景 B 未执行：主机 MemAvailable 小于 ${MEMORY_MAX} 受限 cgroup 加操作系统预留。" \
        "memory_limit_bytes=$memory_limit_bytes" \
        "host_available_bytes=$host_available_bytes" \
        "host_reserve_bytes=$host_reserve_bytes" \
        "请释放其他内存占用后重试；不要通过 overcommit 绕过此检查。" \
        > "$skip_dir/SKIPPED.md"
    echo "SKIPPED: insufficient available host memory; see $skip_dir/SKIPPED.md"
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
    local controller="$3"
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
        "TRACE_PROFILE=control"
        "LLM_MEM_TRACE_TENSOR=0"
        "LLM_MEM_TRACE_KV=0"
        "LLM_MEM_TRACE_EXPERT=0"
        "LLM_MEM_TRACE_MEMORY=1"
        "LLM_MEM_TRACE_RESIDENCY=0"
        "LLM_MEM_TRACE_RESIDENCY_ATTRIBUTION=0"
        "LLM_MEM_TRACE_SMAPS=0"
        "LLM_MEM_TRACE_EXPERT_TASK_MODE=summary"
        "LLM_MEM_TRACE_CONTROL_ONLY=1"
        "LLM_MEM_TRACE_OPT_EXPERT_PROFILE=custom"
        "LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=$controller"
        "LLM_MEM_TRACE_OS_HINTS=1"
        "LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=$([ "$controller" = expert_prefetch ] && echo 1 || echo 0)"
        "LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_BUDGET_MB=$PREFETCH_BUDGET_MB"
        "LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK=$PREFETCH_TOPK"
        "LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_PREFILL_TOPK=$PREFETCH_TOPK"
        "LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_DECODE_TOPK=$PREFETCH_TOPK"
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC=$([ "$controller" = expert_prefetch ] && echo 1 || echo 0)"
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS=1"
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY=0"
        "LLM_MEM_TRACE_OPT_EXPERT_FEEDBACK=1"
        "LLM_MEM_TRACE_OPT_EXPERT_VALUE_GATE=0"
        "LLM_MEM_TRACE_OPT_EXPERT_MADV_COLD_RECLAIM=0"
        "LLM_MEM_TRACE_OPT_EXPERT_MADV_DONTNEED_RECLAIM=0"
        "LLM_MEM_TRACE_OPT_EXPERT_PRELOAD=0"
    )
    echo "[RUN] $group: MemoryMax=$MEMORY_MAX, port=$port, controller=$controller, budget=${PREFETCH_BUDGET_MB}MiB, topk=$PREFETCH_TOPK"
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

run_group baseline "$PORT_BASE" off
run_group performance "$((PORT_BASE + 1))" expert_prefetch

REPORT_DIR="$TRACE_BASE_DIR/${RUN_PREFIX}_report"
python3 "$SCRIPT_DIR/summarize_scenario_b.py" \
    --base-dir "$TRACE_BASE_DIR" \
    --run-prefix "$RUN_PREFIX" \
    --output-dir "$REPORT_DIR"
echo "Report: $REPORT_DIR/SCENARIO_B_REPORT.md"
