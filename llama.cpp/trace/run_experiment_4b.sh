#!/bin/bash
# Experiment 4B: object-level residency attribution.
#
# Default mode is dry-run. Set RUN_EXPERIMENT_4B_EXECUTE=1 to run the fresh
# Unlimited and MemoryMax=7G N=1 measurements.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_FILE="${MODEL_FILE:-$PROJECT_DIR/../models/Qwen3.5-35B-A3B-Q3_K_M.gguf}"
RUN_ROOT="${TRACE_BASE_DIR:-$PROJECT_DIR/trace_output}/experiment_4b"
CGROUP_PARENT_INPUT="${CGROUP_PARENT:-}"
CGROUP_PARENT="${CGROUP_PARENT_INPUT:-/sys/fs/cgroup}"
if [ -z "$CGROUP_PARENT_INPUT" ]; then
    USER_ID="$(id -u)"
    DELEGATED_PARENT="/sys/fs/cgroup/user.slice/user-${USER_ID}.slice/user@${USER_ID}.service"
    # In some containers the cgroup filesystem reports the root as writable
    # even though it is mounted read-only. Prefer the user's delegated systemd
    # subtree when it exists, since that is the normal place to create a child.
    if [ -f "$DELEGATED_PARENT/cgroup.controllers" ] &&
            grep -qw memory "$DELEGATED_PARENT/cgroup.controllers"; then
        CGROUP_PARENT="$DELEGATED_PARENT"
    fi
fi
CGROUP_DIR="$CGROUP_PARENT/llm_mem_trace_experiment_4b_7g"
EXECUTE="${RUN_EXPERIMENT_4B_EXECUTE:-0}"
OVERHEAD="${RUN_EXPERIMENT_4B_OVERHEAD:-0}"
SKIP_UNLIMITED="${RUN_EXPERIMENT_4B_SKIP_UNLIMITED:-0}"

common_env=(
    TRACE_BASE_DIR="$RUN_ROOT"
    TRACE_PROFILE=attribution
    CACHE_MODE=cold
    MODEL_FILE="$MODEL_FILE"
    LLM_MEM_TRACE_RESIDENCY_MAX_PAGES="${LLM_MEM_TRACE_RESIDENCY_MAX_PAGES:-262144}"
    NUM_TOKENS_PREDICT=80
    NUM_THREADS=8
    BATCH_SIZE=512
    CTX_SIZE=2048
    TEMP=0
    SEED=1234
    GPU_LAYERS=0
    LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=off
    LLM_MEM_TRACE_OS_HINTS=0
    LLM_MEM_TRACE_OPT_MADVISE_WILLNEED=0
    LLM_MEM_TRACE_OPT_MADVISE_SEQUENTIAL=0
    LLM_MEM_TRACE_OPT_POSIX_FADVISE=0
    LLM_MEM_TRACE_OPT_THP=0
    LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=0
    LLM_MEM_TRACE_OPT_EXPERT_ASYNC=0
    LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY=0
    LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP=0
    LLM_MEM_TRACE_OPT_EXPERT_DEADLINE_OBSERVE=0
    LLM_MEM_TRACE_OPT_EXPERT_FEEDBACK=0
    LLM_MEM_TRACE_OPT_EXPERT_VALUE_GATE=0
    LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_STEPS=0
    LLM_MEM_TRACE_OPT_EXPERT_MEMORY_OBJECTS=0
    LLM_MEM_TRACE_OPT_EXPERT_MADV_COLD_RECLAIM=0
    LLM_MEM_TRACE_OPT_EXPERT_WORKING_SET_MB=0
    LLM_MEM_TRACE_OPT_EXPERT_RUNTIME_RESCUE=0
)

run_unlimited() {
    echo "[RUN] Unlimited N=1"
    env "${common_env[@]}" RUN_NAME=unlimited_n1 MEMORY_MAX= bash "$SCRIPT_DIR/run_trace_pipeline.sh"
}

run_7g() {
    if [ ! -f "$CGROUP_PARENT/cgroup.controllers" ]; then
        echo "ERROR: cgroup v2 is unavailable at $CGROUP_PARENT" >&2
        return 1
    fi
    echo "[RUN] MemoryMax=7G, MemorySwapMax=0, N=1"
    if ! mkdir -p "$CGROUP_DIR"; then
        echo "ERROR: cannot create a child cgroup under $CGROUP_PARENT" >&2
        return 1
    fi
    echo $((7 * 1024 * 1024 * 1024)) > "$CGROUP_DIR/memory.max"
    if [ -f "$CGROUP_DIR/memory.swap.max" ]; then
        echo 0 > "$CGROUP_DIR/memory.swap.max"
    fi
    set +e
    (
        if ! echo "$BASHPID" > "$CGROUP_DIR/cgroup.procs"; then
            echo "ERROR: cannot migrate the experiment shell into $CGROUP_DIR; refusing to run outside MemoryMax=7G" >&2
            exit 1
        fi
        env "${common_env[@]}" RUN_NAME=memorymax_7g_n1 MEMORY_MAX=7G MEMORY_SWAP_MAX=0 bash "$SCRIPT_DIR/run_trace_pipeline.sh"
    )
    status=$?
    set -e
    rmdir "$CGROUP_DIR" 2>/dev/null || true
    return "$status"
}

run_overhead_pair() {
    echo "[RUN] Unlimited observer OFF x1"
    env "${common_env[@]}" RUN_NAME=overhead_observer_off LLM_MEM_TRACE_RESIDENCY=0 LLM_MEM_TRACE_RESIDENCY_ATTRIBUTION=0 MEMORY_MAX= bash "$SCRIPT_DIR/run_trace_pipeline.sh"
    echo "[RUN] Unlimited observer ON x1"
    env "${common_env[@]}" RUN_NAME=overhead_observer_on LLM_MEM_TRACE_RESIDENCY=1 LLM_MEM_TRACE_RESIDENCY_ATTRIBUTION=1 MEMORY_MAX= bash "$SCRIPT_DIR/run_trace_pipeline.sh"
}

echo "=============================================="
echo "  Experiment 4B: Residency Attribution"
echo "=============================================="
echo "Model: $MODEL_FILE"
echo "Output: $RUN_ROOT"
echo "Mode: $([ "$EXECUTE" = "1" ] && echo execute || echo dry-run)"
echo "Overhead pair: $OVERHEAD"

if [ "$EXECUTE" != "1" ]; then
    echo ""
    echo "Would run the fresh Unlimited and MemoryMax=7G N=1 traces."
    echo "Set RUN_EXPERIMENT_4B_EXECUTE=1 to execute them."
    exit 0
fi

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: model not found: $MODEL_FILE" >&2
    exit 1
fi

if [ "$SKIP_UNLIMITED" != "1" ]; then
    run_unlimited
fi
run_7g

python3 "$SCRIPT_DIR/summarize_experiment_4b.py" --unlimited-dir "$RUN_ROOT/unlimited_n1" --memory-dir "$RUN_ROOT/memorymax_7g_n1" --output-dir "$RUN_ROOT/report"

if [ "$OVERHEAD" = "1" ]; then
    run_overhead_pair
fi
