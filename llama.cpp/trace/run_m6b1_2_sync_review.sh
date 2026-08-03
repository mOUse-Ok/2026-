#!/bin/bash
# M6B1.2 Router Tensor bit-stability gate. Default is dry-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRACE_BASE_DIR="${TRACE_BASE_DIR:-$PROJECT_DIR/trace_output}"
EXECUTE="${RUN_M6B1_2_SYNC_REVIEW_EXECUTE:-0}"
RUN_PREFIX="${RUN_PREFIX:-m6b1_2_sync_review_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
MODEL_SHA256="${MODEL_SHA256:-}"
FIXED_CPU_LIST="${M6B1_2_FIXED_CPU_LIST:-0-7}"
SINGLE_CPU="${M6B1_2_SINGLE_CPU:-0}"

readonly HINT_WORKERS=2
readonly PREDICT_TOKENS=4
readonly SYNC_PROTOCOL=m6b1.2-v1

case "$EXECUTE" in
    0|1) ;;
    *)
        echo "ERROR: RUN_M6B1_2_SYNC_REVIEW_EXECUTE must be 0 or 1" >&2
        exit 1
        ;;
esac
if [ -z "$MODEL_SHA256" ]; then
    echo "ERROR: MODEL_SHA256 must be supplied" >&2
    exit 1
fi
if ! command -v taskset >/dev/null 2>&1; then
    echo "ERROR: taskset is required" >&2
    exit 1
fi
if ! taskset -c "$SINGLE_CPU" true 2>/dev/null; then
    echo "ERROR: single CPU is unavailable: $SINGLE_CPU" >&2
    exit 1
fi
if ! taskset -c "$FIXED_CPU_LIST" true 2>/dev/null; then
    echo "ERROR: fixed CPU list is unavailable: $FIXED_CPU_LIST" >&2
    exit 1
fi

declare -a LABELS=(ST_1 ST_2 MT_1 MT_2 MT_3)
REVIEW_DIR="$TRACE_BASE_DIR/${RUN_PREFIX}_review"
if [ -e "$REVIEW_DIR" ]; then
    echo "ERROR: refusing to overwrite review output: $REVIEW_DIR" >&2
    exit 1
fi
for label in "${LABELS[@]}"; do
    if [ -e "$TRACE_BASE_DIR/${RUN_PREFIX}_${label}" ]; then
        echo "ERROR: refusing to overwrite Run: $TRACE_BASE_DIR/${RUN_PREFIX}_${label}" >&2
        exit 1
    fi
done

run_one() {
    local label="$1"
    local model_threads="$2"
    local affinity="$3"
    local run_name="${RUN_PREFIX}_${label}"
    local cmd=(
        taskset -c "$affinity"
        env
        -u LLM_MEM_TRACE_OPT_EXPERT_MAX_WAIT_THRESHOLD_US
        -u LLM_MEM_TRACE_OPT_EXPERT_URGENT_GUARD_US
        "TRACE_BASE_DIR=$TRACE_BASE_DIR"
        "RUN_NAME=$run_name"
        "MODEL_SHA256=$MODEL_SHA256"
        "NUM_TOKENS_PREDICT=$PREDICT_TOKENS"
        "NUM_THREADS=$model_threads"
        BATCH_SIZE=512
        CTX_SIZE=2048
        TEMP=0.0
        SEED=1234
        GPU_LAYERS=0
        TRACE_PROFILE=custom
        CACHE_MODE=cold
        ALLOW_DIRTY_REPO=1
        LC_ALL=C.UTF-8
        LANG=C.UTF-8
        "OMP_NUM_THREADS=$model_threads"
        OMP_DYNAMIC=FALSE
        "OPENBLAS_NUM_THREADS=$model_threads"
        "MKL_NUM_THREADS=$model_threads"
        MKL_DYNAMIC=FALSE
        "BLIS_NUM_THREADS=$model_threads"
        "NUMEXPR_NUM_THREADS=$model_threads"
        LLM_MEM_TRACE_TENSOR=0
        LLM_MEM_TRACE_KV=0
        LLM_MEM_TRACE_EXPERT=1
        LLM_MEM_TRACE_MEMORY=1
        LLM_MEM_TRACE_RESIDENCY=0
        LLM_MEM_TRACE_SMAPS=0
        LLM_MEM_TRACE_QUEUE_LIMIT=524288
        LLM_MEM_TRACE_ALLOW_DROP=0
        LLM_MEM_TRACE_EXPERT_TASK_MODE=detail
        LLM_MEM_TRACE_ROUTER_SCORE_DIAGNOSTIC=1
        "LLM_MEM_TRACE_ROUTER_TENSOR_SYNC_PROTOCOL=$SYNC_PROTOCOL"
        "LLM_MEM_TRACE_AUDIT_CASE=$label"
        "LLM_MEM_TRACE_AUDIT_MODEL_THREADS=$model_threads"
        "LLM_MEM_TRACE_AUDIT_HINT_WORKERS=$HINT_WORKERS"
        "LLM_MEM_TRACE_AUDIT_CPU_AFFINITY=$affinity"
        LLM_MEM_TRACE_OS_HINTS=1
        LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=1
        LLM_MEM_TRACE_OPT_EXPERT_POLICY=route
        LLM_MEM_TRACE_OPT_EXPERT_CACHE_MB=512
        LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK=0
        LLM_MEM_TRACE_OPT_EXPERT_TTL_STEPS=0
        LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_STEPS=0
        LLM_MEM_TRACE_OPT_EXPERT_COALESCE=0
        LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=off
        LLM_MEM_TRACE_OPT_EXPERT_FEEDBACK=0
        LLM_MEM_TRACE_OPT_EXPERT_SLACK=0
        LLM_MEM_TRACE_OPT_EXPERT_VALUE_GATE=0
        LLM_MEM_TRACE_OPT_EXPERT_CROSS_LAYER_PREDICT=0
        LLM_MEM_TRACE_OPT_EXPERT_SLACK_MODE=off
        LLM_MEM_TRACE_PRESSURE_SHADOW_MODE=off
        LLM_MEM_TRACE_OPT_EXPERT_DEADLINE_OBSERVE=1
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC=1
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE=131072
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS=$HINT_WORKERS"
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY=1
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE=deadline_score
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP=0
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH=1
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_WAIT_US=0
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_COALESCE=0
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_FALLBACK=1
        bash "$SCRIPT_DIR/run_trace_pipeline.sh"
    )
    printf '[%s] ' "$label"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if [ "$EXECUTE" = "1" ]; then
        "${cmd[@]}"
    fi
}

echo "M6B1.2 Router Tensor sync review ($([ "$EXECUTE" = "1" ] && echo execute || echo dry-run))"
echo "Review ID: $RUN_PREFIX"
echo "Gate order: ST x2 -> fixed MT x3 -> bitwise validator"
echo "Sync: producer barrier -> thread 0 Hook -> existing release barrier"
echo "No A/B Smoke, N=8, threshold/guard, search, tuning, or performance ranking"

run_one ST_1 1 "$SINGLE_CPU"
run_one ST_2 1 "$SINGLE_CPU"
run_one MT_1 8 "$FIXED_CPU_LIST"
run_one MT_2 8 "$FIXED_CPU_LIST"
run_one MT_3 8 "$FIXED_CPU_LIST"

validate=(
    python3 "$SCRIPT_DIR/validate_m6b1_2_sync_review.py"
    --review-id "$RUN_PREFIX"
    --single "$TRACE_BASE_DIR/${RUN_PREFIX}_ST_1"
    --single "$TRACE_BASE_DIR/${RUN_PREFIX}_ST_2"
    --multi "$TRACE_BASE_DIR/${RUN_PREFIX}_MT_1"
    --multi "$TRACE_BASE_DIR/${RUN_PREFIX}_MT_2"
    --multi "$TRACE_BASE_DIR/${RUN_PREFIX}_MT_3"
    --output-dir "$REVIEW_DIR"
)
printf '[validate] '
printf '%q ' "${validate[@]}"
printf '\n'
if [ "$EXECUTE" = "1" ]; then
    "${validate[@]}"
    echo "M6B1.2 sync review artifacts: $REVIEW_DIR"
else
    echo "Dry-run complete. Set RUN_M6B1_2_SYNC_REVIEW_EXECUTE=1 to execute Gate 1."
fi
