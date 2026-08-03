#!/bin/bash
# M6B1.2 strict A/B engineering Smoke for workers=2 or workers=4. Default is dry-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRACE_BASE_DIR="${TRACE_BASE_DIR:-$PROJECT_DIR/trace_output}"
EXECUTE="${RUN_M6B1_2_SMOKE_EXECUTE:-0}"
WORKERS="${M6B1_2_SMOKE_WORKERS:-2}"
RUN_PREFIX="${RUN_PREFIX:-m6b1_2_workers${WORKERS}_smoke_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
MODEL_SHA256="${MODEL_SHA256:-}"
FIXED_CPU_LIST="${M6B1_2_FIXED_CPU_LIST:-0-7}"

readonly MODEL_THREADS=8
readonly PREDICT_TOKENS=4
readonly MAX_WAIT_THRESHOLD_US=1
readonly URGENT_GUARD_US=0
readonly SYNC_PROTOCOL=m6b1.2-v1

case "$EXECUTE" in
    0|1) ;;
    *)
        echo "ERROR: RUN_M6B1_2_SMOKE_EXECUTE must be 0 or 1" >&2
        exit 1
        ;;
esac
case "$WORKERS" in
    2|4) ;;
    *)
        echo "ERROR: M6B1_2_SMOKE_WORKERS must be 2 or 4" >&2
        exit 1
        ;;
esac
if [ -z "$MODEL_SHA256" ]; then
    echo "ERROR: MODEL_SHA256 must be supplied" >&2
    exit 1
fi
if ! taskset -c "$FIXED_CPU_LIST" true 2>/dev/null; then
    echo "ERROR: fixed CPU list is unavailable: $FIXED_CPU_LIST" >&2
    exit 1
fi

BASELINE_NAME="${RUN_PREFIX}_deadline_score"
CANDIDATE_NAME="${RUN_PREFIX}_max_wait_protection"
BASELINE_DIR="$TRACE_BASE_DIR/$BASELINE_NAME"
CANDIDATE_DIR="$TRACE_BASE_DIR/$CANDIDATE_NAME"
VALIDATION_FILE="$TRACE_BASE_DIR/${RUN_PREFIX}_validation.json"

for output in "$BASELINE_DIR" "$CANDIDATE_DIR" "$VALIDATION_FILE"; do
    if [ -e "$output" ]; then
        echo "ERROR: refusing to overwrite M6B1.2 evidence: $output" >&2
        exit 1
    fi
done

run_one() {
    local mode="$1"
    local run_name="$2"
    local cmd=(
        taskset -c "$FIXED_CPU_LIST"
        env
        -u LLM_MEM_TRACE_OPT_EXPERT_MAX_WAIT_THRESHOLD_US
        -u LLM_MEM_TRACE_OPT_EXPERT_URGENT_GUARD_US
        "TRACE_BASE_DIR=$TRACE_BASE_DIR"
        "RUN_NAME=$run_name"
        "MODEL_SHA256=$MODEL_SHA256"
        "NUM_TOKENS_PREDICT=$PREDICT_TOKENS"
        "NUM_THREADS=$MODEL_THREADS"
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
        "OMP_NUM_THREADS=$MODEL_THREADS"
        OMP_DYNAMIC=FALSE
        "OPENBLAS_NUM_THREADS=$MODEL_THREADS"
        "MKL_NUM_THREADS=$MODEL_THREADS"
        MKL_DYNAMIC=FALSE
        "BLIS_NUM_THREADS=$MODEL_THREADS"
        "NUMEXPR_NUM_THREADS=$MODEL_THREADS"
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
        "LLM_MEM_TRACE_AUDIT_CASE=workers${WORKERS}_${mode}"
        "LLM_MEM_TRACE_AUDIT_MODEL_THREADS=$MODEL_THREADS"
        "LLM_MEM_TRACE_AUDIT_HINT_WORKERS=$WORKERS"
        "LLM_MEM_TRACE_AUDIT_CPU_AFFINITY=$FIXED_CPU_LIST"
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
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS=$WORKERS"
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY=1
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE=$mode"
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP=0
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH=1
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_WAIT_US=0
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_COALESCE=0
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_FALLBACK=1
    )
    if [ "$mode" = "max_wait_protection" ]; then
        cmd+=(
            "LLM_MEM_TRACE_OPT_EXPERT_MAX_WAIT_THRESHOLD_US=$MAX_WAIT_THRESHOLD_US"
            "LLM_MEM_TRACE_OPT_EXPERT_URGENT_GUARD_US=$URGENT_GUARD_US"
        )
    fi
    cmd+=(bash "$SCRIPT_DIR/run_trace_pipeline.sh")

    printf '[%s] ' "$run_name"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if [ "$EXECUTE" = "1" ]; then
        "${cmd[@]}"
    fi
}

echo "M6B1.2 strict A/B Smoke ($([ "$EXECUTE" = "1" ] && echo execute || echo dry-run))"
echo "Workers=$WORKERS model_threads=$MODEL_THREADS affinity=$FIXED_CPU_LIST"
echo "A=deadline_score B=max_wait_protection threshold=1us guard=0us"
echo "No parameter search, N=8, calibration, or performance ranking"

run_one deadline_score "$BASELINE_NAME"
run_one max_wait_protection "$CANDIDATE_NAME"

validate=(
    python3 "$SCRIPT_DIR/validate_m6b1_smoke.py"
    --baseline "$BASELINE_DIR"
    --candidate "$CANDIDATE_DIR"
    --workers "$WORKERS"
    --output "$VALIDATION_FILE"
)
printf '[validate] '
printf '%q ' "${validate[@]}"
printf '\n'
if [ "$EXECUTE" = "1" ]; then
    "${validate[@]}"
    echo "M6B1.2 workers=$WORKERS validation: $VALIDATION_FILE"
else
    echo "Dry-run complete. Set RUN_M6B1_2_SMOKE_EXECUTE=1 to execute this Gate."
fi
