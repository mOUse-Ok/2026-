#!/bin/bash
# M6B2.1 off/summary engineering equivalence for workers=2 or 4. Default dry-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRACE_BASE_DIR="${TRACE_BASE_DIR:-$PROJECT_DIR/trace_output}"
EXECUTE="${RUN_M6B2_1_SMOKE_EXECUTE:-0}"
WORKERS="${M6B2_1_SMOKE_WORKERS:-2}"
RUN_PREFIX="${RUN_PREFIX:-m6b2_1_workers${WORKERS}_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
MODEL_SHA256="${MODEL_SHA256:-}"
FIXED_CPU_LIST="${M6B2_1_FIXED_CPU_LIST:-0-7}"

readonly MODEL_THREADS=8
readonly PREDICT_TOKENS=4
readonly MAX_WAIT_THRESHOLD_US=1
readonly URGENT_GUARD_US=0
readonly SYNC_PROTOCOL=m6b1.2-v1

case "$EXECUTE" in
    0|1) ;;
    *)
        echo "ERROR: RUN_M6B2_1_SMOKE_EXECUTE must be 0 or 1" >&2
        exit 1
        ;;
esac
case "$WORKERS" in
    2|4) ;;
    *)
        echo "ERROR: M6B2_1_SMOKE_WORKERS must be 2 or 4" >&2
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

for scheduler in deadline_score max_wait_protection; do
    for observation in off summary; do
        output="$TRACE_BASE_DIR/${RUN_PREFIX}_${scheduler}_${observation}"
        if [ -e "$output" ]; then
            echo "ERROR: refusing to overwrite M6B2.1 evidence: $output" >&2
            exit 1
        fi
    done
    validation="$TRACE_BASE_DIR/${RUN_PREFIX}_${scheduler}_validation.json"
    if [ -e "$validation" ]; then
        echo "ERROR: refusing to overwrite M6B2.1 evidence: $validation" >&2
        exit 1
    fi
done

run_one() {
    local scheduler="$1"
    local observation="$2"
    local run_name="${RUN_PREFIX}_${scheduler}_${observation}"
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
        "LLM_MEM_TRACE_QUEUE_OVERHEAD_MODE=$observation"
        "LLM_MEM_TRACE_AUDIT_CASE=m6b2_1_workers${WORKERS}_${scheduler}_${observation}"
        "LLM_MEM_TRACE_AUDIT_MODEL_THREADS=$MODEL_THREADS"
        "LLM_MEM_TRACE_AUDIT_HINT_WORKERS=$WORKERS"
        "LLM_MEM_TRACE_AUDIT_CPU_AFFINITY=$FIXED_CPU_LIST"
        LLM_MEM_TRACE_AUDIT_NOT_M6B2_CALIBRATION=1
        LLM_MEM_TRACE_AUDIT_UNLIMITED_CGROUP_NOT_BASELINE=1
        LLM_MEM_TRACE_AUDIT_PERFORMANCE_CLAIM=0
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
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE=$scheduler"
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP=0
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH=1
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_WAIT_US=0
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_COALESCE=0
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_FALLBACK=1
    )
    if [ "$scheduler" = "max_wait_protection" ]; then
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

echo "M6B2.1 queue overhead engineering smoke ($([ "$EXECUTE" = "1" ] && echo execute || echo dry-run))"
echo "workers=$WORKERS model_threads=$MODEL_THREADS affinity=$FIXED_CPU_LIST"
echo "not_m6b2_calibration=true unlimited_cgroup_not_a_baseline=true performance_claim=false"
echo "No parameter search, formal N=8, calibration, commit, stash, reset, or push"

for scheduler in deadline_score max_wait_protection; do
    run_one "$scheduler" off
    run_one "$scheduler" summary
    validate=(
        python3 "$SCRIPT_DIR/validate_m6b2_1_queue_overhead.py"
        --off "$TRACE_BASE_DIR/${RUN_PREFIX}_${scheduler}_off"
        --summary "$TRACE_BASE_DIR/${RUN_PREFIX}_${scheduler}_summary"
        --scheduler "$scheduler"
        --workers "$WORKERS"
        --output "$TRACE_BASE_DIR/${RUN_PREFIX}_${scheduler}_validation.json"
    )
    printf '[validate:%s] ' "$scheduler"
    printf '%q ' "${validate[@]}"
    printf '\n'
    if [ "$EXECUTE" = "1" ]; then
        "${validate[@]}"
    fi
done

if [ "$EXECUTE" = "0" ]; then
    echo "Dry-run complete. Set RUN_M6B2_1_SMOKE_EXECUTE=1 to execute."
fi
