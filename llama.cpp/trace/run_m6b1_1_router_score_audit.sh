#!/bin/bash
# M6B1.1 Router score determinism diagnostic matrix. Default is dry-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRACE_BASE_DIR="${TRACE_BASE_DIR:-$PROJECT_DIR/trace_output}"
EXECUTE="${RUN_M6B1_1_AUDIT_EXECUTE:-0}"
RUN_PREFIX="${RUN_PREFIX:-m6b1_1_router_score_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
MODEL_SHA256="${MODEL_SHA256:-}"
FIXED_CPU_LIST="${M6B1_1_FIXED_CPU_LIST:-0-7}"

readonly WORKERS=2
readonly PREDICT_TOKENS=4
readonly MAX_WAIT_THRESHOLD_US=1
readonly URGENT_GUARD_US=0

case "$EXECUTE" in
    0|1) ;;
    *)
        echo "ERROR: RUN_M6B1_1_AUDIT_EXECUTE must be 0 or 1" >&2
        exit 1
        ;;
esac

if [ -z "$MODEL_SHA256" ]; then
    echo "ERROR: MODEL_SHA256 must be supplied; hash the frozen model once before cold-cache runs" >&2
    exit 1
fi
if ! command -v taskset >/dev/null 2>&1; then
    echo "ERROR: taskset is required for the fixed-multithread diagnostic" >&2
    exit 1
fi
if ! taskset -c "$FIXED_CPU_LIST" true 2>/dev/null; then
    echo "ERROR: requested fixed CPU list is unavailable: $FIXED_CPU_LIST" >&2
    exit 1
fi

AUDIT_DIR="$TRACE_BASE_DIR/${RUN_PREFIX}_audit"
declare -a LABELS=(
    A_MT_1 A_MT_2 A_MT_3
    B_ST_1 B_ST_2
    C_FIXED_MT_1 C_FIXED_MT_2
    D_MAX_WAIT_1 D_MAX_WAIT_2
)

if [ -e "$AUDIT_DIR" ]; then
    echo "ERROR: refusing to overwrite audit output: $AUDIT_DIR" >&2
    exit 1
fi
for label in "${LABELS[@]}"; do
    if [ -e "$TRACE_BASE_DIR/${RUN_PREFIX}_${label}" ]; then
        echo "ERROR: refusing to overwrite diagnostic run: $TRACE_BASE_DIR/${RUN_PREFIX}_${label}" >&2
        exit 1
    fi
done

run_one() {
    local label="$1"
    local mode="$2"
    local model_threads="$3"
    local affinity="$4"
    local run_name="${RUN_PREFIX}_${label}"
    local affinity_record="$affinity"
    local cmd=(
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
        "LLM_MEM_TRACE_AUDIT_CASE=$label"
        "LLM_MEM_TRACE_AUDIT_MODEL_THREADS=$model_threads"
        "LLM_MEM_TRACE_AUDIT_HINT_WORKERS=$WORKERS"
        "LLM_MEM_TRACE_AUDIT_CPU_AFFINITY=$affinity_record"
        LLM_MEM_TRACE_AUDIT_FLOAT_ROUNDING=default_nearest_unverified
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
    if [ "$affinity" != "unbound" ]; then
        cmd=(taskset -c "$affinity" "${cmd[@]}")
    fi

    printf '[%s] ' "$label"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if [ "$EXECUTE" = "1" ]; then
        "${cmd[@]}"
    fi
}

echo "M6B1.1 Router score determinism audit ($([ "$EXECUTE" = "1" ] && echo execute || echo dry-run))"
echo "Audit ID: $RUN_PREFIX"
echo "Diagnostic runs: A=3 unbound 8-thread deadline_score; B=2 single-thread; C=2 fixed 8-thread; D=2 max_wait self-repeats"
echo "No N=8, performance ranking, threshold search, guard search, or Task identity relaxation"

run_one A_MT_1 deadline_score 8 unbound
run_one A_MT_2 deadline_score 8 unbound
run_one A_MT_3 deadline_score 8 unbound
run_one B_ST_1 deadline_score 1 unbound
run_one B_ST_2 deadline_score 1 unbound
run_one C_FIXED_MT_1 deadline_score 8 "$FIXED_CPU_LIST"
run_one C_FIXED_MT_2 deadline_score 8 "$FIXED_CPU_LIST"
run_one D_MAX_WAIT_1 max_wait_protection 8 unbound
run_one D_MAX_WAIT_2 max_wait_protection 8 unbound

audit=(
    python3 "$SCRIPT_DIR/audit_router_score_determinism.py"
    --audit-id "$RUN_PREFIX"
    --run "A_MT_1=$TRACE_BASE_DIR/${RUN_PREFIX}_A_MT_1"
    --run "A_MT_2=$TRACE_BASE_DIR/${RUN_PREFIX}_A_MT_2"
    --run "A_MT_3=$TRACE_BASE_DIR/${RUN_PREFIX}_A_MT_3"
    --run "B_ST_1=$TRACE_BASE_DIR/${RUN_PREFIX}_B_ST_1"
    --run "B_ST_2=$TRACE_BASE_DIR/${RUN_PREFIX}_B_ST_2"
    --run "C_FIXED_MT_1=$TRACE_BASE_DIR/${RUN_PREFIX}_C_FIXED_MT_1"
    --run "C_FIXED_MT_2=$TRACE_BASE_DIR/${RUN_PREFIX}_C_FIXED_MT_2"
    --run "D_MAX_WAIT_1=$TRACE_BASE_DIR/${RUN_PREFIX}_D_MAX_WAIT_1"
    --run "D_MAX_WAIT_2=$TRACE_BASE_DIR/${RUN_PREFIX}_D_MAX_WAIT_2"
    --comparison A_OFF_OFF_12=A_MT_1,A_MT_2
    --comparison A_OFF_OFF_13=A_MT_1,A_MT_3
    --comparison A_OFF_OFF_23=A_MT_2,A_MT_3
    --comparison B_SINGLE_THREAD=B_ST_1,B_ST_2
    --comparison C_FIXED_MULTI=C_FIXED_MT_1,C_FIXED_MT_2
    --comparison D_MAX_WAIT_SELF=D_MAX_WAIT_1,D_MAX_WAIT_2
    --comparison THREAD_CROSS=A_MT_1,B_ST_1
    --comparison AFFINITY_CROSS=A_MT_1,C_FIXED_MT_1
    --output-dir "$AUDIT_DIR"
)

printf '[audit] '
printf '%q ' "${audit[@]}"
printf '\n'
if [ "$EXECUTE" = "1" ]; then
    "${audit[@]}"
    echo "M6B1.1 audit artifacts: $AUDIT_DIR"
else
    echo "Dry-run complete. Set RUN_M6B1_1_AUDIT_EXECUTE=1 with the same frozen inputs to execute."
fi
