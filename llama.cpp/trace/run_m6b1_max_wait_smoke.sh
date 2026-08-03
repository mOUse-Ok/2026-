#!/bin/bash
# M6B1 two-case engineering smoke. Default behavior is command-only dry-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRACE_BASE_DIR="${TRACE_BASE_DIR:-$PROJECT_DIR/trace_output}"
EXECUTE="${RUN_M6B1_SMOKE_EXECUTE:-0}"
RUN_PREFIX="${RUN_PREFIX:-m6b1_smoke_$(date -u +%Y%m%dT%H%M%SZ)_$$}"

# Frozen path-activation values. They are not calibrated or recommended values.
readonly MAX_WAIT_THRESHOLD_US=1
readonly URGENT_GUARD_US=0
readonly WORKERS=2
readonly PREDICT_TOKENS=4

case "$EXECUTE" in
    0|1) ;;
    *)
        echo "ERROR: RUN_M6B1_SMOKE_EXECUTE must be 0 or 1" >&2
        exit 1
        ;;
esac

BASELINE_NAME="${RUN_PREFIX}_deadline_score"
CANDIDATE_NAME="${RUN_PREFIX}_max_wait_protection"
BASELINE_DIR="$TRACE_BASE_DIR/$BASELINE_NAME"
CANDIDATE_DIR="$TRACE_BASE_DIR/$CANDIDATE_NAME"
VALIDATION_FILE="$TRACE_BASE_DIR/${RUN_PREFIX}_validation.json"

for output in "$BASELINE_DIR" "$CANDIDATE_DIR" "$VALIDATION_FILE"; do
    if [ -e "$output" ]; then
        echo "ERROR: refusing to overwrite existing M6B1 evidence: $output" >&2
        exit 1
    fi
done

run_one() {
    local mode="$1"
    local run_name="$2"
    local cmd=(
        env
        -u LLM_MEM_TRACE_OPT_EXPERT_MAX_WAIT_THRESHOLD_US
        -u LLM_MEM_TRACE_OPT_EXPERT_URGENT_GUARD_US
        "TRACE_BASE_DIR=$TRACE_BASE_DIR"
        "RUN_NAME=$run_name"
        "NUM_TOKENS_PREDICT=$PREDICT_TOKENS"
        NUM_THREADS=8
        BATCH_SIZE=512
        CTX_SIZE=2048
        TEMP=0.0
        SEED=1234
        GPU_LAYERS=0
        TRACE_PROFILE=evidence
        CACHE_MODE=cold
        ALLOW_DIRTY_REPO=1
        LLM_MEM_TRACE_QUEUE_LIMIT=262144
        LLM_MEM_TRACE_ALLOW_DROP=0
        LLM_MEM_TRACE_EXPERT_TASK_MODE=detail
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

echo "M6B1 max-wait engineering smoke ($([ "$EXECUTE" = "1" ] && echo execute || echo dry-run))"
echo "Cases: deadline_score, max_wait_protection"
echo "Frozen trigger-only parameters: threshold=${MAX_WAIT_THRESHOLD_US}us guard=${URGENT_GUARD_US}us"
echo "Workers=$WORKERS predict_tokens=$PREDICT_TOKENS; no repeats, search, or tuning"

run_one deadline_score "$BASELINE_NAME"
run_one max_wait_protection "$CANDIDATE_NAME"

validate=(
    python3 "$SCRIPT_DIR/validate_m6b1_smoke.py"
    --baseline "$BASELINE_DIR"
    --candidate "$CANDIDATE_DIR"
    --output "$VALIDATION_FILE"
)
printf '[validate] '
printf '%q ' "${validate[@]}"
printf '\n'
if [ "$EXECUTE" = "1" ]; then
    "${validate[@]}"
else
    echo "Dry-run complete. Set RUN_M6B1_SMOKE_EXECUTE=1 to execute exactly this A/B smoke."
fi
