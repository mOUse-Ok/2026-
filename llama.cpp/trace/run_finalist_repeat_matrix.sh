#!/bin/bash
# ============================================================================
# Reproduce finalist repeat-run matrix for LLM memory trace experiments.
#
# Default mode is dry-run: commands are printed but not executed.
# Set RUN_REPEAT_MATRIX_EXECUTE=1 to run the full matrix.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRACE_BASE_DIR="${TRACE_BASE_DIR:-$PROJECT_DIR/trace_output}"
RUN_PREFIX="${RUN_PREFIX:-contest_finalist}"
REPEAT_COUNT="${REPEAT_COUNT:-8}"
NUM_TOKENS_PREDICT="${NUM_TOKENS_PREDICT:-80}"
OUTPUT_DIR="${OUTPUT_DIR:-$TRACE_BASE_DIR/contest_runs/repeat_summary}"
EXECUTE="${RUN_REPEAT_MATRIX_EXECUTE:-0}"
TRACE_PROFILE="${TRACE_PROFILE:-benchmark}"
CACHE_MODE="${CACHE_MODE:-cold}"
ORDER_MODE="${ORDER_MODE:-latin}"
ORDER_SEED="${ORDER_SEED:-0}"
MEMORY_MAX="${MEMORY_MAX:-}"
MEMORY_SWAP_MAX="${MEMORY_SWAP_MAX:-}"
CASES_CSV="${CASES_CSV:-baseline,deadline_score,feedback_slack,feedback_slack_predict}"
IFS=',' read -r -a CASES <<< "$CASES_CSV"

if [ "${#CASES[@]}" -eq 0 ]; then
    echo "ERROR: CASES_CSV must contain at least one case" >&2
    exit 1
fi
if [[ ",${CASES_CSV}," != *,baseline,* ]]; then
    echo "ERROR: CASES_CSV must include baseline for relative aggregation" >&2
    exit 1
fi

if ! [[ "$REPEAT_COUNT" =~ ^[0-9]+$ ]] || [ "$REPEAT_COUNT" -le 0 ]; then
    echo "ERROR: REPEAT_COUNT must be a positive integer" >&2
    exit 1
fi

if ! [[ "$ORDER_SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ORDER_SEED must be a non-negative integer" >&2
    exit 1
fi

if [ "$ORDER_MODE" != "latin" ] && [ "$ORDER_MODE" != "fixed" ]; then
    echo "ERROR: ORDER_MODE must be latin or fixed" >&2
    exit 1
fi

if [ "$CACHE_MODE" != "cold" ] && [ "$CACHE_MODE" != "warm" ] && [ "$CACHE_MODE" != "as-is" ]; then
    echo "ERROR: CACHE_MODE must be cold, warm, or as-is" >&2
    exit 1
fi

if [ -n "$MEMORY_SWAP_MAX" ] && [ -z "$MEMORY_MAX" ]; then
    echo "ERROR: MEMORY_SWAP_MAX requires MEMORY_MAX" >&2
    exit 1
fi

if [ "$EXECUTE" = "1" ] && [ -n "$MEMORY_MAX" ] && ! command -v systemd-run >/dev/null 2>&1; then
    echo "ERROR: MEMORY_MAX requires systemd-run for an isolated cgroup v2 scope" >&2
    exit 1
fi

join_runs() {
    local group="$1"
    local out=""
    local i
    for i in $(seq 1 "$REPEAT_COUNT"); do
        if [ -n "$out" ]; then
            out+=","
        fi
        out+="${RUN_PREFIX}_${group}_r${i}"
    done
    echo "$out"
}

run_case() {
    local group="$1"
    local idx="$2"
    local order_position="$3"
    shift 3
    local run_name="${RUN_PREFIX}_${group}_r${idx}"
    local cmd=(
        env
        "TRACE_BASE_DIR=$TRACE_BASE_DIR"
        "RUN_NAME=$run_name"
        "NUM_TOKENS_PREDICT=$NUM_TOKENS_PREDICT"
        "TRACE_PROFILE=$TRACE_PROFILE"
        "CACHE_MODE=$CACHE_MODE"
        "REPEAT_INDEX=$idx"
        "ORDER_POSITION=$order_position"
        "ORDER_MODE=$ORDER_MODE"
        "ORDER_SEED=$ORDER_SEED"
        "MEMORY_MAX=$MEMORY_MAX"
        "MEMORY_SWAP_MAX=$MEMORY_SWAP_MAX"
        "$@"
        bash "$SCRIPT_DIR/run_trace_pipeline.sh"
    )
    local launch=("${cmd[@]}")

    if [ -n "$MEMORY_MAX" ]; then
        local cgroup=(systemd-run --user --scope --quiet -p "MemoryMax=$MEMORY_MAX")
        if [ -n "$MEMORY_SWAP_MAX" ]; then
            cgroup+=(-p "MemorySwapMax=$MEMORY_SWAP_MAX")
        fi
        cgroup+=(--)
        launch=("${cgroup[@]}" "${cmd[@]}")
    fi

    printf '[RUN] %s\n' "$run_name"
    printf '      '
    printf '%q ' "${launch[@]}"
    printf '\n'

    if [ "$EXECUTE" = "1" ]; then
        "${launch[@]}"
    fi
}

run_named_case() {
    local group="$1"
    local idx="$2"
    local order_position="$3"

    case "$group" in
        baseline)
            run_case baseline "$idx" "$order_position" \
                LLM_MEM_TRACE_OS_HINTS=0 \
                LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=0 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC=0 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY=0 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP=0 \
                LLM_MEM_TRACE_OPT_EXPERT_COALESCE=0 \
                LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=off \
                LLM_MEM_TRACE_OPT_EXPERT_FEEDBACK=0 \
                LLM_MEM_TRACE_OPT_EXPERT_SLACK=0 \
                LLM_MEM_TRACE_OPT_EXPERT_VALUE_GATE=0 \
                LLM_MEM_TRACE_OPT_EXPERT_CROSS_LAYER_PREDICT=0 \
                LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_STEPS=0
            ;;
        expert_prefetch)
            run_case expert_prefetch "$idx" "$order_position" \
                LLM_MEM_TRACE_OS_HINTS=1 \
                LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=1 \
                LLM_MEM_TRACE_OPT_EXPERT_POLICY=route \
                LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK=0 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC=0 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY=0 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP=0 \
                LLM_MEM_TRACE_OPT_EXPERT_COALESCE=0 \
                LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=off \
                LLM_MEM_TRACE_OPT_EXPERT_FEEDBACK=0 \
                LLM_MEM_TRACE_OPT_EXPERT_SLACK=0 \
                LLM_MEM_TRACE_OPT_EXPERT_VALUE_GATE=0 \
                LLM_MEM_TRACE_OPT_EXPERT_CROSS_LAYER_PREDICT=0 \
                LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_STEPS=0
            ;;
        deadline_score)
            run_case deadline_score "$idx" "$order_position" \
                LLM_MEM_TRACE_OS_HINTS=1 \
                LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=1 \
                LLM_MEM_TRACE_OPT_EXPERT_POLICY=route \
                LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK=0 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC=1 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE=131072 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS=4 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY=1 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE=deadline_score \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP=0 \
                LLM_MEM_TRACE_OPT_EXPERT_COALESCE=0 \
                LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=off \
                LLM_MEM_TRACE_OPT_EXPERT_FEEDBACK=0 \
                LLM_MEM_TRACE_OPT_EXPERT_SLACK=0 \
                LLM_MEM_TRACE_OPT_EXPERT_VALUE_GATE=0 \
                LLM_MEM_TRACE_OPT_EXPERT_CROSS_LAYER_PREDICT=0 \
                LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_STEPS=0
            ;;
        decode_ttl1)
            run_case decode_ttl1 "$idx" "$order_position" \
                LLM_MEM_TRACE_OS_HINTS=1 \
                LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=1 \
                LLM_MEM_TRACE_OPT_EXPERT_POLICY=route \
                LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK=0 \
                LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_STEPS=0 \
                LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_PREFILL_STEPS=0 \
                LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_DECODE_STEPS=1 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC=1 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE=131072 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS=4 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY=1 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE=deadline_score \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP=0 \
                LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=off \
                LLM_MEM_TRACE_OPT_EXPERT_FEEDBACK=0 \
                LLM_MEM_TRACE_OPT_EXPERT_SLACK=0 \
                LLM_MEM_TRACE_OPT_EXPERT_VALUE_GATE=0 \
                LLM_MEM_TRACE_OPT_EXPERT_CROSS_LAYER_PREDICT=0 \
                LLM_MEM_TRACE_OPT_EXPERT_COALESCE=0
            ;;
        feedback_slack|feedback_slack_predict)
            local controller_profile=feedback_slack
            local predict_enabled=0
            if [ "$group" = "feedback_slack_predict" ]; then
                controller_profile=feedback_slack_predict
                predict_enabled=1
            fi
            run_case "$group" "$idx" "$order_position" \
                LLM_MEM_TRACE_OS_HINTS=1 \
                LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=1 \
                LLM_MEM_TRACE_OPT_EXPERT_POLICY=route \
                LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK=0 \
                LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER="$controller_profile" \
                LLM_MEM_TRACE_OPT_EXPERT_FEEDBACK=1 \
                LLM_MEM_TRACE_OPT_EXPERT_SLACK=1 \
                LLM_MEM_TRACE_OPT_EXPERT_VALUE_GATE=1 \
                LLM_MEM_TRACE_OPT_EXPERT_CROSS_LAYER_PREDICT="$predict_enabled" \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC=1 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE=131072 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS=4 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY=1 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE=deadline_score \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP=1 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH=8 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_WAIT_US=100 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_COALESCE=1 \
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_FALLBACK=0 \
                LLM_MEM_TRACE_OPT_EXPERT_PREDICT_TOPK=2 \
                LLM_MEM_TRACE_OPT_EXPERT_PREDICT_MIN_SAMPLES=8 \
                LLM_MEM_TRACE_OPT_EXPERT_PREDICT_MIN_CONFIDENCE=0.10 \
                LLM_MEM_TRACE_OPT_EXPERT_COALESCE=0 \
                LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_STEPS=0
            ;;
        *)
            echo "ERROR: unknown case: $group" >&2
            exit 1
            ;;
    esac
}

summarize_matrix() {
    local cmd=(
        python3 "$SCRIPT_DIR/summarize_repeat_runs.py"
        --base-dir "$TRACE_BASE_DIR"
        --baseline-group baseline
    )
    local group
    for group in "${CASES[@]}"; do
        cmd+=(--group "$group=$(join_runs "$group")")
    done
    cmd+=(--output-dir "$OUTPUT_DIR")

    printf '[SUMMARY]\n'
    printf '      '
    printf '%q ' "${cmd[@]}"
    printf '\n'

    if [ "$EXECUTE" = "1" ]; then
        "${cmd[@]}"
    fi
}

echo "=============================================="
echo "  Finalist Repeat Matrix"
echo "=============================================="
echo "Mode: $([ "$EXECUTE" = "1" ] && echo execute || echo dry-run)"
echo "Run prefix: $RUN_PREFIX"
echo "Repeat count: $REPEAT_COUNT"
echo "Trace base: $TRACE_BASE_DIR"
echo "Trace profile: $TRACE_PROFILE"
echo "Cache mode: $CACHE_MODE"
echo "Cases: $CASES_CSV"
echo "Order mode: $ORDER_MODE (seed=$ORDER_SEED)"
echo "Memory max: ${MEMORY_MAX:-unlimited by this script}"
echo "Memory+swap max: ${MEMORY_SWAP_MAX:-unlimited by this script}"
echo ""

for i in $(seq 1 "$REPEAT_COUNT"); do
    shift_by=0
    if [ "$ORDER_MODE" = "latin" ]; then
        shift_by=$(( (ORDER_SEED + i - 1) % ${#CASES[@]} ))
    fi
    for position in $(seq 0 $((${#CASES[@]} - 1))); do
        case_index=$(( (position + shift_by) % ${#CASES[@]} ))
        run_named_case "${CASES[$case_index]}" "$i" "$((position + 1))"
    done
done

summarize_matrix

if [ "$EXECUTE" != "1" ]; then
    echo ""
    echo "Dry-run complete. Set RUN_REPEAT_MATRIX_EXECUTE=1 to run these commands."
fi
