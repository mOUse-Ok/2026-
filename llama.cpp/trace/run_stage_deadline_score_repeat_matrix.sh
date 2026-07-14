#!/bin/bash
# Compare legacy deadline_score with runtime stage_deadline_score without
# changing the finalist matrix. Default mode is a dry-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRACE_BASE_DIR="${TRACE_BASE_DIR:-$PROJECT_DIR/trace_output}"
RUN_PREFIX="${RUN_PREFIX:-stage_deadline_score}"
REPEAT_COUNT="${REPEAT_COUNT:-8}"
NUM_TOKENS_PREDICT="${NUM_TOKENS_PREDICT:-80}"
OUTPUT_DIR="${OUTPUT_DIR:-$TRACE_BASE_DIR/stage_deadline_score/repeat_summary}"
EXECUTE="${RUN_STAGE_PRIORITY_EXECUTE:-0}"
TRACE_PROFILE="${TRACE_PROFILE:-benchmark}"
CACHE_MODE="${CACHE_MODE:-cold}"
ORDER_SEED="${ORDER_SEED:-0}"
MEMORY_MAX="${MEMORY_MAX:-}"
MEMORY_SWAP_MAX="${MEMORY_SWAP_MAX:-}"
MODES=(deadline_score stage_deadline_score)

if ! [[ "$REPEAT_COUNT" =~ ^[0-9]+$ ]] || [ "$REPEAT_COUNT" -le 0 ]; then
    echo "ERROR: REPEAT_COUNT must be a positive integer" >&2
    exit 1
fi
if ! [[ "$ORDER_SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: ORDER_SEED must be a non-negative integer" >&2
    exit 1
fi
if [ -n "$MEMORY_SWAP_MAX" ] && [ -z "$MEMORY_MAX" ]; then
    echo "ERROR: MEMORY_SWAP_MAX requires MEMORY_MAX" >&2
    exit 1
fi
if [ "$EXECUTE" = "1" ] && [ -n "$MEMORY_MAX" ] && ! command -v systemd-run >/dev/null 2>&1; then
    echo "ERROR: MEMORY_MAX requires systemd-run" >&2
    exit 1
fi

join_runs() {
    local mode="$1"
    local result=""
    local i
    for i in $(seq 1 "$REPEAT_COUNT"); do
        [ -z "$result" ] || result+=","
        result+="${RUN_PREFIX}_${mode}_r${i}"
    done
    echo "$result"
}

run_one() {
    local mode="$1"
    local index="$2"
    local position="$3"
    local run_name="${RUN_PREFIX}_${mode}_r${index}"
    local cmd=(
        env
        "TRACE_BASE_DIR=$TRACE_BASE_DIR"
        "RUN_NAME=$run_name"
        "NUM_TOKENS_PREDICT=$NUM_TOKENS_PREDICT"
        "TRACE_PROFILE=$TRACE_PROFILE"
        "CACHE_MODE=$CACHE_MODE"
        "REPEAT_INDEX=$index"
        "ORDER_POSITION=$position"
        "ORDER_MODE=latin"
        "ORDER_SEED=$ORDER_SEED"
        "MEMORY_MAX=$MEMORY_MAX"
        "MEMORY_SWAP_MAX=$MEMORY_SWAP_MAX"
        LLM_MEM_TRACE_OS_HINTS=1
        LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=1
        LLM_MEM_TRACE_OPT_EXPERT_POLICY=route
        LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK=0
        LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=off
        LLM_MEM_TRACE_OPT_EXPERT_FEEDBACK=0
        LLM_MEM_TRACE_OPT_EXPERT_SLACK=0
        LLM_MEM_TRACE_OPT_EXPERT_DEADLINE_OBSERVE=1
        LLM_MEM_TRACE_OPT_EXPERT_VALUE_GATE=0
        LLM_MEM_TRACE_OPT_EXPERT_CROSS_LAYER_PREDICT=0
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC=1
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE=131072
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS=4
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY=1
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE=$mode"
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP=1
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH=1
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_WAIT_US=0
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_COALESCE=0
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_FALLBACK=1
        LLM_MEM_TRACE_OPT_EXPERT_COALESCE=0
        LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_STEPS=0
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

    printf '[RUN] %s\n      ' "$run_name"
    printf '%q ' "${launch[@]}"
    printf '\n'
    if [ "$EXECUTE" = "1" ]; then
        "${launch[@]}"
    fi
}

echo "=============================================="
echo "  Stage Deadline Score A/B Matrix"
echo "=============================================="
echo "Mode: $([ "$EXECUTE" = "1" ] && echo execute || echo dry-run)"
echo "Repeat count: $REPEAT_COUNT"
echo "Cases: ${MODES[*]}"
echo "Top-K: all routed experts (PREFETCH_TOPK=0)"
echo "Admission: legacy only (Slack/feedback/value gates disabled)"
echo "Chunking: disabled (no chunk mechanism configured)"
echo "Finalist matrix: untouched"
echo ""

for i in $(seq 1 "$REPEAT_COUNT"); do
    shift_by=$(( (ORDER_SEED + i - 1) % ${#MODES[@]} ))
    for position in $(seq 0 $((${#MODES[@]} - 1))); do
        mode_index=$(( (position + shift_by) % ${#MODES[@]} ))
        run_one "${MODES[$mode_index]}" "$i" "$((position + 1))"
    done
done

summary=(
    python3 "$SCRIPT_DIR/summarize_repeat_runs.py"
    --base-dir "$TRACE_BASE_DIR"
    --baseline-group deadline_score
    --group "deadline_score=$(join_runs deadline_score)"
    --group "stage_deadline_score=$(join_runs stage_deadline_score)"
    --output-dir "$OUTPUT_DIR"
    --metric process_wall_time_s
    --metric decode_p95_latency_us
    --metric decode_throughput_tokens_per_s
    --metric total_major_faults
    --metric rss_peak_gb
    --metric expert_task_early_enqueued
    --metric expert_task_early_issued
    --metric expert_task_early_deadline_late_count
    --metric expert_task_early_queue_wait_avg_us
    --metric expert_task_early_queue_wait_max_us
    --metric expert_task_late_enqueued
    --metric expert_task_late_issued
    --metric expert_task_late_deadline_late_count
    --metric expert_task_late_queue_wait_avg_us
    --metric expert_task_late_queue_wait_max_us
    --metric expert_task_unknown_enqueued
    --metric expert_task_unknown_issued
    --metric expert_task_unknown_deadline_late_count
    --metric expert_task_unknown_queue_wait_avg_us
    --metric expert_task_unknown_queue_wait_max_us
)
printf '[SUMMARY]\n      '
printf '%q ' "${summary[@]}"
printf '\n'
if [ "$EXECUTE" = "1" ]; then
    "${summary[@]}"
else
    echo ""
    echo "Dry-run complete. Set RUN_STAGE_PRIORITY_EXECUTE=1 to execute."
fi
