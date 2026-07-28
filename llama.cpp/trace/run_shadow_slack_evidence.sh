#!/bin/bash
# M4A.1 evidence runner. Default mode is a dry-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRACE_BASE_DIR="${TRACE_BASE_DIR:-$PROJECT_DIR/trace_output}"
RUN_PREFIX="${RUN_PREFIX:-m4a1_shadow_slack}"
DETAIL_REPEATS="${DETAIL_REPEATS:-1}"
SUMMARY_REPEATS="${SUMMARY_REPEATS:-1}"
WORKER_COUNTS="${WORKER_COUNTS:-2 4}"
NUM_TOKENS_PREDICT="${NUM_TOKENS_PREDICT:-80}"
EXECUTE="${RUN_SHADOW_SLACK_EXECUTE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$TRACE_BASE_DIR/shadow_slack/evidence_summary}"
CACHE_MODE="${CACHE_MODE:-cold}"
MEMORY_MAX="${MEMORY_MAX:-}"
MEMORY_SWAP_MAX="${MEMORY_SWAP_MAX:-}"

for value in "$DETAIL_REPEATS" "$SUMMARY_REPEATS"; do
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -le 0 ]; then
        echo "ERROR: DETAIL_REPEATS and SUMMARY_REPEATS must be positive integers" >&2
        exit 1
    fi
done
for workers in $WORKER_COUNTS; do
    if ! [[ "$workers" =~ ^[0-9]+$ ]] || [ "$workers" -le 0 ]; then
        echo "ERROR: WORKER_COUNTS must contain positive integers" >&2
        exit 1
    fi
done
if [ -n "$MEMORY_SWAP_MAX" ] && [ -z "$MEMORY_MAX" ]; then
    echo "ERROR: MEMORY_SWAP_MAX requires MEMORY_MAX" >&2
    exit 1
fi
if [ "$EXECUTE" = "1" ] && [ -n "$MEMORY_MAX" ] && ! command -v systemd-run >/dev/null 2>&1; then
    echo "ERROR: MEMORY_MAX requires systemd-run" >&2
    exit 1
fi

run_one() {
    local run_name="$1"
    local shadow_mode="$2"
    local trace_profile="$3"
    local task_mode="$4"
    local repeat_index="$5"
    local order_position="$6"
    local workers="$7"
    local cmd=(
        env
        "TRACE_BASE_DIR=$TRACE_BASE_DIR"
        "RUN_NAME=$run_name"
        "NUM_TOKENS_PREDICT=$NUM_TOKENS_PREDICT"
        "TRACE_PROFILE=$trace_profile"
        "CACHE_MODE=$CACHE_MODE"
        "REPEAT_INDEX=$repeat_index"
        "ORDER_POSITION=$order_position"
        ORDER_MODE=alternating
        "MEMORY_MAX=$MEMORY_MAX"
        "MEMORY_SWAP_MAX=$MEMORY_SWAP_MAX"
        ALLOW_DIRTY_REPO=1
        LLM_MEM_TRACE_SMAPS=0
        LLM_MEM_TRACE_RESIDENCY=0
        LLM_MEM_TRACE_OS_HINTS=1
        "LLM_MEM_TRACE_EXPERT_TASK_MODE=$task_mode"
        LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=1
        LLM_MEM_TRACE_OPT_EXPERT_POLICY=route
        LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK=0
        LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=off
        LLM_MEM_TRACE_OPT_EXPERT_FEEDBACK=0
        LLM_MEM_TRACE_OPT_EXPERT_SLACK=0
        "LLM_MEM_TRACE_OPT_EXPERT_SLACK_MODE=$shadow_mode"
        LLM_MEM_TRACE_OPT_EXPERT_DEADLINE_OBSERVE=1
        LLM_MEM_TRACE_OPT_EXPERT_VALUE_GATE=0
        LLM_MEM_TRACE_OPT_EXPERT_CROSS_LAYER_PREDICT=0
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC=1
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE=131072
        "LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS=$workers"
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY=1
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE=deadline_score
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP=1
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH=1
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_WAIT_US=0
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_BATCH_COALESCE=0
        LLM_MEM_TRACE_OPT_EXPERT_ASYNC_FALLBACK=1
        LLM_MEM_TRACE_OPT_EXPERT_COALESCE=0
        LLM_MEM_TRACE_OPT_EXPERT_TTL_STEPS=0
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
echo "  M4A.1 Aligned Shadow Slack Evidence"
echo "=============================================="
echo "Mode: $([ "$EXECUTE" = "1" ] && echo execute || echo dry-run)"
echo "Comparator: deadline_score (unchanged)"
echo "Active gates: feedback=0 active_slack=0 value=0 cross_layer_predict=0"
echo "Detail repeats: $DETAIL_REPEATS"
echo "Summary off/on repeats: $SUMMARY_REPEATS"
echo "Worker counts: $WORKER_COUNTS"
echo "Memory max: ${MEMORY_MAX:-unlimited by this script}"
echo "Memory swap max: ${MEMORY_SWAP_MAX:-unlimited by this script}"
echo ""

summary_off_args=()
summary_shadow_args=()
equivalence_args=()
for workers in $WORKER_COUNTS; do
    for index in $(seq 1 "$SUMMARY_REPEATS"); do
        if [ $((index % 2)) -eq 1 ]; then
            modes=(off shadow)
        else
            modes=(shadow off)
        fi
        position=0
        for mode in "${modes[@]}"; do
            position=$((position + 1))
            run_one "${RUN_PREFIX}_w${workers}_summary_${mode}_r${index}" \
                "$mode" benchmark summary "$index" "$position" "$workers"
        done
        summary_off_args+=(--summary-off \
            "$TRACE_BASE_DIR/${RUN_PREFIX}_w${workers}_summary_off_r${index}")
        summary_shadow_args+=(--summary-shadow \
            "$TRACE_BASE_DIR/${RUN_PREFIX}_w${workers}_summary_shadow_r${index}")
        equivalence_args+=(--equivalence \
            "$OUTPUT_DIR/equivalence_w${workers}_r${index}.json")
    done
done

detail_args=()
detail_report_args=()
for workers in $WORKER_COUNTS; do
    for index in $(seq 1 "$DETAIL_REPEATS"); do
        run_name="${RUN_PREFIX}_w${workers}_detail_shadow_r${index}"
        run_one "$run_name" shadow evidence detail "$index" 1 "$workers"
        detail_args+=(--run-dir "$TRACE_BASE_DIR/$run_name")
        detail_report_args+=(--detail-run-dir "$TRACE_BASE_DIR/$run_name")
    done
done

if [ "$EXECUTE" = "1" ]; then
    for workers in $WORKER_COUNTS; do
        for index in $(seq 1 "$SUMMARY_REPEATS"); do
            python3 "$SCRIPT_DIR/validate_shadow_slack_equivalence.py" \
                --off "$TRACE_BASE_DIR/${RUN_PREFIX}_w${workers}_summary_off_r${index}" \
                --shadow "$TRACE_BASE_DIR/${RUN_PREFIX}_w${workers}_summary_shadow_r${index}" \
                --detail "$TRACE_BASE_DIR/${RUN_PREFIX}_w${workers}_detail_shadow_r1" \
                --output "$OUTPUT_DIR/equivalence_w${workers}_r${index}.json"
        done
    done
    python3 "$SCRIPT_DIR/aggregate_shadow_slack_runs.py" \
        "${detail_args[@]}" \
        --output "$OUTPUT_DIR/detail_model_comparison.json"
    python3 "$SCRIPT_DIR/summarize_shadow_slack_results.py" \
        --input "$OUTPUT_DIR/detail_model_comparison.json" \
        --output "$OUTPUT_DIR/M4A1_shadow_slack_report.md" \
        --final-json "$OUTPUT_DIR/M4A1_shadow_slack_full.json" \
        "${detail_report_args[@]}" \
        "${summary_off_args[@]}" \
        "${summary_shadow_args[@]}" \
        "${equivalence_args[@]}"
    echo "M4A.1 evidence written to $OUTPUT_DIR"
else
    echo ""
    echo "Dry-run complete. Set RUN_SHADOW_SLACK_EXECUTE=1 to execute."
fi
