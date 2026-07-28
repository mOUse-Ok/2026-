#!/bin/bash
# M5A Pressure Shadow runner. Default mode is a dry-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRACE_BASE_DIR="${TRACE_BASE_DIR:-$PROJECT_DIR/trace_output}"
RUN_PREFIX="${RUN_PREFIX:-m5a_pressure_shadow}"
OUTPUT_DIR="${OUTPUT_DIR:-$TRACE_BASE_DIR/pressure_shadow/evidence_summary}"
EXECUTE="${RUN_PRESSURE_SHADOW_EXECUTE:-0}"
FORMAL="${PRESSURE_SHADOW_FORMAL:-0}"
ALLOW_DIRTY_REPO="${ALLOW_DIRTY_REPO:-0}"
WORKER_COUNTS="${WORKER_COUNTS:-2 4}"
MEMORY_CASES="${PRESSURE_MEMORY_CASES:-8G:2G}"
NUM_TOKENS_PREDICT="${NUM_TOKENS_PREDICT:-80}"
CACHE_MODE="${CACHE_MODE:-cold}"
SAMPLE_MS="${PRESSURE_SHADOW_SAMPLE_MS:-25}"
PSS_SAMPLE_MS="${PRESSURE_SHADOW_PSS_SAMPLE_MS:-2000}"
ORDER_SEED_VALUE="${ORDER_SEED:-260726}"
MODEL_FILE_OVERRIDE="${MODEL_FILE:-}"
LLAMA_CLI_OVERRIDE="${LLAMA_CLI:-}"
EQUIVALENCE_REPEATS="${EQUIVALENCE_REPEATS:-1}"
EVIDENCE_REPEATS="${EVIDENCE_REPEATS:-$([ "$FORMAL" = "1" ] && echo 8 || echo 1)}"
OVERHEAD_REPEATS="${OVERHEAD_REPEATS:-$([ "$FORMAL" = "1" ] && echo 8 || echo 1)}"

positive_integer() {
    [[ "$1" =~ ^[0-9]+$ ]] && [ "$1" -gt 0 ]
}

for value in "$SAMPLE_MS" "$PSS_SAMPLE_MS" "$EQUIVALENCE_REPEATS" \
        "$EVIDENCE_REPEATS" "$OVERHEAD_REPEATS"; do
    if ! positive_integer "$value"; then
        echo "ERROR: sample intervals and repeat counts must be positive integers" >&2
        exit 1
    fi
done
if [ "$SAMPLE_MS" -lt 10 ] || [ "$SAMPLE_MS" -gt 50 ]; then
    echo "ERROR: PRESSURE_SHADOW_SAMPLE_MS must be in 10..50" >&2
    exit 1
fi
overhead_off_args=()
overhead_summary_args=()
for workers in $WORKER_COUNTS; do
    if ! positive_integer "$workers"; then
        echo "ERROR: WORKER_COUNTS must contain positive integers" >&2
        exit 1
    fi
done

memory_case_count=0
for memory_case in $MEMORY_CASES; do
    if [[ "$memory_case" != *:* ]]; then
        echo "ERROR: PRESSURE_MEMORY_CASES entries must be memory_max:memory_swap_max" >&2
        exit 1
    fi
    memory_case_count=$((memory_case_count + 1))
done
if [ "$FORMAL" = "1" ]; then
    if [ "$ALLOW_DIRTY_REPO" = "1" ]; then
        echo "ERROR: formal mode refuses ALLOW_DIRTY_REPO=1" >&2
        exit 1
    fi
    if [ "$EVIDENCE_REPEATS" -lt 8 ] || [ "$OVERHEAD_REPEATS" -lt 8 ]; then
        echo "ERROR: formal mode requires EVIDENCE_REPEATS>=8 and OVERHEAD_REPEATS>=8" >&2
        exit 1
    fi
    if [ "$memory_case_count" -lt 2 ]; then
        echo "ERROR: formal mode requires a verified safe baseline and a separately smoke-tested tighter memory case" >&2
        exit 1
    fi
fi
if [ "$EXECUTE" = "1" ]; then
    if ! command -v systemd-run >/dev/null 2>&1; then
        echo "ERROR: delegated systemd-run --user --scope is required" >&2
        exit 1
    fi
    if ! systemctl --user is-system-running >/dev/null 2>&1; then
        echo "ERROR: systemd user manager is unavailable" >&2
        exit 1
    fi
fi

run_one() {
    local run_name="$1"
    local pressure_mode="$2"
    local trace_profile="$3"
    local task_mode="$4"
    local repeat_index="$5"
    local order_position="$6"
    local workers="$7"
    local memory_max="$8"
    local memory_swap_max="$9"
    local order_mode="${10:-alternating}"
    local cmd=(
        env
        "TRACE_BASE_DIR=$TRACE_BASE_DIR"
        "RUN_NAME=$run_name"
        "LLM_MEM_TRACE_RUN_ID=$run_name"
        "NUM_TOKENS_PREDICT=$NUM_TOKENS_PREDICT"
        "TRACE_PROFILE=$trace_profile"
        "CACHE_MODE=$CACHE_MODE"
        "REPEAT_INDEX=$repeat_index"
        "ORDER_POSITION=$order_position"
        "ORDER_MODE=$order_mode"
        "ORDER_SEED=$ORDER_SEED_VALUE"
        "MEMORY_MAX=$memory_max"
        "MEMORY_SWAP_MAX=$memory_swap_max"
        "ALLOW_DIRTY_REPO=$ALLOW_DIRTY_REPO"
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
        LLM_MEM_TRACE_OPT_EXPERT_SLACK_MODE=off
        LLM_MEM_TRACE_OPT_EXPERT_VALUE_GATE=0
        LLM_MEM_TRACE_OPT_EXPERT_CROSS_LAYER_PREDICT=0
        LLM_MEM_TRACE_OPT_EXPERT_DEADLINE_OBSERVE=1
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
        "LLM_MEM_TRACE_PRESSURE_SHADOW_MODE=$pressure_mode"
        "LLM_MEM_TRACE_PRESSURE_SHADOW_SAMPLE_MS=$SAMPLE_MS"
        "LLM_MEM_TRACE_PRESSURE_SHADOW_PSS_SAMPLE_MS=$PSS_SAMPLE_MS"
    )
    if [ -n "$MODEL_FILE_OVERRIDE" ]; then
        cmd+=("MODEL_FILE=$MODEL_FILE_OVERRIDE")
    fi
    if [ -n "$LLAMA_CLI_OVERRIDE" ]; then
        cmd+=("LLAMA_CLI=$LLAMA_CLI_OVERRIDE")
    fi
    cmd+=(bash "$SCRIPT_DIR/run_trace_pipeline.sh")
    local launch=(
        systemd-run
        --user
        --scope
        --quiet
        -p "MemoryMax=$memory_max"
        -p "MemorySwapMax=$memory_swap_max"
        --
        "${cmd[@]}"
    )
    printf '[RUN] %s\n      ' "$run_name"
    printf '%q ' "${launch[@]}"
    printf '\n'
    if [ "$EXECUTE" = "1" ]; then
        "${launch[@]}"
    fi
}

echo "=============================================="
echo "  M5A Pressure Shadow Evidence"
echo "=============================================="
echo "Mode: $([ "$EXECUTE" = "1" ] && echo execute || echo dry-run)"
echo "Evidence class: $([ "$FORMAL" = "1" ] && echo formal || echo engineering/informal)"
echo "Comparator: deadline_score (unchanged)"
echo "Active gates: controller=off feedback=0 active_slack=0 value=0 cross_layer_predict=0"
echo "Workers: $WORKER_COUNTS"
echo "Memory cases: $MEMORY_CASES"
echo "Sample cadence: ${SAMPLE_MS}ms; PSS cadence: ${PSS_SAMPLE_MS}ms"
echo "Repeats: equivalence=$EQUIVALENCE_REPEATS evidence=$EVIDENCE_REPEATS overhead=$OVERHEAD_REPEATS"
echo ""

first_memory_case="${MEMORY_CASES%% *}"
baseline_memory="${first_memory_case%%:*}"
baseline_swap="${first_memory_case#*:}"
equivalence_args=()
for workers in $WORKER_COUNTS; do
    for index in $(seq 1 "$EQUIVALENCE_REPEATS"); do
        if [ $((index % 2)) -eq 1 ]; then
            modes=(off summary)
        else
            modes=(summary off)
        fi
        position=0
        for pressure_mode in "${modes[@]}"; do
            position=$((position + 1))
            run_one \
                "${RUN_PREFIX}_equiv_w${workers}_${pressure_mode}_r${index}" \
                "$pressure_mode" evidence detail "$index" "$position" "$workers" \
                "$baseline_memory" "$baseline_swap"
        done
        equivalence_args+=(
            "$OUTPUT_DIR/equivalence_w${workers}_r${index}.json"
        )
    done
done

detail_args=()
evidence_cases=()
for memory_case in $MEMORY_CASES; do
    memory_max="${memory_case%%:*}"
    memory_swap_max="${memory_case#*:}"
    memory_tag="$(printf '%s_%s' "$memory_max" "$memory_swap_max" | tr -c '[:alnum:]' '_')"
    for workers in $WORKER_COUNTS; do
        evidence_cases+=("$memory_max|$memory_swap_max|$memory_tag|$workers")
    done
done
evidence_case_count="${#evidence_cases[@]}"
for index in $(seq 1 "$EVIDENCE_REPEATS"); do
    offset=$(((index - 1) % evidence_case_count))
    for position in $(seq 1 "$evidence_case_count"); do
        case_index=$(((offset + position - 1) % evidence_case_count))
        IFS='|' read -r memory_max memory_swap_max memory_tag workers \
            <<< "${evidence_cases[$case_index]}"
        run_name="${RUN_PREFIX}_evidence_${memory_tag}_w${workers}_detail_r${index}"
        run_one "$run_name" detail evidence detail "$index" "$position" "$workers" \
            "$memory_max" "$memory_swap_max" cyclic_rotation
        detail_args+=(--run-dir "$TRACE_BASE_DIR/$run_name")
    done
done

overhead_off_args=()
overhead_summary_args=()
for workers in $WORKER_COUNTS; do
    for index in $(seq 1 "$OVERHEAD_REPEATS"); do
        if [ $((index % 2)) -eq 1 ]; then
            modes=(off summary)
        else
            modes=(summary off)
        fi
        position=0
        for pressure_mode in "${modes[@]}"; do
            position=$((position + 1))
            run_one \
                "${RUN_PREFIX}_overhead_w${workers}_${pressure_mode}_r${index}" \
                "$pressure_mode" benchmark summary "$index" "$position" "$workers" \
                "$baseline_memory" "$baseline_swap"
        done
        overhead_off_args+=(
            --overhead-off
            "$TRACE_BASE_DIR/${RUN_PREFIX}_overhead_w${workers}_off_r${index}"
        )
        overhead_summary_args+=(
            --overhead-summary
            "$TRACE_BASE_DIR/${RUN_PREFIX}_overhead_w${workers}_summary_r${index}"
        )
    done
done

if [ "$EXECUTE" = "1" ]; then
    mkdir -p "$OUTPUT_DIR"
    for workers in $WORKER_COUNTS; do
        for index in $(seq 1 "$EQUIVALENCE_REPEATS"); do
            python3 "$SCRIPT_DIR/validate_pressure_shadow_equivalence.py" \
                --off "$TRACE_BASE_DIR/${RUN_PREFIX}_equiv_w${workers}_off_r${index}" \
                --summary "$TRACE_BASE_DIR/${RUN_PREFIX}_equiv_w${workers}_summary_r${index}" \
                --output "$OUTPUT_DIR/equivalence_w${workers}_r${index}.json"
        done
    done
    python3 "$SCRIPT_DIR/pressure_shadow_analysis.py" \
        "${detail_args[@]}" \
        --strict \
        --output-dir "$OUTPUT_DIR"
    summary_args=()
    for equivalence_path in "${equivalence_args[@]}"; do
        summary_args+=(--equivalence "$equivalence_path")
    done
    python3 "$SCRIPT_DIR/summarize_pressure_shadow_results.py" \
        --analysis "$OUTPUT_DIR/M5A_pressure_shadow_full.json" \
        --candidate "$OUTPUT_DIR/M5A_pressure_state_candidates.json" \
        --output-dir "$OUTPUT_DIR" \
        "${summary_args[@]}" \
        "${overhead_off_args[@]}" \
        "${overhead_summary_args[@]}"
    echo "M5A Pressure Shadow evidence written to $OUTPUT_DIR"
else
    echo ""
    echo "Dry-run complete. Set RUN_PRESSURE_SHADOW_EXECUTE=1 to execute."
    echo "Formal mode also requires PRESSURE_SHADOW_FORMAL=1, a clean commit,"
    echo "and at least two independently smoke-tested PRESSURE_MEMORY_CASES."
fi
