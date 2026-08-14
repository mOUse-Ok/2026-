#!/usr/bin/env bash
#
# Three-group completion experiment for Scenario A (memory constrained).
# It intentionally reports a negative result when Plain also completes.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRACE_BASE_DIR="${TRACE_BASE_DIR:-$PROJECT_DIR/trace_output/scenario_a}"
RUN_PREFIX="${RUN_PREFIX:-scenario_a}"
MEMORY_MAX="${MEMORY_MAX:-512M}"
MEMORY_SWAP_MAX="${MEMORY_SWAP_MAX:-0}"
MODEL_FILE="${MODEL_FILE:-$PROJECT_DIR/../models/Qwen3.5-35B-A3B-Q3_K_M.gguf}"
TRACE_LLAMA_CLI="${TRACE_LLAMA_CLI:-$PROJECT_DIR/build/bin/llama-cli}"
PLAIN_LLAMA_CLI="${PLAIN_LLAMA_CLI:-$PROJECT_DIR/build-plain/bin/llama-cli}"
NUM_TOKENS_PREDICT="${NUM_TOKENS_PREDICT:-80}"
NUM_THREADS="${NUM_THREADS:-8}"
ALLOW_DIRTY_REPO="${ALLOW_DIRTY_REPO:-0}"
WORKING_SET_MB="${WORKING_SET_MB:-128}"
RECLAIM_MAX_MB_PER_STEP="${RECLAIM_MAX_MB_PER_STEP:-64}"

if ! command -v systemd-run >/dev/null 2>&1; then
    echo "ERROR: systemd-run is required for the scoped memory limit" >&2
    exit 1
fi
for f in "$MODEL_FILE" "$TRACE_LLAMA_CLI" "$PLAIN_LLAMA_CLI"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file is missing: $f" >&2
        exit 1
    fi
done
if [ ! -x "$TRACE_LLAMA_CLI" ] || [ ! -x "$PLAIN_LLAMA_CLI" ]; then
    echo "ERROR: both llama-cli binaries must be executable" >&2
    exit 1
fi

TRACE_BASE_DIR="$(realpath -m "$TRACE_BASE_DIR")"
for group in plain survival_static survival_reclaim; do
    if [ -e "$TRACE_BASE_DIR/${RUN_PREFIX}_${group}" ]; then
        echo "ERROR: refusing to overwrite existing result: $TRACE_BASE_DIR/${RUN_PREFIX}_${group}" >&2
        exit 1
    fi
done
mkdir -p "$TRACE_BASE_DIR"

run_scope() {
    local group="$1"
    shift
    local run_dir="$TRACE_BASE_DIR/${RUN_PREFIX}_${group}"
    local status
    set +e
    systemd-run --user --scope --quiet \
        -p "MemoryMax=$MEMORY_MAX" \
        -p "MemorySwapMax=$MEMORY_SWAP_MAX" \
        -p "OOMPolicy=continue" \
        -- env \
            "MEMORY_MAX=$MEMORY_MAX" \
            "MEMORY_SWAP_MAX=$MEMORY_SWAP_MAX" \
            "ALLOW_DIRTY_REPO=$ALLOW_DIRTY_REPO" \
            "$@"
    status=$?
    set -e
    mkdir -p "$run_dir"
    printf '%s\n' "$status" > "$run_dir/launch_status.txt"
    echo "[RESULT] $group: scope exit=$status"
}

echo "=============================================="
echo "  Scenario A: constrained-memory completion"
echo "=============================================="
echo "MemoryMax=$MEMORY_MAX, MemorySwapMax=$MEMORY_SWAP_MAX"
echo "Tokens=$NUM_TOKENS_PREDICT, threads=$NUM_THREADS"
echo "Output=$TRACE_BASE_DIR"

# First run also materializes the canonical prompt consumed by the true Plain
# binary.  Every case uses cold cache preparation inside its own scope.
run_scope survival_static \
    "TRACE_BASE_DIR=$TRACE_BASE_DIR" \
    "RUN_NAME=${RUN_PREFIX}_survival_static" \
    "MODEL_FILE=$MODEL_FILE" \
    "NUM_TOKENS_PREDICT=$NUM_TOKENS_PREDICT" \
    "NUM_THREADS=$NUM_THREADS" \
    "TRACE_PROFILE=control" \
    "CACHE_MODE=cold" \
    "LLM_MEM_TRACE_OPT_EXPERT_PROFILE=survival" \
    "LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=off" \
    "LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=0" \
    "LLM_MEM_TRACE_OPT_EXPERT_MADV_COLD_RECLAIM=0" \
    "LLM_MEM_TRACE_OPT_EXPERT_MADV_DONTNEED_RECLAIM=0" \
    "LLM_MEM_TRACE_OPT_EXPERT_WORKING_SET_MB=0" \
    "LLM_MEM_TRACE_INFERENCE_OOM_SCORE_ADJ=1000" \
    bash "$SCRIPT_DIR/run_trace_pipeline.sh"

run_scope plain \
    "RUN_DIR=$TRACE_BASE_DIR/${RUN_PREFIX}_plain" \
    "SCENARIO_A_BASE_DIR=$TRACE_BASE_DIR" \
    "RUN_NAME=${RUN_PREFIX}_plain" \
    "MODEL_FILE=$MODEL_FILE" \
    "PROMPT_FILE_INPUT=$TRACE_BASE_DIR/${RUN_PREFIX}_survival_static/test_prompt.txt" \
    "LLAMA_CLI=$PLAIN_LLAMA_CLI" \
    "INFERENCE_OOM_SCORE_ADJ=1000" \
    "NUM_TOKENS_PREDICT=$NUM_TOKENS_PREDICT" \
    "NUM_THREADS=$NUM_THREADS" \
    bash "$SCRIPT_DIR/run_scenario_a_plain_case.sh"

run_scope survival_reclaim \
    "TRACE_BASE_DIR=$TRACE_BASE_DIR" \
    "RUN_NAME=${RUN_PREFIX}_survival_reclaim" \
    "MODEL_FILE=$MODEL_FILE" \
    "NUM_TOKENS_PREDICT=$NUM_TOKENS_PREDICT" \
    "NUM_THREADS=$NUM_THREADS" \
    "TRACE_PROFILE=control" \
    "CACHE_MODE=cold" \
    "LLM_MEM_TRACE_OPT_EXPERT_PROFILE=survival" \
    "LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER=off" \
    "LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=0" \
    "LLM_MEM_TRACE_OPT_EXPERT_MADV_COLD_RECLAIM=0" \
    "LLM_MEM_TRACE_OPT_EXPERT_MEMORY_OBJECTS=1" \
    "LLM_MEM_TRACE_OPT_EXPERT_WORKING_SET_MB=$WORKING_SET_MB" \
    "LLM_MEM_TRACE_OPT_EXPERT_MADV_DONTNEED_RECLAIM=1" \
    "LLM_MEM_TRACE_OPT_EXPERT_RECLAIM_GRACE_STEPS=3" \
    "LLM_MEM_TRACE_OPT_EXPERT_RECLAIM_MAX_MB_PER_STEP=$RECLAIM_MAX_MB_PER_STEP" \
    "LLM_MEM_TRACE_OPT_EXPERT_RECLAIM_MEMORY_RATIO_PCT=85" \
    "LLM_MEM_TRACE_OPT_EXPERT_RECLAIM_REFAULT_DELTA=1024" \
    "LLM_MEM_TRACE_INFERENCE_OOM_SCORE_ADJ=1000" \
    bash "$SCRIPT_DIR/run_trace_pipeline.sh"

REPORT_DIR="$TRACE_BASE_DIR/${RUN_PREFIX}_report"
if [ -e "$REPORT_DIR" ]; then
    echo "ERROR: report directory already exists: $REPORT_DIR" >&2
    exit 1
fi
python3 "$SCRIPT_DIR/summarize_scenario_a.py" \
    --base-dir "$TRACE_BASE_DIR" \
    --run-prefix "$RUN_PREFIX" \
    --output-dir "$REPORT_DIR"
echo "Report: $REPORT_DIR/SCENARIO_A_REPORT.md"
