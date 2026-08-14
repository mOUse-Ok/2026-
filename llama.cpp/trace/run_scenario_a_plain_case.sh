#!/usr/bin/env bash
# Run one actual non-Trace llama-cli baseline inside an already-created scope.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$PROJECT_DIR")"

RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
SCENARIO_A_BASE_DIR="${SCENARIO_A_BASE_DIR:?SCENARIO_A_BASE_DIR is required}"
MODEL_FILE="${MODEL_FILE:?MODEL_FILE is required}"
PROMPT_FILE_INPUT="${PROMPT_FILE_INPUT:?PROMPT_FILE_INPUT is required}"
LLAMA_CLI="${LLAMA_CLI:?LLAMA_CLI is required}"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
MEMORY_MAX="${MEMORY_MAX:?MEMORY_MAX is required}"
MEMORY_SWAP_MAX="${MEMORY_SWAP_MAX:?MEMORY_SWAP_MAX is required}"
NUM_TOKENS_PREDICT="${NUM_TOKENS_PREDICT:-80}"
NUM_THREADS="${NUM_THREADS:-8}"
ALLOW_DIRTY_REPO="${ALLOW_DIRTY_REPO:-0}"
INFERENCE_OOM_SCORE_ADJ="${INFERENCE_OOM_SCORE_ADJ:-}"
OOM_SCORE_PREFIX=()
if [ -n "$INFERENCE_OOM_SCORE_ADJ" ]; then
    OOM_SCORE_PREFIX=("$SCRIPT_DIR/run_with_oom_score_adj.sh" "$INFERENCE_OOM_SCORE_ADJ")
fi

RUN_DIR="$(realpath -m "$RUN_DIR")"
SCENARIO_A_BASE_DIR="$(realpath -m "$SCENARIO_A_BASE_DIR")"
case "$RUN_DIR" in
    "$SCENARIO_A_BASE_DIR"/*) ;;
    *)
        echo "ERROR: RUN_DIR must be a child of SCENARIO_A_BASE_DIR" >&2
        exit 1
        ;;
esac
if [ -e "$RUN_DIR" ]; then
    echo "ERROR: refusing to overwrite existing run directory: $RUN_DIR" >&2
    exit 1
fi
if [ ! -f "$MODEL_FILE" ] || [ ! -f "$PROMPT_FILE_INPUT" ] || [ ! -x "$LLAMA_CLI" ]; then
    echo "ERROR: plain case prerequisite is missing" >&2
    exit 1
fi

mkdir -p "$RUN_DIR"
cp -- "$PROMPT_FILE_INPUT" "$RUN_DIR/test_prompt.txt"

MANIFEST_ARGS=(
    --output "$RUN_DIR/run_manifest.json"
    --project "$REPO_DIR"
    --model "$MODEL_FILE"
    --prompt "$RUN_DIR/test_prompt.txt"
    --llama-cli "$LLAMA_CLI"
    --run-name "$RUN_NAME"
    --trace-profile plain
    --cache-mode cold
    --expert-profile plain
    --cache-type-k f16
    --cache-type-v f16
    --flash-attn auto
    --batch-size 512
    --ubatch-size 512
    --ctx-size 2048
    --memory-max "$MEMORY_MAX"
    --memory-swap-max "$MEMORY_SWAP_MAX"
)
if [ "$ALLOW_DIRTY_REPO" != "1" ]; then
    MANIFEST_ARGS+=(--require-clean)
fi
python3 "$SCRIPT_DIR/write_run_manifest.py" "${MANIFEST_ARGS[@]}"
python3 "$SCRIPT_DIR/capture_cgroup_snapshot.py" \
    --output "$RUN_DIR/cgroup_before_inference.json" \
    --stage before_inference \
    --expected-memory-max "$MEMORY_MAX"
python3 "$SCRIPT_DIR/prepare_model_cache.py" \
    --model "$MODEL_FILE" --mode cold > "$RUN_DIR/cache_preparation.json"

set +e
/usr/bin/time -q \
    -f '{"wall_time_s":%e,"user_time_s":%U,"system_time_s":%S,"max_rss_kb":%M,"major_faults":%F,"minor_faults":%R,"file_inputs":%I,"file_outputs":%O,"exit_code":%x}' \
    -o "$RUN_DIR/process_metrics.json" \
    "${OOM_SCORE_PREFIX[@]}" "$LLAMA_CLI" \
    -m "$MODEL_FILE" \
    -f "$RUN_DIR/test_prompt.txt" \
    -n "$NUM_TOKENS_PREDICT" \
    -t "$NUM_THREADS" \
    -b 512 \
    -ub 512 \
    -c 2048 \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --flash-attn auto \
    --gpu-layers 0 \
    --temp 0 \
    --seed 1234 \
    --no-display-prompt \
    --simple-io \
    --single-turn \
    --no-warmup \
    --no-perf \
    --no-show-timings \
    > "$RUN_DIR/inference_output.txt" 2>"$RUN_DIR/inference_stderr.txt"
INFERENCE_STATUS=$?
set -e

python3 "$SCRIPT_DIR/capture_cgroup_snapshot.py" \
    --output "$RUN_DIR/cgroup_after_inference.json" \
    --stage after_inference \
    --expected-memory-max "$MEMORY_MAX"
if [ "$INFERENCE_STATUS" -ne 0 ]; then
    exit "$INFERENCE_STATUS"
fi
sha256sum "$RUN_DIR/inference_output.txt" | awk '{print $1}' > "$RUN_DIR/output.sha256"
