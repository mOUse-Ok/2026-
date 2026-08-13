#!/bin/bash
# ============================================================================
# LLM Memory Trace Automation Pipeline
# ============================================================================
# 1. Run llama-cli with MEM_TRACE enabled on a designed test case
# 2. Parse JSONL trace output
# 3. Generate visualizations and analysis report
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_DIR="$(dirname "$PROJECT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
MODEL_DIR="$PROJECT_DIR/../models"
TRACE_BASE_DIR="${TRACE_BASE_DIR:-$PROJECT_DIR/trace_output}"
RUN_NAME="${RUN_NAME:-latest}"
TRACE_OUT_DIR="${TRACE_OUT_DIR:-$TRACE_BASE_DIR/$RUN_NAME}"
MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/Qwen3.5-35B-A3B-Q3_K_M.gguf}"
ANALYSIS_SCRIPT="$SCRIPT_DIR/analyze_trace.py"
LLAMA_CLI="${LLAMA_CLI:-$BUILD_DIR/bin/llama-cli}"
TIME_BIN="${TIME_BIN:-/usr/bin/time}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/llm_mem_trace_matplotlib}"

# ------------------------------------------------------------------
# Test Case Design
# ------------------------------------------------------------------
# A carefully designed prompt to exercise diverse memory access patterns:
# - Multi-topic content to activate diverse MoE experts
# - Mixed short/long sentences for varied token generation
# - Enough context length to observe KV cache growth
# ------------------------------------------------------------------
TEST_PROMPT="Below is a comprehensive technical analysis of modern computer systems.

## 1. CPU Architecture
Modern CPUs employ superscalar execution with out-of-order instruction scheduling. The front-end fetches and decodes instructions into micro-operations, which are then dispatched to execution units. Key performance factors include branch prediction accuracy, cache hierarchy design (L1/L2/L3), and TLB coverage. Memory-level parallelism allows multiple outstanding cache misses, hiding DRAM latency.

## 2. Memory Hierarchy
The memory subsystem consists of registers (sub-nanosecond), L1 cache (~1ns, 32-64KB), L2 cache (~5ns, 256-512KB), L3 cache (~15ns, 8-32MB), main memory (~80ns, GBs), and storage (microseconds, TBs). Each level trades capacity for latency. Modern systems use prefetchers that observe access patterns and speculate to bring data closer to the CPU before it is explicitly requested.

## 3. GPU Computing
GPUs feature thousands of simpler cores organized into streaming multiprocessors. They excel at data-parallel workloads like matrix multiplication. The CUDA programming model exposes a hierarchy of threads, warps, blocks, and grids. Shared memory and registers are precious resources that determine occupancy and throughput.

## 4. Operating System Memory Management
The OS provides virtual memory through paging. Each process has its own address space, with page tables mapping virtual to physical addresses. Demand paging lazily allocates physical frames on first access. The page replacement algorithm (e.g., LRU approximation) determines which pages to evict under memory pressure. Transparent huge pages reduce TLB misses for large workloads.

## 5. Machine Learning Inference
Large language models process tokens sequentially through transformer layers. Each token attends to all previous tokens via the self-attention mechanism, whose memory complexity is O(n^2) for context length n. The KV cache stores key-value pairs to avoid recomputation during autoregressive decoding. Mixture-of-Experts models route each token to a subset of specialized feed-forward networks, trading memory for compute efficiency.

Question: Based on the above analysis, what are the three most critical memory bottlenecks in LLM inference, and how would you optimize each one?"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
# A profile selects only startup-time inference parameters.  KV type, context
# capacity and ubatch allocation are fixed when llama_context is created, so
# they must not be changed by an in-flight pressure controller.
#
# "custom" preserves the historical defaults and lets callers supply every
# parameter directly.  Explicit BATCH_SIZE/UBATCH_SIZE/CTX_SIZE/
# KV_CACHE_TYPE_K/KV_CACHE_TYPE_V/FLASH_ATTN values always override a profile.
EXPERT_PROFILE="${LLM_MEM_TRACE_OPT_EXPERT_PROFILE:-custom}"
case "$EXPERT_PROFILE" in
    custom)
        PROFILE_CACHE_TYPE_K=f16
        PROFILE_CACHE_TYPE_V=f16
        PROFILE_BATCH_SIZE=512
        PROFILE_UBATCH_SIZE=512
        PROFILE_CTX_SIZE=2048
        PROFILE_FLASH_ATTN=auto
        ;;
    survival)
        # 1024 is deliberately larger than the old 512 proposal: the bundled
        # prompt plus generation budget can exceed 512 model tokens.
        PROFILE_CACHE_TYPE_K=q8_0
        PROFILE_CACHE_TYPE_V=f16
        PROFILE_BATCH_SIZE=512
        PROFILE_UBATCH_SIZE=64
        PROFILE_CTX_SIZE=1024
        PROFILE_FLASH_ATTN=auto
        ;;
    balanced)
        PROFILE_CACHE_TYPE_K=q8_0
        PROFILE_CACHE_TYPE_V=f16
        PROFILE_BATCH_SIZE=512
        PROFILE_UBATCH_SIZE=128
        PROFILE_CTX_SIZE=1536
        PROFILE_FLASH_ATTN=auto
        ;;
    performance)
        PROFILE_CACHE_TYPE_K=f16
        PROFILE_CACHE_TYPE_V=f16
        PROFILE_BATCH_SIZE=512
        PROFILE_UBATCH_SIZE=512
        PROFILE_CTX_SIZE=2048
        PROFILE_FLASH_ATTN=auto
        ;;
    *)
        echo "ERROR: LLM_MEM_TRACE_OPT_EXPERT_PROFILE must be custom, survival, balanced, or performance" >&2
        exit 1
        ;;
esac

NUM_TOKENS_PREDICT="${NUM_TOKENS_PREDICT:-80}"  # Decode enough tokens to observe decode behavior
NUM_THREADS="${NUM_THREADS:-8}"                 # CPU threads
BATCH_SIZE="${BATCH_SIZE:-$PROFILE_BATCH_SIZE}" # Logical batch size for prefill
UBATCH_SIZE="${UBATCH_SIZE:-$PROFILE_UBATCH_SIZE}" # Physical batch / compute buffer size
CTX_SIZE="${CTX_SIZE:-$PROFILE_CTX_SIZE}"       # Context window size
KV_CACHE_TYPE_K="${KV_CACHE_TYPE_K:-$PROFILE_CACHE_TYPE_K}"
KV_CACHE_TYPE_V="${KV_CACHE_TYPE_V:-$PROFILE_CACHE_TYPE_V}"
FLASH_ATTN="${FLASH_ATTN:-$PROFILE_FLASH_ATTN}"
TEMP="${TEMP:-0.0}"                             # Deterministic output for reproducibility
SEED="${SEED:-1234}"                            # Fix sampler RNG
GPU_LAYERS="${GPU_LAYERS:-0}"                  # Route tracing requires CPU-resident experts
CACHE_MODE="${CACHE_MODE:-cold}"                # cold, warm, or as-is
TRACE_PROFILE="${TRACE_PROFILE:-evidence}"      # evidence, benchmark, control, attribution, 5a, or custom
ALLOW_DROPPED_EVENTS="${ALLOW_DROPPED_EVENTS:-0}"
ALLOW_DIRTY_REPO="${ALLOW_DIRTY_REPO:-0}"
EXPERT_CONTROLLER="${LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER:-off}"

case "$EXPERT_CONTROLLER" in
    off)
        CONTROLLER_FEEDBACK=0
        CONTROLLER_VALUE_GATE=0
        CONTROLLER_ASYNC=0
        CONTROLLER_PRIORITY=0
        CONTROLLER_PRIORITY_HEAP=0
        CONTROLLER_PRIORITY_MODE=score
        CONTROLLER_DEADLINE_OBSERVE=0
        CONTROLLER_WORKERS=1
        CONTROLLER_OS_HINTS=0
        CONTROLLER_PREFETCH=0
        ;;
    expert_prefetch)
        CONTROLLER_FEEDBACK=1
        CONTROLLER_VALUE_GATE=1
        CONTROLLER_ASYNC=1
        CONTROLLER_PRIORITY=1
        CONTROLLER_PRIORITY_HEAP=1
        CONTROLLER_PRIORITY_MODE=deadline_score
        CONTROLLER_DEADLINE_OBSERVE=1
        CONTROLLER_WORKERS=4
        CONTROLLER_OS_HINTS=1
        CONTROLLER_PREFETCH=1
        ;;
    *)
        echo "ERROR: LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER must be off or expert_prefetch" >&2
        exit 1
        ;;
esac

PROFILE_CONTROL_ONLY=0
case "$TRACE_PROFILE" in
    evidence)
        PROFILE_TENSOR=1
        PROFILE_KV=1
        PROFILE_EXPERT=1
        PROFILE_MEMORY=1
        PROFILE_RESIDENCY=1
        PROFILE_ATTRIBUTION=0
        PROFILE_SMAPS=1
        PROFILE_EXPERT_TASK_MODE=detail
        ;;
    benchmark)
        PROFILE_TENSOR=0
        PROFILE_KV=0
        PROFILE_EXPERT=1
        PROFILE_MEMORY=1
        PROFILE_RESIDENCY=0
        PROFILE_ATTRIBUTION=0
        PROFILE_SMAPS=0
        PROFILE_EXPERT_TASK_MODE=summary
        ;;
    control)
        # Keep the MEMORY sink available to the controller, while its writer
        # accepts only process-end *_SUMMARY records.  Router parsing and
        # pressure feedback remain live; Task-1 evidence events do not.
        PROFILE_TENSOR=0
        PROFILE_KV=0
        PROFILE_EXPERT=0
        PROFILE_MEMORY=1
        PROFILE_RESIDENCY=0
        PROFILE_ATTRIBUTION=0
        PROFILE_SMAPS=0
        PROFILE_EXPERT_TASK_MODE=off
        PROFILE_CONTROL_ONLY=1
        ;;
    attribution)
        PROFILE_TENSOR=1
        PROFILE_KV=0
        PROFILE_EXPERT=0
        PROFILE_MEMORY=1
        PROFILE_RESIDENCY=1
        PROFILE_ATTRIBUTION=1
        PROFILE_SMAPS=0
        PROFILE_EXPERT_TASK_MODE=off
        ;;
    5a)
        PROFILE_TENSOR=0
        PROFILE_KV=0
        PROFILE_EXPERT=1
        PROFILE_MEMORY=1
        PROFILE_RESIDENCY=0
        PROFILE_ATTRIBUTION=0
        PROFILE_SMAPS=0
        PROFILE_EXPERT_TASK_MODE=detail
        ;;
    custom)
        PROFILE_TENSOR=1
        PROFILE_KV=1
        PROFILE_EXPERT=1
        PROFILE_MEMORY=1
        PROFILE_RESIDENCY=1
        PROFILE_ATTRIBUTION=0
        PROFILE_SMAPS=1
        PROFILE_EXPERT_TASK_MODE=detail
        ;;
    *)
        echo "ERROR: TRACE_PROFILE must be evidence, benchmark, control, attribution, 5a, or custom" >&2
        exit 1
        ;;
esac

# `control` is intentionally strict: allowing a caller to turn a raw sink
# back on would silently invalidate an overhead comparison labelled control.
if [ "$TRACE_PROFILE" = "control" ]; then
    if [ "${LLM_MEM_TRACE_TENSOR:-0}" != "0" ] ||
       [ "${LLM_MEM_TRACE_KV:-0}" != "0" ] ||
       [ "${LLM_MEM_TRACE_EXPERT:-0}" != "0" ] ||
       [ "${LLM_MEM_TRACE_MEMORY:-1}" = "0" ] ||
       [ "${LLM_MEM_TRACE_CONTROL_ONLY:-1}" = "0" ]; then
        echo "ERROR: TRACE_PROFILE=control requires TENSOR/KV/EXPERT=0, MEMORY=1, and CONTROL_ONLY=1" >&2
        exit 1
    fi
fi

case "$CACHE_MODE" in
    cold|warm|as-is) ;;
    *)
        echo "ERROR: CACHE_MODE must be cold, warm, or as-is" >&2
        exit 1
        ;;
esac

# ------------------------------------------------------------------
# Step 0: Check prerequisites
# ------------------------------------------------------------------
echo "=============================================="
echo "  LLM Memory Trace Pipeline"
echo "=============================================="
echo ""

if [ ! -f "$LLAMA_CLI" ]; then
    echo "ERROR: llama-cli not found at $LLAMA_CLI"
    echo "Please rebuild with: cd $BUILD_DIR && cmake .. -DLLAMA_MEM_TRACE=ON && make -j\$(nproc) llama-cli"
    exit 1
fi

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model not found at $MODEL_FILE"
    exit 1
fi

if [ ! -x "$TIME_BIN" ]; then
    echo "ERROR: GNU time not found at $TIME_BIN (install package: time)" >&2
    exit 1
fi

python3 -c 'import matplotlib, numpy, pandas' 2>/dev/null || {
    echo "ERROR: analysis dependencies are missing." >&2
    echo "Install with: python3 -m pip install -r $SCRIPT_DIR/requirements-analysis.txt" >&2
    exit 1
}

# ------------------------------------------------------------------
# Step 1: Clean previous trace output & run inference
# ------------------------------------------------------------------
echo "[1/4] Running LLM inference with memory tracing..."
echo "      Model: $(basename "$MODEL_FILE")"
echo "      Predict tokens: $NUM_TOKENS_PREDICT"
echo "      Trace profile: $TRACE_PROFILE"
echo "      Cache mode: $CACHE_MODE"
echo "      Expert controller: $EXPERT_CONTROLLER"
echo "      Expert startup profile: $EXPERT_PROFILE"
echo "      Startup memory config: K=$KV_CACHE_TYPE_K, V=$KV_CACHE_TYPE_V, ctx=$CTX_SIZE, batch=$BATCH_SIZE, ubatch=$UBATCH_SIZE, flash-attn=$FLASH_ATTN"
echo "      Output dir: $TRACE_OUT_DIR"
echo ""

TRACE_BASE_DIR="$(realpath -m "$TRACE_BASE_DIR")"
TRACE_OUT_DIR="$(realpath -m "$TRACE_OUT_DIR")"
case "$TRACE_OUT_DIR" in
    "$TRACE_BASE_DIR"/*) ;;
    *)
        echo "ERROR: TRACE_OUT_DIR must be a child of TRACE_BASE_DIR" >&2
        exit 1
        ;;
esac
if [ "$TRACE_OUT_DIR" = "/" ] || [ "$TRACE_OUT_DIR" = "$TRACE_BASE_DIR" ]; then
    echo "ERROR: refusing to replace unsafe trace output directory: $TRACE_OUT_DIR" >&2
    exit 1
fi
rm -rf -- "$TRACE_OUT_DIR"
mkdir -p "$TRACE_OUT_DIR"

export LLM_MEM_TRACE=1
export LLM_MEM_TRACE_DIR="$TRACE_OUT_DIR"
export LLM_MEM_TRACE_RUN_ID="${LLM_MEM_TRACE_RUN_ID:-$RUN_NAME}"
export TRACE_PROFILE CACHE_MODE
export NUM_TOKENS_PREDICT NUM_THREADS BATCH_SIZE UBATCH_SIZE CTX_SIZE TEMP SEED GPU_LAYERS
export EXPERT_PROFILE KV_CACHE_TYPE_K KV_CACHE_TYPE_V FLASH_ATTN
export LLM_MEM_TRACE_OPT_EXPERT_PROFILE="$EXPERT_PROFILE"
export LLM_MEM_TRACE_TENSOR="${LLM_MEM_TRACE_TENSOR:-$PROFILE_TENSOR}"
export LLM_MEM_TRACE_KV="${LLM_MEM_TRACE_KV:-$PROFILE_KV}"
export LLM_MEM_TRACE_EXPERT="${LLM_MEM_TRACE_EXPERT:-$PROFILE_EXPERT}"
export LLM_MEM_TRACE_MEMORY="${LLM_MEM_TRACE_MEMORY:-$PROFILE_MEMORY}"
export LLM_MEM_TRACE_CONTROL_ONLY="${LLM_MEM_TRACE_CONTROL_ONLY:-$PROFILE_CONTROL_ONLY}"
export LLM_MEM_TRACE_QUEUE_LIMIT="${LLM_MEM_TRACE_QUEUE_LIMIT:-65536}"
export LLM_MEM_TRACE_ALLOW_DROP="${LLM_MEM_TRACE_ALLOW_DROP:-0}"
export LLM_MEM_TRACE_EXPERT_TASK_MODE="${LLM_MEM_TRACE_EXPERT_TASK_MODE:-$PROFILE_EXPERT_TASK_MODE}"
export LLM_MEM_TRACE_RESIDENCY="${LLM_MEM_TRACE_RESIDENCY:-$PROFILE_RESIDENCY}"
export LLM_MEM_TRACE_RESIDENCY_ATTRIBUTION="${LLM_MEM_TRACE_RESIDENCY_ATTRIBUTION:-$PROFILE_ATTRIBUTION}"
export LLM_MEM_TRACE_RESIDENCY_MAX_PAGES="${LLM_MEM_TRACE_RESIDENCY_MAX_PAGES:-4096}"
export LLM_MEM_TRACE_SMAPS="${LLM_MEM_TRACE_SMAPS:-$PROFILE_SMAPS}"
export LLM_MEM_TRACE_OS_HINTS="${LLM_MEM_TRACE_OS_HINTS:-$CONTROLLER_OS_HINTS}"
export LLM_MEM_TRACE_OPT_MADVISE_WILLNEED="${LLM_MEM_TRACE_OPT_MADVISE_WILLNEED:-0}"
export LLM_MEM_TRACE_OPT_MADVISE_SEQUENTIAL="${LLM_MEM_TRACE_OPT_MADVISE_SEQUENTIAL:-0}"
export LLM_MEM_TRACE_OPT_POSIX_FADVISE="${LLM_MEM_TRACE_OPT_POSIX_FADVISE:-0}"
export LLM_MEM_TRACE_OPT_THP="${LLM_MEM_TRACE_OPT_THP:-0}"
export LLM_MEM_TRACE_OPT_EXPERT_PREFETCH="${LLM_MEM_TRACE_OPT_EXPERT_PREFETCH:-$CONTROLLER_PREFETCH}"
export LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_BUDGET_MB="${LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_BUDGET_MB:-512}"
export LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK="${LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK:-0}"
export LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_PREFILL_TOPK="${LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_PREFILL_TOPK:-}"
export LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_DECODE_TOPK="${LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_DECODE_TOPK:-}"
export LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_STEPS="${LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_STEPS:-0}"
export LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_PREFILL_STEPS="${LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_PREFILL_STEPS:-}"
export LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_DECODE_STEPS="${LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_DECODE_STEPS:-}"
export LLM_MEM_TRACE_OPT_EXPERT_ASYNC="${LLM_MEM_TRACE_OPT_EXPERT_ASYNC:-$CONTROLLER_ASYNC}"
export LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE="${LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE:-65536}"
export LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS="${LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS:-$CONTROLLER_WORKERS}"
export LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY="${LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY:-$CONTROLLER_PRIORITY}"
export LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE="${LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE:-$CONTROLLER_PRIORITY_MODE}"
export LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP="${LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP:-$CONTROLLER_PRIORITY_HEAP}"
export LLM_MEM_TRACE_OPT_EXPERT_DEADLINE_OBSERVE="${LLM_MEM_TRACE_OPT_EXPERT_DEADLINE_OBSERVE:-$CONTROLLER_DEADLINE_OBSERVE}"
export LLM_MEM_TRACE_OPT_EXPERT_CONTROLLER="$EXPERT_CONTROLLER"
export LLM_MEM_TRACE_OPT_EXPERT_FEEDBACK="${LLM_MEM_TRACE_OPT_EXPERT_FEEDBACK:-$CONTROLLER_FEEDBACK}"
export LLM_MEM_TRACE_OPT_EXPERT_VALUE_GATE="${LLM_MEM_TRACE_OPT_EXPERT_VALUE_GATE:-$CONTROLLER_VALUE_GATE}"
export LLM_MEM_TRACE_OPT_EXPERT_ASYNC_FALLBACK="${LLM_MEM_TRACE_OPT_EXPERT_ASYNC_FALLBACK:-0}"
export LLM_MEM_TRACE_OPT_EXPERT_PRESSURE_SAMPLE_MS="${LLM_MEM_TRACE_OPT_EXPERT_PRESSURE_SAMPLE_MS:-50}"
export LLM_MEM_TRACE_OPT_EXPERT_WORKING_SET_MB="${LLM_MEM_TRACE_OPT_EXPERT_WORKING_SET_MB:-0}"
export LLM_MEM_TRACE_OPT_EXPERT_MADV_DONTNEED_RECLAIM="${LLM_MEM_TRACE_OPT_EXPERT_MADV_DONTNEED_RECLAIM:-0}"
export LLM_MEM_TRACE_OPT_EXPERT_RECLAIM_GRACE_STEPS="${LLM_MEM_TRACE_OPT_EXPERT_RECLAIM_GRACE_STEPS:-3}"
export LLM_MEM_TRACE_OPT_EXPERT_RECLAIM_MAX_MB_PER_STEP="${LLM_MEM_TRACE_OPT_EXPERT_RECLAIM_MAX_MB_PER_STEP:-64}"
export LLM_MEM_TRACE_OPT_EXPERT_RECLAIM_MEMORY_RATIO_PCT="${LLM_MEM_TRACE_OPT_EXPERT_RECLAIM_MEMORY_RATIO_PCT:-85}"
export LLM_MEM_TRACE_OPT_EXPERT_RECLAIM_REFAULT_DELTA="${LLM_MEM_TRACE_OPT_EXPERT_RECLAIM_REFAULT_DELTA:-1024}"
export LLM_MEM_TRACE_OPT_TARGETS="${LLM_MEM_TRACE_OPT_TARGETS:-token_embd.weight,output.weight,ffn_down_exps.weight}"
export LLM_MEM_TRACE_OPT_MAX_BYTES="${LLM_MEM_TRACE_OPT_MAX_BYTES:-536870912}"

# Write prompt to temp file for -f flag (avoids pipe issues)
PROMPT_FILE="$TRACE_OUT_DIR/test_prompt.txt"
echo "$TEST_PROMPT" > "$PROMPT_FILE"

echo "      Prompt token count (approx): $(wc -w < "$PROMPT_FILE") words"
echo ""

MANIFEST_ARGS=(
    --output "$TRACE_OUT_DIR/run_manifest.json"
    --project "$REPO_DIR"
    --model "$MODEL_FILE"
    --prompt "$PROMPT_FILE"
    --llama-cli "$LLAMA_CLI"
    --run-name "$RUN_NAME"
    --trace-profile "$TRACE_PROFILE"
    --cache-mode "$CACHE_MODE"
    --expert-profile "$EXPERT_PROFILE"
    --cache-type-k "$KV_CACHE_TYPE_K"
    --cache-type-v "$KV_CACHE_TYPE_V"
    --flash-attn "$FLASH_ATTN"
    --batch-size "$BATCH_SIZE"
    --ubatch-size "$UBATCH_SIZE"
    --ctx-size "$CTX_SIZE"
    --repeat-index "${REPEAT_INDEX:-}"
    --order-position "${ORDER_POSITION:-}"
    --order-mode "${ORDER_MODE:-}"
    --order-seed "${ORDER_SEED:-}"
    --memory-max "${MEMORY_MAX:-}"
    --memory-swap-max "${MEMORY_SWAP_MAX:-}"
    --model-sha256 "${MODEL_SHA256:-}"
)
if [ "$ALLOW_DIRTY_REPO" != "1" ]; then
    MANIFEST_ARGS+=(--require-clean)
fi
python3 "$SCRIPT_DIR/write_run_manifest.py" "${MANIFEST_ARGS[@]}"

python3 "$SCRIPT_DIR/prepare_model_cache.py" \
    --model "$MODEL_FILE" \
    --mode "$CACHE_MODE" \
    > "$TRACE_OUT_DIR/cache_preparation.json"

set +e
"$TIME_BIN" -q \
    -f '{"wall_time_s":%e,"user_time_s":%U,"system_time_s":%S,"max_rss_kb":%M,"major_faults":%F,"minor_faults":%R,"file_inputs":%I,"file_outputs":%O,"exit_code":%x}' \
    -o "$TRACE_OUT_DIR/process_metrics.json" \
    "$LLAMA_CLI" \
    -m "$MODEL_FILE" \
    -f "$PROMPT_FILE" \
    -n "$NUM_TOKENS_PREDICT" \
    -t "$NUM_THREADS" \
    -b "$BATCH_SIZE" \
    -ub "$UBATCH_SIZE" \
    -c "$CTX_SIZE" \
    --cache-type-k "$KV_CACHE_TYPE_K" \
    --cache-type-v "$KV_CACHE_TYPE_V" \
    --flash-attn "$FLASH_ATTN" \
    --gpu-layers "$GPU_LAYERS" \
    --temp "$TEMP" \
    --seed "$SEED" \
    --no-display-prompt \
    --simple-io \
    --single-turn \
    --no-warmup \
    --no-perf \
    --no-show-timings \
    > "$TRACE_OUT_DIR/inference_output.txt" 2>"$TRACE_OUT_DIR/inference_stderr.txt"
INFERENCE_STATUS=$?
set -e

if [ "$INFERENCE_STATUS" -ne 0 ]; then
    echo "ERROR: llama-cli exited with status $INFERENCE_STATUS" >&2
    exit "$INFERENCE_STATUS"
fi

if ! command -v sha256sum >/dev/null 2>&1; then
    echo "ERROR: sha256sum is required for output consistency validation" >&2
    exit 1
fi
sha256sum "$TRACE_OUT_DIR/inference_output.txt" | awk '{print $1}' > "$TRACE_OUT_DIR/output.sha256"

echo "      Inference completed."
echo ""

# ------------------------------------------------------------------
# Step 2: Check trace output files
# ------------------------------------------------------------------
echo "[2/4] Checking trace output files..."

declare -A TRACE_FILES=(
    ["tensor"]="tensor_trace.jsonl"
    ["kv"]="kv_trace.jsonl"
    ["expert"]="expert_trace.jsonl"
    ["memory"]="memory_trace.jsonl"
)
declare -A LINE_COUNTS
declare -A SINK_ENABLED=(
    ["tensor"]="$LLM_MEM_TRACE_TENSOR"
    ["kv"]="$LLM_MEM_TRACE_KV"
    ["expert"]="$LLM_MEM_TRACE_EXPERT"
    ["memory"]="$LLM_MEM_TRACE_MEMORY"
)

for key in "${!TRACE_FILES[@]}"; do
    f="$TRACE_OUT_DIR/${TRACE_FILES[$key]}"
    if [ -f "$f" ]; then
        count=$(wc -l < "$f")
        LINE_COUNTS[$key]=$count
        echo "      ${TRACE_FILES[$key]}: $count events"
    else
        LINE_COUNTS[$key]=0
        echo "      ${TRACE_FILES[$key]}: NOT FOUND (check if sink is enabled)"
        if [ "${SINK_ENABLED[$key]}" != "0" ]; then
            echo "ERROR: enabled trace sink did not create ${TRACE_FILES[$key]}" >&2
            exit 1
        fi
    fi
done

SUMMARY_ARGS=("$TRACE_OUT_DIR/summary.json")
for key in tensor kv expert memory; do
    if [ "${SINK_ENABLED[$key]}" != "0" ]; then
        SUMMARY_ARGS+=(--expect-sink "$key")
    fi
done
if [ "$ALLOW_DROPPED_EVENTS" = "1" ]; then
    SUMMARY_ARGS+=(--allow-dropped)
fi
python3 "$SCRIPT_DIR/validate_trace_summary.py" "${SUMMARY_ARGS[@]}"
echo "      summary.json: $(cat "$TRACE_OUT_DIR/summary.json")"

echo ""

# ------------------------------------------------------------------
# Step 3: Run Python analysis
# ------------------------------------------------------------------
echo "[3/4] Running trace analysis & visualization..."
echo ""

if [ "$TRACE_PROFILE" = "attribution" ]; then
    python3 "$SCRIPT_DIR/analyze_residency_attribution.py" \
        --trace-dir "$TRACE_OUT_DIR" \
        --output-dir "$TRACE_OUT_DIR/analysis" \
        --condition-label "${MEMORY_MAX:+MemoryMax=7G}"
else
    python3 "$ANALYSIS_SCRIPT" \
        --trace-dir "$TRACE_OUT_DIR" \
        --output-dir "$TRACE_OUT_DIR/analysis" \
        --num-generate "$NUM_TOKENS_PREDICT"
fi

echo ""
echo "[4/4] Pipeline complete!"
echo ""
echo "Output files:"
echo "  Trace data:       $TRACE_OUT_DIR/"
echo "  Analysis results: $TRACE_OUT_DIR/analysis/"
echo "  Inference output: $TRACE_OUT_DIR/inference_output.txt"
echo "  Output hash:      $TRACE_OUT_DIR/output.sha256"
echo ""
echo "Open $TRACE_OUT_DIR/analysis/analysis_report.html to view the full report."
