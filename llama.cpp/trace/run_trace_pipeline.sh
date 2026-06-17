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
BUILD_DIR="$PROJECT_DIR/build"
MODEL_DIR="$PROJECT_DIR/../models"
TRACE_BASE_DIR="${TRACE_BASE_DIR:-$PROJECT_DIR/trace_output}"
RUN_NAME="${RUN_NAME:-latest}"
TRACE_OUT_DIR="${TRACE_OUT_DIR:-$TRACE_BASE_DIR/$RUN_NAME}"
MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/Qwen3.5-35B-A3B-Q3_K_M.gguf}"
ANALYSIS_SCRIPT="$SCRIPT_DIR/analyze_trace.py"
LLAMA_CLI="${LLAMA_CLI:-$BUILD_DIR/bin/llama-cli}"

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
NUM_TOKENS_PREDICT="${NUM_TOKENS_PREDICT:-80}"  # Decode enough tokens to observe decode behavior
NUM_THREADS="${NUM_THREADS:-8}"                 # CPU threads
BATCH_SIZE="${BATCH_SIZE:-512}"                 # Batch size for prefill
CTX_SIZE="${CTX_SIZE:-2048}"                    # Context window size
TEMP="${TEMP:-0.0}"                             # Deterministic output for reproducibility

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

# ------------------------------------------------------------------
# Step 1: Clean previous trace output & run inference
# ------------------------------------------------------------------
echo "[1/4] Running LLM inference with memory tracing..."
echo "      Model: $(basename "$MODEL_FILE")"
echo "      Predict tokens: $NUM_TOKENS_PREDICT"
echo "      Output dir: $TRACE_OUT_DIR"
echo ""

rm -rf "$TRACE_OUT_DIR"
mkdir -p "$TRACE_OUT_DIR"

export LLM_MEM_TRACE=1
export LLM_MEM_TRACE_DIR="$TRACE_OUT_DIR"
export LLM_MEM_TRACE_TENSOR="${LLM_MEM_TRACE_TENSOR:-1}"
export LLM_MEM_TRACE_KV="${LLM_MEM_TRACE_KV:-1}"
export LLM_MEM_TRACE_EXPERT="${LLM_MEM_TRACE_EXPERT:-1}"
export LLM_MEM_TRACE_MEMORY="${LLM_MEM_TRACE_MEMORY:-1}"
export LLM_MEM_TRACE_RESIDENCY="${LLM_MEM_TRACE_RESIDENCY:-1}"
export LLM_MEM_TRACE_RESIDENCY_MAX_PAGES="${LLM_MEM_TRACE_RESIDENCY_MAX_PAGES:-4096}"
export LLM_MEM_TRACE_SMAPS="${LLM_MEM_TRACE_SMAPS:-1}"
export LLM_MEM_TRACE_OS_HINTS="${LLM_MEM_TRACE_OS_HINTS:-0}"
export LLM_MEM_TRACE_OPT_MADVISE_WILLNEED="${LLM_MEM_TRACE_OPT_MADVISE_WILLNEED:-0}"
export LLM_MEM_TRACE_OPT_MADVISE_SEQUENTIAL="${LLM_MEM_TRACE_OPT_MADVISE_SEQUENTIAL:-0}"
export LLM_MEM_TRACE_OPT_POSIX_FADVISE="${LLM_MEM_TRACE_OPT_POSIX_FADVISE:-0}"
export LLM_MEM_TRACE_OPT_THP="${LLM_MEM_TRACE_OPT_THP:-0}"
export LLM_MEM_TRACE_OPT_EXPERT_PREFETCH="${LLM_MEM_TRACE_OPT_EXPERT_PREFETCH:-0}"
export LLM_MEM_TRACE_OPT_EXPERT_POLICY="${LLM_MEM_TRACE_OPT_EXPERT_POLICY:-route}"
export LLM_MEM_TRACE_OPT_EXPERT_CACHE_MB="${LLM_MEM_TRACE_OPT_EXPERT_CACHE_MB:-512}"
export LLM_MEM_TRACE_OPT_EXPERT_TTL_STEPS="${LLM_MEM_TRACE_OPT_EXPERT_TTL_STEPS:-4}"
export LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK="${LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK:-0}"
export LLM_MEM_TRACE_OPT_EXPERT_EVICT="${LLM_MEM_TRACE_OPT_EXPERT_EVICT:-cold}"
export LLM_MEM_TRACE_OPT_TARGETS="${LLM_MEM_TRACE_OPT_TARGETS:-token_embd.weight,output.weight,ffn_down_exps.weight}"
export LLM_MEM_TRACE_OPT_MAX_BYTES="${LLM_MEM_TRACE_OPT_MAX_BYTES:-536870912}"

# Write prompt to temp file for -f flag (avoids pipe issues)
PROMPT_FILE="$TRACE_OUT_DIR/test_prompt.txt"
echo "$TEST_PROMPT" > "$PROMPT_FILE"

echo "      Prompt token count (approx): $(wc -w < "$PROMPT_FILE") words"
echo ""

"$LLAMA_CLI" \
    -m "$MODEL_FILE" \
    -f "$PROMPT_FILE" \
    -n "$NUM_TOKENS_PREDICT" \
    -t "$NUM_THREADS" \
    -b "$BATCH_SIZE" \
    -c "$CTX_SIZE" \
    --temp "$TEMP" \
    --no-display-prompt \
    --simple-io \
    --single-turn \
    --no-warmup \
    --no-perf \
    --no-show-timings \
    > "$TRACE_OUT_DIR/inference_output.txt" 2>"$TRACE_OUT_DIR/inference_stderr.txt"

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

for key in "${!TRACE_FILES[@]}"; do
    f="$TRACE_OUT_DIR/${TRACE_FILES[$key]}"
    if [ -f "$f" ]; then
        count=$(wc -l < "$f")
        LINE_COUNTS[$key]=$count
        echo "      ${TRACE_FILES[$key]}: $count events"
    else
        LINE_COUNTS[$key]=0
        echo "      ${TRACE_FILES[$key]}: NOT FOUND (check if sink is enabled)"
    fi
done

if [ -f "$TRACE_OUT_DIR/summary.json" ]; then
    echo "      summary.json: $(cat "$TRACE_OUT_DIR/summary.json")"
fi

echo ""

# ------------------------------------------------------------------
# Step 3: Run Python analysis
# ------------------------------------------------------------------
echo "[3/4] Running trace analysis & visualization..."
echo ""

python3 "$ANALYSIS_SCRIPT" \
    --trace-dir "$TRACE_OUT_DIR" \
    --output-dir "$TRACE_OUT_DIR/analysis" \
    --num-generate "$NUM_TOKENS_PREDICT"

echo ""
echo "[4/4] Pipeline complete!"
echo ""
echo "Output files:"
echo "  Trace data:       $TRACE_OUT_DIR/"
echo "  Analysis results: $TRACE_OUT_DIR/analysis/"
echo "  Inference output: $TRACE_OUT_DIR/inference_output.txt"
echo ""
echo "Open $TRACE_OUT_DIR/analysis/analysis_report.html to view the full report."
