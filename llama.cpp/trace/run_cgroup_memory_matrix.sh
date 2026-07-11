#!/bin/bash
# ============================================================================
# Reproduce memory-pressure trace runs with Linux cgroup v2.
#
# Default mode is dry-run: commands are printed but not executed.
# Set RUN_MEMORY_PRESSURE_EXECUTE=1 to create cgroups and run the matrix.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TRACE_BASE_DIR="${TRACE_BASE_DIR:-$PROJECT_DIR/trace_output}"
RUN_PREFIX="${RUN_PREFIX:-cgroup_pressure}"
REPEAT_COUNT="${REPEAT_COUNT:-1}"
NUM_TOKENS_PREDICT="${NUM_TOKENS_PREDICT:-80}"
TRACE_PROFILE="${TRACE_PROFILE:-benchmark}"
CACHE_MODE="${CACHE_MODE:-cold}"
MEMORY_LIMITS_MB="${MEMORY_LIMITS_MB:-4096,5120,6144}"
RUN_GROUPS="${RUN_GROUPS:-baseline,deadline_score}"
EXECUTE="${RUN_MEMORY_PRESSURE_EXECUTE:-0}"
CGROUP_PARENT="${CGROUP_PARENT:-/sys/fs/cgroup}"
CGROUP_NAME_PREFIX="${CGROUP_NAME_PREFIX:-llm_mem_trace}"
CGROUP_SWAP_MAX="${CGROUP_SWAP_MAX:-max}"
CGROUP_MEMORY_HIGH_RATIO="${CGROUP_MEMORY_HIGH_RATIO:-0}"
DROP_CACHES="${DROP_CACHES:-0}"

if ! [[ "$REPEAT_COUNT" =~ ^[0-9]+$ ]] || [ "$REPEAT_COUNT" -le 0 ]; then
    echo "ERROR: REPEAT_COUNT must be a positive integer" >&2
    exit 1
fi

csv_to_array() {
    local value="$1"
    local -n out_ref="$2"
    local old_ifs="$IFS"
    IFS=","
    read -r -a out_ref <<< "$value"
    IFS="$old_ifs"
}

validate_limit_mb() {
    local limit="$1"
    if ! [[ "$limit" =~ ^[0-9]+$ ]] || [ "$limit" -le 0 ]; then
        echo "ERROR: memory limit must be a positive MiB integer: $limit" >&2
        exit 1
    fi
}

bytes_from_mb() {
    local mb="$1"
    echo $((mb * 1024 * 1024))
}

case_env() {
    local group="$1"
    CASE_ENV=()
    case "$group" in
        baseline)
            CASE_ENV=(
                LLM_MEM_TRACE_OS_HINTS=0
                LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=0
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC=0
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY=0
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP=0
                LLM_MEM_TRACE_OPT_EXPERT_COALESCE=0
                LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_STEPS=0
            )
            ;;
        expert_prefetch)
            CASE_ENV=(
                LLM_MEM_TRACE_OS_HINTS=1
                LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=1
                LLM_MEM_TRACE_OPT_EXPERT_POLICY=route
                LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK=0
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC=0
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY=0
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP=0
                LLM_MEM_TRACE_OPT_EXPERT_COALESCE=0
                LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_STEPS=0
            )
            ;;
        deadline_score)
            CASE_ENV=(
                LLM_MEM_TRACE_OS_HINTS=1
                LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=1
                LLM_MEM_TRACE_OPT_EXPERT_POLICY=route
                LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK=0
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC=1
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE=131072
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS=4
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY=1
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE=deadline_score
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP=0
                LLM_MEM_TRACE_OPT_EXPERT_COALESCE=0
                LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_STEPS=0
            )
            ;;
        decode_ttl1)
            CASE_ENV=(
                LLM_MEM_TRACE_OS_HINTS=1
                LLM_MEM_TRACE_OPT_EXPERT_PREFETCH=1
                LLM_MEM_TRACE_OPT_EXPERT_POLICY=route
                LLM_MEM_TRACE_OPT_EXPERT_PREFETCH_TOPK=0
                LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_STEPS=0
                LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_PREFILL_STEPS=0
                LLM_MEM_TRACE_OPT_EXPERT_ROUTE_HINT_TTL_DECODE_STEPS=1
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC=1
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_QUEUE=131072
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_WORKERS=4
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY=1
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_MODE=deadline_score
                LLM_MEM_TRACE_OPT_EXPERT_ASYNC_PRIORITY_HEAP=0
                LLM_MEM_TRACE_OPT_EXPERT_COALESCE=0
            )
            ;;
        *)
            echo "ERROR: unknown RUN_GROUP: $group" >&2
            echo "Known groups: baseline, expert_prefetch, deadline_score, decode_ttl1" >&2
            exit 1
            ;;
    esac
}

check_cgroup_ready() {
    if [ ! -f /sys/fs/cgroup/cgroup.controllers ]; then
        echo "ERROR: cgroup v2 unified hierarchy not found at /sys/fs/cgroup" >&2
        exit 1
    fi
    if [ ! -d "$CGROUP_PARENT" ]; then
        echo "ERROR: CGROUP_PARENT does not exist: $CGROUP_PARENT" >&2
        exit 1
    fi
}

maybe_drop_caches() {
    if [ "$DROP_CACHES" != "1" ]; then
        return
    fi
    if [ "$(id -u)" != "0" ]; then
        echo "ERROR: DROP_CACHES=1 requires running as root" >&2
        exit 1
    fi
    sync
    echo 3 > /proc/sys/vm/drop_caches
}

write_cgroup_knobs() {
    local cgdir="$1"
    local limit_mb="$2"
    local memory_max
    memory_max="$(bytes_from_mb "$limit_mb")"

    mkdir -p "$cgdir"
    if [ ! -f "$cgdir/memory.max" ]; then
        echo "ERROR: memory.max is missing in $cgdir; enable the memory controller on the parent cgroup" >&2
        exit 1
    fi

    echo "$memory_max" > "$cgdir/memory.max"
    if [ "$CGROUP_MEMORY_HIGH_RATIO" != "0" ] && [ -f "$cgdir/memory.high" ]; then
        echo $((memory_max * CGROUP_MEMORY_HIGH_RATIO / 100)) > "$cgdir/memory.high"
    fi
    if [ -f "$cgdir/memory.swap.max" ]; then
        echo "$CGROUP_SWAP_MAX" > "$cgdir/memory.swap.max"
    fi
}

collect_cgroup_metrics() {
    local cgdir="$1"
    local run_name="$2"
    local limit_mb="$3"
    local out_dir="$TRACE_BASE_DIR/$run_name"
    local path="$out_dir/cgroup_metrics_after.txt"
    local f

    mkdir -p "$out_dir"
    {
        echo "run_name=$run_name"
        echo "memory_limit_mb=$limit_mb"
        echo "cgroup=$cgdir"
        echo "memory_swap_max=$CGROUP_SWAP_MAX"
        echo ""
        for f in \
            memory.current \
            memory.peak \
            memory.events \
            memory.events.local \
            memory.stat \
            memory.swap.current \
            memory.swap.peak \
            memory.swap.events \
            memory.pressure; do
            if [ -f "$cgdir/$f" ]; then
                echo "[$f]"
                cat "$cgdir/$f"
                echo ""
            fi
        done
    } > "$path"
}

print_run() {
    local run_name="$1"
    local limit_mb="$2"
    local cgdir="$3"
    shift 3
    local cmd=(env "$@" bash "$SCRIPT_DIR/run_trace_pipeline.sh")

    printf '[RUN] %s memory=%sMiB\n' "$run_name" "$limit_mb"
    printf '      cgroup: %s\n' "$cgdir"
    printf '      '
    printf '%q ' "${cmd[@]}"
    printf '\n'
}

run_case() {
    local group="$1"
    local limit_mb="$2"
    local idx="$3"
    local run_name="${RUN_PREFIX}_${group}_${limit_mb}mb_r${idx}"
    local cgdir="${CGROUP_PARENT%/}/${CGROUP_NAME_PREFIX}_${run_name}"
    local cmd_env=(
        "TRACE_BASE_DIR=$TRACE_BASE_DIR"
        "RUN_NAME=$run_name"
        "NUM_TOKENS_PREDICT=$NUM_TOKENS_PREDICT"
        "TRACE_PROFILE=$TRACE_PROFILE"
        "CACHE_MODE=$CACHE_MODE"
        "REPEAT_INDEX=$idx"
        "MEMORY_MAX=${limit_mb}M"
    )
    local status

    case_env "$group"
    if [ "$CGROUP_SWAP_MAX" != "max" ]; then
        cmd_env+=("MEMORY_SWAP_MAX=$CGROUP_SWAP_MAX")
    fi
    cmd_env+=("${CASE_ENV[@]}")
    print_run "$run_name" "$limit_mb" "$cgdir" "${cmd_env[@]}"

    if [ "$EXECUTE" != "1" ]; then
        return
    fi

    maybe_drop_caches
    write_cgroup_knobs "$cgdir" "$limit_mb"

    set +e
    (
        echo "$BASHPID" > "$cgdir/cgroup.procs"
        env "${cmd_env[@]}" bash "$SCRIPT_DIR/run_trace_pipeline.sh"
    )
    status=$?
    set -e

    collect_cgroup_metrics "$cgdir" "$run_name" "$limit_mb"
    rmdir "$cgdir" 2>/dev/null || true
    return "$status"
}

MEMORY_LIMITS=()
GROUP_ARRAY=()
csv_to_array "$MEMORY_LIMITS_MB" MEMORY_LIMITS
csv_to_array "$RUN_GROUPS" GROUP_ARRAY

echo "=============================================="
echo "  Cgroup Memory Pressure Matrix"
echo "=============================================="
echo "Mode: $([ "$EXECUTE" = "1" ] && echo execute || echo dry-run)"
echo "Run prefix: $RUN_PREFIX"
echo "Repeat count: $REPEAT_COUNT"
echo "Memory limits: $MEMORY_LIMITS_MB MiB"
echo "Run groups: $RUN_GROUPS"
echo "Trace base: $TRACE_BASE_DIR"
echo "Trace profile: $TRACE_PROFILE"
echo "Cache mode: $CACHE_MODE"
echo "Cgroup parent: $CGROUP_PARENT"
echo ""

if [ "$EXECUTE" = "1" ]; then
    check_cgroup_ready
fi

for limit in "${MEMORY_LIMITS[@]}"; do
    limit="${limit//[[:space:]]/}"
    validate_limit_mb "$limit"
    for group in "${GROUP_ARRAY[@]}"; do
        group="${group//[[:space:]]/}"
        for i in $(seq 1 "$REPEAT_COUNT"); do
            run_case "$group" "$limit" "$i"
        done
    done
done

if [ "$EXECUTE" != "1" ]; then
    echo ""
    echo "Dry-run complete. Set RUN_MEMORY_PRESSURE_EXECUTE=1 to run these commands."
    echo "If /sys/fs/cgroup is not writable, prepare a delegated cgroup parent and set CGROUP_PARENT to it."
fi
