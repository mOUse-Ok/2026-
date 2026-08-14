#!/usr/bin/env sh
# Prefer the inference child, rather than its metrics wrapper, as the cgroup
# OOM victim. Intended only for controlled completion experiments.
set -eu

score="${1:?oom score is required}"
shift
case "$score" in
    ''|*[!0-9]*)
        echo "ERROR: oom score must be an integer from 0 through 1000" >&2
        exit 1
        ;;
esac
if [ "$score" -gt 1000 ]; then
    echo "ERROR: oom score must be an integer from 0 through 1000" >&2
    exit 1
fi
printf '%s\n' "$score" > /proc/self/oom_score_adj
exec "$@"
