#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: $0 JOB_ID [SOURCE_NODE]" >&2
    exit 2
fi

JOB_ID="$1"
SOURCE_NODE="${2:-babel-s5-32}"
CACHE_DIR=/scratch/junyizh3/waymo_sky_masks
POLL_SECONDS=10

if [[ ! "$JOB_ID" =~ ^[0-9]+$ ]]; then
    echo "JOB_ID must be numeric, got: $JOB_ID" >&2
    exit 2
fi

echo "Waiting for Slurm job $JOB_ID to receive a node..."
while true; do
    JOB_INFO="$(squeue --noheader --jobs="$JOB_ID" --format='%T|%N' 2>/dev/null || true)"
    JOB_INFO="${JOB_INFO#${JOB_INFO%%[![:space:]]*}}"
    JOB_INFO="${JOB_INFO%${JOB_INFO##*[![:space:]]}}"

    if [[ -z "$JOB_INFO" ]]; then
        echo "Job $JOB_ID is no longer in the Slurm queue before a node was detected." >&2
        exit 1
    fi

    JOB_STATE="${JOB_INFO%%|*}"
    TARGET_NODE="${JOB_INFO#*|}"
    if [[ "$JOB_STATE" == "RUNNING" && -n "$TARGET_NODE" && "$TARGET_NODE" != "(null)" ]]; then
        break
    fi
    case "$JOB_STATE" in
        COMPLETED|CANCELLED|FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
            echo "Job $JOB_ID reached terminal state $JOB_STATE before cache sync." >&2
            exit 1
            ;;
    esac
    echo "$(date '+%F %T') job=$JOB_ID state=$JOB_STATE node=$TARGET_NODE"
    sleep "$POLL_SECONDS"
done

if [[ "$TARGET_NODE" == "$SOURCE_NODE" ]]; then
    echo "Job $JOB_ID is running on $SOURCE_NODE; the cache is already local."
    exit 0
fi

echo "Job $JOB_ID is running on $TARGET_NODE."
echo "Pulling sky-mask cache from $SOURCE_NODE to $TARGET_NODE..."
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new "$TARGET_NODE" \
    "mkdir -p '$CACHE_DIR' && rsync -a --ignore-existing --partial --info=progress2 --exclude='*.tmp.*' -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new' '$SOURCE_NODE:$CACHE_DIR/' '$CACHE_DIR/'"

echo "Sky-mask cache sync to $TARGET_NODE completed."
