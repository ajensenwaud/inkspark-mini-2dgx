#!/usr/bin/env bash
# Copy the pack from the head to the worker over the link ($MODEL_DIR, the same path on both nodes), then check
# that both copies hold the same files at the same sizes.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
need_pair
[[ -f "$MODEL_DIR/config.json" ]] || die "no pack in $MODEL_DIR on the head"

listing="cd '$MODEL_DIR' && find . -type f ! -name '.*' -printf '%P %s\n' | sort"   # dot files are the fetcher's
log "copying $MODEL_DIR to the worker"
ssh "$WORKER" "mkdir -p '$MODEL_DIR'" </dev/null
rsync -a --partial --exclude '.*' "$MODEL_DIR/" "$WORKER:$MODEL_DIR/"
head_files=$(bash -c "$listing")
[[ "$(ssh "$WORKER" "$listing" </dev/null)" == "$head_files" ]] || die "the worker's copy differs from the head's"
log "both nodes: $(grep -c . <<< "$head_files") files, $(awk '{s += $2} END {printf "%.1f GB", s / 1e9}' <<< "$head_files")"
