#!/usr/bin/env bash
# Stop both ranks: save each rank's log to logs/, remove the containers and drop the serve's GPU locks.
set -euo pipefail
cd "$(dirname "$0")"
source scripts/config.sh
need_pair

mkdir -p logs   # the server's per-request lines (speed, prompt cache, drafting) go with the container
ts=$(date +%Y%m%d-%H%M%S)
docker logs "$CONTAINER-r0" > "logs/$CONTAINER-r0-$ts.log" 2>&1 || true
ssh "$WORKER" "docker logs $CONTAINER-r1" > "logs/$CONTAINER-r1-$ts.log" 2>&1 </dev/null || true
docker rm -f "$CONTAINER-r0" >/dev/null 2>&1 && log "stopped rank 0" || true
ssh "$WORKER" "docker rm -f $CONTAINER-r1 >/dev/null 2>&1" </dev/null && log "stopped rank 1 on the worker" || true
drop='f=$HOME/'"$GPU_LOCK"'; if grep -qs "^serve inkmini" "$f"; then rm -f "$f"; fi'
bash -c "$drop"
ssh "$WORKER" "$drop" </dev/null || true
log "logs in logs/*-$ts.log"
