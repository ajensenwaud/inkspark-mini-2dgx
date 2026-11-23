#!/usr/bin/env bash
# Start Inkling 2 Mini on both Sparks: rank 1 on the worker, then rank 0 here, which serves the API on $HOST:$PORT.
# SHAPE=long (the default) gives one stream of 1M tokens; SHAPE=agents gives eight streams of 128K.
# ./start.sh restart bounces both ranks. DRY_RUN=1 prints the two docker commands and starts nothing.
# SKIP_LOCK=1 is for scripts that already hold the GPU lock on both nodes (tools/g3-equal/window.sh).
set -euo pipefail
cd "$(dirname "$0")"
source scripts/config.sh
need_pair
[[ -n "${PARALLEL:-}" && -n "${CONTEXT:-}" ]] || die "SHAPE=$SHAPE: use long or agents"

rank_cmd() {   # RANK: the docker command that starts one rank
  local serve="serve /model --name $SERVED_NAME --host $HOST --port $PORT --context $CONTEXT --parallel $PARALLEL ${SERVE_EXTRA:-}"
  [[ "$VISION" == 1 ]] && serve+=" --vision"
  [[ "$DRAFTER" == 1 ]] || serve+=" --no-drafts"
  echo "docker run -d --name $CONTAINER-r$1 --gpus all --network host --ipc host" \
       "--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1:-1" \
       "-e TF_INKLING2_RANK=$1 -e TF_INKLING2_WORLD=2 -e NCCL_COMM_ID=$MASTER:$MASTER_PORT" \
       "-e NCCL_SOCKET_IFNAME=$NCCL_IFNAME -e NCCL_IB_DISABLE=1 -e NCCL_RUNTIME_CONNECT=0" \
       "-e NCCL_CUMEM_ENABLE=0 -e NCCL_CUMEM_HOST_ENABLE=0 -e TF_ROCE=1 -e NCCL_IB_HCA=$NCCL_HCA" \
       "-e TENSORFOLD_MEMORY_RESERVE_GIB=$RESERVE_GIB ${TF_INKLING2_PROFILE:+-e TF_INKLING2_PROFILE=$TF_INKLING2_PROFILE}" \
       "${DRAFT_WEIGHTS:+-e TF_INKLING2_DRAFT_WEIGHTS=$DRAFT_WEIGHTS} ${KV:+-e TF_INKLING2_KV=$KV}" \
       "${PLAIN_STREAMS:+-e TF_INKLING2_PLAIN_STREAMS=$PLAIN_STREAMS}" \
       "-v $MODEL_DIR:/model:ro -v $TF_DIR/zig-out:/tf:ro" \
       "--entrypoint /tf/native/bin/tensorfold-native $IMAGE $serve"
}

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "# worker: $(rank_cmd 1)"
  echo "# head:   $(rank_cmd 0)"
  exit 0
fi

take_lock() {   # GPU_LOCK MIN_FREE_GIB: refuse a busy node, else write the serve's lock
  local avail
  mkdir -p "$(dirname "$HOME/$1")"
  [[ -e "$HOME/$1" ]] && { echo "locked: $(cat "$HOME/$1")"; return 1; }
  [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || { echo "the GPU is in use"; return 1; }
  avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  (( avail >= $2 )) || { echo "only $avail GiB of memory available, $2 needed"; return 1; }
  ( set -o noclobber; echo "serve inkmini $(date -Is)" > "$HOME/$1" ) 2>/dev/null || { echo "lost the lock race"; return 1; }
}

[[ "${1:-}" == restart ]] && ./stop.sh
[[ -x "$TF_DIR/zig-out/native/bin/tensorfold-native" ]] || die "no build in $TF_DIR: run scripts/prepare.sh"
[[ -f "$MODEL_DIR/config.json" ]] || die "no pack in $MODEL_DIR: run scripts/fetch_pack.sh"
if [[ "${SKIP_LOCK:-0}" != 1 ]]; then
  msg=$(take_lock "$GPU_LOCK" "$MIN_FREE_GIB") || die "head: $msg"
  msg=$(ssh "$WORKER" "$(declare -f take_lock); take_lock '$GPU_LOCK' '$MIN_FREE_GIB'" </dev/null) \
    || { rm -f "$HOME/$GPU_LOCK"; die "worker: $msg"; }
fi

log "rank 1 on the worker ($SHAPE: $PARALLEL x $CONTEXT tokens, drafter $DRAFTER, KV $KV)"
ssh "$WORKER" "$(rank_cmd 1) >/dev/null" </dev/null || die "rank 1 did not start: ./stop.sh"
log "rank 0 here"
# shellcheck disable=SC2046
$(rank_cmd 0) >/dev/null || die "rank 0 did not start: ./stop.sh"
log "loading: docker logs -f $CONTAINER-r0"
for _ in $(seq 1 360); do
  if curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    log "warming up"
    curl -fsS -m 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":16,\"temperature\":0}" \
      -o /dev/null || die "warm-up request failed: docker logs $CONTAINER-r0"
    log "serving $SERVED_NAME at http://$(hostname):$PORT/v1 ($PARALLEL x $CONTEXT tokens)"
    exit 0
  fi
  docker inspect -f '{{.State.Running}}' "$CONTAINER-r0" 2>/dev/null | grep -q true || {
    docker logs --tail 30 "$CONTAINER-r0"; die "rank 0 exited: ./stop.sh saves both logs"; }
  sleep 5
done
die "not up after 30 min: docker logs $CONTAINER-r0"
