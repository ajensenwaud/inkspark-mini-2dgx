#!/usr/bin/env bash
# InkSpark-Mini: Inkling 2 Mini on two DGX Sparks via TensorFold. Shared settings; override any in scripts/local.sh
# (gitignored), which holds the node addresses.
# shellcheck disable=SC2034

# --- nodes (set in scripts/local.sh) -------------------------------------------------------------------------------
WORKER="${WORKER:-}"                          # rank 1, user@address on the link between the Sparks
MASTER="${MASTER:-}"                          # rank 0's address on that link (rendezvous + NCCL)
MASTER_PORT="${MASTER_PORT:-29592}"

# --- checkpoints ---------------------------------------------------------------------------------------------------
SRC_REPO="thinkingmachines/Inkling-2-Mini"    # NVFP4/MXFP8 experts, bf16 elsewhere, as released
SRC_REV="ed40e7da625f594fc9c4990f5f66be24873c22d7"
PACK_REPO="ajensenwaud/inkspark-mini-2dgx"
PACK_REV="${PACK_REV:-}"                      # empty until the pack is on the Hub, then its commit; fetch_pack.sh needs it
MODELS_ROOT="${MODELS_ROOT:-$HOME/models/inkspark-mini}"
SRC_DIR="${SRC_DIR:-$MODELS_ROOT/Inkling-2-Mini}"
MODEL_DIR="${MODEL_DIR:-$MODELS_ROOT/inkspark-mini-2dgx}"   # the same path on both nodes
SERVED_NAME="${SERVED_NAME:-Inkling-2-Mini}"

# --- engine --------------------------------------------------------------------------------------------------------
TF_REPO="https://github.com/ashhart/TensorFold"
TF_TAG="v1.3.3"
TF_COMMIT="${TF_COMMIT:-2071d0a219eda1df1e75874193444060ad5b8e70}"   # v1.3.3 (tag object; commit 1e4a32cb5)
ZIG_TREE="a33d8ed73058a389fa9cd2b345c0c886658687b9"   # v1.3.3 + patches-zig/ (git tree hash; prepare.sh checks it)
TF_DIR="${TF_DIR:-$HOME/inkspark-mini/tensorfold}"   # the patched tree and its build (zig-out/), same path on both nodes
EXL3_REPO="https://github.com/turboderp-org/exllamav3"
EXL3_TAG="v1.6.6"                             # the compiled image on both nodes; the architecture is a Python overlay

# --- image (docker/Dockerfile.build: builds the engine and runs it) -----------------------------------------------
BASE_IMAGE="${BASE_IMAGE:-ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:14f15591eae5d6a540f09218d3852068962fe5381371bbfefe0e9194cd834529}"
IMAGE="${IMAGE:-inkspark-zig:0.17.0}"
CONTAINER="${CONTAINER:-inkmini}"             # inkmini-r0 on the head, inkmini-r1 on the worker

# --- serving -------------------------------------------------------------------------------------------------------
PORT="${PORT:-8888}"
HOST="${HOST:-0.0.0.0}"                       # the API's listen address; 127.0.0.1 keeps it on the head
SHAPE="${SHAPE:-long}"                        # long: 1 stream at 1M; agents: 8 streams at 128K
case "$SHAPE" in
  long)   PARALLEL="${PARALLEL:-1}"; CONTEXT="${CONTEXT:-1048576}" ;;
  agents) PARALLEL="${PARALLEL:-8}"; CONTEXT="${CONTEXT:-131072}" ;;
esac
DRAFTER="${DRAFTER:-1}"                       # 1: the checkpoint's DSpark drafter
VISION="${VISION:-1}"                         # 1: images (the vision tower on the head); 0: text only
DRAFT_WEIGHTS="${DRAFT_WEIGHTS:-nvfp4}"        # the drafter's projections: nvfp4 (quantised at load) or bf16
KV="${KV:-fp8}"                               # the global layers' KV cache: fp8 or bf16
PLAIN_STREAMS="${PLAIN_STREAMS:-}"            # shared rounds this wide decode without drafts; empty: the engine's (4: drafts at 1-3)
RESERVE_GIB="${RESERVE_GIB:-3}"               # memory each rank leaves free
MIN_FREE_GIB="${MIN_FREE_GIB:-60}"            # start.sh refuses a node with less memory available than this
GPU_LOCK="${GPU_LOCK:-inkspark-mini/gpu.lock}"   # under $HOME on each node

# --- fabric (the direct link between the Sparks) ---------------------------------------------------------------------
NCCL_IFNAME="${NCCL_IFNAME:-enp1s0f0np0}"
NCCL_HCA="${NCCL_HCA:-rocep1s0f0}"

[[ -f "$(dirname "${BASH_SOURCE[0]}")/local.sh" ]] && source "$(dirname "${BASH_SOURCE[0]}")/local.sh"

log() { printf '[%s] %s\n' "$(basename "$0")" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }
need_pair() { [[ -n "$WORKER" && -n "$MASTER" ]] || die "set WORKER and MASTER in scripts/local.sh"; }
