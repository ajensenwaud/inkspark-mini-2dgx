#!/usr/bin/env bash
# Download the published pack on the head, pinned to its revision, with curl only (scripts/fetch_hf_pinned.sh checks
# every large file against the Hub's sha256), then copy it to the worker over the link (scripts/sync_pack.sh).
# Resumable: run it again after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
need_pair
[[ -n "$PACK_REV" ]] || die "no PACK_REV pinned in scripts/config.sh"

log "$PACK_REPO @ ${PACK_REV:0:12} -> $MODEL_DIR"
for pass in 1 2 3; do   # a file that fails its sha256 is fetched again on the next pass
  scripts/fetch_hf_pinned.sh "$PACK_REPO" "$PACK_REV" "$MODEL_DIR" && break
  [[ $pass == 3 ]] && die "the download did not complete: run scripts/fetch_pack.sh again"
done
scripts/sync_pack.sh
