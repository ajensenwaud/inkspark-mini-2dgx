#!/usr/bin/env bash
# Fetch a Hugging Face repo at a pinned revision with curl only (no host installs, no image needed),
# JOBS files in parallel (default 4), resumable, each LFS file checked against the Hub's sha256 before it is kept.
# Files over SEG_MIN (2 GiB) download as SEGS byte ranges in parallel (default 8): one stream is slow on the home
# link (two 101 GB files went from 6 to 11 MB/s total).
# Small (non-LFS) files are fetched first, so tokenizer/config/index exist early for asset steps.
# INCLUDE (a regex on the path) limits the files, e.g. two Engram shards of a large base repo.
# Prints "DONE have/n files" last; exits non-zero if any file is missing (rerun to resume; BAD files are refetched).
# Stop a running fetch by killing its whole session (kill -- -<sid>): killing only the curls lets the per-file
# loops start new ones, and a relaunch then has two writers on the same .part files (BAD sha256).
# Usage: [INCLUDE=regex] [JOBS=n] [SEGS=n] fetch_hf_pinned.sh <repo> <revision> <dest dir>   (add -H auth for a gated repo)
set -euo pipefail
REPO=$1; REV=$2; DEST=$3; JOBS=${JOBS:-4}; INCLUDE=${INCLUDE:-.}; export SEGS=${SEGS:-8}
mkdir -p "$DEST"; cd "$DEST"
curl -sfL "https://huggingface.co/api/models/$REPO/tree/$REV?recursive=1" > .tree.json
python3 - "$REPO" "$REV" "$INCLUDE" > .list <<'EOF'
import json, re, sys
repo, rev, inc = sys.argv[1:4]
files = [x for x in json.load(open(".tree.json")) if x["type"] == "file" and re.search(inc, x["path"])]
files.sort(key=lambda x: ("lfs" in x, x.get("size", 0)))  # small files first
for x in files:
    lfs = x.get("lfs", {})
    print(x["path"], lfs.get("oid", "-"), f"https://huggingface.co/{repo}/resolve/{rev}/{x['path']}",
          lfs.get("size", x.get("size", 0)))
EOF

fsize() { [ -f "$1" ] && stat -c %s "$1" || echo 0; }

# One byte range [a, b] into its own file, resumed by asking for the bytes it lacks.
# Never curl --retry with -C -: its internal retry resumes from a stale offset and corrupts the file.
seg() {
  local url=$1 out=$2 a=$3 b=$4 try have
  for try in $(seq 1 30); do
    have=$(fsize "$out")
    [ "$have" -eq $((b - a + 1)) ] && return 0
    [ "$have" -gt $((b - a + 1)) ] && { echo "oversize $out"; : > "$out"; continue; }
    curl -sfL -r "$((a + have))-$b" "$url" >> "$out" || { echo "retry $try $out"; sleep 10; }
  done
  [ "$(fsize "$out")" -eq $((b - a + 1)) ]
}

fetch() {
  local path=$1 oid=$2 url=$3 size=${4:-0}
  [ -f "$path" ] && { echo "have $path"; return 0; }
  mkdir -p "$(dirname "$path")"
  if [ "$size" -gt "${SEG_MIN:-$((2 * 1024 ** 3))}" ] && [ "$SEGS" -gt 1 ]; then
    # Segments start where .part ends, so a rerun with the same .part and SEGS resumes the same segments.
    local base step i a b ok=1 pids=()
    base=$(fsize "$path.part"); step=$(( (size - base + SEGS - 1) / SEGS ))
    for i in $(seq 0 $((SEGS - 1))); do
      a=$((base + i * step)); b=$((a + step - 1)); [ "$b" -ge "$size" ] && b=$((size - 1))
      [ "$a" -le "$b" ] || continue
      seg "$url" "$path.seg$i" "$a" "$b" & pids+=($!)
    done
    for i in "${pids[@]}"; do wait "$i" || ok=0; done
    [ "$ok" = 1 ] || { echo "segments incomplete $path"; return 1; }
    for i in $(seq 0 $((SEGS - 1))); do [ -f "$path.seg$i" ] && cat "$path.seg$i" >> "$path.part" && rm -f "$path.seg$i"; done
  else
    for try in $(seq 1 20); do curl -sfL -C - -o "$path.part" "$url" && break; echo "retry $try $path"; sleep 10; done
  fi
  if [ "$oid" != "-" ] && [ "$(sha256sum "$path.part" | cut -d' ' -f1)" != "$oid" ]; then
    echo "BAD sha256 $path"; rm -f "$path.part"; return 1
  fi
  mv "$path.part" "$path"; echo "ok $path"
}
export -f fsize seg fetch
xargs -P "$JOBS" -L 1 bash -c 'fetch "$@"' _ < .list || true
n=$(wc -l < .list); have=0
while read -r p _; do [ -f "$p" ] && have=$((have + 1)); done < .list
echo "DONE $have/$n files $(date -Is)"
[ "$have" -eq "$n" ]
