#!/usr/bin/env bash
# One-time, no GPU needed. Builds the engine image (docker/Dockerfile.build) on the head and loads the same image on
# the worker, checks out the pinned TensorFold release (TF_TAG) with patches-zig/ on a fresh branch, compiles the server and the family tools
# for GB10 (sm_121) inside that image, and copies the build to the worker over the link. Nothing is installed on the
# hosts. BUILD_LIMITS (e.g. "--cpus 8 --memory 20g") caps the compile container.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/config.sh
need_pair
RECIPE=$PWD
[[ -n "$ZIG_TREE" ]] || die "no ZIG_TREE pinned in scripts/config.sh"

# 1. the image, built once here and copied to the worker, so both ranks run the same bits
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  log "building $IMAGE"
  docker build -q -t "$IMAGE" --build-arg BASE_IMAGE="$BASE_IMAGE" -f docker/Dockerfile.build docker >/dev/null
fi
layers='{{range .RootFS.Layers}}{{println .}}{{end}}'   # layers, not .Id: the Id differs between Docker stores
hl=$(docker image inspect --format "$layers" "$IMAGE")
if [[ "$(ssh "$WORKER" "docker image inspect --format '$layers' $IMAGE 2>/dev/null" </dev/null)" != "$hl" ]]; then
  log "copying $IMAGE to the worker"
  docker save "$IMAGE" | ssh "$WORKER" docker load >/dev/null
  [[ "$(ssh "$WORKER" "docker image inspect --format '$layers' $IMAGE" </dev/null)" == "$hl" ]] \
    || die "the worker's $IMAGE differs from the head's"
fi
log "$IMAGE is the same on both nodes ($(grep -c . <<< "$hl") layers)"

# 2. the source: TensorFold at the pinned release, the patches on a fresh branch
[[ -d "$TF_DIR/.git" ]] || git clone -q "$TF_REPO" "$TF_DIR"
cd "$TF_DIR"
git fetch -q --force --tags origin   # --force: a release tag moved upstream must not stop the fetch; TF_COMMIT decides
[[ "$(git rev-parse "$TF_TAG^{}")" == "$(git rev-parse "$TF_COMMIT^{}")" ]] || die "$TF_TAG is not $TF_COMMIT"
git checkout -q -f -B inkmini "$TF_TAG^{commit}"
log "applying $(ls "$RECIPE"/patches-zig/*.patch | wc -l) patches to TensorFold $TF_TAG"
git -c user.name=recipe -c user.email=recipe@localhost am -q "$RECIPE"/patches-zig/*.patch \
  || { git am --abort; die "patches-zig/ did not apply"; }
tree=$(git rev-parse 'HEAD^{tree}')   # the tree, not the commit: git am stamps its own dates
[[ "$tree" == "$ZIG_TREE" ]] || die "patched tree $tree is not the pinned $ZIG_TREE"
log "patched tree $tree"

# 3. compile: the server (native/bin/tensorfold-native) and the family's check tools
log "compiling (a first build takes a while)"
rm -rf zig-out
# shellcheck disable=SC2086
docker run --rm ${BUILD_LIMITS:-} --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e ZIG_GLOBAL_CACHE_DIR=/work/.zig-cache-global -v "$TF_DIR:/work" -w /work --entrypoint bash "$IMAGE" -c '
    set -e
    NVCC=$(command -v nvcc)
    zig build native tf-inkling2-engine tf-inkling2-vision -Dnvcc="$NVCC" -Dtp-nvcc="$NVCC" -Dsm=121 \
      -Doptimize=ReleaseFast --prefix /work/zig-out'
for b in native/bin/tensorfold-native bin/tf-inkling2-engine bin/tf-inkling2-vision; do
  [[ -x "zig-out/$b" ]] || die "build incomplete: no zig-out/$b"
done

# 4. the same build on the worker
log "copying the build to the worker ($TF_DIR/zig-out)"
ssh "$WORKER" "mkdir -p '$TF_DIR'" </dev/null
rsync -a --delete "$TF_DIR/zig-out/" "$WORKER:$TF_DIR/zig-out/"
log "built tree $tree in $TF_DIR/zig-out on both nodes: scripts/fetch_pack.sh, then ./start.sh"
