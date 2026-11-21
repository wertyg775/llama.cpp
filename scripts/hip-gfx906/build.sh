#!/usr/bin/env bash
# Build llama.cpp with HIP for Radeon VII (gfx906) inside the ROCm container.
# usage: scripts/hip-gfx906/build.sh [targets...]   (default: llama-cli llama-server llama-bench)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image="${LLAMA_HIP_IMAGE:-llama-hip-gfx906:7.14}"
preset=x64-linux-hip-gfx906-release
targets=("${@:-llama-cli llama-server llama-bench}")

if ! docker image inspect "$image" >/dev/null 2>&1; then
    docker build -t "$image" "$root/scripts/hip-gfx906"
fi

# same path inside and outside, so compile_commands.json works on the host
docker run --rm --pull never --network none \
    --user "$(id -u):$(id -g)" --env HOME=/tmp \
    --mount "type=bind,src=$root,dst=$root" --workdir "$root" \
    "$image" bash -c "cmake --preset $preset >/dev/null && cmake --build build-$preset --parallel $(nproc) --target ${targets[*]}"

echo "binaries: $root/build-$preset/bin  (run them with scripts/hip-gfx906/run.sh)"
