#!/usr/bin/env bash
# Run a HIP gfx906 build tool in the ROCm container, with the GPU attached.
# usage: scripts/hip-gfx906/run.sh llama-server -m ~/inference/models/x.gguf -ngl 99 --port 8080
# Model paths work as on the host: $LLAMA_MODELS (default ~/inference/models) is mounted read-only at the same path.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image="${LLAMA_HIP_IMAGE:-llama-hip-gfx906:7.14}"
bin="$root/build-x64-linux-hip-gfx906-release/bin"
models="$(realpath "${LLAMA_MODELS:-$HOME/inference/models}")"

tool="${1:?usage: run.sh <llama-tool> [args...]}"
shift
[[ -x "$bin/$tool" ]] || { echo "missing $bin/$tool - run scripts/hip-gfx906/build.sh $tool" >&2; exit 1; }

tty=()
[[ -t 0 && -t 1 ]] && tty=(-it)

exec docker run --rm "${tty[@]}" --pull never --network host \
    --user "$(id -u):$(id -g)" --env HOME=/tmp \
    --device /dev/kfd --device /dev/dri \
    --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
    --mount "type=bind,src=$bin,dst=$bin,readonly" \
    --mount "type=bind,src=$models,dst=$models,readonly" \
    --env "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}" \
    "$image" "$bin/$tool" "$@"
