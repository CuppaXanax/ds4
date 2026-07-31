#!/usr/bin/env bash
# run-test.sh — rebuild the Vulkan backend and run a quick inference smoke test.
#
# Usage:
#   ./run-test.sh                       # default model + prompt
#   ./run-test.sh <model.gguf> [prompt]
#   SKIP_BUILD=1 ./run-test.sh ...      # skip the rebuild (use existing binary)
#   DS4_VULKAN_WEIGHT_BUDGET_GB=46 ./run-test.sh ...
#
# What to look for:
#   - "ds4: VULKAN device: ..."  -> GPU init OK
#   - "ds4: VULKAN loaded N shaders" / "backend ready" -> kernel load OK
#   - no "Metal prefill kernel warmup failed" -> weight lazy-upload works
#   - a generated reply (even if nonsense) -> full pipeline runs
set -uo pipefail
cd "$(dirname "$0")" || exit 1

MODEL="${1:-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}"
PROMPT="${2:-Hello}"
CTX="${CTX:-2048}"
NTOK="${NTOK:-8}"
export DS4_VULKAN_WEIGHT_BUDGET_GB="${DS4_VULKAN_WEIGHT_BUDGET_GB:-40}"

if [ ! -f "$MODEL" ]; then
    echo "model not found: $MODEL" >&2
    exit 1
fi

if [ "${SKIP_BUILD:-0}" != "1" ]; then
    echo "==> rebuilding Vulkan backend..."
    g++ -O3 -g -std=c++17 -pthread -I. -Ivulkan -Ivulkan/include -march=native \
        -DDS4_VULKAN_BUILD -c -o ds4_vulkan.o vulkan/vulkan_backend.cpp || exit 1
    g++ -O3 -g -std=c++17 -pthread -I. -Ivulkan -Ivulkan/include -march=native \
        -DDS4_VULKAN_BUILD -o ds4 ds4_cli.o ds4_help.o linenoise.o ds4_gpu_args.o \
        ds4.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_vulkan.o ds4_layer_pack.o \
        -lm -pthread -lvulkan || exit 1
    echo "==> build OK"
else
    echo "==> SKIP_BUILD=1: using existing ./ds4"
fi

echo "==> ./ds4 -m $MODEL --vulkan -c $CTX -n $NTOK -p \"$PROMPT\" --temp 0"
echo "    (weight budget ${DS4_VULKAN_WEIGHT_BUDGET_GB} GiB)"
echo "------------------------------------------------------------"
exec ./ds4 -m "$MODEL" --vulkan -c "$CTX" -n "$NTOK" -p "$PROMPT" --temp 0
