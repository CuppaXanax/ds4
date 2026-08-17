#!/usr/bin/env bash
# run-kernel-tests.sh — build and run the Vulkan kernel test harness.
# Needs GPU access (/dev/dri). Writes results to vulkan/tests/results.txt.
set -uo pipefail
cd "$(dirname "$0")" || exit 1   # repo root (this script lives at the repo root)

echo "==> building kernel test harness..."
python3 vulkan/shaders/compile.py || exit 1
g++ -O2 -g -std=c++17 -pthread -I. -Ivulkan -Ivulkan/include \
    -DDS4_VULKAN_BUILD \
    vulkan/tests/harness.cpp vulkan/tests/tests.cpp \
    vulkan/tests/tests/*.cpp \
    vulkan/vulkan_backend.cpp vulkan/q8_aligned_artifact.cpp \
    vulkan/execution_artifact.cpp \
    -lm -pthread -lvulkan -o vulkan-tests || exit 1
echo "==> build OK"

echo "==> running..."
mkdir -p vulkan/tests
./vulkan-tests 2>&1 | tee vulkan/tests/results.txt
exit ${PIPESTATUS[0]}
