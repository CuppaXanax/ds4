#!/usr/bin/env python3
"""Compile all .comp GLSL shaders to .spv SPIR-V binaries.
Called from Makefile during build. Outputs to shaders/spv/."""

import subprocess, sys, os, pathlib
import re

GLSLANG = os.environ.get("GLSLANG", "glslangValidator")
SRC_DIR = pathlib.Path(__file__).parent
OUT_DIR = SRC_DIR / "spv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Keep the Vulkan IQ2 lookup data byte-for-byte tied to the repository CUDA
# tables.  The shader must not carry a reduced or placeholder table.
cuda_tables = SRC_DIR.parent.parent / "ds4_iq2_tables_cuda.inc"
iq2_include = SRC_DIR / "iq2_tables.glsl"
source_text = cuda_tables.read_text(encoding="utf-8")
signs = re.search(r"cuda_ksigns_iq2xs\[128\].*?\{(.*?)\};", source_text, re.S)
grid = re.search(r"cuda_iq2xxs_grid\[256\].*?\{(.*?)\};", source_text, re.S)
if not signs or not grid:
    print(f"Could not extract IQ2 tables from {cuda_tables}", file=sys.stderr)
    sys.exit(1)
numbers = lambda block: re.findall(r"0x[0-9a-fA-F]+|\d+", block)
sign_values = numbers(signs.group(1))
grid_values = numbers(grid.group(1))
if len(sign_values) != 128 or len(grid_values) != 256:
    print(f"Invalid IQ2 table counts: signs={len(sign_values)} grid={len(grid_values)}", file=sys.stderr)
    sys.exit(1)
grid_bytes = [str((int(value, 0) >> shift) & 0xff)
              for value in grid_values for shift in range(0, 64, 8)]
iq2_include.write_text(
    "const uint iq2_signs[128] = uint[](" + ",".join(sign_values) + ");\n" +
    "const uint iq2_grid[2048] = uint[](" + ",".join(grid_bytes) + ");\n",
    encoding="utf-8",
)

shaders = sorted(SRC_DIR.glob("*.comp"))
if not shaders:
    print("No .comp shaders found, skipping.")
    sys.exit(0)

# Keep a manual-unpack copy in the same binary solely for correctness and
# performance A/B measurements.  matmul_f16.spv is the opt-in native
# unpackHalf2x16 candidate until BC-250 qualification promotes it.
jobs = [(src, OUT_DIR / f"{src.stem}.spv", []) for src in shaders]
matmul_f16 = SRC_DIR / "matmul_f16.comp"
if matmul_f16 in shaders:
    jobs.append((matmul_f16, OUT_DIR / "matmul_f16_legacy.spv",
                 ["-DDS4_F16_MANUAL_UNPACK=1"]))

compiled = 0
failed = 0
for src, spv, defines in jobs:
    result = subprocess.run(
        [GLSLANG, "-V", "--target-env", "vulkan1.2", *defines,
         f"-I{SRC_DIR}", str(src), "-o", str(spv)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        compiled += 1
    else:
        failed += 1
        diagnostics = "\n".join(part for part in
                                (result.stdout.strip(), result.stderr.strip()) if part)
        print(f"FAIL {src.name} -> {spv.name}: {diagnostics}", file=sys.stderr)

print(f"Compiled {compiled} shaders{' (with {failed} failures)' if failed else ''} to {OUT_DIR}")
sys.exit(1 if failed else 0)
