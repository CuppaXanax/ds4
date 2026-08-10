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
grid_values = [value + "ul" if value.startswith("0x") else value
               for value in grid_values]
iq2_include.write_text(
    "const uint iq2_signs[128] = uint[](" + ",".join(sign_values) + ");\n" +
    "const uint64_t iq2_grid[256] = uint64_t[](" + ",".join(grid_values) + ");\n",
    encoding="utf-8",
)

shaders = sorted(SRC_DIR.glob("*.comp"))
if not shaders:
    print("No .comp shaders found, skipping.")
    sys.exit(0)

compiled = 0
failed = 0
for src in shaders:
    spv = OUT_DIR / f"{src.stem}.spv"
    result = subprocess.run(
        [GLSLANG, "-V", "--target-env", "vulkan1.2", f"-I{SRC_DIR}", str(src), "-o", str(spv)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        compiled += 1
    else:
        failed += 1
        diagnostics = "\n".join(part for part in
                                (result.stdout.strip(), result.stderr.strip()) if part)
        print(f"FAIL {src.name}: {diagnostics}", file=sys.stderr)

print(f"Compiled {compiled} shaders{' (with {failed} failures)' if failed else ''} to {OUT_DIR}")
sys.exit(1 if failed else 0)
