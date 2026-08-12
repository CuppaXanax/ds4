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

# The portable routed shader is the permanent fallback and intentionally stays
# byte-for-byte unchanged.  Derive the Wave64 source from it so the large common
# body cannot drift; only the subgroup extensions, reducer, and the two
# projection coordinate mappings are allowed to differ.
routed_src = SRC_DIR / "routed_moe.comp"
routed_wave64_src = SRC_DIR / "routed_moe_wave64.comp"
routed_text = routed_src.read_text(encoding="utf-8")

extension_anchor = "#extension GL_GOOGLE_include_directive : require\n"
wave64_extensions = (
    extension_anchor
    + "#extension GL_KHR_shader_subgroup_basic : require\n"
    + "#extension GL_KHR_shader_subgroup_shuffle : require\n"
)
portable_reducer = """float reduce_row_sum(float value, uint lanes_per_row) {
    uint lid = gl_LocalInvocationIndex;
    uint lane = lid & (lanes_per_row - 1u);
    q8_values[lid] = value;
    barrier();
    for (uint stride = lanes_per_row >> 1u; stride != 0u; stride >>= 1u) {
        if (lane < stride) q8_values[lid] += q8_values[lid + stride];
        barrier();
    }
    return q8_values[lid - lane];
}"""
wave64_reducer = """/* Host dispatch restricts this variant to power-of-two rows no wider than
 * Wave64.  Subgroup coordinates define both work assignment and shuffle
 * partners; Vulkan does not relate subgroup order to local-invocation order. */
uint wave64_logical_id() {
    return gl_SubgroupID * gl_SubgroupSize + gl_SubgroupInvocationID;
}

float reduce_row_sum(float value, uint lanes_per_row) {
    uint lane = wave64_logical_id() & (lanes_per_row - 1u);
    for (uint stride = lanes_per_row >> 1u; stride != 0u; stride >>= 1u) {
        float paired = subgroupShuffleXor(value, stride);
        if (lane < stride) value += paired;
    }
    return value;
}"""
portable_mapping = """        uint lane = lid & (lanes_per_row - 1u);
        uint rows_per_group = 256u / lanes_per_row;
        uint row = gl_WorkGroupID.x * rows_per_group + lid / lanes_per_row;"""
wave64_mapping = """        uint logical_id = wave64_logical_id();
        uint lane = logical_id & (lanes_per_row - 1u);
        uint rows_per_group = 256u / lanes_per_row;
        uint row = gl_WorkGroupID.x * rows_per_group + logical_id / lanes_per_row;"""

if routed_text.count(extension_anchor) != 1 or routed_text.count(portable_reducer) != 1 \
        or routed_text.count(portable_mapping) != 2:
    print("Portable routed shader no longer matches the Wave64 derivation anchors",
          file=sys.stderr)
    sys.exit(1)
routed_wave64_text = routed_text.replace(extension_anchor, wave64_extensions, 1)
routed_wave64_text = routed_wave64_text.replace(portable_reducer, wave64_reducer, 1)
routed_wave64_text = routed_wave64_text.replace(portable_mapping, wave64_mapping, 2)
routed_wave64_src.write_text(
    routed_wave64_text,
    encoding="utf-8",
    newline="\n",
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
