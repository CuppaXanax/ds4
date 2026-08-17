#!/usr/bin/env python3
"""Compile all .comp GLSL shaders to .spv SPIR-V binaries.
Called from Makefile during build. Outputs to shaders/spv/."""

import subprocess, sys, os, pathlib

GLSLANG = os.environ.get("GLSLANG", "glslangValidator")
SRC_DIR = pathlib.Path(__file__).parent
OUT_DIR = SRC_DIR / "spv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

# routed_moe.comp is intentionally one source: the runtime's mode switch is
# useful for fallback/debugging, but it makes every canonical pipeline carry
# all quantizers, projections, and reduction paths. Build the same source
# with one mode selected so RADV can compile out the other paths. The push
# constant and descriptor ABI are unchanged.
routed = SRC_DIR / "routed_moe.comp"
for mode in range(6):
    spv = OUT_DIR / f"routed_moe_mode{mode}.spv"
    result = subprocess.run(
        [GLSLANG, "-V", "--target-env", "vulkan1.2", f"-I{SRC_DIR}",
         f"-DDS4_ROUTED_MODE={mode}", str(routed), "-o", str(spv)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        compiled += 1
    else:
        failed += 1
        diagnostics = "\n".join(part for part in
                                (result.stdout.strip(), result.stderr.strip()) if part)
        print(f"FAIL {routed.name} mode {mode}: {diagnostics}", file=sys.stderr)

# BC-250's GFX1013 path has Wave64 subgroups.  These variants preserve the
# scalar arithmetic and reduction order but replace repeated workgroup
# barriers with subgroup shuffles; the runtime admits them only on a proven
# subgroup-size-64 device and keeps the ordinary shaders as fallback.
for source_name in ("routed_moe_fused_mid", "routed_moe_down_reduce_q2"):
    src = SRC_DIR / f"{source_name}.comp"
    spv = OUT_DIR / f"{source_name}_wave64.spv"
    result = subprocess.run(
        [GLSLANG, "-V", "--target-env", "vulkan1.2", f"-I{SRC_DIR}",
         "-DDS4_ROUTED_WAVE64=1", str(src), "-o", str(spv)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        compiled += 1
    else:
        failed += 1
        diagnostics = "\n".join(part for part in
                                (result.stdout.strip(), result.stderr.strip()) if part)
        print(f"FAIL {src.name} Wave64: {diagnostics}", file=sys.stderr)

# Opt-in cooperative IQ2 candidate.  It retains the exact shader ABI and
# fallback while changing only the Wave64 IQ2 block execution representation.
src = SRC_DIR / "routed_moe_fused_mid.comp"
spv = OUT_DIR / "routed_moe_fused_mid_iq2_coop_wave64.spv"
result = subprocess.run(
    [GLSLANG, "-V", "--target-env", "vulkan1.2", f"-I{SRC_DIR}",
     "-DDS4_ROUTED_WAVE64=1", "-DDS4_ROUTED_IQ2_COOP=1", str(src), "-o", str(spv)],
    capture_output=True, text=True)
if result.returncode == 0:
    compiled += 1
else:
    failed += 1
    diagnostics = "\n".join(part for part in
                            (result.stdout.strip(), result.stderr.strip()) if part)
    print(f"FAIL {src.name} IQ2 cooperative Wave64: {diagnostics}", file=sys.stderr)

print(f"Compiled {compiled} shaders{' (with {failed} failures)' if failed else ''} to {OUT_DIR}")
sys.exit(1 if failed else 0)
