#!/usr/bin/env python3
"""Compile GLSL shaders without silently replacing frozen SPIR-V artifacts.

By default every shader is compiled to a temporary file and compared with the
existing output. Missing outputs are installed and byte-identical outputs are
kept. A byte-different existing output is never overwritten unless its exact
output stem is named with ``--allow-recompile``.

Pass one or more output stems to compile only those jobs, for example::

    python3 compile.py attention_indexed_online_wave64_512

This lets an LKG promotion add one new shader without recompiling unrelated
artifacts. An intentional replacement remains explicit and reviewable::

    python3 compile.py rope_tail --allow-recompile rope_tail
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass


GLSLANG = os.environ.get("GLSLANG", "glslangValidator")
SRC_DIR = Path(__file__).resolve().parent
OUT_DIR = SRC_DIR / "spv"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class Job:
    source: Path
    output_stem: str
    defines: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "shaders",
        nargs="*",
        metavar="OUTPUT_STEM",
        help="compile only these output stems (default: all)",
    )
    parser.add_argument(
        "--allow-recompile",
        action="append",
        default=[],
        metavar="OUTPUT_STEM",
        help="allow this existing output to be replaced when bytes differ",
    )
    return parser.parse_args()


def all_jobs() -> list[Job]:
    sources = sorted(SRC_DIR.glob("*.comp"))
    jobs = [Job(source=source, output_stem=source.stem) for source in sources]

    routed = SRC_DIR / "routed_moe.comp"
    if routed.is_file():
        jobs.extend(
            Job(routed, f"routed_moe_mode{mode}", (f"DS4_ROUTED_MODE={mode}",))
            for mode in range(6)
        )

    # BC-250's GFX1013 path has Wave64 subgroups. These variants preserve the
    # descriptor ABI while specializing the source so RADV can remove the
    # ordinary workgroup-barrier path.
    for source_name in ("routed_moe_fused_mid", "routed_moe_down_reduce_q2"):
        source = SRC_DIR / f"{source_name}.comp"
        if source.is_file():
            jobs.append(
                Job(source, f"{source_name}_wave64", ("DS4_ROUTED_WAVE64=1",))
            )

    return jobs


def validate_names(names: set[str], label: str) -> None:
    invalid = sorted(name for name in names if not SAFE_NAME.fullmatch(name))
    if invalid:
        raise SystemExit(f"invalid {label}: {', '.join(invalid)}")


def compile_job(job: Job, destination: Path) -> subprocess.CompletedProcess[str]:
    command = [
        GLSLANG,
        "-V",
        "--target-env",
        "vulkan1.2",
        f"-I{SRC_DIR}",
    ]
    command.extend(f"-D{define}" for define in job.defines)
    command.extend((str(job.source), "-o", str(destination)))
    return subprocess.run(command, capture_output=True, text=True)


def main() -> int:
    args = parse_args()
    selected = set(args.shaders)
    allowed = set(args.allow_recompile)
    validate_names(selected, "shader selector")
    validate_names(allowed, "recompile allowlist entry")

    jobs = all_jobs()
    if not jobs:
        print("No .comp shaders found, skipping.")
        return 0

    known_outputs = {job.output_stem for job in jobs}
    unknown_selected = selected - known_outputs
    unknown_allowed = allowed - known_outputs
    if unknown_selected:
        raise SystemExit(
            "unknown shader output stem(s): " + ", ".join(sorted(unknown_selected))
        )
    if unknown_allowed:
        raise SystemExit(
            "unknown --allow-recompile output stem(s): "
            + ", ".join(sorted(unknown_allowed))
        )
    if allowed - selected and selected:
        raise SystemExit("--allow-recompile entries must also be selected")

    if selected:
        jobs = [job for job in jobs if job.output_stem in selected]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    identical = 0
    installed = 0
    updated = 0
    drifted: list[str] = []
    failed: list[str] = []

    with tempfile.TemporaryDirectory(prefix=".compile-", dir=OUT_DIR) as temp_dir:
        temp_root = Path(temp_dir)
        for job in jobs:
            temporary = temp_root / f"{job.output_stem}.spv"
            result = compile_job(job, temporary)
            if result.returncode != 0:
                diagnostics = "\n".join(
                    part
                    for part in (result.stdout.strip(), result.stderr.strip())
                    if part
                )
                print(f"FAIL {job.output_stem}: {diagnostics}", file=sys.stderr)
                failed.append(job.output_stem)
                continue

            output = OUT_DIR / f"{job.output_stem}.spv"
            if not output.exists():
                os.replace(temporary, output)
                installed += 1
                print(f"INSTALL {output.name}")
            elif output.read_bytes() == temporary.read_bytes():
                identical += 1
            elif job.output_stem in allowed:
                os.replace(temporary, output)
                updated += 1
                print(f"UPDATE {output.name}")
            else:
                drifted.append(job.output_stem)
                print(
                    f"REFUSE {output.name}: compiled bytes differ; existing artifact preserved",
                    file=sys.stderr,
                )

    print(
        "Shader compile summary: "
        f"jobs={len(jobs)} identical={identical} installed={installed} "
        f"updated={updated} drifted={len(drifted)} failed={len(failed)}"
    )
    if drifted:
        print(
            "Refusing silent SPIR-V replacement. Re-run only the intentionally "
            "changed output with --allow-recompile after qualification: "
            + ", ".join(sorted(drifted)),
            file=sys.stderr,
        )
    return 1 if drifted or failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
