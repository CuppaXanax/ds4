from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import compile as shader_compile


class FrozenShaderCompileTests(unittest.TestCase):
    def run_compile(
        self,
        root: Path,
        compiled_bytes: bytes,
        *,
        allow_recompile: bool = False,
    ) -> int:
        source = root / "probe.comp"
        source.write_text("#version 450\n", encoding="utf-8")
        job = shader_compile.Job(source, "probe")

        def fake_compile(
            unused_job: shader_compile.Job, destination: Path
        ) -> subprocess.CompletedProcess[str]:
            destination.write_bytes(compiled_bytes)
            return subprocess.CompletedProcess([], 0, "", "")

        args = argparse.Namespace(
            shaders=["probe"],
            allow_recompile=["probe"] if allow_recompile else [],
        )
        with (
            mock.patch.object(shader_compile, "OUT_DIR", root / "spv"),
            mock.patch.object(shader_compile, "all_jobs", return_value=[job]),
            mock.patch.object(shader_compile, "compile_job", side_effect=fake_compile),
            mock.patch.object(shader_compile, "parse_args", return_value=args),
        ):
            return shader_compile.main()

    def test_missing_output_is_installed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(self.run_compile(root, b"new"), 0)
            self.assertEqual((root / "spv" / "probe.spv").read_bytes(), b"new")

    def test_different_existing_output_is_preserved_and_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "spv" / "probe.spv"
            output.parent.mkdir()
            output.write_bytes(b"lkg")
            self.assertEqual(self.run_compile(root, b"drift"), 1)
            self.assertEqual(output.read_bytes(), b"lkg")

    def test_allowlisted_output_can_be_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "spv" / "probe.spv"
            output.parent.mkdir()
            output.write_bytes(b"lkg")
            self.assertEqual(
                self.run_compile(root, b"qualified", allow_recompile=True), 0
            )
            self.assertEqual(output.read_bytes(), b"qualified")


if __name__ == "__main__":
    unittest.main()
