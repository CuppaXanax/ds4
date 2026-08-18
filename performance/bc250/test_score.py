import csv
import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("score.py")
SPEC = importlib.util.spec_from_file_location("bc250_score", MODULE_PATH)
assert SPEC and SPEC.loader
score = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = score
SPEC.loader.exec_module(score)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = json.loads(Path(__file__).with_name("lkg.json").read_text())
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))

    def tearDown(self):
        self.temp.cleanup()

    def make_run(self, name, commit, binary, tps, shader_count=None):
        path = self.root / name
        path.mkdir()
        bench = self.manifest["benchmark"]
        coordinator_profile = self.manifest["runtime_lkg"][
            "runtime_shader_profiles"
        ]["coordinator"]
        meta = {
            "commit": commit,
            "binary_sha256": binary * 64 if len(binary) == 1 else binary,
            "fleet_binary_sha256": self.manifest["runtime_lkg"]["fleet_binary_sha256"],
            "fleet_commit": commit,
            "fleet_identity_sha256": "d" * 64,
            "fleet_topology_sha256": "e" * 64,
            "fleet_env_sha256": "f" * 64,
            "fleet_node_count": self.manifest["topology"]["blades"],
            "runtime_shader_count": (
                coordinator_profile["shader_count"]
                if shader_count is None
                else shader_count
            ),
            "runtime_shader_manifest_sha256": coordinator_profile[
                "shader_manifest_sha256"
            ],
            "model_sha256": "a" * 64,
            "prompt_sha256": bench["prompt_sha256"],
            "ctx_start": bench["ctx_start"],
            "ctx_max": bench["ctx_max"],
            "ctx_alloc": bench["ctx_alloc"],
            "gen_tokens": bench["gen_tokens"],
            "weight_budget_gib": bench["weight_budget_gib"],
            "dist_activation_bits": bench["dist_activation_bits"],
            "kernel": "test-kernel",
            "governor_state": "test-governor",
            "host": "coordinator",
            "return_code": 0,
        }
        meta["runtime_commit"] = commit
        (path / "meta.json").write_text(json.dumps(meta))
        with (path / "bench.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "ctx_tokens",
                    "prefill_tokens",
                    "prefill_tps",
                    "gen_tokens",
                    "gen_tps",
                    "gen_first_ms",
                    "gen_steady_tokens",
                    "gen_steady_tps",
                    "kvcache_bytes",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "ctx_tokens": bench["ctx_max"],
                    "prefill_tokens": bench["ctx_max"],
                    "prefill_tps": 25,
                    "gen_tokens": bench["gen_tokens"],
                    "gen_tps": tps,
                    "gen_first_ms": 200,
                    "gen_steady_tokens": bench["gen_steady_tokens"],
                    "gen_steady_tps": tps,
                    "kvcache_bytes": 0,
                }
            )
        return path

    def make_group(self, prefix, commit, binary, values, shader_count=None):
        return [
            self.make_run(
                f"{prefix}-{i}", commit, binary, value, shader_count=shader_count
            )
            for i, value in enumerate(values)
        ]

    def invoke_score(self, argv):
        with contextlib.redirect_stdout(io.StringIO()):
            return score.main(argv)

    def test_promote_eligible(self):
        lkg = self.manifest["runtime_lkg"]["commit"]
        baseline = self.make_group("b", lkg, "b", [5.0, 5.1, 5.2])
        candidate = self.make_group(
            "c", "c" * 40, "c", [5.4, 5.5, 5.6], shader_count=63
        )
        control = self.root / "control.json"
        cand = self.root / "candidate.json"
        activation = self.root / "activation.log"
        control.write_text("same")
        cand.write_text("same")
        activation.write_text("candidate path active")
        output = self.root / "verdict.json"
        argv = ["--manifest", str(self.manifest_path)]
        for run in baseline:
            argv += ["--baseline-run", str(run)]
        for run in candidate:
            argv += ["--candidate-run", str(run)]
        argv += [
            "--control-artifact", str(control),
            "--candidate-artifact", str(cand),
            "--activation-evidence", str(activation),
            "--output", str(output),
        ]
        self.assertEqual(self.invoke_score(argv), 0)
        self.assertEqual(json.loads(output.read_text())["verdict"], "promote_eligible")

    def test_regression_rejected(self):
        lkg = self.manifest["runtime_lkg"]["commit"]
        baseline = self.make_group("b", lkg, "b", [5.0, 5.1, 5.2])
        candidate = self.make_group("c", "c" * 40, "c", [4.7, 4.8, 4.9])
        control = self.root / "control.json"
        cand = self.root / "candidate.json"
        activation = self.root / "activation.log"
        control.write_text("same")
        cand.write_text("same")
        activation.write_text("candidate path active")
        output = self.root / "verdict.json"
        argv = ["--manifest", str(self.manifest_path)]
        for run in baseline:
            argv += ["--baseline-run", str(run)]
        for run in candidate:
            argv += ["--candidate-run", str(run)]
        argv += [
            "--control-artifact", str(control),
            "--candidate-artifact", str(cand),
            "--activation-evidence", str(activation),
            "--output", str(output),
        ]
        self.assertEqual(self.invoke_score(argv), 1)
        self.assertEqual(json.loads(output.read_text())["verdict"], "reject")

    def test_output_mismatch_rejected(self):
        lkg = self.manifest["runtime_lkg"]["commit"]
        baseline = self.make_group("b", lkg, "b", [5.0, 5.1, 5.2])
        candidate = self.make_group("c", "c" * 40, "c", [5.4, 5.5, 5.6])
        control = self.root / "control.json"
        cand = self.root / "candidate.json"
        activation = self.root / "activation.log"
        control.write_text("control")
        cand.write_text("different")
        activation.write_text("candidate path active")
        output = self.root / "verdict.json"
        argv = ["--manifest", str(self.manifest_path)]
        for run in baseline:
            argv += ["--baseline-run", str(run)]
        for run in candidate:
            argv += ["--candidate-run", str(run)]
        argv += [
            "--control-artifact", str(control),
            "--candidate-artifact", str(cand),
            "--activation-evidence", str(activation),
            "--output", str(output),
        ]
        self.assertEqual(self.invoke_score(argv), 1)
        verdict = json.loads(output.read_text())
        self.assertEqual(verdict["verdict"], "reject")
        self.assertFalse(verdict["exactness"]["match"])

    def test_baseline_shader_manifest_mismatch_fails_closed(self):
        lkg = self.manifest["runtime_lkg"]["commit"]
        baseline = self.make_group("b", lkg, "b", [5.0, 5.1, 5.2])
        meta_path = baseline[0] / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["runtime_shader_manifest_sha256"] = "0" * 64
        meta_path.write_text(json.dumps(meta))
        argv = ["--manifest", str(self.manifest_path)]
        for run in baseline:
            argv += ["--baseline-run", str(run)]
        argv += ["--output", str(self.root / "verdict.json")]
        with self.assertRaisesRegex(ValueError, "mixed runtime identity"):
            self.invoke_score(argv)


if __name__ == "__main__":
    unittest.main()
