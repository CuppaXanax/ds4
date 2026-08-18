import importlib.util
import json
import os
import tempfile
import time
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("fleet_identity.py")
SPEC = importlib.util.spec_from_file_location("bc250_fleet_identity", MODULE_PATH)
assert SPEC and SPEC.loader
fleet_identity = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fleet_identity)


class FleetIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = json.loads(Path(__file__).with_name("lkg.json").read_text())
        self.commit = self.manifest["runtime_lkg"]["commit"]
        self.binary = self.manifest["runtime_lkg"]["fleet_binary_sha256"]
        self.shader_profiles = self.manifest["runtime_lkg"][
            "runtime_shader_profiles"
        ]
        self.audit = self.root / "fleet.txt"
        topology = self.manifest["topology"]
        nodes = [
            (topology["coordinator"]["host"], "coordinator-ready", topology["coordinator"]["layers"])
        ] + [(node["host"], "worker", node["layers"]) for node in topology["workers"]]
        lines = []
        for host, role, layers in nodes:
            profile_name = "coordinator" if role == "coordinator-ready" else "worker"
            profile = self.shader_profiles[profile_name]
            env = "coordinator-managed" if role == "coordinator-ready" else "f" * 64
            lines.append(
                "BC250_IDENTITY|"
                f"host={host}|commit={self.commit}|binary_sha256={self.binary}|"
                f"source_clean=1|role={role}|layers={layers}|ctx=128000|"
                f"weight_budget_gib=11|shader_count={profile['shader_count']}|"
                f"shader_manifest_sha256={profile['shader_manifest_sha256']}|"
                f"model={self.manifest['model']['default_path']}|env_sha256={env}"
            )
        self.audit.write_text("\n".join(lines) + "\n")

    def tearDown(self):
        self.temp.cleanup()

    def validate(self, now=None):
        return fleet_identity.validate(
            self.manifest,
            self.audit,
            self.commit,
            self.binary,
            self.shader_profiles["coordinator"]["shader_count"],
            self.shader_profiles["coordinator"]["shader_manifest_sha256"],
            self.shader_profiles["worker"]["shader_count"],
            self.shader_profiles["worker"]["shader_manifest_sha256"],
            now=now,
        )

    def test_valid_exact_fleet(self):
        result = self.validate()
        self.assertEqual(result["node_count"], 12)
        self.assertEqual(result["worker_env_sha256"], "f" * 64)

    def test_wrong_context_fails_closed(self):
        text = self.audit.read_text().replace("ctx=128000", "ctx=2048", 1)
        self.audit.write_text(text)
        with self.assertRaisesRegex(ValueError, "ctx='2048'"):
            self.validate()

    def test_stale_audit_fails_closed(self):
        old = time.time() - 901
        os.utime(self.audit, (old, old))
        with self.assertRaisesRegex(ValueError, "audit age"):
            self.validate(now=time.time())

    def test_wrong_shader_manifest_fails_closed(self):
        coordinator_manifest = self.shader_profiles["coordinator"][
            "shader_manifest_sha256"
        ]
        text = self.audit.read_text().replace(coordinator_manifest, "0" * 64, 1)
        self.audit.write_text(text)
        with self.assertRaisesRegex(ValueError, "shader_manifest_sha256"):
            self.validate()

    def test_wrong_worker_shader_manifest_fails_closed(self):
        worker_manifest = self.shader_profiles["worker"][
            "shader_manifest_sha256"
        ]
        text = self.audit.read_text().replace(worker_manifest, "0" * 64, 1)
        self.audit.write_text(text)
        with self.assertRaisesRegex(ValueError, "shader_manifest_sha256"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
