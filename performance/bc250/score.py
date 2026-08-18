#!/usr/bin/env python3
"""Fail-closed scorer for the canonical BC-250 decode benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn


@dataclass(frozen=True)
class Run:
    path: Path
    meta: dict[str, Any]
    row: dict[str, str]
    tps: float


def fail(message: str) -> NoReturn:
    raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lf_normalized(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(path: Path, manifest: dict[str, Any]) -> Run:
    meta_path = path / "meta.json"
    csv_path = path / "bench.csv"
    if not meta_path.is_file() or not csv_path.is_file():
        fail(f"{path}: expected meta.json and bench.csv")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        fail(f"{csv_path}: expected exactly one benchmark row, found {len(rows)}")
    row = rows[0]
    bench = manifest["benchmark"]
    expected = {
        "ctx_tokens": int(bench["ctx_max"]),
        "gen_tokens": int(bench["gen_tokens"]),
        "gen_steady_tokens": int(bench["gen_steady_tokens"]),
    }
    for key, value in expected.items():
        try:
            actual = int(row[key])
        except (KeyError, ValueError) as exc:
            fail(f"{csv_path}: invalid {key}: {exc}")
        if actual != value:
            fail(f"{csv_path}: {key}={actual}, expected {value}")
    meta_expected = {
        "prompt_sha256": str(bench["prompt_sha256"]).lower(),
        "ctx_start": int(bench["ctx_start"]),
        "ctx_max": int(bench["ctx_max"]),
        "ctx_alloc": int(bench["ctx_alloc"]),
        "gen_tokens": int(bench["gen_tokens"]),
        "weight_budget_gib": int(bench["weight_budget_gib"]),
        "dist_activation_bits": int(bench["dist_activation_bits"]),
        "return_code": 0,
    }
    for key, value in meta_expected.items():
        actual = meta.get(key)
        if isinstance(value, str):
            actual = str(actual).lower()
        if actual != value:
            fail(f"{meta_path}: {key}={actual!r}, expected {value!r}")
    try:
        tps = float(row[str(bench["primary_metric"])])
    except (KeyError, ValueError) as exc:
        fail(f"{csv_path}: invalid primary metric: {exc}")
    if not (0.0 < tps < 1000.0):
        fail(f"{csv_path}: implausible TPS {tps}")
    if meta.get("runtime_commit") != meta.get("commit"):
        fail(
            f"{meta_path}: runtime_commit does not match scored commit: "
            f"{meta.get('runtime_commit')!r} vs {meta.get('commit')!r}"
        )
    commit = str(meta.get("commit", "")).lower()
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        fail(f"{meta_path}: invalid commit: {commit!r}")
    if meta.get("fleet_commit") != meta.get("commit"):
        fail(f"{meta_path}: fleet commit does not match scored commit")
    if meta.get("fleet_node_count") != int(manifest["topology"]["blades"]):
        fail(f"{meta_path}: incomplete fleet identity")
    if not isinstance(meta.get("runtime_shader_count"), int) or meta["runtime_shader_count"] <= 0:
        fail(f"{meta_path}: invalid runtime_shader_count")
    for key in (
        "binary_sha256",
        "fleet_binary_sha256",
        "fleet_identity_sha256",
        "fleet_topology_sha256",
        "fleet_env_sha256",
        "model_sha256",
        "runtime_shader_manifest_sha256",
    ):
        value = str(meta.get(key, ""))
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            fail(f"{meta_path}: invalid {key}: {value!r}")
    for key in ("host", "kernel", "governor_state"):
        if not str(meta.get(key, "")).strip():
            fail(f"{meta_path}: missing {key}")
    validate_fleet_artifact(path, meta, manifest)
    return Run(path=path, meta=meta, row=row, tps=tps)


def validate_fleet_artifact(
    run_path: Path, meta: dict[str, Any], manifest: dict[str, Any]
) -> None:
    audit_path = run_path / "fleet-identity.txt"
    if not audit_path.is_file():
        fail(f"{audit_path}: missing captured fleet identity")
    if sha256(audit_path) != meta["fleet_identity_sha256"]:
        fail(f"{audit_path}: hash does not match run metadata")

    topology = manifest["topology"]
    expected = {
        topology["coordinator"]["host"]: (
            "coordinator-ready",
            topology["coordinator"]["layers"],
        )
    }
    expected.update(
        (node["host"], ("worker", node["layers"]))
        for node in topology["workers"]
    )
    records: dict[str, dict[str, str]] = {}
    for raw in audit_path.read_text(encoding="utf-8", errors="strict").splitlines():
        marker = raw.find("BC250_IDENTITY|")
        if marker < 0:
            continue
        fields: dict[str, str] = {}
        for item in raw[marker:].split("|")[1:]:
            key, separator, value = item.partition("=")
            if not separator or key in fields:
                fail(f"{audit_path}: malformed fleet identity field {item!r}")
            fields[key] = value
        host = fields.get("host", "")
        if host in records:
            fail(f"{audit_path}: duplicate fleet identity for {host}")
        records[host] = fields
    if set(records) != set(expected):
        fail(f"{audit_path}: incomplete fleet identity")

    worker_envs: set[str] = set()
    for host, (role, layers) in expected.items():
        row = records[host]
        required = {
            "commit": str(meta["commit"]),
            "binary_sha256": str(meta["fleet_binary_sha256"]),
            "source_clean": "1",
            "role": role,
            "layers": layers,
            "ctx": str(manifest["benchmark"]["ctx_alloc"]),
            "weight_budget_gib": str(manifest["benchmark"]["weight_budget_gib"]),
            "shader_count": str(meta["runtime_shader_count"]),
            "shader_manifest_sha256": str(meta["runtime_shader_manifest_sha256"]),
            "model": str(manifest["model"]["default_path"]),
            "model_size_bytes": str(manifest["model"]["size_bytes"]),
            "model_sample_sha256": str(manifest["model"]["sample_sha256"]),
            "model_sha256": str(manifest["model"]["sha256"]),
        }
        for key, expected_value in required.items():
            if row.get(key) != expected_value:
                fail(
                    f"{audit_path}: {host} {key}={row.get(key)!r}, "
                    f"expected {expected_value!r}"
                )
        if role == "worker":
            worker_envs.add(row.get("env_sha256", ""))
    expected_worker_env = str(manifest["runtime_lkg"]["fleet_worker_env_sha256"])
    if worker_envs != {expected_worker_env}:
        fail(f"{audit_path}: mixed or unexpected worker environments")


def validate_group(name: str, runs: list[Run], expected_count: int) -> None:
    if len(runs) != expected_count:
        fail(f"{name}: expected {expected_count} runs, found {len(runs)}")
    fingerprint_keys = (
        "commit",
        "binary_sha256",
        "fleet_binary_sha256",
        "fleet_topology_sha256",
        "fleet_env_sha256",
        "fleet_node_count",
        "runtime_shader_count",
        "runtime_shader_manifest_sha256",
        "model_sha256",
        "prompt_sha256",
        "ctx_alloc",
        "weight_budget_gib",
        "dist_activation_bits",
        "kernel",
        "governor_state",
        "host",
    )
    first = runs[0].meta
    for run in runs[1:]:
        for key in fingerprint_keys:
            if run.meta.get(key) != first.get(key):
                fail(
                    f"{name}: mixed runtime identity for {key}: "
                    f"{first.get(key)!r} vs {run.meta.get(key)!r}"
                )


def validate_semantic_evidence(
    path: Path, manifest: dict[str, Any], candidate: Run
) -> dict[str, Any]:
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"{path}: invalid semantic evidence: {exc}")

    qualification = manifest["runtime_lkg"]["promotion_evidence"][
        "semantic_qualification"
    ]
    expected_identity = {
        "status": "pass",
        "fixture": qualification["fixture"],
        "fixture_sha256": qualification["fixture_sha256"],
        "runtime_commit": candidate.meta["commit"],
        "fleet_binary_sha256": candidate.meta["fleet_binary_sha256"],
        "runtime_shader_count": candidate.meta["runtime_shader_count"],
        "runtime_shader_manifest_sha256": candidate.meta[
            "runtime_shader_manifest_sha256"
        ],
        "model_path": manifest["model"]["default_path"],
        "model_size_bytes": manifest["model"]["size_bytes"],
        "expected_model_sha256": manifest["model"]["sha256"],
        "model_sample_sha256": manifest["model"]["sample_sha256"],
        "fleet_identity_sha256": candidate.meta["fleet_identity_sha256"],
    }
    for key, expected in expected_identity.items():
        if evidence.get(key) != expected:
            fail(
                f"{path}: semantic {key}={evidence.get(key)!r}, "
                f"expected {expected!r}"
            )

    fixture_path = Path(__file__).resolve().parents[2] / qualification["fixture"]
    if (
        not fixture_path.is_file()
        or sha256_lf_normalized(fixture_path) != qualification["fixture_sha256"]
    ):
        fail(f"official semantic fixture is missing or changed: {fixture_path}")

    cases = evidence.get("cases")
    if not isinstance(cases, dict):
        fail(f"{path}: semantic cases are missing")
    required_cases = ("short_reasoning_plain", "short_italian_fact")
    if set(cases) != set(required_cases):
        fail(f"{path}: semantic cases differ from the required official cases")
    normalized_cases: dict[str, Any] = {}
    for case in required_cases:
        expected_case = qualification[case]
        actual_case = cases.get(case)
        if not isinstance(actual_case, dict):
            fail(f"{path}: semantic case {case} is malformed")
        expected_text = expected_case["expected"]
        if (
            actual_case.get("expected") != expected_text
            or actual_case.get("actual") != expected_text
            or actual_case.get("token_ids") != expected_case["token_ids"]
            or actual_case.get("prompt_sha256") != expected_case["prompt_sha256"]
        ):
            fail(f"{path}: semantic case {case} does not match the official result")
        prompt_path = Path(__file__).resolve().parents[2] / expected_case["prompt"]
        if (
            not prompt_path.is_file()
            or sha256_lf_normalized(prompt_path) != expected_case["prompt_sha256"]
        ):
            fail(f"official semantic prompt is missing or changed: {prompt_path}")
        raw_name = str(actual_case.get("raw_artifact", ""))
        log_name = str(actual_case.get("log", ""))
        if Path(raw_name).name != raw_name or Path(log_name).name != log_name:
            fail(f"{path}: semantic case {case} uses a non-local artifact path")
        raw_path = path.parent / raw_name
        log_path = path.parent / log_name
        if (
            not raw_path.is_file()
            or sha256(raw_path) != actual_case.get("raw_artifact_sha256")
        ):
            fail(f"{path}: semantic case {case} raw artifact is missing or changed")
        if not log_path.is_file() or sha256(log_path) != actual_case.get("log_sha256"):
            fail(f"{path}: semantic case {case} log is missing or changed")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        steps = raw.get("steps", [])
        raw_text = "".join(
            step.get("selected", {}).get("text", "") for step in steps
        )
        raw_ids = [step.get("selected", {}).get("id") for step in steps]
        if raw_text != expected_text or raw_ids != expected_case["token_ids"]:
            fail(f"{path}: semantic case {case} raw output is incorrect")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if re.search(
            r"shader not found|fallback[^0-9]*[1-9]|out of memory|"
            r"oom-kill|gpu reset|device lost",
            log_text,
            flags=re.IGNORECASE,
        ):
            fail(f"{path}: semantic case {case} log contains a runtime failure marker")
        normalized_cases[case] = actual_case

    return {
        "status": "pass",
        "artifact_path": str(path),
        "artifact_sha256": sha256(path),
        "fixture_sha256": qualification["fixture_sha256"],
        "runtime_commit": candidate.meta["commit"],
        "fleet_binary_sha256": candidate.meta["fleet_binary_sha256"],
        "runtime_shader_manifest_sha256": candidate.meta[
            "runtime_shader_manifest_sha256"
        ],
        "model_sha256": candidate.meta["model_sha256"],
        "fleet_identity_sha256": candidate.meta["fleet_identity_sha256"],
        "cases": normalized_cases,
    }


def summary(runs: list[Run]) -> dict[str, Any]:
    values = [run.tps for run in runs]
    return {
        "runs": [str(run.path) for run in runs],
        "commit": runs[0].meta["commit"],
        "binary_sha256": runs[0].meta["binary_sha256"],
        "fleet_binary_sha256": runs[0].meta["fleet_binary_sha256"],
        "fleet_identity_sha256": runs[0].meta["fleet_identity_sha256"],
        "runtime_shader_manifest_sha256": runs[0].meta[
            "runtime_shader_manifest_sha256"
        ],
        "model_sha256": runs[0].meta["model_sha256"],
        "samples": values,
        "median_tps": statistics.median(values),
        "min_tps": min(values),
        "max_tps": max(values),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("lkg.json"),
    )
    parser.add_argument("--baseline-run", action="append", type=Path, required=True)
    parser.add_argument("--candidate-run", action="append", type=Path, default=[])
    parser.add_argument("--control-artifact", type=Path)
    parser.add_argument("--candidate-artifact", type=Path)
    parser.add_argument("--activation-evidence", type=Path)
    parser.add_argument("--semantic-evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    manifest = load_manifest(args.manifest)
    repetitions = int(manifest["benchmark"]["repetitions"])
    baseline = [load_run(path, manifest) for path in args.baseline_run]
    validate_group("baseline", baseline, repetitions)
    lkg_commit = str(manifest["runtime_lkg"]["commit"])
    if baseline[0].meta.get("commit") != lkg_commit:
        fail(
            f"baseline commit {baseline[0].meta.get('commit')} does not match LKG {lkg_commit}"
        )
    coordinator_profile = manifest["runtime_lkg"]["runtime_shader_profiles"][
        "coordinator"
    ]
    lkg_shader_count = int(coordinator_profile["shader_count"])
    if baseline[0].meta.get("runtime_shader_count") != lkg_shader_count:
        fail(
            "baseline runtime shader count "
            f"{baseline[0].meta.get('runtime_shader_count')!r} does not match LKG "
            f"{lkg_shader_count}"
        )
    lkg_shader_manifest = str(coordinator_profile["shader_manifest_sha256"]).lower()
    if str(
        baseline[0].meta.get("runtime_shader_manifest_sha256", "")
    ).lower() != lkg_shader_manifest:
        fail("baseline runtime shader manifest does not match the LKG")
    lkg_binary = str(manifest["runtime_lkg"]["fleet_binary_sha256"]).lower()
    if str(baseline[0].meta.get("fleet_binary_sha256", "")).lower() != lkg_binary:
        fail("baseline fleet binary hash does not match the LKG")
    if str(baseline[0].meta.get("model_sha256", "")).lower() != str(
        manifest["model"]["sha256"]
    ).lower():
        fail("baseline model hash does not match the LKG")
    profiles = manifest["runtime_lkg"]["runtime_shader_profiles"]
    if manifest["promotion"].get("requires_uniform_shader_manifest") and (
        profiles["coordinator"] != profiles["worker"]
    ):
        fail("LKG role shader profiles differ despite the uniform-manifest gate")

    result: dict[str, Any] = {
        "schema_version": 1,
        "metric": manifest["benchmark"]["primary_metric"],
        "baseline": summary(baseline),
        "targets": {
            "first_milestone_tps": manifest["benchmark"]["first_milestone_tps"],
            "target_tps": manifest["benchmark"]["target_tps"],
        },
        "verdict": "baseline_recorded",
        "user_approval_required": True,
    }

    if args.candidate_run:
        if not (
            args.control_artifact
            and args.candidate_artifact
            and args.activation_evidence
            and args.semantic_evidence
        ):
            fail(
                "candidate scoring requires control/candidate artifacts, activation "
                "evidence, and semantic evidence"
            )
        for evidence in (
            args.control_artifact,
            args.candidate_artifact,
            args.activation_evidence,
            args.semantic_evidence,
        ):
            if not evidence.is_file() or evidence.stat().st_size == 0:
                fail(f"missing or empty evidence: {evidence}")
        candidate = [load_run(path, manifest) for path in args.candidate_run]
        validate_group("candidate", candidate, repetitions)
        if candidate[0].meta.get("commit") == lkg_commit:
            fail("candidate commit is identical to the LKG")
        if candidate[0].meta.get("model_sha256") != baseline[0].meta.get("model_sha256"):
            fail("candidate and baseline model hashes differ")
        for key in ("host", "kernel", "governor_state", "fleet_topology_sha256"):
            if candidate[0].meta.get(key) != baseline[0].meta.get(key):
                fail(f"candidate and baseline {key} differ")

        control_hash = sha256(args.control_artifact)
        candidate_hash = sha256(args.candidate_artifact)
        exact = control_hash == candidate_hash
        semantic = validate_semantic_evidence(
            args.semantic_evidence, manifest, candidate[0]
        )
        base_median = statistics.median(run.tps for run in baseline)
        candidate_median = statistics.median(run.tps for run in candidate)
        ratio = candidate_median / base_median
        promote_ratio = float(manifest["promotion"]["automatic_eligibility_ratio"])
        regression_ratio = float(manifest["promotion"]["regression_ratio"])
        if not exact or ratio < regression_ratio:
            verdict = "reject"
        elif ratio >= promote_ratio:
            verdict = "promote_eligible"
        else:
            verdict = "hold"

        result.update(
            {
                "candidate": summary(candidate),
                "ratio": ratio,
                "percent_change": (ratio - 1.0) * 100.0,
                "exactness": {
                    "match": exact,
                    "control_sha256": control_hash,
                    "candidate_sha256": candidate_hash,
                },
                "activation_evidence": {
                    "path": str(args.activation_evidence),
                    "sha256": sha256(args.activation_evidence),
                },
                "semantic_qualification": semantic,
                "fleet_qualification": {
                    "uniform_shader_manifest": True,
                    "model_identity_match": True,
                    "identity_sha256": candidate[0].meta["fleet_identity_sha256"],
                },
                "verdict": verdict,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] != "reject" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"score error: {exc}", file=sys.stderr)
        raise SystemExit(2)
