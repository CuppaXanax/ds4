#!/usr/bin/env python3
"""Validate a fresh, read-only BC-250 fleet identity audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, NoReturn


def fail(message: str) -> NoReturn:
    raise ValueError(message)


def validate(
    manifest: dict[str, Any],
    audit_path: Path,
    expected_commit: str,
    expected_binary: str,
    expected_shader_count: int,
    expected_shader_manifest: str,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    now = time.time() if now is None else now
    max_age = int(manifest["benchmark"]["fleet_identity_max_age_seconds"])
    age = now - audit_path.stat().st_mtime
    if age < -60 or age > max_age:
        fail(f"fleet identity audit age {age:.0f}s is outside 0..{max_age}s")

    coordinator = manifest["topology"]["coordinator"]
    expected = {
        coordinator["host"]: ("coordinator-ready", coordinator["layers"])
    }
    expected.update(
        (node["host"], ("worker", node["layers"]))
        for node in manifest["topology"]["workers"]
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
                fail(f"malformed fleet identity field: {item!r}")
            fields[key] = value
        host = fields.get("host", "")
        if host in records:
            fail(f"duplicate fleet identity for {host}")
        records[host] = fields

    if set(records) != set(expected):
        fail(f"fleet hosts differ: found={sorted(records)} expected={sorted(expected)}")
    if len(records) != int(manifest["topology"]["blades"]):
        fail(f"fleet node count {len(records)} does not match topology")

    worker_envs: set[str] = set()
    for host, (role, layers) in expected.items():
        row = records[host]
        checks = {
            "binary_sha256": expected_binary,
            "source_clean": "1",
            "role": role,
            "layers": layers,
            "ctx": str(manifest["benchmark"]["ctx_alloc"]),
            "weight_budget_gib": str(manifest["benchmark"]["weight_budget_gib"]),
            "shader_count": str(expected_shader_count),
            "shader_manifest_sha256": expected_shader_manifest,
            "model": manifest["model"]["default_path"],
        }
        if role == "worker":
            checks["commit"] = expected_commit
            worker_envs.add(row.get("env_sha256", ""))
        for key, wanted in checks.items():
            if row.get(key) != wanted:
                fail(f"{host}: {key}={row.get(key)!r}, expected {wanted!r}")

    worker_env = next(iter(worker_envs), "")
    if len(worker_envs) != 1 or not re.fullmatch(r"[0-9a-f]{64}", worker_env):
        fail(f"worker environments are mixed or invalid: {sorted(worker_envs)}")

    topology = [
        {"host": host, "role": expected[host][0], "layers": expected[host][1]}
        for host in sorted(expected)
    ]
    topology_hash = hashlib.sha256(
        json.dumps(topology, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "identity_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "topology_sha256": topology_hash,
        "worker_env_sha256": worker_env,
        "node_count": len(records),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-binary", required=True)
    parser.add_argument("--expected-shader-count", type=int, required=True)
    parser.add_argument("--expected-shader-manifest", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = validate(
        manifest,
        args.audit,
        args.expected_commit,
        args.expected_binary,
        args.expected_shader_count,
        args.expected_shader_manifest,
    )
    print(result["identity_sha256"])
    print(result["topology_sha256"])
    print(result["worker_env_sha256"])
    print(result["node_count"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"fleet identity error: {exc}", file=sys.stderr)
        raise SystemExit(2)
