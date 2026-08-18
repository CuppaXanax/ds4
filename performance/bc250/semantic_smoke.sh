#!/usr/bin/env bash
set -euo pipefail

if [[ "${DS4_USER_APPROVED_SEMANTIC_SMOKE:-}" != 1 ]]; then
    echo 'Refusing semantic smoke without DS4_USER_APPROVED_SEMANTIC_SMOKE=1.' >&2
    exit 2
fi
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 FLEET_IDENTITY_AUDIT [OUTPUT_JSON]" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$script_dir/../.." && pwd)"
manifest="$script_dir/lkg.json"
fleet_audit="$1"
output="${2:-/tmp/bc250-semantic-verdict.json}"
[[ -f "$fleet_audit" ]] || {
    echo "Missing fleet identity audit: $fleet_audit" >&2
    exit 2
}
if pgrep -x ds4 >/dev/null || pgrep -x ds4-bench >/dev/null; then
    echo 'Refusing semantic smoke while the coordinator is busy.' >&2
    exit 3
fi

cd "$repo"
non_shader_status="$(git status --porcelain --untracked-files=all |
    grep -vE '^.. vulkan/shaders/spv/[^/]+\.spv$' || true)"
if [[ -n "$non_shader_status" ]]; then
    echo 'Refusing semantic smoke on a dirty runtime checkout.' >&2
    printf '%s\n' "$non_shader_status" >&2
    exit 4
fi

mapfile -t cfg < <(python3 - "$manifest" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
s = m["runtime_lkg"]["promotion_evidence"]["semantic_qualification"]
print(m["model"]["default_path"])
print(m["model"]["size_bytes"])
print(m["model"]["sha256"])
print(m["model"]["sample_sha256"])
print(s["fixture"])
print(s["fixture_sha256"])
print(s["short_reasoning_plain"]["prompt"])
print(s["short_reasoning_plain"]["prompt_sha256"])
print(s["short_italian_fact"]["prompt"])
print(s["short_italian_fact"]["prompt_sha256"])
PY
)
model="${cfg[0]}"
expected_model_size="${cfg[1]}"
expected_model_hash="${cfg[2]}"
expected_model_sample_hash="${cfg[3]}"
fixture="${cfg[4]}"
expected_fixture_hash="${cfg[5]}"
reasoning_prompt="${cfg[6]}"
expected_reasoning_prompt_hash="${cfg[7]}"
italian_prompt="${cfg[8]}"
expected_italian_prompt_hash="${cfg[9]}"

[[ -f "$model" ]] || { echo "Missing model: $model" >&2; exit 5; }
[[ -f "$fixture" ]] || { echo "Missing fixture: $fixture" >&2; exit 5; }
[[ -f "$reasoning_prompt" ]] || { echo "Missing prompt: $reasoning_prompt" >&2; exit 5; }
[[ -f "$italian_prompt" ]] || { echo "Missing prompt: $italian_prompt" >&2; exit 5; }
actual_model_size="$(stat -c '%s' "$model")"
[[ "$actual_model_size" == "$expected_model_size" ]] || {
    echo "Model size mismatch: $actual_model_size" >&2
    exit 5
}
actual_fixture_hash="$(sha256sum "$fixture" | awk '{print tolower($1)}')"
[[ "$actual_fixture_hash" == "$expected_fixture_hash" ]] || {
    echo "Fixture hash mismatch: $actual_fixture_hash" >&2
    exit 5
}
actual_reasoning_prompt_hash="$(sha256sum "$reasoning_prompt" | awk '{print tolower($1)}')"
actual_italian_prompt_hash="$(sha256sum "$italian_prompt" | awk '{print tolower($1)}')"
[[ "$actual_reasoning_prompt_hash" == "$expected_reasoning_prompt_hash" ]] || {
    echo "Reasoning prompt hash mismatch: $actual_reasoning_prompt_hash" >&2
    exit 5
}
[[ "$actual_italian_prompt_hash" == "$expected_italian_prompt_hash" ]] || {
    echo "Italian prompt hash mismatch: $actual_italian_prompt_hash" >&2
    exit 5
}
sample_bytes=4194304
sample_mid=$((actual_model_size / 2 - sample_bytes / 2))
((sample_mid < 0)) && sample_mid=0
actual_model_sample_hash="$({
    dd if="$model" iflag=count_bytes count="$sample_bytes" status=none
    dd if="$model" iflag=skip_bytes,count_bytes skip="$sample_mid" \
        count="$sample_bytes" status=none
    tail -c "$sample_bytes" "$model"
} | sha256sum | awk '{print tolower($1)}')"
[[ "$actual_model_sample_hash" == "$expected_model_sample_hash" ]] || {
    echo "Model sample hash mismatch: $actual_model_sample_hash" >&2
    exit 5
}

shader_list="$(find vulkan/shaders/spv -maxdepth 1 -type f -name '*.spv' -print0 |
    sort -z | xargs -0 sha256sum)"
shader_count="$(printf '%s\n' "$shader_list" | sed '/^$/d' | wc -l)"
shader_manifest="$(printf '%s\n' "$shader_list" | sha256sum | awk '{print tolower($1)}')"
runtime_commit="$(git rev-parse HEAD)"
binary_hash="$(sha256sum ./ds4 | awk '{print tolower($1)}')"
audit_summary="$(python3 "$script_dir/fleet_identity.py" \
    --manifest "$manifest" \
    --audit "$fleet_audit" \
    --expected-commit "$runtime_commit" \
    --expected-binary "$binary_hash" \
    --coordinator-shader-count "$shader_count" \
    --coordinator-shader-manifest "$shader_manifest" \
    --worker-shader-count "$shader_count" \
    --worker-shader-manifest "$shader_manifest")"
mapfile -t audit_cfg <<<"$audit_summary"
fleet_identity_hash="${audit_cfg[0]}"
mkdir -p "$(dirname "$output")"
semantic_prefix="${output%.json}"

run_case() {
    local id=$1 prompt=$2 tokens=$3 expected=$4
    local artifact="${semantic_prefix}-${id}-raw.json"
    local log="${semantic_prefix}-${id}.log"
    rm -f -- "$artifact" "$log" /tmp/ds4.lock
    set +e
    env -i HOME="$HOME" USER="${USER:-xander}" PATH="$PATH" LANG=C.UTF-8 \
        DS4_VULKAN_WEIGHT_BUDGET_GB=11 \
        DS4_MTP_SPEC_DISABLE=1 \
        timeout 600 ./ds4 --vulkan -m "$model" \
        --role coordinator --layers 0:3 \
        --listen 192.168.42.42 1234 --dist-activation-bits 32 \
        --ctx 128000 --nothink --temp 0 -n "$tokens" \
        --prompt-file "$prompt" \
        --dump-logprobs "$artifact" --logprobs-top-k 20 \
        >"$log" 2>&1
    local rc=$?
    set -e
    [[ $rc -eq 0 && -s "$artifact" ]] || {
        printf 'BC250_SEMANTIC|id=%s|pass=0|rc=%s|log=%s\n' "$id" "$rc" "$log"
        return 1
    }
    if grep -Eqi 'shader not found|fallback[^0-9]*[1-9]|out of memory|oom-kill|gpu reset|device lost' "$log"; then
        printf 'BC250_SEMANTIC|id=%s|pass=0|reason=runtime-marker|log=%s\n' "$id" "$log"
        return 1
    fi
    python3 - "$id" "$artifact" "$expected" <<'PY'
import json, sys
case, path, expected = sys.argv[1:]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
steps = data.get("steps", [])
text = "".join(step.get("selected", {}).get("text", "") for step in steps)
ids = [step.get("selected", {}).get("id") for step in steps]
passed = text == expected
print(f"BC250_SEMANTIC|id={case}|pass={int(passed)}|text={text!r}|ids={ids}|expected={expected!r}")
raise SystemExit(0 if passed else 1)
PY
}

run_case short_reasoning_plain \
    "$reasoning_prompt" \
    1 '16'
run_case short_italian_fact \
    "$italian_prompt" \
    4 'Ada Lovelace'

python3 - "$output" "$fixture" "$actual_fixture_hash" "$runtime_commit" \
    "$binary_hash" "$shader_count" "$shader_manifest" "$model" \
    "$actual_model_size" "$expected_model_hash" "$actual_model_sample_hash" \
    "$fleet_identity_hash" "$actual_reasoning_prompt_hash" \
    "$actual_italian_prompt_hash" \
    "${semantic_prefix}-short_reasoning_plain-raw.json" \
    "${semantic_prefix}-short_reasoning_plain.log" \
    "${semantic_prefix}-short_italian_fact-raw.json" \
    "${semantic_prefix}-short_italian_fact.log" <<'PY'
import hashlib, json, os, sys, tempfile

(
    _, output, fixture, fixture_hash, commit, binary_hash, shader_count,
    shader_manifest, model, model_size, expected_model_hash, model_sample_hash,
    fleet_identity_hash, reasoning_prompt_hash, italian_prompt_hash,
    reasoning_path, reasoning_log, italian_path, italian_log,
) = sys.argv

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def result(path, log, expected):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    steps = data.get("steps", [])
    actual = "".join(step.get("selected", {}).get("text", "") for step in steps)
    token_ids = [step.get("selected", {}).get("id") for step in steps]
    if actual != expected:
        raise SystemExit(f"semantic result changed while recording: {actual!r}")
    return {
        "expected": expected,
        "actual": actual,
        "token_ids": token_ids,
        "raw_artifact": os.path.basename(path),
        "raw_artifact_sha256": sha256(path),
        "log": os.path.basename(log),
        "log_sha256": sha256(log),
    }

evidence = {
    "schema_version": 1,
    "status": "pass",
    "fixture": fixture,
    "fixture_sha256": fixture_hash,
    "runtime_commit": commit,
    "fleet_binary_sha256": binary_hash,
    "runtime_shader_count": int(shader_count),
    "runtime_shader_manifest_sha256": shader_manifest,
    "model_path": model,
    "model_size_bytes": int(model_size),
    "expected_model_sha256": expected_model_hash,
    "model_sample_sha256": model_sample_hash,
    "fleet_identity_sha256": fleet_identity_hash,
    "cases": {
        "short_reasoning_plain": {
            **result(reasoning_path, reasoning_log, "16"),
            "prompt_sha256": reasoning_prompt_hash,
        },
        "short_italian_fact": {
            **result(italian_path, italian_log, "Ada Lovelace"),
            "prompt_sha256": italian_prompt_hash,
        },
    },
}
directory = os.path.dirname(os.path.abspath(output))
os.makedirs(directory, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=".semantic-", suffix=".json", dir=directory)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(temporary, output)
except BaseException:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
    raise
PY

printf 'BC250_SEMANTIC|result=PASS|evidence=%s\n' "$output"
