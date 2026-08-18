#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 OUTPUT_DIRECTORY FLEET_IDENTITY_AUDIT" >&2
    exit 2
fi
if [[ "${DS4_USER_APPROVED_BENCHMARK:-}" != "1" ]]; then
    echo "refusing to own the coordinator without DS4_USER_APPROVED_BENCHMARK=1" >&2
    exit 3
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
manifest="$script_dir/lkg.json"
output_dir="$1"
fleet_audit="$2"

if [[ -e "$output_dir" ]]; then
    echo "output already exists: $output_dir" >&2
    exit 4
fi
if [[ ! -f "$fleet_audit" ]]; then
    echo "missing fleet identity audit: $fleet_audit" >&2
    exit 4
fi
if pgrep -x ds4 >/dev/null || pgrep -x ds4-bench >/dev/null; then
    echo "a coordinator process already exists; no process was touched" >&2
    pgrep -a -x ds4 || true
    pgrep -a -x ds4-bench || true
    exit 5
fi

cd "$repo_dir"
non_shader_status="$(git status --porcelain --untracked-files=all |
    grep -vE '^.. vulkan/shaders/spv/[^/]+\.spv$' || true)"
if [[ -n "$non_shader_status" ]]; then
    echo "refusing to benchmark a dirty checkout" >&2
    printf '%s\n' "$non_shader_status" >&2
    exit 6
fi

mapfile -t cfg < <(python3 - "$manifest" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
print(m["runtime_lkg"]["commit"])
print(m["runtime_lkg"]["fleet_binary_sha256"])
profiles = m["runtime_lkg"]["runtime_shader_profiles"]
print(profiles["coordinator"]["shader_count"])
print(profiles["coordinator"]["shader_manifest_sha256"])
print(profiles["worker"]["shader_count"])
print(profiles["worker"]["shader_manifest_sha256"])
print(m["model"]["default_path"])
print(m["benchmark"]["prompt_path"])
print(m["benchmark"]["prompt_sha256"])
print(m["benchmark"]["ctx_start"])
print(m["benchmark"]["ctx_max"])
print(m["benchmark"]["ctx_alloc"])
print(m["benchmark"]["gen_tokens"])
print(m["benchmark"]["weight_budget_gib"])
print(m["benchmark"]["dist_activation_bits"])
print(m["model"]["size_bytes"])
print(m["model"]["sha256"])
print(m["model"]["sample_sha256"])
print(int(bool(m["promotion"].get("requires_uniform_shader_manifest"))))
PY
)

expected_commit="${DS4_EXPECTED_COMMIT:-${cfg[0]}}"
expected_lkg_binary="${cfg[1]}"
expected_coordinator_shaders="${DS4_EXPECTED_COORDINATOR_SHADER_COUNT:-${cfg[2]}}"
expected_coordinator_manifest="${DS4_EXPECTED_COORDINATOR_SHADER_MANIFEST:-${cfg[3]}}"
expected_worker_shaders="${DS4_EXPECTED_WORKER_SHADER_COUNT:-${cfg[4]}}"
expected_worker_manifest="${DS4_EXPECTED_WORKER_SHADER_MANIFEST:-${cfg[5]}}"
model_path="${DS4_BENCH_MODEL:-${cfg[6]}}"
prompt_path="${cfg[7]}"
expected_prompt_hash="${cfg[8]}"
ctx_start="${cfg[9]}"
ctx_max="${cfg[10]}"
ctx_alloc="${cfg[11]}"
gen_tokens="${cfg[12]}"
weight_budget="${cfg[13]}"
dist_bits="${cfg[14]}"
expected_model_size="${cfg[15]}"
expected_model_hash="${cfg[16]}"
expected_model_sample_hash="${cfg[17]}"
uniform_shader_manifest_required="${cfg[18]}"

actual_commit="$(git rev-parse HEAD)"
if ! git cat-file -e "$expected_commit^{commit}" 2>/dev/null; then
    echo "unknown expected runtime commit: $expected_commit" >&2
    exit 7
fi
if ! git merge-base --is-ancestor "$expected_commit" "$actual_commit"; then
    echo "checkout $actual_commit does not descend from expected runtime $expected_commit" >&2
    exit 7
fi
runtime_delta="$(git diff --name-only "$expected_commit" -- |
    grep -Ev '^(AGENTS\.md|\.gitattributes|\.gitignore|\.githooks/|performance/bc250/|vulkan/shaders/(compile\.py|test_compile\.py)$)' || true)"
if [[ -n "$runtime_delta" ]]; then
    echo "checkout contains runtime changes beyond expected commit $expected_commit:" >&2
    echo "$runtime_delta" >&2
    exit 7
fi
if [[ ! -x ./ds4-bench ]]; then
    echo "missing executable ./ds4-bench; build the exact checkout first" >&2
    exit 8
fi
if [[ ! -x ./ds4 ]]; then
    echo "missing executable ./ds4; build the exact checkout first" >&2
    exit 8
fi
if [[ ! -f "$model_path" ]]; then
    echo "missing model: $model_path" >&2
    exit 9
fi
actual_model_size="$(stat -c '%s' "$model_path")"
if [[ "$actual_model_size" != "$expected_model_size" ]]; then
    echo "model size mismatch: $actual_model_size" >&2
    exit 9
fi
sample_bytes=4194304
sample_mid=$((actual_model_size / 2 - sample_bytes / 2))
((sample_mid < 0)) && sample_mid=0
actual_model_sample_hash="$({
    dd if="$model_path" iflag=count_bytes count="$sample_bytes" status=none
    dd if="$model_path" iflag=skip_bytes,count_bytes skip="$sample_mid" \
        count="$sample_bytes" status=none
    tail -c "$sample_bytes" "$model_path"
} | sha256sum | awk '{print tolower($1)}')"
if [[ "$actual_model_sample_hash" != "$expected_model_sample_hash" ]]; then
    echo "model sample hash mismatch: $actual_model_sample_hash" >&2
    exit 9
fi
if [[ ! -f "$prompt_path" ]]; then
    echo "missing prompt: $prompt_path" >&2
    exit 10
fi
actual_prompt_hash="$(sha256sum "$prompt_path" | awk '{print tolower($1)}')"
if [[ "$actual_prompt_hash" != "$expected_prompt_hash" ]]; then
    echo "prompt hash mismatch: $actual_prompt_hash" >&2
    exit 11
fi

shader_list="$(find vulkan/shaders/spv -maxdepth 1 -type f -name '*.spv' -print0 |
    sort -z | xargs -0 sha256sum)"
disk_shader_count="$(printf '%s\n' "$shader_list" | sed '/^$/d' | wc -l)"
actual_shader_manifest="$(printf '%s\n' "$shader_list" | sha256sum |
    awk '{print tolower($1)}')"
if [[ "$disk_shader_count" != "$expected_coordinator_shaders" ]]; then
    echo "expected $expected_coordinator_shaders coordinator shader files; found $disk_shader_count" >&2
    exit 14
fi
if [[ "$actual_shader_manifest" != "$expected_coordinator_manifest" ]]; then
    echo "shader manifest mismatch: $actual_shader_manifest" >&2
    exit 14
fi
if [[ "$uniform_shader_manifest_required" == 1 ]] && {
    [[ "$expected_coordinator_shaders" != "$expected_worker_shaders" ]] ||
    [[ "$expected_coordinator_manifest" != "$expected_worker_manifest" ]]
}; then
    echo "uniform shader manifest required, but role profiles differ" >&2
    exit 14
fi

unexpected_env="$(env | cut -d= -f1 | grep '^DS4_' | grep -Ev '^(DS4_USER_APPROVED_BENCHMARK|DS4_EXPECTED_COMMIT|DS4_EXPECTED_COORDINATOR_SHADER_COUNT|DS4_EXPECTED_COORDINATOR_SHADER_MANIFEST|DS4_EXPECTED_WORKER_SHADER_COUNT|DS4_EXPECTED_WORKER_SHADER_MANIFEST|DS4_BENCH_MODEL)$' || true)"
if [[ -n "$unexpected_env" ]]; then
    echo "unexpected DS4 environment variables:" >&2
    echo "$unexpected_env" >&2
    exit 12
fi

binary_hash="$(sha256sum ./ds4-bench | awk '{print tolower($1)}')"
fleet_binary_hash="$(sha256sum ./ds4 | awk '{print tolower($1)}')"
if [[ "$expected_commit" == "${cfg[0]}" && "$fleet_binary_hash" != "$expected_lkg_binary" ]]; then
    echo "LKG fleet binary hash mismatch: $fleet_binary_hash" >&2
    exit 15
fi

audit_summary="$(python3 "$script_dir/fleet_identity.py" \
    --manifest "$manifest" \
    --audit "$fleet_audit" \
    --expected-commit "$expected_commit" \
    --expected-binary "$fleet_binary_hash" \
    --coordinator-shader-count "$expected_coordinator_shaders" \
    --coordinator-shader-manifest "$expected_coordinator_manifest" \
    --worker-shader-count "$expected_worker_shaders" \
    --worker-shader-manifest "$expected_worker_manifest")"
mapfile -t audit_cfg <<<"$audit_summary"
fleet_identity_hash="${audit_cfg[0]}"
fleet_topology_hash="${audit_cfg[1]}"
fleet_env_hash="${audit_cfg[2]}"
fleet_node_count="${audit_cfg[3]}"

mkdir -p "$output_dir"
cp -- "$fleet_audit" "$output_dir/fleet-identity.txt"

command=(
    ./ds4-bench --vulkan
    -m "$model_path"
    --prompt-file "$prompt_path"
    --role coordinator
    --layers 0:3
    --listen 192.168.42.42 1234
    --dist-activation-bits "$dist_bits"
    --ctx-start "$ctx_start"
    --ctx-max "$ctx_max"
    --ctx-alloc "$ctx_alloc"
    --gen-tokens "$gen_tokens"
    --csv "$output_dir/bench.csv"
)

set +e
env -i \
    HOME="$HOME" USER="${USER:-xander}" PATH="$PATH" LANG=C.UTF-8 \
    DS4_VULKAN_WEIGHT_BUDGET_GB="$weight_budget" \
    DS4_MTP_SPEC_DISABLE=1 \
    "${command[@]}" >"$output_dir/run.log" 2>&1
run_rc=$?
set -e
if [[ $run_rc -ne 0 ]]; then
    echo "benchmark failed with exit code $run_rc; evidence retained in $output_dir" >&2
    exit "$run_rc"
fi

if grep -Eqi 'shader not found|fallback[^0-9]*[1-9]|out of memory|oom-kill|gpu reset|device lost' "$output_dir/run.log"; then
    echo "benchmark log contains a fail-closed runtime marker" >&2
    exit 13
fi
shader_count="$(grep -Eo 'VULKAN loaded [0-9]+ shaders' "$output_dir/run.log" | tail -n1 | awk '{print $3}')"
if [[ "$shader_count" != "$expected_coordinator_shaders" ]]; then
    echo "expected $expected_coordinator_shaders coordinator runtime shaders; found ${shader_count:-none}" >&2
    exit 14
fi

model_hash="$(sha256sum "$model_path" | awk '{print tolower($1)}')"
if [[ "$model_hash" != "$expected_model_hash" ]]; then
    echo "full model hash mismatch: $model_hash" >&2
    exit 9
fi
kernel="$(uname -r)"
host="$(hostname)"
governor_state="$(cat /sys/class/drm/card0/device/pp_dpm_sclk 2>/dev/null | tr '\n' ';' || true)"

python3 - "$output_dir/meta.json" "$expected_commit" "$actual_commit" \
    "$binary_hash" "$fleet_binary_hash" "$fleet_identity_hash" \
    "$fleet_topology_hash" "$fleet_env_hash" "$fleet_node_count" \
    "$shader_count" "$actual_shader_manifest" "$model_path" "$model_hash" "$prompt_path" \
    "$actual_prompt_hash" "$ctx_start" "$ctx_max" "$ctx_alloc" \
    "$gen_tokens" "$weight_budget" "$dist_bits" "$host" "$kernel" \
    "$governor_state" <<'PY'
import json
import sys

(
    _, output_path, commit, source_head, binary_hash, fleet_binary_hash,
    fleet_identity_hash, fleet_topology_hash, fleet_env_hash, fleet_node_count,
    shader_count, shader_manifest, model_path, model_hash, prompt_path,
    prompt_hash, ctx_start, ctx_max, ctx_alloc, gen_tokens, weight_budget,
    dist_bits, host, kernel, governor_state,
) = sys.argv
meta = {
  "schema_version": 1,
  "commit": commit,
  "runtime_commit": commit,
  "source_head": source_head,
  "binary_sha256": binary_hash,
  "fleet_binary_sha256": fleet_binary_hash,
  "fleet_commit": commit,
  "fleet_identity_sha256": fleet_identity_hash,
  "fleet_topology_sha256": fleet_topology_hash,
  "fleet_env_sha256": fleet_env_hash,
  "fleet_node_count": int(fleet_node_count),
  "runtime_shader_profile": "coordinator",
  "runtime_shader_count": int(shader_count),
  "runtime_shader_manifest_sha256": shader_manifest,
  "model_path": model_path,
  "model_sha256": model_hash,
  "prompt_path": prompt_path,
  "prompt_sha256": prompt_hash,
  "ctx_start": int(ctx_start),
  "ctx_max": int(ctx_max),
  "ctx_alloc": int(ctx_alloc),
  "gen_tokens": int(gen_tokens),
  "weight_budget_gib": int(weight_budget),
  "dist_activation_bits": int(dist_bits),
  "host": host,
  "kernel": kernel,
  "governor_state": governor_state,
  "return_code": 0
}
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, sort_keys=True)
    f.write("\n")
PY

echo "captured $output_dir"
cat "$output_dir/bench.csv"
