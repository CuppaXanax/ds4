#!/usr/bin/env bash
set -euo pipefail

repo=/home/xander/dev/ds4
ip="$(hostname -I | tr ' ' '\n' | grep '^192\.168\.42\.' | head -n1)"
octet="${ip##*.}"
case "$octet" in
    42) expected_layers='0:3' ;;
    43) expected_layers='4:7' ;;
    44) expected_layers='8:11' ;;
    45) expected_layers='12:15' ;;
    46) expected_layers='16:19' ;;
    47) expected_layers='20:23' ;;
    48) expected_layers='24:27' ;;
    49) expected_layers='28:31' ;;
    50) expected_layers='32:35' ;;
    51) expected_layers='36:38' ;;
    52) expected_layers='39:40' ;;
    53) expected_layers='41:output' ;;
    *) echo "identity probe refused unknown host $ip" >&2; exit 2 ;;
esac

cd "$repo"
commit="$(git rev-parse HEAD)"
if git diff --quiet -- . && git diff --cached --quiet -- .; then
    source_clean=1
else
    source_clean=0
fi

mapfile -t pids < <(pgrep -x ds4 || true)
if [[ "$octet" == 42 ]]; then
    if [[ "${#pids[@]}" -ne 0 ]] || pgrep -x ds4-bench >/dev/null; then
        echo "coordinator must be idle during its identity probe" >&2
        exit 3
    fi
    [[ -x ./ds4 ]] || { echo "missing coordinator ./ds4" >&2; exit 4; }
    binary_hash="$(sha256sum ./ds4 | awk '{print tolower($1)}')"
    role=coordinator-ready
    layers="$expected_layers"
    ctx=128000
    weight_budget=11
    shader_count=unloaded
    model=/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
    env_hash=coordinator-managed
else
    if [[ "${#pids[@]}" -ne 1 ]]; then
        echo "expected one ds4 worker on $ip; found ${#pids[@]}" >&2
        exit 5
    fi
    pid="${pids[0]}"
    mapfile -d '' -t argv < "/proc/$pid/cmdline"
    arg_value() {
        local wanted="$1" i
        for ((i = 0; i + 1 < ${#argv[@]}; i++)); do
            if [[ "${argv[$i]}" == "$wanted" ]]; then
                printf '%s' "${argv[$((i + 1))]}"
                return 0
            fi
        done
        return 1
    }
    role="$(arg_value --role || true)"
    layers="$(arg_value --layers || true)"
    ctx="$(arg_value --ctx || true)"
    model="$(arg_value -m || true)"
    [[ "$role" == worker && "$layers" == "$expected_layers" ]] || {
        echo "wrong worker role/layers on $ip: role=$role layers=$layers" >&2
        exit 6
    }
    binary_hash="$(sha256sum "/proc/$pid/exe" | awk '{print tolower($1)}')"
    process_env="$(tr '\0' '\n' < "/proc/$pid/environ" | grep '^DS4_' | sort || true)"
    weight_budget="$(printf '%s\n' "$process_env" |
        sed -n 's/^DS4_VULKAN_WEIGHT_BUDGET_GB=//p' | tail -n1)"
    env_hash="$(printf '%s' "$process_env" | sha256sum | awk '{print tolower($1)}')"
    shader_count="$(grep -Eo 'VULKAN loaded [0-9]+ shaders' /tmp/ds4-worker.log 2>/dev/null |
        tail -n1 | awk '{print $3}')"
    [[ -n "$shader_count" ]] || { echo "missing shader count on $ip" >&2; exit 7; }
fi

printf 'BC250_IDENTITY|host=%s|commit=%s|binary_sha256=%s|source_clean=%s|role=%s|layers=%s|ctx=%s|weight_budget_gib=%s|shader_count=%s|model=%s|env_sha256=%s\n' \
    "$ip" "$commit" "$binary_hash" "$source_clean" "$role" "$layers" "$ctx" \
    "$weight_budget" "$shader_count" "$model" "$env_hash"
