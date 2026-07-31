#!/usr/bin/env python3
"""Classify every ds4_gpu_* function as REAL (implemented in the Vulkan
backend) or STUB (no-op placeholder from _impl_gen.cpp).

Outputs a Markdown table.  Used to maintain vulkan/STATUS.md.

Usage: python3 vulkan/kernel_status.py
"""
import re
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gen_impl import parse_header, backend_defined_names, COMPAT_IMPL, HEADERS, BACKEND

# Category classification by function name (order matters: first match wins).
RULES = [
    ("matmul",    re.compile(r"matmul|add3_tensor|^ds4_gpu_add_tensor")),
    ("attention", re.compile(r"attention")),
    ("moe",       re.compile(r"moe|routed|shared_|swiglu|router|sinkhorn|gated")),
    ("norm",      re.compile(r"rms_norm|norm")),
    ("rope",      re.compile(r"rope")),
    ("embed",     re.compile(r"embed")),
    ("indexer",   re.compile(r"indexer|topk|argmax|markov|dspark")),
    ("kv",        re.compile(r"kv|compressor|fp8|store_raw|raw_kv|rope_tail")),
    ("hc",        re.compile(r"hc_|_hc")),
    ("head",      re.compile(r"head_rms|output_|logits|vocab")),
    ("stream",    re.compile(r"stream_expert|streaming|preload|residency|q4_expert")),
    ("tensor",    re.compile(r"tensor_|lookup_cache|register_|device_cache")),
    ("commands",  re.compile(r"commands|flush_|synchronize|readback|begin_cmd")),
    ("model",     re.compile(r"model_|cache_model|q8_cache")),
    ("tp",        re.compile(r"_tp_|tp_")),
    ("mgpu",      re.compile(r"xdev|alloc_on|set_current_device|tier_free_vram|init_multi|peer")),
    ("misc",      re.compile(r".*")),
]

# Kernels verified correct by the test harness (vulkan/tests, run-kernel-tests.sh).
VERIFIED = {
    "ds4_gpu_matmul_f16_tensor",
    "ds4_gpu_router_select_tensor",
    "ds4_gpu_routed_moe_one_tensor",
    "ds4_gpu_attention_decode_heads_tensor",
    "ds4_gpu_embed_token_q8_0_tensor",
    "ds4_gpu_embed_tokens_q8_0_tensor",
    "ds4_gpu_embed_token_quant_tensor",
    "ds4_gpu_embed_tokens_quant_tensor",
    "ds4_gpu_indexer_score_one_tensor",
    "ds4_gpu_indexer_scores_decode_batch_tensor",
    "ds4_gpu_indexer_topk_tensor",
    "ds4_gpu_attention_output_q8_batch_tensor",
    "ds4_gpu_attention_output_low_q8_tensor",
    "ds4_gpu_rms_norm_plain_rows_tensor",
    "ds4_gpu_rms_norm_weight_rows_tensor",
    "ds4_gpu_rope_tail_tensor",
    "ds4_gpu_head_rms_norm_tensor",
    "ds4_gpu_head_rms_norm_rope_tail_tensor",
    "ds4_gpu_matmul_q8_0_tensor",
    "ds4_gpu_routed_moe_batch_tensor",
    "ds4_gpu_dsv4_qkv_rms_norm_rows_tensor",
    "ds4_gpu_dsv4_fp8_kv_quantize_tensor",
    "ds4_gpu_store_raw_kv_batch_tensor",
    "ds4_gpu_attention_prefill_raw_heads_tensor",
}


def category(name):
    for label, rx in RULES:
        if rx.search(name):
            return label
    return "misc"


def main():
    funcs = {}
    for h in HEADERS:
        for k, v in parse_header(h).items():
            funcs[k] = v
    real = backend_defined_names(BACKEND) | COMPAT_IMPL

    rows = []
    for name in sorted(funcs):
        if name in VERIFIED:
            status = "VERIFIED"
        elif name in real:
            status = "REAL"
        else:
            status = "STUB"
        cat = category(name)
        rows.append((cat, name, status))

    # Group by category for readability.
    by_cat = {}
    for cat, name, status in rows:
        by_cat.setdefault(cat, []).append((name, status))

    for cat in sorted(by_cat):
        n_real = sum(1 for _, s in by_cat[cat] if s == "REAL")
        n_stub = sum(1 for _, s in by_cat[cat] if s == "STUB")
        print(f"## {cat} ({n_real} real / {n_stub} stub)")
        print()
        print("| function | status |")
        print("|---|---|")
        for name, status in by_cat[cat]:
            print(f"| `{name}` | {status} |")
        print()

    total = len(rows)
    n_real = sum(1 for _, _, s in rows if s == "REAL")
    n_stub = sum(1 for _, _, s in rows if s == "STUB")
    print(f"TOTAL: {total} functions, {n_real} real, {n_stub} stub")


if __name__ == "__main__":
    main()
