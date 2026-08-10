#!/usr/bin/env python3
"""Regenerate vulkan/_impl_gen.cpp from the current ds4_gpu.h / ds4_gpu_mgpu.h.

The Vulkan backend implements a subset of the ds4_gpu_* contract in
vulkan/vulkan_backend.cpp.  Everything else must still be defined so the
engine links; this generator emits no-op placeholder definitions for every
function NOT implemented by the backend.

Return-value convention for unsupported implementations:
    integer / bool-like -> 0, pointer -> NULL, void -> {}

Usage:
    python3 vulkan/gen_impl.py
"""

import re
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
BACKEND = ROOT / "vulkan" / "vulkan_backend.cpp"
HEADERS = [ROOT / "ds4_gpu.h", ROOT / "ds4_gpu_mgpu.h"]
OUT = ROOT / "vulkan" / "_impl_gen.cpp"

# Names implemented by hand in the mgpu-compat section of vulkan_backend.cpp.
COMPAT_IMPL = {
    "ds4_gpu_init_multi",
    "ds4_gpu_set_current_device",
    "ds4_gpu_set_current_device_fenced",
    "ds4_gpu_tier_free_vram",
    "ds4_gpu_tensor_alloc_on",
    "ds4_gpu_tensor_free_in_place",
    "ds4_gpu_tensor_alloc_ptr_on",
    "ds4_gpu_tensor_alloc_managed_on",
    "ds4_gpu_tensor_copy_async",
    "ds4_gpu_tensor_copy_xdev",
    "ds4_gpu_tensor_copy_xdev_default",
    "ds4_gpu_tensor_copy_xdev_ordered",
    "ds4_gpu_tensor_copy_xdev3",
    "ds4_gpu_tensor_copy_xdev3_default_dst",
    "ds4_gpu_tensor_wait_xdev",
    "ds4_gpu_tensor_wait_xdev_default",
    "ds4_gpu_tensor_device",
    "ds4_gpu_add_xdev_tensor",
    "ds4_gpu_register_model_map_no_copy",
    "ds4_gpu_lookup_cache_strict",
    "ds4_gpu_args_probe_auto_cuda",
    "ds4_gpu_enable_q8_dequant_gemm",
    "ds4_gpu_set_decode_fast_attention",
    "ds4_gpu_set_decode_score_vec4",
    "ds4_gpu_register_support_map",
    "ds4_gpu_device_cache_tensors",
    "ds4_gpu_device_cache_support_tensors",
    "ds4_gpu_lookup_cache",
    "ds4_gpu_lookup_cache_device",
    "ds4_gpu_q8_cache_suppressed",
    "ds4_gpu_set_q8_cache_suppressed",
    # Non-Apple builds provide these as static inline definitions in ds4_gpu.h.
    "ds4_gpu_device_is_m5_apple_silicon",
    "ds4_gpu_device_is_pre_m5_apple_silicon",
}

# Status queries whose honest unsupported value is nonzero.
SCALAR_RETURNS = {
    "ds4_gpu_tp_failed": "1",
}


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def parse_header(path):
    """Return {name: (return_type, params)} for every ds4_gpu_* prototype."""
    src = strip_comments(path.read_text())
    funcs = {}
    for m in re.finditer(r"\b(ds4_gpu_[A-Za-z0-9_]+)\s*\(", src):
        name = m.group(1)
        if name in funcs:
            continue
        # Return type: text from start of the current line up to the name.
        line_start = src.rfind("\n", 0, m.start()) + 1
        ret = src[line_start:m.start()].strip()
        if not ret or ret.endswith(("typedef", "struct", "enum", "}")):
            continue
        # Parameters: match the enclosing parens.
        depth = 0
        j = m.end() - 1  # points at '('
        while j < len(src):
            c = src[j]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        params = src[m.end():j].strip()
        # Must be a prototype: next non-space char is ';'.
        k = j + 1
        while k < len(src) and src[k] in " \t\n":
            k += 1
        if k >= len(src) or src[k] != ";":
            continue
        funcs[name] = (ret, params)
    return funcs


def backend_defined_names(path):
    """Names of ds4_gpu_* functions whose definition starts at column 0."""
    src = path.read_text()
    names = set()
    for m in re.finditer(
            r"^(?:extern \"C\"[ \t]+)?[A-Za-z_][A-Za-z0-9_ \t*]*"
            r"\b(ds4_gpu_[A-Za-z0-9_]+)[ \t]*\(", src, re.M):
        names.add(m.group(1))
    return names


def main():
    funcs = {}
    for h in HEADERS:
        for k, v in parse_header(h).items():
            funcs[k] = v
    defined = backend_defined_names(BACKEND) | COMPAT_IMPL
    missing = sorted(k for k in funcs if k not in defined)

    lines = [
        "/* AUTO-GENERATED from ds4_gpu.h / ds4_gpu_mgpu.h - do not edit.",
        " * Regenerate with: python3 vulkan/gen_impl.py",
        " * Vulkan backend: integer/bool-like -> 0, pointer -> NULL, void -> {}. */",
    ]
    for name in missing:
        ret, params = funcs[name]
        log = f'if (getenv("DS4_VULKAN_LOG_STUBS")) fprintf(stderr, "ds4: STUB {name}\\n");'
        if ret == "void":
            body = f"{{ {log} }}"
        elif ret.endswith("*"):
            body = f"{{ {log} return NULL; }}"
        elif name in SCALAR_RETURNS:
            body = f"{{ {log} return {SCALAR_RETURNS[name]}; }}"
        elif "bool" in ret or "uint" in ret or ret == "size_t":
            body = f"{{ {log} return 0; }}"
        else:
            body = f"{{ {log} return 0; }}"
        lines.append(f"{ret} {name}({params}) {body}")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"generated {len(missing)} placeholders -> {OUT}")


if __name__ == "__main__":
    main()
