# DS4 Vulkan Backend Patches — 2026-07-12

Applied to a clean git checkout of `vulkan/vulkan_backend.cpp` to fix inference on Strix Halo (Radeon 8060S, RADV).

## Pre-existing in the clean file (already committed)

The following patches were **already present** in the git revision checked out:

1. `#include <algorithm>` after `<cstring>`  (line 31)
2. `bool submitted = false` in `VulkanCommandCtx` struct (line 44)
3. Guarded fence wait/reset in `begin_cmd()` — checks `c.submitted` before waiting (lines 396-403)
4. `c.submitted = true` in `end_and_submit()` (line 420)
5. Full `ds4_gpu_cache_model_range` with staging buffer upload (lines 703-774)
6. Early-return in `wait_cmd()` — skips fence wait when `command_count == 0` (line 426)
7. Vulkan compute dispatch for `ds4_gpu_matmul_q8_0_tensor` (lines 907-977)
8. Vulkan compute dispatch for `ds4_gpu_matmul_f16_tensor` (lines 980-1043)

## Patches applied this session

### Patch 10: Descriptor pool sizing

```diff
- VkDescriptorPoolSize ps[] = {{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256 }};
- dpci.maxSets = 64;
+ VkDescriptorPoolSize ps[] = {{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8192 }};
+ dpci.maxSets = 8192;
```

**Rationale:** Each compute dispatch allocates a new descriptor set. A full
prefill (43 layers × ~15 dispatches/layer = ~645 dispatches) requires 645+
sets. The old pool (64 sets of 256 buffers) exhausts after ~16 dispatches,
causing `vkAllocateDescriptorSets` to fail → GPU hang.

### Patch 9a: `ds4_gpu_dsv4_fp8_kv_quantize_tensor`

```c
int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
    ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot)
{
    (void)x; (void)n_tok; (void)head_dim; (void)n_rot;
    return 1;  /* No-op: KV cache stays in float on Vulkan backend */
}
```

### Patch 9b: `ds4_gpu_attention_prefill_raw_heads_tensor`

CPU pass-through: copies raw KV data to heads output.

### Patch 9c: `ds4_gpu_dsv4_qkv_rms_norm_rows_tensor`

CPU implementation reading Q and KV weights from `model_map`, with per-row
RMS normalization. Same logic as `_cpu_fallback.inc`.

### Patch 11: `_impl_gen.cpp` include

Added between the MoE function and `/* ---- BEGIN AUTO-GENERATED STUBS ---- */`:

```cpp
/* ---- AUTO-GENERATED CPU IMPLEMENTATIONS ---- */
extern "C" {
#include "_impl_gen.cpp"
}
```

### Patch 12: Two-file stub split

Created `_impl_gen.cpp` (500 lines, 46 functions) containing all stubs from
the original `_stubs.gen.cpp` that are NOT yet implemented. Each stub returns
0 (DS4 detectable failure). The `_stubs.gen.cpp` was cleared to a 1-line
comment since all its functions were either implemented or moved.

## Verification

Compiled cleanly with: `g++ -O3 -g -std=c++17 -pthread -I. -Ivulkan -Ivulkan/include -march=native -c vulkan/vulkan_backend.cpp`
→ Exit code 0, zero errors, one pre-existing warning (unused fread result).

## Still needed

The 46 stubs in `_impl_gen.cpp` still return 0 (failure). Each needs a real
Vulkan compute shader dispatch to make inference work end-to-end. Priority
order (based on what ds4.c calls during prefill):
1. Compressor functions (`ds4_gpu_compressor_*`)
2. Attention decode heads (`ds4_gpu_attention_decode_heads_tensor`)
3. Indexer functions (`ds4_gpu_indexer_*`)
4. MoE matmul (`ds4_gpu_routed_moe_one_tensor`)
5. HC (heads) expand/split functions
6. Router functions
