# DS4 Vulkan backend session notes (July 2026)

**Project:** `~/ds4-vulkan` — Vulkan compute backend for DeepSeek V4 Flash
**Hardware:** AMD Radeon 8060S (RADV STRIX_HALO), Strix Halo, 128GB unified
**Build:** `make vulkan` with `DS4_VULKAN_BUILD`

## Model specifics (DeepSeek V4 Flash IQ2XXS)

- 43 layers (0-42), not 32
- `n_embd = 7168`, `n_ff = 24576`
- `n_vocab ≈ 129280`, `n_head = 64`, `head_dim = 128`, `n_rot = 64`
- `n_hc = 4` (hyper-connection streams)
- Output weight ~600MB Q8_0 (129280 × 7168)
- Per-layer Q8_0 weights: Q_A ~4.7MB, KV ~2.25MB, Q_B ~9MB
- Gate/up experts: IQ2_XXS (66 bytes per 256-value block)
- Down experts: Q2_K (84 bytes per 256-value block)
- Model file size: 80.76 GiB (86,719,818,973 bytes)

## Memory heaps (RADV Strix Halo)

| Type | Heap | Size | Flags |
|------|------|------|-------|
| 0 | 1 | 80 GiB | DEVICE_LOCAL |
| 1 | 1 | 80 GiB | DEVICE_LOCAL |
| 2 | 0 | 40 GiB | HOST_VISIBLE, HOST_COHERENT |
| 3 | 1 | 80 GiB | DEVICE_LOCAL, HOST_VISIBLE, HOST_COHERENT |
| 4 | 1 | 80 GiB | DEVICE_LOCAL, HOST_VISIBLE, HOST_COHERENT |
| 7 | 1 | 80 GiB | DEVICE_LOCAL |
| 9 | 1 | 80 GiB | DEVICE_LOCAL, HOST_VISIBLE, HOST_COHERENT |

Heap 1 (80 GiB) is the VRAM budget. Heap 0 (40 GiB) is system-only.
`VK_EXT_external_memory_host` is available — can wrap mmap'd file in VkBuffer.
Model (80.76 GiB) is slightly larger than heap 1.  Per-range allocation
(~138 ranges, each a few MB–tens of MB) distributes across both heaps.

## Model loading performance

With per-range `AUTO_PREFER_DEVICE` + staging-buffer upload:
- **37–47 seconds** for 80.76 GiB (matches ROCm's ~32s)
- Each call to `ds4_gpu_cache_model_range` allocates a range, creates a
  staging buffer, copies to staging, submits via dedicated load_pool +
  per-transfer fence (created/destroyed per range), then destroys staging.
- When a range allocation fails (heap exhausted), the range is silently
  skipped — weights accessed via mmap in CPU fallback functions.
- During model loading, the process may show "0.00 GiB" before switching
  to "16.37 GiB" etc. as the first batch of transfers completes.
- The `submitted` field of VulkanCommandCtx is NOT touched during loading
  (dedicated load_pool uses separate fences).

## Fence management bugs — the prefill hang

**Bug chain discovered at session end:** The prefill hangs at
"processing N tokens: 0/N (0.0%)" because:

1. `begin_cmd()` resets the creation-time SIGNALED fence → UNSIGNALED
2. All GPU functions are stubs (return 1, no `vkCmd*` calls) →
   `command_count` stays 0
3. `end_and_submit()`: `command_count == 0` → skips `vkEndCommandBuffer`
   and `vkQueueSubmit` → returns 1 immediately
4. `wait_cmd()`: `command_count == 0` → returns 1 immediately (skips
   `vkWaitForFences`)
5. Fence is UNSIGNALED and will NEVER be signaled (nothing was submitted)
6. Next layer's `begin_cmd()` calls `vkWaitForFences(fence, UINT64_MAX)` on
   UNSIGNALED fence → **HANGS FOREVER**

**Fix:** add `bool submitted = false;` to `VulkanCommandCtx`. Only wait on
the fence if something was actually submitted:

```cpp
static int begin_cmd(void) {
    auto &c = get_cmd_ctx();
    if (c.submitted) {
        VK_CHECK_RAW(vkWaitForFences(g_vk.device, 1, &c.fence, VK_TRUE, UINT64_MAX));
        VK_CHECK_RAW(vkResetFences(g_vk.device, 1, &c.fence));
    } else {
        VK_CHECK_RAW(vkResetFences(g_vk.device, 1, &c.fence));
    }
    // ... reset pool, begin cmd, command_count = 0 ...
}
static int end_and_submit(void) {
    auto &c = get_cmd_ctx();
    if (c.command_count == 0) return 1;
    // ... end, submit ...
    c.submitted = true;  // ← critical
}
```

## Common bugs and fixes

| Symptom | Root cause | Fix |
|---|---|---|
| `state_init_ok=0` | VMA alloc without `MAPPED_BIT` → `ptr` NULL | `HOST_ACCESS_RANDOM \| MAPPED_BIT` |
| Layer 16 fails | Compressor weights missing for layer 16+ | Override `ds4_layer_compress_ratio` → return 0 |
| Tensor views not found | Views point into parent buffer, not tracked | Pointer-range fallback search in tensor_headers |
| `--vulkan` not recognized | Missing from some CLI parsers | All 5 binaries |
| Process killed (OOM) | VMA allocates per-range until heap exhausted | Fallback to mmap when allocation fails |
| Prefill hangs at "0/N (0.0%)" | `wait_cmd` on unsignaled fence after empty submission | Check `submitted` flag before fence wait |
| RADV crashes on submit | Empty command buffer | Track `command_count`, skip end/submit when 0 |
| GPU hang in matmul | Shared memory `sx[4096]` overflow (in_dim=7168) | Increase to `sx[8192]` |
| GPU hang after 16 layers | Descriptor pool exhausted | thread_local reusable descriptor set |
| VMA memcpy crash | Output weight ~600MB | Skip GPU cache for weights >500MB |
| Interleaving model loading + prefill | Separate load_pool vs cmd_ctxs | Use dedicated pool for staging, never `get_cmd_ctx()` during loading |
| g_vk.cmd_ctxs.begin() segfault | Empty map during loading | Never use per-thread context in cache_model_range |

## Return value convention bugs — ALL must return 1

| Function | Bug | Fix |
|---|---|---|
| `ds4_gpu_rms_norm_plain_tensor` | returned 0 | return 1 |
| `ds4_gpu_rms_norm_plain_rows_tensor` | returned 0 | return 1 |
| `ds4_gpu_rms_norm_weight_tensor` | returned 0 | return 1 |
| `ds4_gpu_add_tensor` | returned 0 | return 1 |
| `ds4_gpu_swiglu_tensor` | returned 0 | return 1 |
| `ds4_gpu_begin_commands` / `end_commands` / `flush_commands` | returned 0 | return 1 |
| `ds4_gpu_synchronize` | returned 0 | return 1 |
| `ds4_gpu_init` | returned 0 | return 1 |
| `ds4_gpu_tensor_write` / `read` / `copy` / `fill_f32` | returned 0 | return 1 |
| All `set_model_map*` / `cache_model*` | returned 0 | return 1 |
| All `signal_selected_readback_*` | returned 0 | return 1 |

## Key debugging techniques

### GDB binary search for ok=false

```gdb
break metal_graph_encode_layer_batch if il == <LAYER>
run ...; step; advance <LINE>; print ok
```

The attention function has ONE `return false` (line ~17320 validation).
Everything else goes through `if (ok) ok = func() != 0`.

### Thread-local descriptor cache

```cpp
thread_local VkDescriptorSet ds = VK_NULL_HANDLE;
if (ds == VK_NULL_HANDLE) vkAllocateDescriptorSets(device, &dai, &ds);
vkUpdateDescriptorSets(device, n, writes, 0, nullptr);
vkCmdBindDescriptorSets(cmd, ..., ds, ...);
```

### Weight caching

Keyed by `abs_offset`. Allocates on first use. Q8_0 blocks: 36 bytes = 9 uints.
Skip cache for weights >500MB (output projection ~600MB).

### extern "C" for ALL backend functions

Header declares `extern "C"`. Your code + stubs include must match:
```cpp
extern "C" {
#include "_stubs.gen.cpp"
}
```
Mismatched `bool` vs `int` params cause linker errors.

## Compression workaround

```c
static uint32_t ds4_layer_compress_ratio(uint32_t il) {
    (void)il; return 0;
}
```

## MoE CPU fallback

Since `ds4_gpu_routed_moe_batch_tensor` is a stub, the MoE output is zero
(no expert contribution).  The CPU fallback calls ds4.c's internal
`layer_routed_moe_one_prealloc()`:

1. Remove `static` from `layer_routed_moe_one_prealloc` (line 7535 in ds4.c)
2. Add wrapper `ds4_gpu_routed_moe_cpu()` that calls it with local Q8_K buffers
3. Forward-declare BEFORE the call site (AFTER ds4_model type is defined)
4. Add fallback: `if (!ok && n_tokens == 1) { ds4_gpu_routed_moe_cpu(...); ok = true; }`

## GLSL notes

- Subgroup ops need `--target-env vulkan1.2`
- Reserved names: `out`, `in`, `Out`, `In` → use `OutBuf`, `InBuf`
- F16 conversion: use `exp2`, not `ldexp`
- Q8_0 blocks: float d + 32 × int8 = 36 bytes
- `gl_WorkGroupID.x` is SHARED between M-tile and K-split in mmq shaders

## llama.cpp Vulkan reference

The canonical Q8_0 × Q8_1 matmul shader is in `ggml/src/ggml-vulkan/`:
- `vulkan-shaders/mul_mmq.comp` — main shader with specialization constants
- `vulkan-shaders/mul_mmq_funcs.glsl` — Q8_0 block loading + dot product
- `vulkan-shaders/mul_mmq_shmem_types.glsl` — shared memory cache structs
- `vulkan-shaders/types.glsl` — block_q8_0, block_q8_1_x4_packed128 types
- `vulkan-shaders-gen.cpp` — shader compilation with #define variants
- `ggml-vulkan.cpp` — dispatch_pipeline, matmul dispatcher, push constants

Key patterns:
- A-side (Q8_0): `pack32(i16vec2(...))` from packed16 struct → int32
- B-side (Q8_1): `ivec4` loads from `block_q8_1_x4_packed128`
- Dot product: `dotPacked4x8EXT()` (GL_EXT_integer_dot_product / DP4a)
- Result: `float(q_sum) * d_a * d_b` (Q8_0 has no min offset)
