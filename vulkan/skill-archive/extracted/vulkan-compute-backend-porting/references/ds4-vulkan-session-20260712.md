# DS4 Vulkan Backend — Session Notes (2026-07-12)

## Error patterns and fixes encountered

### 1. Fence hang: prefill stuck at 0/N tokens

**Symptom:** "processing N input tokens: 0/N (0.0%)" — never advances.

**Root cause:** `begin_cmd()` resets the fence → all functions are stubs (command_count stays 0) → `end_and_submit()` skips submission → fence stays UNSIGNALED → next `begin_cmd()` calls `vkWaitForFences(UINT64_MAX)` on unsignaled fence → **infinite hang**.

**Fix:** Add `submitted` flag to command context. Only wait/reset fence if `submitted == true`. Set `submitted = true` in `end_and_submit()` when `command_count > 0`.

```cpp
struct VulkanCommandCtx {
    bool submitted = false;  // ← ADD
};
static int begin_cmd(void) {
    auto &c = get_cmd_ctx();
    if (c.submitted) {
        VK_CHECK_RAW(vkWaitForFences(...));
        VK_CHECK_RAW(vkResetFences(...));
    } else {
        VK_CHECK_RAW(vkResetFences(g_vk.device, 1, &c.fence));  // just reset, don't wait
    }
    ...
}
static int end_and_submit(void) {
    ...
    if (c.command_count == 0) return 1;
    VK_CHECK_RAW(vkQueueSubmit(...));
    c.submitted = true;  // ← SIGNAL that fence needs waiting next round
    ...
}
```

### 2. Staging copy crashes during model loading

**Symptom:** SIGSEGV at 0.00 GiB during model loading.

**Root cause:** `ds4_gpu_cache_model_range()` called `g_vk.cmd_ctxs.begin()->second.pool` but the per-thread command contexts map was EMPTY (no context created yet during model loading). Accessing `begin()` on an empty map is UB → segfault.

**Fix:** Use a dedicated `VkCommandPool load_pool` (static local) for all staging transfers during model loading. Create it once, reuse for each transfer. Never access `g_vk.cmd_ctxs` during model loading.

```cpp
static VkCommandPool load_pool = VK_NULL_HANDLE;
if (load_pool == VK_NULL_HANDLE) {
    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = g_vk.queue_family;
    cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(g_vk.device, &cpci, nullptr, &load_pool);
}
```

### 3. Model output is complete gibberish

**Symptom:** Prefill completes at 1.25 t/s, generation at 4.98 t/s, but output is random-looking characters with AI-typical phrases ("rectionaffected...", "explainsvet...", "political..."). Memory shows correct 80 GiB loading.

**Root cause:** Multiple functions are stubs that return 1 without computing. The attention uses argmax (picks single best KV) instead of softmax. The MoE returns zero (no FFN contribution). Pipeline produces garbage because most critical operations don't execute.

**Fix:** Replace ALL stubs with real Vulkan compute dispatches. UNIMPLEMENTED functions should return 0 (detectable failure) — the pipeline stops at the first unimpl function with a clear error, which is debuggable. Silent 1-returning stubs are WORSE than returning 0.

### 4. GPU driver crash in vkEndCommandBuffer

**Symptom:** Crash in `libvulkan_radeon.so` during `vkEndCommandBuffer()`.

**Root cause:** RADV driver crashes when the command buffer contains no recorded commands. An empty command buffer is submitted (no dispatches recorded because all functions are stubs).

**Fix:** Track `command_count`. Skip `vkEndCommandBuffer` + `vkQueueSubmit` + `vkWaitForFences` when `command_count == 0`. (Now subsumed by the `submitted` flag fix above.)

### 5. VMA allocation fails for 80 GiB model buffer

**Symptom:** "VULKAN failed to allocate model buffer (80.8 GiB)" or process killed by OOM.

**Root cause:** Heap 1 is 82261 MiB (~80 GiB), but the model is 80.76 GiB (~86.7 GB = 86719818973 bytes). Heap 1 in MiB: 82261 × 1048576 = 86,267,662,336 bytes. Model: 86,719,818,973 bytes. Model exceeds heap 1 by ~452 MiB.

**Fix:** Allocate PER-RANGE (each tensor individually) with `VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE`. VMA allocates from heap 1 first, then falls back to heap 0 (40 GiB HOST_VISIBLE). Individual ranges are small (a few MB to 37 MB) and fit in any heap.

### 6. OOM kill during model loading

**Symptom:** "killed" at ~10 GiB progress during model loading. A background ds4 process continues running.

**Root cause:** VMA per-range allocations consume HOST_VISIBLE memory rapidly. Once the HOST_VISIBLE heap is exhausted, VMA fails. Combined with the mmap'd file (which is also faulting in pages), the system runs low on available memory.

**Fix:** For the DS4 project, return 1 from `cache_model_range` without allocating (weights stay mmap'd). The CPU fallback functions read from mmap directly. Future GPU shader dispatch will need proper VkBuffers; switch to `VK_EXT_external_memory_host` when moving to full GPU compute.

### 7. "ds4_gpu_routed_moe_cpu" forward declaration must come AFTER type definitions

**Symptom:** Compile error: "unknown type name 'ds4_model'" or "incompatible pointer type".

**Root cause:** `ds4_model` and `ds4_layer_weights` are defined INSIDE ds4.c at ~line 1635, but the forward declaration at the top of the file (line 47) is too early.

**Fix:** Move the forward declaration to just before the call site (~line 19147), or define the wrapper function before the call site, or place it at the end of ds4.c (after line 27791).

## Key memory type info

```
Device: Radeon 8060S Graphics (RADV STRIX_HALO)
VK_EXT_external_memory_host: AVAILABLE
Type 0: heap=1 size=82261MiB DEVICE_LOCAL
Type 1: heap=1 size=82261MiB DEVICE_LOCAL
Type 2: heap=0 size=41130MiB HOST_VISIBLE HOST_COHERENT
Type 3: heap=1 size=82261MiB DEVICE_LOCAL HOST_VISIBLE HOST_COHERENT
Type 4: heap=1 size=82261MiB DEVICE_LOCAL HOST_VISIBLE HOST_COHERENT
Type 5: heap=0 size=41130MiB HOST_VISIBLE HOST_COHERENT HOST_CACHED
Type 6: heap=0 size=41130MiB HOST_VISIBLE HOST_COHERENT HOST_CACHED
Type 7: heap=1 size=82261MiB DEVICE_LOCAL
Type 8: heap=0 size=41130MiB HOST_VISIBLE HOST_COHERENT
Type 9: heap=1 size=82261MiB DEVICE_LOCAL HOST_VISIBLE HOST_COHERENT
Type 10: heap=0 size=41130MiB HOST_VISIBLE HOST_COHERENT HOST_CACHED
```

## ROCm reference

The ROCm backend in `~/ds4/rocm/ds4_rocm_runtime.cuh` implements
`ds4_gpu_cache_model_range` at line 4726 via:
```c
if (!cuda_model_range_ptr(model_map, offset, bytes, label)) return 0;
return cuda_model_range_is_cached(model_map, offset, bytes);
```

The CUDA backend allocates per-range GPU memory and copies data.  Total
load time: ~32s for 80.76 GiB, matching the Vulkan staging approach
(~37-46s).  The model loading progress prints at 16 GiB intervals.
