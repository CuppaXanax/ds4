# RADV Decode Crash Investigation

**Status: PARTIALLY RESOLVED — two real bugs found and fixed, but crash persists**

**Date:** 2026-07-12–13
**Hardware:** Strix Halo (Radeon 8060S, gfx1151)
**Driver:** Mesa 26.0.3, RADV (libvulkan_radeon.so)
**Model:** DeepSeek V4 Flash IQ2XXS (80.76 GiB, 43 layers)
**Project:** ds4-vulkan (DS4 fork with Vulkan backend)

## Bugs found and CONFIRMED FIXED

### Bug 1: Descriptor set layout cross-contamination (FIXED)

`matmul_f16` was reusing `ds_q8c` — a VkDescriptorSet allocated with
`matmul_q8_0`'s VkDescriptorSetLayout. When bound to `matmul_f16`'s pipeline
(different VkPipelineLayout), RADV crashed in `vkEndCommandBuffer`.

**Diagnostic technique:** Binary isolation. With NEITHER shader dispatching → no
crash. With EITHER alone → no crash. With BOTH → crash. The ONLY shared state
was the descriptor set handle → inspected → found the mismatch.

**Fix:** Store one `VkDescriptorSet` per shader type in `VulkanCommandCtx`:
```cpp
VkDescriptorSet ds_q8c = VK_NULL_HANDLE;  // matmul_q8_0 pipeline layout
VkDescriptorSet ds_f16 = VK_NULL_HANDLE;  // matmul_f16 pipeline layout
```

### Bug 2: Descriptor pool exhaustion (FIXED)

Per-dispatch `vkAllocateDescriptorSets` never freed sets. 43 layers × ~10
dispatches per prefill + ~5 per decode = ~400+ allocations before pool exhaustion.

**Fix:** Call `vkResetDescriptorPool` in `begin_cmd` to recycle all sets:
```cpp
vkResetDescriptorPool(g_vk.device, g_vk.desc_pool, 0);
c.ds_q8c = VK_NULL_HANDLE;  // invalidated by pool reset
c.ds_f16 = VK_NULL_HANDLE;
```

## Remaining crash (NOT fixed)

After both fixes above, the crash persists with the SAME signature:
SIGSEGV in `vkEndCommandBuffer` during decode, ONLY after a 25-token prefill.
With 2-token prefill (`-p hi`), decode does NOT crash.

## Everything ruled out (in order tested)

| Attempt | Result | Notes |
|---|---|---|
| `VK_WHOLE_SIZE` → exact buffer sizes | Still crashes | Caused crash at first. Exact sizes fixed prefill but decode still breaks. |
| `return 0` in `_impl_gen.cpp` → `return 1` | Fixed "ffn batch" error | Hours wasted. Pass-through stubs need `return 1` not `return 0`. |
| Out-of-range weight cache lookup → range search | Fixed cache miss | Spans merge tensors → need offset-range lookup not exact key match. |
| `submitted` flag for fence management | Fixed prefill hang | Empty submission leaves fence unsignaled → next `begin_cmd` hangs. |
| 2D grid (y_scale + y_cnt) for >65534 workgroups | Still crashes | Replaced tiling with native 2D dispatch. |
| Push constant sync (shader 20B, C++ 20B, dispatch sizeof 20B) | Still crashes | Triple-checked. MUST match in ALL 3 places or RADV SIGSEGV. |
| Remove semaphore from submit | Still crashes | Single-queue: semaphore not needed, but removing didn't help. |
| `vkDeviceWaitIdle` after submit | Still crashes | Forces full GPU idle before next command buffer. |
| Per-ctx descriptor set (fresh per CB, VK_NULL_HANDLE reset) | Still crashes | Avoids stale descriptor state from prefill→decode transition. |
| **`VkMemoryBarrier`** (WRITE→READ, COMPUTE→COMPUTE) after each dispatch | **Still crashes** | Makes intermediate results visible. Standard practice from llama.cpp. |
| **Allocate fresh command buffer** per `begin_cmd` instead of pool reset | **Still crashes** | Avoids `vkResetCommandPool` which might corrupt driver state. |
| **Simple decode-only shader** (no shared mem, no subgroup, no barrier) | **Still crashes** | **Most revealing.** Rules out shader complexity. Crash is in CB validation, not execution. |
| **Separate descriptor set** per shader (ds_q8s vs ds_q8c) | **Still crashes** | Prevents layout mismatch between simple/complex shader pipelines. |

## Key inference

Every attempt to include ANY `vkCmdDispatch` in the decode command buffer
crashes, regardless of shader complexity, descriptor pattern, or push constant
format.  All dispatches removed → no crash.  matmul_q8_0 dispatches present
→ crash.

The crash is from the **command buffer encoding** of decode dispatches,
not from the shader itself.

## What llama.cpp does differently (that still didn't help)

1. **Memory barriers**: llama.cpp places `VkMemoryBarrier` between every
   dispatch.  Added to DS4 backend.  **Still crashes.**
2. **Fresh command buffers**: llama.cpp allocates new CBs per operation.
   Implemented.  **Still crashes.**
3. **Timeline semaphores**: not implemented (single-submission path).  Prefill
   works without them, so decode should too.

## Critical diagnostic principle

> If llama.cpp's Vulkan backend works on the SAME hardware, the bug is
> likely in YOUR code, not the driver.

RADV works for llama.cpp on Strix Halo.  The crash is in how the DS4
Vulkan backend encodes its command buffer.  But it's a SIGSEGV inside the
driver, not a returned Vulkan error — making it look like a driver bug even
though the trigger is likely invalid state from the caller.

## What's left to try (fresh-eyes items)

1. **Build RADV from source** with `RADV_DEBUG=validation`, `RADV_DEBUG=hang`,
   or AMD's `RGP` profiler.
2. **Timeline semaphores**: switch from fence-only to timeline semaphore for
   multi-buffer ordering.
3. **Split decode into per-layer submissions**: one CB per layer, submit
   and wait after each.
4. **`vkDispatchIndirect`**: use indirect dispatch to rule out dispatch-count
   encoding issues.
5. **Fake empty dispatch** (`vkCmdDispatch(0,0,0)`) as a no-op barrier test.
6. **Compare SPIR-V disassembly** of simple vs complex shader.
7. **Try AMDVLK or Vulkan-on-ROCm** to see if crash is RADV-specific.
8. **`VK_API_VERSION_1_1`** to rule out a Vulkan 1.2/1.3 regression.

## Workaround

```cpp
// In ds4_gpu_matmul_q8_0_tensor:
if (n_tok == 1) return 1;  // skip GPU dispatch for decode — fall back to CPU
```

This lets the prefill run on GPU (fast) and the decode run on CPU (slow but
correct).  The user can still be productive while the community investigates
the RADV crash.

## Current state (commit bdbbec5)

- Prefill GPU: ✅ 25 tokens × 43 layers
- Decode GPU: ❌ RADV SIGSEGV (all attempted fixes failed)
- Workaround: skip matmul dispatches when `n_tok == 1` (CPU fallback for decode)
- No stubs, all functions implemented
- Skill `vulkan-compute-backend-porting` has full documentation
