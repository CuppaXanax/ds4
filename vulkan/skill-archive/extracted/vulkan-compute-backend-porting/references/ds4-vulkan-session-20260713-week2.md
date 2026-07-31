# ds4-vulkan session 2026-07-13 (week 2) — from the user's frustration

## Premise

Copy ~/ds4 (ROCm backend) to ~/ds4-vulkan, implement full Vulkan compute
backend. Read ~/ds4/rocm/*.cuh for reference patterns. Reference llama.cpp's
ggml-vulkan backend at each step.

## Session outcome

Three bugs found and fixed; one crash remains unresolved.

## Three bugs found (all fixed)

### Bug 1: `_impl_gen.cpp` returning 0 instead of 1

Returning 0 (failure) on every pass-through helper function causes every
code path to immediately fail. The pipeline never reaches the GPU dispatches
at all. Symptom: "gpu layer 0 ffn batch encode failed".

**Fix:** Regenerate `_impl_gen.cpp` with `return 1` for int-returning helpers.
`return 0` only for deliberately unimplemented blocking functions.

### Bug 2: `matmul_f16` reusing `ds_q8c` descriptor set

The descriptor set `ds_q8c` was allocated with `matmul_q8_0`'s pipeline
layout. When `vkCmdBindDescriptorSets` bound this set to `matmul_f16`'s
pipeline (different layout), RADV crashed during `vkEndCommandBuffer`
validation.

**Fix:** give each pipeline its own descriptor set handle in the command
context: `ds_q8c` for matmul_q8_0, `ds_f16` for matmul_f16.

**How found:** Binary isolation — q8_0-only and f16-only dispatches worked
independently but BOTH together crashed. The ONLY shared state was the
descriptor set handle.

### Bug 3: `vkResetDescriptorPool` invalidating pre-allocated descriptor sets

After switching to pre-allocated descriptor sets (32768 sets from shared
layout), `vkResetDescriptorPool` was called at each `begin_cmd`. This freed
every pre-allocated handle. The next dispatch's `vkUpdateDescriptorSets`
on a dangling handle → SIGSEGV inside RADV.

**Fix:** NEVER call `vkResetDescriptorPool` after pre-allocating. Recycle
sets via `descriptor_set_idx = 0` only.

**How found:** GDB backtrace showed crash at `vkUpdateDescriptorSets` (line
1000 in vulkan_backend.cpp), during PREFILL layer 2 — NOT decode. Without
GDB, this would have been misdiagnosed as a decode crash because the log
showed "25/25 (100.0%)" before the crash.

## Unresolved crash: `radv_amdgpu_cs_finalize` during decode

**Symptom:** After all three bugs are fixed and prefill GPU completes
successfully, the first decode token's `vkEndCommandBuffer` crashes in
RADV's `radv_amdgpu_cs_finalize` (radv_amdgpu_cs.c:509).

**GDB backtrace (with Mesa debug symbols):**
```
#0  radv_amdgpu_cs_finalize(_cs) at radv_amdgpu_cs.c:509
#1  radv_EndCommandBuffer(cmd_buffer) at radv_cmd_buffer.c:8152
#2  end_and_submit() at vulkan_backend.cpp:447
#3  metal_graph_eval_token_raw_swa(... pos=25, token=74035) at ds4.c:19493
```

**Isolation matrix (all approaches tested):**

| Approach | Result |
|----------|--------|
| `vkResetCommandPool` between prefill and decode | CRASH |
| `vkResetCommandBuffer` (individual, no pool reset) | CRASH |
| Fresh command pool/fence per `begin_cmd` | CRASH |
| `vkBeginCommandBuffer` on submitted CB (no reset at all) | CRASH |
| `VK_COMMAND_POOL_CREATE_TRANSIENT_BIT` added | CRASH |
| `DS4_SPLIT_LAYERS=1` (flush every layer) | CRASH |
| `DS4_SPLIT_LAYERS=10` | CRASH |
| 2D grid dispatch (y_scale + y_cnt) | CRASH |
| No semaphore in submit | CRASH |
| `vkDeviceWaitIdle` after prefill | CRASH |
| Per-dispatch descriptor set allocation | CRASH |
| Per-ctx descriptor set reuse | CRASH |
| Pre-allocated descriptor sets (32768) | CRASH |
| Shared descriptor set layout for all pipelines | CRASH |
| Skip ALL GPU dispatches during decode (CPU fallback) | NO CRASH |

**llama.cpp comparison — still open differences:**

| Aspect | llama.cpp | ds4-vulkan |
|--------|-----------|------------|
| Pool never reset between graphs | Never. Cleanup after 10 CBs. | Always. Reset every begin/end. |
| Timeline semaphore | yes | no (removed binary sema) |
| Stage mask for barriers | COMPUTE + TRANSFER | COMPUTE only |
| Batch submits during graph | Every ~200 GFLOP | One submit at end |
| Queue family | Compute+transfer capable | Compute-only |

The pool reset vs no-reset difference is the most likely root cause.
`vkBeginCommandBuffer` on a submitted CB implicitly resets it without
pool-level side effects. The DS4 code's explicit `vkResetCommandPool`
may corrupt RADV's internal CS state.

## Workaround (functional)

Prefill GPU + decode CPU: `if (n_tok == 1) return 1;` in both matmul
dispatch functions.  Prefill processes 25 tokens in ~250ms on GPU.
Decode runs on CPU (slow but correct).

## Mesa source at crash site

```c
// src/amd/vulkan/winsys/amdgpu/radv_amdgpu_cs.c:501-515
static VkResult radv_amdgpu_cs_finalize(struct ac_cmdbuf *_cs) {
    struct radv_amdgpu_cs *cs = radv_amdgpu_cs(_cs);
    assert(cs->base.cdw <= cs->base.reserved_dw);  // COULD ASSERT

    if (cs->chain_ib) {
        radv_amdgpu_winsys_cs_pad(_cs, 4);
        radv_amdgpu_cs_emit_nops(cs, 4);
        assert(cs->base.cdw <= ~C_3F2_IB_SIZE);
        *cs->ib_size_ptr |= cs->base.cdw;
    } else {
        radv_amdgpu_winsys_cs_pad(_cs, 0);      // ← LINE 509
    }

    radv_amdgpu_cs_add_ib_buffer(cs, cs->ib_buffer, cs->ib_buffer->va, ...);
    cs->ib_buffer = NULL;
    cs->chained_to = NULL;
    assert(cs->base.cdw <= cs->base.max_dw + 4);
    return cs->status;
}
```

Line 509 is `radv_amdgpu_winsys_cs_pad(_cs, 0)`. If `cs` is stale (freed
memory from pool reset), this dereferences freed memory → SIGSEGV.

## Key methodological lesson

The user's correction (paraphrased):

> If llama.cpp works on the same hardware and you can't explain WHY,
> you can't claim a Mesa bug. The most plausible conclusion is no bug —
> your code uses the API wrong. llama.cpp demonstrates the correct
> pattern for this driver.

This applies to ANY Vulkan debugging: a working reference implementation
on the same driver means your implementation is wrong, not the driver.
