# llama.cpp Vulkan Submission Pipeline — Architecture Analysis

Analysis of `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (llama.cpp, ~18K lines), focusing
on how the Vulkan backend manages command buffers, descriptor sets, and submission
across prefill/decode transitions.  Hardware: Radeon 8060S (Strix Halo, RADV 26.0.3).

## Key architectural decisions

### 1. Command context destroy-and-recreate per graph

llama.cpp does NOT reuse command buffers via `vkResetCommandPool`.  Instead,
each graph submission gets a fresh command context:

```cpp
// ggml_vk_build_graph() → line 14999-15012
if (submit || last_node) {
    ggml_vk_ctx_end(compute_ctx);   // vkEndCommandBuffer
    ctx->compute_ctx.reset();        // DESTROY shared_ptr, old context freed
    ggml_vk_compute_forward(ctx, ...); // SUBMIT + wait on fence
}

// Next call → ggml_vk_get_compute_ctx() → line 7679-7688
if (ctx->compute_ctx.expired()) {
    result = ggml_vk_create_context(ctx, ctx->compute_cmd_pool); // NEW struct
    ggml_vk_ctx_begin(ctx->device, result);  // allocate NEW CB from pool
}
```

The command POOL (`ctx->compute_cmd_pool`) IS reused across the entire run.
Only the `vk_context` wrapper (shared_ptr) is destroyed and recreated.
New command buffers are allocated from the persistent pool each time.

### 2. Command pool: reused, reset periodically

`ctx->compute_cmd_pool` persists. New CBs allocated each time. Cleanup every ~10:

```cpp
// line 3044
static void ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p) {
    device->device.resetCommandPool(p.pool);
    for (auto& cmd_buffer : p.cmd_buffers) { cmd_buffer.in_use = false; }
}
```

### 3. Descriptor sets: pre-allocated pool with idx-based recycling

llama.cpp pre-allocates descriptor sets upfront and reuses via `descriptor_set_idx++`:

```cpp
// In ggml_vk_dispatch_pipeline (line 7640-7646):
GGML_ASSERT(ctx->descriptor_set_idx < ctx->descriptor_sets.size());
vk::DescriptorSet& descriptor_set = ctx->descriptor_sets[ctx->descriptor_set_idx++];
ctx->device->device.updateDescriptorSets({ write_descriptor_set }, {});
```

Reset at graph end: `ctx->descriptor_set_idx = 0;` (line 15101).
Pool grows dynamically via `ggml_pipeline_allocate_descriptor_sets`.
This avoids per-dispatch `vkAllocateDescriptorSets` and prevents pool exhaustion.

### 4. Batch submissions during graph compute

llama.cpp submits in batches based on FLOP count, not once at graph end:

```cpp
// line 16312-16316
// Submit after enough work to overlap CPU cmdbuffer gen with GPU execution.
// Estimate compute work using flops, submit every ~200 GFLOP.
uint64_t flops_per_submit = std::min(flops_cap, ctx->last_total_flops / 40u);

// line 16323-16329 — AMD weak GPU mitigation:
if (vendor == VK_VENDOR_ID_AMD && shader_core_count < 24)
    flops_cap = 2'000'000'000ULL * shader_core_count;
```

### 5. Shared descriptor set layout

ONE `VkDescriptorSetLayout` for ALL 3-SSBO pipelines. All pipelines are
compatible with the same descriptor sets — no per-shader DS needed.

## DS4 Vulkan backend — what was tried

| Pattern | llama.cpp | DS4 Vulkan tried? | Result |
|---------|-----------|-------------------|--------|
| Destroy+recreate CB context per graph | Yes | Yes | ❌ crash persists |
| Reuse pool, allocate fresh CBs | Yes | Yes | ❌ crash persists |
| Pre-allocate DS + idx recycling | Yes (pre-alloc) | Yes (4096 sets) | ❌ crash persists |
| Shared DS layout | Yes | Yes (`g_shared_dsl`) | ✅ fixed one crash source |
| `vkResetDescriptorPool` in begin_cmd | Via pool cleanup | Yes | ✅ |
| Batch submits | Yes (FLOP-based) | No (monolithic) | NOT TRIED |
| Timeline semaphores | Yes (multi-queue) | No (single queue) | N/A |
| `vkDeviceWaitIdle` between graphs | Yes | Yes | ❌ crash persists |
| Memory barriers between dispatches | Yes | Yes | ❌ crash persists |

## DS4 Vulkan — 2026-07-13 session results

Two real bugs found and fixed:
1. **DS layout mismatch**: `matmul_f16` was reusing `ds_q8c` (matmul_q8_0's layout)
2. **DS pool exhaustion**: per-dispatch allocation saturated the pool

After fixing both, the RADV crash persists: prefill GPU (25 tokens × 43 layers)
works, decode GPU (n_tok=1) crashes in `vkEndCommandBuffer`. The crash happens
even with `vkDeviceWaitIdle` between graphs, fresh command contexts, and
pre-allocated descriptor sets.

The crash is NOT reproducible with 2-token prefill (`-p hi`), only with 25-token
prefill. Binary isolation (disable one shader at a time) confirms the crash only
occurs when BOTH matmul_q8_0 AND matmul_f16 dispatches run in the decode path
after a GPU prefill.

Workaround: skip GPU dispatches during decode (return 1, CPU fallback).
Prefill runs on GPU, decode on CPU.

## Key takeaway

The pattern of destroying and recreating command contexts between graphs avoids
state contamination. However, even with this pattern fully implemented, the DS4
Vulkan backend still crashes — suggesting a RADV driver bug specific to the
sequence of dispatches in the DS4 decode path after a large GPU prefill.
