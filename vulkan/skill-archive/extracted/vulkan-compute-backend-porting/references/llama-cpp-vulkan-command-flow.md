# llama.cpp Vulkan command flow (annotated)

Source: `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (Mesa-26.0.3 era)

## Command pool creation

```cpp
// ggml-vulkan.cpp:1009-1016
void vk_command_pool::init(vk_device& device, vk_queue *q_) {
    vk::CommandPoolCreateInfo command_pool_create_info(
        vk::CommandPoolCreateFlags(
            VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
        ),
        q->queue_family_index);
    pool = device->device.createCommandPool(command_pool_create_info);
}
```

Key detail: `TRANSIENT_BIT`. Without it, RADV may crash in
`radv_amdgpu_cs_finalize` when the pool is reused after submission.

## Command buffer lifecycle

```cpp
// ggml-vulkan.cpp:7579-7598

// 1. Get or create: scans for a free CB (in_use=false), or allocates new
static vk_command_buffer* ggml_vk_get_or_create_cmd_buffer(vk_device& device, vk_command_pool& pool) {
    for (auto& cmd_buffer : pool.cmd_buffers) {
        if (!cmd_buffer.in_use) {
            cmd_buffer.use_counter++;
            cmd_buffer.in_use = true;
            return &cmd_buffer;                  // ← reuse existing CB
        }
    }
    return ggml_vk_create_cmd_buffer(device, pool);  // or allocate new
}

// 2. Begin — starts recording on the CB (implicitly resets if previously submitted)
static vk_submission ggml_vk_begin_submission(vk_device& device, vk_command_pool& p, bool one_time = true) {
    vk_submission s;
    s.buffer = ggml_vk_get_or_create_cmd_buffer(device, p);
    s.buffer->buf.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
    return s;
}

// 3. End
static void ggml_vk_ctx_end(vk_context& ctx) {
    if (ctx->s == nullptr) return;
    ctx->s->buffer->buf.end();                   // vkEndCommandBuffer
    ctx->s = nullptr;
}

// 4. Submit
// ggml_vk_submit(ctx, fence) — constructs SubmitInfo, calls QueueSubmit
// Each submission includes wait/signal timeline semaphores

// 5. Cleanup (periodic, NOT between graphs)
static void ggml_vk_command_pool_cleanup(vk_device& device, vk_command_pool& p) {
    device->device.resetCommandPool(p.pool);      // vkResetCommandPool
    for (auto& cmd_buffer : p.cmd_buffers) {
        cmd_buffer.in_use = false;                // mark all as reusable
    }
}
```

Cleanup happens every 10 command buffers (`cleanup_frequency = 10`).
NOT between prefill and decode.

## Context lifecycle (per graph compute)

```cpp
// ggml-vulkan.cpp:7679-7696
static vk_context ggml_vk_get_compute_ctx(ggml_backend_vk_context * ctx) {
    vk_context result;
    if (!ctx->compute_ctx.expired()) {
        result = ctx->compute_ctx.lock();     // ← reuse if still alive
    } else {
        result = ggml_vk_create_context(ctx, ctx->compute_cmd_pool);
        ctx->compute_ctx = result;
        ggml_vk_ctx_begin(ctx->device, result);  // ← begin new CB
    }
    return result;
}
```

The compute context is a shared_ptr. After each batch submission,
`ctx->compute_ctx.reset()` destroys the old context. The next
`ggml_vk_get_compute_ctx` creates a fresh one with a new command buffer.

## Dispatch pipeline (every compute operation)

```cpp
// ggml-vulkan.cpp:7628-7656
template <typename T>
static void ggml_vk_dispatch_pipeline(
    ggml_backend_vk_context* ctx,
    vk_context& subctx,
    vk_pipeline& pipeline,
    std::initializer_list<vk::DescriptorBufferInfo> const& descriptor_buffer_infos,
    const T &push_constants,
    std::array<uint32_t, 3> elements)
{
    // 1. Compute workgroup counts
    const uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
    const uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
    const uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);

    // 2. Assert limits
    GGML_ASSERT(wg0 <= ctx->device->properties.limits.maxComputeWorkGroupCount[0] && ...);

    // 3. Get pre-allocated descriptor set by index
    vk::DescriptorSet& descriptor_set = ctx->descriptor_sets[ctx->descriptor_set_idx++];

    // 4. Update descriptor set bindings (overwrites previous)
    vk::WriteDescriptorSet write_descriptor_set{
        descriptor_set, 0, 0,
        pipeline->parameter_count,
        vk::DescriptorType::eStorageBuffer,
        nullptr, descriptor_buffer_infos.begin()
    };
    ctx->device->device.updateDescriptorSets({ write_descriptor_set }, {});

    // 5. Record commands into the active command buffer
    subctx->s->buffer->buf.pushConstants(pipeline->layout, ...);
    subctx->s->buffer->buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
    subctx->s->buffer->buf.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
        pipeline->layout, 0, { descriptor_set }, {});
    subctx->s->buffer->buf.dispatch(wg0, wg1, wg2);
}
```

## Pipeline barrier (sync_buffers)

```cpp
// ggml-vulkan.cpp:3306-3326
static void ggml_vk_sync_buffers(ggml_backend_vk_context* ctx, vk_context& subctx) {
    const bool transfer_queue = subctx->p->q->transfer_only;

    subctx->s->buffer->buf.pipelineBarrier(
        subctx->p->q->stage_flags,      // VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | TRANSFER
        subctx->p->q->stage_flags,      // same for dst
        {},
        {{                              // ONE memory barrier
            !transfer_queue
                ? (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                   VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT)
                : (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT),
            !transfer_queue
                ? (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                   VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT)
                : (VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT)
        }},
        {},
        {}
    );
}
```

Barriers are CONDITIONAL — only inserted when tensor buffer overlap
is detected in the graph, not after every dispatch.

## Graph compute submission loop (batched submits)

```cpp
// ggml-vulkan.cpp:16239-16596
static ggml_status ggml_backend_vk_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    // Per-submission flop budget (AMD-specific scaling)
    uint32_t submitted_nodes = 0;
    uint32_t submit_count = 0;
    uint64_t batch_flops = 0;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        // ... accumulate flops ...

        bool submit = (submitted_nodes >= ctx->device->max_nodes_per_submit) ||
                      (flops_per_submit != 0 && batch_flops >= flops_per_submit) ||
                      (i + ctx->num_additional_fused_ops >= last_node) ||
                      (almost_ready && !ctx->almost_ready_fence_pending);

        bool enqueued = ggml_vk_build_graph(ctx, cgraph, i, ...);  // records ops

        if (submit && enqueued) {
            first_node_in_batch = true;
            submitted_nodes = 0;
            batch_flops = 0;                                          // ← reset counters
            // ggml_vk_build_graph already called ctx_end + submit internally
        }
    }
}
```

Inside `ggml_vk_build_graph`, when `submit=true`:
1. `ggml_vk_ctx_end(compute_ctx)` — ends command buffer
2. `ctx->compute_ctx.reset()` — destroys the context
3. `ggml_vk_compute_forward(...)` — submits via `ggml_vk_submit`
4. Next node: `ggml_vk_get_compute_ctx` creates fresh context + CB
