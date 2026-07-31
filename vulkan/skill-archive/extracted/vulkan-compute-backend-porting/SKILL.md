---
name: vulkan-compute-backend-porting
description: >-
  Port CUDA/HIP GPU compute backends to Vulkan 1.3 compute shaders for
  LLM inference. Covers project structure, VMA, GLSL shaders, pipeline
  creation, command infrastructure, descriptor set management, and the
  systematic comparison methodology against working references like
  llama.cpp's ggml-vulkan backend. Specific to DS4-style single-file
  GPU backends but patterns apply generally.
tags: [vulkan, gpu, compute, porting, cuda, hip, inference, radv, amd]
---

# Vulkan Compute Backend Porting

## When to use

- You need to add a Vulkan compute backend to an existing CUDA/HIP project.
- The project has a clear GPU-API abstraction layer (like `ds4_gpu.h`).
- You target AMD RADV on unified-memory hardware (Strix Halo, etc.).

## User preferences (Strix Halo / DS4 / 2026-07)

| Rule | Rationale |
|------|-----------|
| **NO stubs, NO CPU fallback** | Stubs returning 1 produce SILENT GARBAGE that wastes hours. Unimplemented = return 0 (detectable failure). |
| **ALWAYS verify with exact user command** | `-p hi` (2 tok) masks crashes that `--prompt-file prompt_test.txt` (25 tok) exposes. User WILL catch the lie. Run the exact command. |
| **NEVER commit without explicit permission** | User rejected unauthorized commits: "não commita nada antes de eu testar e confirmar". |
| **NEVER claim a Mesa bug without proof** | If llama.cpp's Vulkan backend works on the same hardware, the bug is in YOUR code. Systematic comparison with the reference is mandatory before blaming the driver. |
| **GDB backtrace is authoritative, not log output** | "25/25 completed" can come from a DIFFERENT begin/end pair. GDB revealed a prefill crash that logs misattributed to decode. |
| **Binary isolation for shared-state bugs** | When q8_0-only and f16-only work independently but BOTH crash together, shared state is the culprit. |
| pt-BR, informal | Communicate in Portuguese, casual tone. |

## Architecture overview

```
project/
  ds4_gpu.h              ← GPU API header (unchanged)
  ds4_vulkan.h            ← Vulkan state + capability structs
  vulkan/
    vulkan_backend.cpp     ← Unified implementation
    _stubs.gen.cpp          ← CPU-fallback stubs (return 0)
    _impl_gen.cpp           ← Pass-through stubs (return 1)
    glslangValidator        ← GLSL→SPIR-V from Khronos glslang
    include/
      vulkan/              ← Vulkan-Headers v1.3.231 (no video stubs)
      vk_mem_alloc.h       ← VMA
    shaders/
      compile.py           ← Batch SPIR-V compilation
      spv/                 ← Compiled .spv files
      *.comp               ← GLSL compute shaders
```

## llama.cpp reference patterns (ggml-vulkan backend)

llama.cpp's Vulkan backend (at `ggml/src/ggml-vulkan/ggml-vulkan.cpp`)
works on the same RADV / Strix Halo hardware. Every difference from its
pattern is a potential root cause. Use it as the authoritative reference.

### Command pool lifecycle (the most critical difference)

**llama.cpp NEVER calls `vkResetCommandPool` during normal operation.**
Pool cleanup (`vkResetCommandPool`) happens periodically (every ~10 CBs)
in a dedicated cleanup function, NOT between prefill and decode.

Instead, llama.cpp calls `vkBeginCommandBuffer` on an already-submitted
command buffer. Per the Vulkan spec, calling `vkBeginCommandBuffer` on a
CB in the `VK_COMMAND_BUFFER_STATE_EXECUTABLE` state (after submit)
**implicitly resets** it — without the pool-level side effects.

```
// MY CODE (WRONG): resets the pool between each graph
begin_cmd:
  vkResetCommandPool(pool)              ← CAN CRASH RADV
  vkBeginCommandBuffer(cmd, ...)

// LLAMA.CPP (RIGHT): just begins on the submitted CB
begin_cmd:
  vkBeginCommandBuffer(cmd, ...)        ← implicitly resets, no pool side effects
```

**Pitfall:** `vkResetCommandPool` after submission can crash RADV's
`radv_amdgpu_cs_finalize` when a new command buffer is allocated from the
same pool. The CS buffer object may not be properly recycled.  Fix: remove
`vkResetCommandPool` from `begin_cmd`. Let `vkBeginCommandBuffer` handle
the reset implicitly.

**Pool creation flags (llama.cpp pattern):**
```cpp
VkCommandPoolCreateInfo cpci(
    VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |  // ← CRITICAL: hints short-lived CBs
    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    queue_family_index);
```

Without `TRANSIENT_BIT`, RADV may use a different internal allocation
strategy for the CS buffer. Always include it for compute pools.

### Descriptor set management (pre-allocate, never reset the pool)

llama.cpp pre-allocates ALL descriptor sets upfront from a SINGLE shared
`VkDescriptorSetLayout`, then recycles them via an index counter.

```cpp
// 1. ONE shared layout for ALL 3-SSBO shaders
static VkDescriptorSetLayout g_shared_dsl = VK_NULL_HANDLE;
if (g_shared_dsl == VK_NULL_HANDLE) {
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.bindingCount = 3; dslci.pBindings = bindings;
    vkCreateDescriptorSetLayout(device, &dslci, nullptr, &g_shared_dsl);
}
entry.desc_layout = g_shared_dsl;  // ← all pipelines share this

// 2. Pre-allocate 32768 sets (half the 65536 pool)
g_vk.pre_allocated_sets.resize(32768);
vkAllocateDescriptorSets(device, &dai, g_vk.pre_allocated_sets.data());

// 3. In begin_cmd: reset idx, NEVER reset descriptor pool
g_vk.descriptor_set_idx = 0;
// DO NOT: vkResetDescriptorPool(...) — frees pre-allocated handles!

// 4. In each dispatch:
VkDescriptorSet ds = g_vk.pre_allocated_sets[g_vk.descriptor_set_idx++];
vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);  // overwrite bindings
vkCmdBindDescriptorSets(cmd, ..., &ds, ...);
```

**Pitfall: `vkResetDescriptorPool` + pre-allocated sets = SIGSEGV.**
Pool reset frees every pre-allocated handle. `vkUpdateDescriptorSets` on
a dangling handle crashes RADV.  GDB backtrace shows the crash at
`vkUpdateDescriptorSets` (NOT `vkEndCommandBuffer`), during prefill
(NOT decode as logs suggest).

### Pipeline barriers (compare with llama.cpp)

llama.cpp uses `ggml_vk_sync_buffers` with ALL access types and the
compute+transfer stage mask:

```cpp
subctx->s->buffer->buf.pipelineBarrier(
    stage_flags,  // VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT
    stage_flags,  // same for dst
    {},
    {{
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT
    }},
    {},
    {}
);
```

Key differences from naive barriers:
- **Includes `VK_PIPELINE_STAGE_TRANSFER_BIT`** — handles CP DMA operations
  that AMD GPUs use for internal buffer management
- **Full read+write on both sides** — not just WRITE→READ, which can miss
  read-after-read hazards from previous dispatches
- **Conditional** — only called when overlapping tensor buffers are detected,
  not after every dispatch

### Timeline semaphores (multi-submission orchestration)

llama.cpp uses timeline semaphores (not fences) for ordering between
multiple submissions within a graph compute. Each submission chain
increments the timeline value, establishing a happens-before relationship.

```cpp
vk::SemaphoreTypeCreateInfo tci{ vk::SemaphoreType::eTimeline, 0 };
vk::SemaphoreCreateInfo ci{};
ci.setPNext(&tci);
vk::Semaphore semaphore = device.createSemaphore(ci);
// Submit with wait=last_value, signal=last_value+1
```

Timeline semaphores are more robust than single fences when multiple
submissions pile up. For single-queue work, they're not strictly required,
but removing the binary semaphore (that was never waited on) is essential.

## Systematic comparison methodology (for RADV debugging)

When a crash reproduces on RADV but llama.cpp works on the same hardware,
the bug is in YOUR code until proven otherwise. Follow this checklist:

1. **Clone and build llama.cpp with Vulkan.** Verify it works with the
   same model format (or at least dispatches compute shaders).

2. **Extract llama.cpp's Vulkan sequence.** Read `ggml/src/ggml-vulkan/`
   and identify their exact patterns for:
   - Command pool creation flags
   - Command buffer lifecycle (get/begin/end/submit)
   - Descriptor set allocation and recycling
   - Pipeline barrier setup and frequency
   - Queue family selection and stage masks

3. **Compare call-by-call with GDB.** Use `gdb -batch -ex run -ex "bt 30"`
   to capture the crash site. Cross-reference the llama.cpp source to see
   what they do differently at the equivalent point.

4. **Install Mesa debug symbols** (not for blaming, for isolating):
   ```bash
   echo "deb http://ddebs.ubuntu.com $(lsb_release -cs) main restricted" \
     | sudo tee /etc/apt/sources.list.d/ddebs.list
   sudo apt install -y ubuntu-dbgsym-keyring && sudo apt update
   sudo apt install -y mesa-vulkan-drivers-dbgsym libvulkan1-dbgsym
   ```
   Then: `gdb -batch -ex run -ex "bt full 30" ./ds4`

5. **Mesa source at crash site**:
   ```bash
   git clone --depth 1 --branch mesa-26.0.3 \
     https://gitlab.freedesktop.org/mesa/mesa.git
   # Crash: src/amd/vulkan/winsys/amdgpu/radv_amdgpu_cs.c:509
   # Caller: src/amd/vulkan/radv_cmd_buffer.c:8152
   ```

6. **Binary isolation for shared-state bugs:** Disable dispatches one shader
   at a time. If q8_0-only and f16-only work but BOTH together crash, the
   shared state (descriptor set handle, pipeline layout, push constant) is
   the root cause.

7. **Never claim a fix without testing the exact user command:**
   `-p hi` (2 tokens) succeeds while `--prompt-file prompt_test.txt`
   (25 tokens) crashes. Simplified tests lie.

## Known traps (quick reference)

| Pitfall | Fix |
|---------|------|
| `vkResetCommandPool` between prefill and decode | Remove it. `vkBeginCommandBuffer` on submitted CB implicitly resets. |
| Missing `VK_COMMAND_POOL_CREATE_TRANSIENT_BIT` | Add it. RADV uses different CS allocation strategy without it. |
| `vkResetDescriptorPool` invalidating pre-allocated sets | NEVER reset after pre-allocation. Recycle via idx only. |
| Descriptor set shared across different pipelines | Each pipeline needs its OWN descriptor set handle. Sharing `ds_q8c` with `matmul_f16` causes SIGSEGV in `vkEndCommandBuffer`. |
| `_impl_gen.cpp` returning 0 | Use `return 1` for pass-through helpers. `return 0` = detectable failure that halts the pipeline. |
| Push constant size mismatch: shader vs C++ | Verify `entry.push_size == sizeof(pc) == shader struct`. Mismatch = RADV SIGSEGV. |
| `ds4_gpu_recommended_working_set_size()` returns 0 | Return ~85 GiB for Strix Halo. 0 → silent early exit. |
| Weight cache exact-match lookup on merged spans | Use range-based search (offset within span) instead of `find(offset)`. |
| `#if !defined(DS4_ROCM_BUILD)` code runs for Vulkan | That condition reads "NOT ROCM" — which IS Vulkan. Functions labeled "cuda_stream" run. |
| Empty command buffer + unsignaled fence | Track `submitted` flag; skip `vkWaitForFences` when nothing submitted. |
| Descriptor pool exhaustion with per-dispatch allocation | Pre-allocate 32768 sets from shared layout; recycle via idx. |
| Workgroup count > 65535 | Use 2D grid (y_scale + y_cnt) instead of tiling loops. |
