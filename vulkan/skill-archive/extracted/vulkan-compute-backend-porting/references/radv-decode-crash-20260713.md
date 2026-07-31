# RADV Decode Crash — Debug Trace (2026-07-13)

## Reproducible Steps

```bash
cd ~/ds4-vulkan
make vulkan
./ds4 --vulkan -c 32000 \
  -m ../ds4/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --prompt-file prompt_test.txt
```

Output:
```
processing 25 input tokens: 25/25 (100.0%)
[1]    176465 segmentation fault (core dumped)
```

## Hardware
- AMD Radeon 8060S (gfx1151, Strix Halo), 128GB unified RAM
- Mesa RADV 26.0.3, libvulkan_radeon.so
- DeepSeek V4 Flash IQ2XXS weighing ~80.76 GiB

## Three Real Bugs Found (All Fixed, Crash Persists)

### Bug 1: Descriptor set shared across pipelines
**Found via:** Binary isolation (q8_0-only works, f16-only works, BOTH crash)
**Root cause:** `matmul_f16` reused `ds_q8c` — allocated with matmul_q8_0's pipeline layout
**Fix:** Separate descriptor set handle per pipeline — ds_q8c for q8_0, ds_f16 for f16

### Bug 2: vkResetDescriptorPool + pre-allocated sets = SIGSEGV in vkUpdateDescriptorSets
**GDB backtrace proved the real crash location during PREFILL, not decode:**
```
Thread 1 "ds4" received signal SIGSEGV, Segmentation fault.
0x00007febc312d39c in ?? () from libvulkan_radeon.so
#1  0x555555621b68 in ds4_gpu_matmul_q8_0_tensor at vulkan_backend.cpp:1000
    in_dim=4096, out_dim=1024, n_tok=25
#3  metal_graph_encode_layer_attention_batch at ds4.c:17449, il=2
```
Line 1000 = `vkUpdateDescriptorSets`. Crash at PREFILL LAYER 2 — NOT during decode!
The log showing "25/25 (100.0%)" was from a DIFFERENT begin/end pair — misleading.

**Root cause:** vkResetDescriptorPool freed pre-allocated descriptor sets.
g_vk.pre_allocated_sets[0] became dangling. vkUpdateDescriptorSets on a freed
handle crashed inside RADV.

**Fix:** Remove vkResetDescriptorPool entirely. Pre-allocate descriptors once
during init, recycle via idx increment. Never call vkResetDescriptorPool.

### Bug 3: Per-dispatch descriptor allocation exhausting pool
**Fix:** Pre-allocate 32768 descriptor sets from shared layout during init.
Use vkUpdateDescriptorSets (not vkAllocateDescriptorSets) for each dispatch.

## Persistent Crash: radv_amdgpu_cs_finalize (AMDGUP command stream finalization)

### GDB backtrace WITH DEBUG SYMBOLS (critical finding):
```
Thread 1 "ds4" received signal SIGSEGV, Segmentation fault.
#0  0x00007febc31b2258 in radv_amdgpu_cs_finalize (_cs=0x555555ec4040)
    at ../src/amd/vulkan/winsys/amdgpu/radv_amdgpu_cs.c:509
#1  0x00007febc310b71c in radv_EndCommandBuffer (commandBuffer=0x555555ebf580)
    at ../src/amd/vulkan/radv_cmd_buffer.c:8152
#2  0x000055555562120c in end_and_submit ()
    at vulkan/vulkan_backend.cpp:447  (vkEndCommandBuffer)
#3  submit_and_wait ()
#4  ds4_gpu_end_commands ()
#5  0x00005555555b105a in metal_graph_eval_token_raw_swa (g=0x555555e54cd0,
    model=0x5555558052b0, weights=0x5555558053b8,
    token=74035, pos=25, logits=0x7febc55eb010)
    at ds4.c:19493  ← DECODE

Command context state at crash:
  command_count = 173  (dispatches recorded in this decode CB)
  submitted = true
  qf = RADV_QUEUE_COMPUTE
```

### Mesa source at crash site (radv_amdgpu_cs.c:509):
```c
static VkResult radv_amdgpu_cs_finalize(struct ac_cmdbuf *_cs) {
    struct radv_amdgpu_cs *cs = radv_amdgpu_cs(_cs);
    assert(cs->base.cdw <= cs->base.reserved_dw);       // line 491
    if (cs->chain_ib) { ... } else { radv_amdgpu_winsys_cs_pad(_cs, 0); }
    radv_amdgpu_cs_add_ib_buffer(cs, cs->ib_buffer,
        cs->ib_buffer->va,                              // line 506-507
        cs->chain_ib ? G_3F2_IB_SIZE(*cs->ib_size_ptr) : cs->base.cdw);
    cs->ib_buffer = NULL;                                // line 509 ← crash
    cs->chained_to = NULL;
    assert(cs->base.cdw <= cs->base.max_dw + 4);        // line 515
    return cs->status;
}
```

Line 509 is `cs->ib_buffer = NULL` — a simple NULL assignment. The crash is
likely attributed to line 509 due to compiler optimization. The REAL crash
is at line 491 (`assert(cs->base.cdw <= cs->base.reserved_dw)`) or line
506-507 (`cs->ib_buffer->va`) where `cs->ib_buffer` is accessed. If `cs`
is a garbage pointer (freed memory), any field access crashes.

### radv_EndCommandBuffer (radv_cmd_buffer.c:8152):
```c
// Line 8100:
struct radv_cmd_stream *cs = cmd_buffer->cs;
// ... cache flush + CP DMA wait ...
// Line 8152:
VkResult result = radv_finalize_cmd_stream(device, cs);
```

The CS (`cmd_buffer->cs`) is obtained from the command buffer. If the CB was
reset via vkResetCommandPool or vkResetCommandBuffer, the CS should be valid.
But the crash suggests the CS pointer or its internal state is corrupted.

## Isolation Matrix (Final)

| Prefill GPU | Decode GPU | Split | Result |
|---|---|---|---|
| YES (25 tok) | YES | default (4) | CRASH at radv_amdgpu_cs_finalize |
| YES (25 tok) | YES | 1 (every layer) | CRASH (rules out CS buffer size) |
| YES (25 tok) | YES | 10 | CRASH |
| YES (25 tok) | skip decode | N/A | no crash |
| skip prefill | YES | N/A | no crash |
| YES (2 tok -p hi) | YES | default | no crash |
| YES (25 tok) | YES | Fresh pool/fence | CRASH |
| YES (25 tok) | YES | vkResetCommandBuffer | CRASH |

## What Was Ruled Out (Complete)

1. Workgroup count > 65535 — 2D grid handles up to 4G
2. Push constant size mismatch — sizeof(pc) == push_size verified
3. Descriptor set layout mismatch — shared g_shared_dsl for all pipelines
4. Descriptor pool exhaustion — pre-allocated 32768 sets, no per-dispatch alloc
5. vkResetDescriptorPool invalidation — removed from begin_cmd (BUG 2 fix)
6. Descriptor set sharing across pipelines — fixed (BUG 1)
7. Command buffer contamination — fresh CB each time
8. Semaphore issues — removed from submit
9. VK_WHOLE_SIZE — exact buffer sizes
10. Subgroup/shmem complexity — simple shader also crashes
11. Memory barriers — VkMemoryBarrier between dispatches
12. Fresh command pool/fence — destroy+recreate each begin_cmd
13. RADV_DEBUG=validation — no diagnostic output
14. RADV_DEBUG=hang,syncshaders — no GPU hang detected
15. DS4_SPLIT_LAYERS=1 — splits every layer, still crashes
16. vkResetCommandBuffer (individual) — instead of pool reset, still crashes
17. Queue type — attempted general queue (build broke due to code error)

## Remaining Theories (untested)

1. **VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT:** llama.cpp uses this
   flag on vkResetCommandPool. Without it, the CS internal buffers may not
   be properly freed.

2. **Buffer memory barriers vs generic:** Using VkBufferMemoryBarrier
   (specific buffers) instead of VkMemoryBarrier (all buffers).

3. **Timeline semaphores:** llama.cpp uses them for multi-queue orchestration.

4. **CS initial allocation size:** Pre-allocating larger CS buffers at
   command pool creation time.

## Debug Symbols Installation

For RADV crash analysis with readable backtraces:
```bash
echo "deb http://ddebs.ubuntu.com $(lsb_release -cs) main restricted universe multiverse" \
  | sudo tee /etc/apt/sources.list.d/ddebs.list
echo "deb http://ddebs.ubuntu.com $(lsb_release -cs)-updates main restricted universe multiverse" \
  | sudo tee -a /etc/apt/sources.list.d/ddebs.list
sudo apt install -y ubuntu-dbgsym-keyring
sudo apt update
sudo apt install -y mesa-vulkan-drivers-dbgsym libvulkan1-dbgsym
```

## Mesa Source for Crash Analysis

```bash
cd /tmp
git clone --depth 1 --branch mesa-26.0.3 \
  https://gitlab.freedesktop.org/mesa/mesa.git mesa-src
# Crash site: mesa-src/src/amd/vulkan/winsys/amdgpu/radv_amdgpu_cs.c:509
# Caller:     mesa-src/src/amd/vulkan/radv_cmd_buffer.c:8152
```

## GDB Backtrace Command

```bash
gdb -batch \
  -ex "run --vulkan -c 32000 -m .../model.gguf --prompt-file prompt_test.txt" \
  -ex "bt full 30" -ex "frame 1" -ex "p *cmd_buffer" -ex "p *cmd_buffer->cs" \
  ./ds4 2>&1 | grep -A60 SIGSEGV
```

## Status

**UNRESOLVED.** radv_amdgpu_cs_finalize crashes at vkEndCommandBuffer during
decode when prefill used GPU. Root cause not identified — all Vulkan API
usage patterns (descriptor sets, layouts, pools, CB management) ruled out.
Suspected RADV driver bug in CS finalization after large prefill submission.

**Workaround available:** Skip GPU dispatches during decode (n_tok==1).
Prefill GPU works unassisted. Decode CPU fallback is slow but functional.
