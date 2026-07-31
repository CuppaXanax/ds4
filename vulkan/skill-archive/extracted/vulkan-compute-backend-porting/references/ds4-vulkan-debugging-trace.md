# DS4 Vulkan Backend: Debugging Trace

## Session: 2026-07-12 — DS4 V4 Flash IQ2XXS on Strix Halo / RADV

### GDB trace: finding where `ok` becomes false

The DS4 prefill/decode uses a local `bool ok`. Functions chain:
`if (ok) ok = gpu_func(...) != 0;`

The first function that returns 0 stops the pipeline. Use GDB binary search:

```gdb
# Enter attention function for failing layer
break metal_graph_encode_layer_batch if il == 0
run --vulkan -c 32000 -m model.gguf -p hi

# Advance to key checkpoints
advance 17460   # after matmul_f16 + HC split → ok=true
advance 17495   # after dsv4_qkv_rms_norm_rows → ok=true
advance 17510   # after fused block → ok=false
```

### Functions found to be stubs (returning 0) during tracing

These were identified one at a time by GDB tracing and reg-stub detection:

| Function | Called from (ds4.c line) | Role |
|---|---|---|
| `ds4_gpu_dsv4_qkv_rms_norm_rows_tensor` | 17479 | Fused Q+KV norm |
| `ds4_gpu_dsv4_fp8_kv_quantize_tensor` | 17639 | KV cache quantize |
| `ds4_gpu_attention_prefill_raw_heads_tensor` | 17666 | Causal attention for batch prefill |

### Chain of failure (attention path, il=0, n_tokens=10)

```
rms_norm_plain_rows @ 17370 → OK
matmul_f16 @ 17378 → OK (GDB confirmed returns 1)
hc_split_weighted_sum @ 17410 → OK (CPU)
rms_norm_weight_rows @ 17492 → OK (CPU)
matmul_q8_0 "attn_q_a" @ 17451 → OK (Vulkan dispatch)
matmul_q8_0 "attn_kv" @ 17467 → OK (Vulkan dispatch)
dsv4_qkv_rms_norm_rows @ 17479 → OK (CPU, just implemented)
rope_tail(KV) @ 17621 → OK (CPU)
dsv4_fp8_kv_quantize @ 17639 → **STUB (return 0)** ← FAIL
```

After fixing `dsv4_fp8_kv_quantize`:
```
attention_prefill_raw_heads @ 17666 → **STUB (return 0)** ← FAIL
```

### RADV driver crash during decode

Stack trace:
```
#0  libvulkan_radeon.so  (SIGSEGV)
#1  libvulkan_radeon.so
#2  end_and_submit()  (vulkan_backend.cpp:415)
#3  submit_and_wait()
#4  ds4_gpu_end_commands()
#5  metal_graph_eval_token_raw_swa (ds4.c:19493)
```

Cause: `VK_WHOLE_SIZE` in descriptor bindings. The RADV driver's
validation layer crashes (SIGSEGV) when a shader accesses beyond the
exact buffer range, instead of returning VK_ERROR. Fix: compute exact
buffer sizes from tensor dimensions.

### Two-bug chain causing hang at "processing 0/N (0.0%)"

1. `begin_cmd()` resets fence (SIGNALED→UNSIGNALED)
2. All functions are stubs → `command_count = 0`
3. `end_and_submit()` skips submission (command_count==0)
4. `wait_cmd()` skips (command_count==0)
5. Fence is UNSIGNALED, never to be signaled
6. Next `begin_cmd()` calls `vkWaitForFences(UNSIGNALED,
   UINT64_MAX)` → **infinite hang**

Fix: add `bool submitted;` flag to struct. Only wait/reset fence if
submitted==true.
