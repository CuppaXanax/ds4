# DS4 Vulkan Backend — Session Notes (2026-07-12, Part 2)

## Critical findings

### 1. `_impl_gen.cpp` returning 0 causes every pipeline path to fail

The auto-generated `_impl_gen.cpp` contained 42 functions all returning `return 0;`.
Every `if (ok) ok = func() != 0` check evaluated `0 != 0` = `false`, setting `ok = false`.
This affected ALL pipeline functions (attention, FFN, router, MoE, HC expand).

**Lesson:** `_impl_gen.cpp` pass-through stubs MUST return 1, not 0. Return-0 is for
`_stubs.gen.cpp` functions you KNOW are unimplemented and want to crash on.

The symptom was `ok=0` at the end of `metal_graph_encode_layer_ffn_batch(il=0)`
with ALL breakpoints showing real implementations returning 1 — because the failure
was in a different function (e.g., `ds4_gpu_expert_count_selected_tensor` → 0).

### 2. `keep_ffn_out` and `shared_down_f16` determine the HC expand path

The FFN post-MoE code branches on `shared_down_f16`:

```c
if (ok && shared_down_f16) {
    ok = ds4_gpu_hc_expand_add_split_half_add_tensor(next_hc_view,
        g->batch_routed_out, g->batch_q_half, g->batch_after_attn_hc,
        hc_split_view, DS4_N_EMBD, DS4_N_HC) != 0;
}
else if (ok) {
    ok = ds4_gpu_hc_expand_add_split_tensor(next_hc_view,
        g->batch_routed_out, g->batch_shared_out, g->batch_after_attn_hc,
        hc_split_view, DS4_N_EMBD, DS4_N_HC) != 0;
}
```

If `DS4_METAL_TRY_SHARED_DOWN_F16()` ran (sets `shared_down_f16 = true`), the
"half_add" variant is used, which takes `g->batch_q_half` as the shared output.
If `shared_down_f16` is false, the plain `_add_split_tensor` variant takes
`g->batch_shared_out`.

### 3. Decode CRASHES in RADV driver at vkEndCommandBuffer

The segfault at `end_and_submit()` → `vkEndCommandBuffer` happens ONLY during
decode (not prefill). Prefill for all 43 layers completes successfully.

Stack trace:
```
#0 libvulkan_radeon.so (driver crash during CB validation)
#1 libvulkan_radeon.so
#2 end_and_submit → vkEndCommandBuffer
#3 ds4_gpu_end_commands
#4 metal_graph_eval_token_raw_swa (decode, pos=10)
```

**Current hypothesis:** The decode path dispatches matmuls with `out_dim`
exceeding the max workgroup count (Q_B = 65536, output head = 129280).
Even with tiling to 65534 chunks, the driver still crashes — possibly
because the tiled dispatches are in the same CB and the driver validates
them together.

**Alternative hypothesis:** Shared memory overflow in a decode-specific
shader path (different dimensions from prefill).

### 4. GDB breakpoints don't fire with -O3

The compiler inlines and reorders aggressively. Breakpoints on source lines
inside `metal_graph_encode_layer_ffn_batch` often fail to fire even though
the function IS called (confirmed by the error message).

**Workaround:** Set breakpoints on the GPU functions directly
(e.g., `break ds4_gpu_matmul_q8_0_tensor`) rather than on lines in ds4.c.

### 5. `ds4_gpu_tensor_free` called on view tensors

The function frees views created by `ds4_gpu_tensor_view()`. These views
have `owner = 0`, so `ds4_gpu_tensor_free` skips VMA buffer destruction
and just calls `free(t)`. This is correct — views are lightweight.

### 6. void-returning functions don't affect `ok`

Functions with `void` return type (`ds4_gpu_tensor_free`, debug dumps, etc.)
can NOT set `ok = false`. Only `int`/`bool`-returning functions assigned
via `if (ok) ok = func() != 0` can change `ok`.

### 7. Weight cache lookup by range (not exact offset)

The weight cache stores entries keyed by SPAN offset (from
`cache_model_range`), but matmul dispatches look up by TENSOR offset.
Since multiple tensors can share a single span, the lookup must search
by range: `weight_offset >= span_off && weight_offset < span_off + span_size`.

Without this fix, the dispatch returns 0 (weight not found) for any tensor
that is NOT the first tensor in its span.
