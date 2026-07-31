# DS4 Vulkan backend — status & kernel roadmap

> Last updated: 2026-07-31 · Regenerate the kernel table with
> `python3 vulkan/kernel_status.py`.

## What works (2026-07-31)

- **Build**: `make vulkan` compiles all five binaries (`ds4`, `ds4-server`,
  `ds4-bench`, `ds4-eval`, `ds4-agent`) on Linux with the Vulkan backend
  (C++17 + vendored headers/volk + VMA).
- **GPU init**: instance/device creation, GPU info printed
  (`ds4: VULKAN device: ...`), 12 compute shaders loaded, pipelines created.
- **Full pipeline on GPU, Vulkan only (no CPU fallback)**: an 81 GiB DeepSeek
  V4 Flash model runs end-to-end (prefill + decode), streamed through a
  ~47 GiB iGPU heap:
  `prefill: 2.18 t/s, generation: 50.71 t/s`.
- **Weight streaming**: `cache_model_range` records metadata only; kernels
  upload weights lazily from the model mmap (`ensure_weight`) with LRU
  eviction (`DS4_VULKAN_WEIGHT_BUDGET_GB`, default 40 GiB).
- **RADV crash fixed** (`radv_amdgpu_cs_finalize` on decode after large
  prefill): the engine calls `ds4_gpu_commit_and_wait_selected_readback`
  mid-decode and keeps recording kernels without `begin_commands`.
  `commit_and_wait` / `flush_commands` now re-begin a fresh command buffer
  after submit (Metal encoder semantics).

## Why the output is only `<｜begin▁of▁sentence｜>`

Of the 261 `ds4_gpu_*` contract functions, **157 are no-op stubs**
(`vulkan/_impl_gen.cpp`) that return 1 (success) **without writing their
output tensors**. The engine therefore runs on garbage activations:

- router / MoE / attention / indexer / compressor / KV kernels are stubs →
  selected experts, attention heads and KV state are garbage;
- `ds4_gpu_matmul_f16_tensor` was reading f16 weights as uint32 and using a
  subgroup reduction that RADV mishandled — both fixed and **verified** by the
  kernel harness (2026-07-31);
- the final logits are garbage, so greedy argmax degenerates to the BOS token.

The pipeline is complete and stable; the math is not.

## How to test

```sh
./run-test.sh                              # rebuild + 10-token smoke test
DS4_VULKAN_WEIGHT_BUDGET_GB=46 ./run-test.sh ...
DS4_VULKAN_DEBUG=1 ./run-test.sh ...       # verbose begin/end/submit trace
```

## Kernel implementation order (critical path to correct output)

1. ✅ `ds4_gpu_matmul_f16_tensor` — f16 element size + reduction fixed,
   verified by `vulkan/tests/tests/t_matmul_f16.cpp`.
2. ✅ `ds4_gpu_router_select_tensor` (decode) — host-side top-k + softmax,
   verified by `vulkan/tests/tests/t_router_select.cpp`.
3. ✅ `ds4_gpu_routed_moe_one_tensor` (decode) — MoE FFN host-side con
   dequant Q8_0/Q2_K/IQ2_XXS, verified by
   `vulkan/tests/tests/t_routed_moe_one.cpp` (interno-coerente col reference
   ds4.c; validazione finale contro il motore da fare).
4. ✅ `ds4_gpu_attention_decode_heads_tensor` (decode) — host-side attention
   (raw circolare + comp f16/f32 + mask), verified by
   `vulkan/tests/tests/t_attention_decode.cpp`.
5. ✅ `ds4_gpu_embed_token_q8_0/quant_tensor` + batch — input embeddings
   (Q8_0/F16), verified by `vulkan/tests/tests/t_embed.cpp`.
6. ✅ Indexer (`score_one`, `scores_decode_batch`, `topk`) — verified by
   `vulkan/tests/tests/t_indexer.cpp`. Compressor (`compressor_*`) ancora STUB.
7. Prefill-side kernels: `routed_moe_batch_tensor` azzera l'output (fix
   necessario), `attention_prefill_*`/KV da verificare, `matmul_q8_0` da
   verificare (probabile bug formato come f16).
8. HC expand family (`ds4_gpu_hc_expand_*`) — STUB.

Each real kernel must also be audited for format bugs (f16 vs f32, block
layouts) before it is trusted.

## Kernel table

Legend: `REAL` = implemented in `vulkan/vulkan_backend.cpp` (may still be
numerically wrong); `STUB` = no-op placeholder from `vulkan/_impl_gen.cpp`
(returns 1 without touching outputs).

---

## attention (2 real / 26 stub)

| function | status |
|---|---|
| `ds4_gpu_attention_decode_heads_rope_tensor` | STUB |
| `ds4_gpu_attention_decode_heads_tensor` | VERIFIED |
| `ds4_gpu_attention_decode_mixed_batch_heads_tensor` | STUB |
| `ds4_gpu_attention_decode_raw_batch_heads_tensor` | REAL |
| `ds4_gpu_attention_decode_rows_rope_tensor` | STUB |
| `ds4_gpu_attention_indexed_mixed_batch_heads_tensor` | STUB |
| `ds4_gpu_attention_noncausal_raw_batch_heads_tensor` | STUB |
| `ds4_gpu_attention_output_low_q4_K_slice_tensor` | STUB |
| `ds4_gpu_attention_output_low_q8_rows_exact_tensor` | STUB |
| `ds4_gpu_attention_output_low_q8_tensor` | VERIFIED |
| `ds4_gpu_attention_output_q4_K_batch_tensor` | STUB |
| `ds4_gpu_attention_output_q8_batch_f16_tensor` | STUB |
| `ds4_gpu_attention_output_q8_batch_tensor` | VERIFIED |
| `ds4_gpu_attention_output_q8_tp_tensor` | STUB |
| `ds4_gpu_attention_prefill_masked_mixed_heads_tensor` | STUB |
| `ds4_gpu_attention_prefill_raw_heads_range_tensor` | STUB |
| `ds4_gpu_attention_prefill_raw_heads_tensor` | VERIFIED |
| `ds4_gpu_attention_prefill_static_mixed_heads_range_tensor` | STUB |
| `ds4_gpu_attention_prefill_static_mixed_heads_tensor` | STUB |
| `ds4_gpu_glm_attention_flash_staged_tensor` | STUB |
| `ds4_gpu_glm_attention_flash_tensor` | STUB |
| `ds4_gpu_glm_attention_full_tensor` | STUB |
| `ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor` | STUB |
| `ds4_gpu_glm_attention_indexed_batch_lora_tensor` | STUB |
| `ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor` | STUB |
| `ds4_gpu_glm_attention_indexed_batch_tensor` | STUB |
| `ds4_gpu_glm_attention_indexed_batch_typed_tensor` | STUB |
| `ds4_gpu_glm_attention_indexed_decode_split_group8_tensor` | STUB |
| `ds4_gpu_glm_attention_indexed_decode_split_group8_typed_tensor` | STUB |
| `ds4_gpu_glm_attention_indexed_decode_tensor` | STUB |
| `ds4_gpu_glm_attention_indexed_decode_typed_tensor` | STUB |
| `ds4_gpu_set_decode_fast_attention` | REAL |

## commands (7 real / 2 stub)

| function | status |
|---|---|
| `ds4_gpu_begin_commands` | REAL |
| `ds4_gpu_commands_active` | STUB |
| `ds4_gpu_commit_and_wait_selected_readback` | REAL |
| `ds4_gpu_end_commands` | REAL |
| `ds4_gpu_flush_commands` | REAL |
| `ds4_gpu_flush_encoder` | STUB |
| `ds4_gpu_signal_selected_readback_ready` | REAL |
| `ds4_gpu_synchronize` | REAL |
| `ds4_gpu_wait_selected_readback_ready` | REAL |

## embed (2 real / 0 stub)

| function | status |
|---|---|
| `ds4_gpu_embed_token_hc_tensor` | REAL |
| `ds4_gpu_embed_token_q8_0_tensor` | VERIFIED |
| `ds4_gpu_embed_token_quant_tensor` | VERIFIED |
| `ds4_gpu_embed_tokens_hc_tensor` | REAL |
| `ds4_gpu_embed_tokens_q8_0_tensor` | VERIFIED |
| `ds4_gpu_embed_tokens_quant_tensor` | VERIFIED |

## hc (7 real / 6 stub)

| function | status |
|---|---|
| `ds4_gpu_hc_expand_add_split_half_add_tensor` | STUB |
| `ds4_gpu_hc_expand_add_split_tensor` | STUB |
| `ds4_gpu_hc_expand_add_tensor` | REAL |
| `ds4_gpu_hc_expand_split_half_tensor` | STUB |
| `ds4_gpu_hc_expand_split_tensor` | REAL |
| `ds4_gpu_hc_expand_tensor` | REAL |
| `ds4_gpu_hc_rms_scale_project_f16_tensor` | STUB |
| `ds4_gpu_hc_split_weighted_sum_tensor` | REAL |
| `ds4_gpu_hc_weighted_sum_split_tensor` | REAL |
| `ds4_gpu_hc_weighted_sum_tensor` | REAL |
| `ds4_gpu_output_hc_weights_tensor` | REAL |
| `ds4_gpu_repeat_hc_rows_tensor` | STUB |
| `ds4_gpu_repeat_hc_tensor` | STUB |

## indexer (0 real / 9 stub)

| function | status |
|---|---|
| `ds4_gpu_argmax_tensor` | STUB |
| `ds4_gpu_dspark_markov_argmax_tensor` | STUB |
| `ds4_gpu_dsv4_indexer_qat_tensor` | STUB |
| `ds4_gpu_dsv4_topk_mask_tensor` | STUB |
| `ds4_gpu_glm_indexer_score_one_tensor` | STUB |
| `ds4_gpu_glm_indexer_scores_batch_tensor` | STUB |
| `ds4_gpu_glm_store_indexer_k_tensor` | STUB |
| `ds4_gpu_indexer_score_one_tensor` | VERIFIED |
| `ds4_gpu_indexer_scores_decode_batch_tensor` | VERIFIED |
| `ds4_gpu_indexer_scores_prefill_tensor` | STUB |
| `ds4_gpu_indexer_top1_value_tensor` | STUB |
| `ds4_gpu_indexer_topk_tensor` | VERIFIED |

## kv (4 real / 8 stub)

| function | status |
|---|---|
| `ds4_gpu_compressor_prefill_ratio4_replay_tensor` | STUB |
| `ds4_gpu_compressor_prefill_state_ratio4_tensor` | STUB |
| `ds4_gpu_compressor_prefill_tensor` | REAL |
| `ds4_gpu_compressor_store_batch_tensor` | REAL |
| `ds4_gpu_compressor_update_tensor` | REAL |
| `ds4_gpu_dsv4_fp8_kv_quantize_tensor` | VERIFIED |
| `ds4_gpu_flash_kv_stage_f16_tensor` | STUB |
| `ds4_gpu_glm_build_kv_cache_flash_tensor` | STUB |
| `ds4_gpu_glm_build_kv_cache_tensor` | STUB |
| `ds4_gpu_glm_store_compact_kv_tensor` | STUB |
| `ds4_gpu_kv_fp8_store_raw_decode_rows_tensor` | STUB |
| `ds4_gpu_kv_fp8_store_raw_tensor` | VERIFIED |
| `ds4_gpu_should_use_managed_kv_cache` | REAL |
| `ds4_gpu_store_raw_kv_batch_tensor` | VERIFIED |
| `ds4_gpu_store_raw_kv_tensor` | STUB |

## matmul (1 real / 16 stub)

| function | status |
|---|---|
| `ds4_gpu_add3_tensor` | STUB |
| `ds4_gpu_add_tensor` | REAL |
| `ds4_gpu_matmul_f16_pair_compressor_store_tensor` | VERIFIED |
| `ds4_gpu_matmul_f16_pair_tensor` | STUB |
| `ds4_gpu_matmul_f16_router_rows_exact_tensor` | STUB |
| `ds4_gpu_matmul_f16_tensor` | VERIFIED |
| `ds4_gpu_matmul_f32_tensor` | VERIFIED |
| `ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor` | STUB |
| `ds4_gpu_matmul_q8_0_decode_mpp_tensor` | STUB |
| `ds4_gpu_matmul_q8_0_decode_rows_exact_tensor` | STUB |
| `ds4_gpu_matmul_q8_0_f16_out_tensor` | STUB |
| `ds4_gpu_matmul_q8_0_hc_expand_tensor` | VERIFIED |
| `ds4_gpu_matmul_q8_0_kslice_hc_expand_add_tensor` | STUB |
| `ds4_gpu_matmul_q8_0_kslice_rows_tensor` | STUB |
| `ds4_gpu_matmul_q8_0_kslice_tensor` | STUB |
| `ds4_gpu_matmul_q8_0_pair_decode_rows_exact_tensor` | STUB |
| `ds4_gpu_matmul_q8_0_pair_tensor` | VERIFIED |
| `ds4_gpu_matmul_q8_0_rows_scalar_tensor` | STUB |
| `ds4_gpu_matmul_q8_0_tensor` | VERIFIED |
| `ds4_gpu_matmul_q8_0_top1_tensor` | STUB |
| `ds4_gpu_matmul_quant_decode_mpp_model_view_tensor` | STUB |
| `ds4_gpu_matmul_quant_kslice_tensor` | STUB |
| `ds4_gpu_matmul_quant_rows_scalar_tensor` | STUB |
| `ds4_gpu_matmul_quant_tensor` | VERIFIED |

## mgpu (5 real / 0 stub)

| function | status |
|---|---|
| `ds4_gpu_add_xdev_tensor` | REAL |
| `ds4_gpu_init_multi` | REAL |
| `ds4_gpu_set_current_device` | REAL |
| `ds4_gpu_set_current_device_fenced` | REAL |
| `ds4_gpu_tier_free_vram` | REAL |

## misc (9 real / 15 stub)

| function | status |
|---|---|
| `ds4_gpu_cache_q8_f16_range` | REAL |
| `ds4_gpu_cleanup` | REAL |
| `ds4_gpu_directional_steering_project_tensor` | STUB |
| `ds4_gpu_enable_q8_dequant_gemm` | REAL |
| `ds4_gpu_glm_fill_selected_range_batch_tensor` | STUB |
| `ds4_gpu_glm_fill_selected_range_tensor` | STUB |
| `ds4_gpu_glm_k_b_project_tensor` | STUB |
| `ds4_gpu_glm_k_b_project_typed_tensor` | STUB |
| `ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor` | STUB |
| `ds4_gpu_glm_qk_lowrank_q8_0_tensor` | STUB |
| `ds4_gpu_glm_qk_lowrank_typed_batch_tensor` | STUB |
| `ds4_gpu_glm_qk_lowrank_typed_tensor` | STUB |
| `ds4_gpu_glm_value_project_q8_0_batch_heads_tensor` | STUB |
| `ds4_gpu_glm_value_project_typed_batch_heads_tensor` | STUB |
| `ds4_gpu_init` | REAL |
| `ds4_gpu_pack_slot_rows_f32_tensor` | STUB |
| `ds4_gpu_print_memory_report` | REAL |
| `ds4_gpu_recommended_working_set_size` | REAL |
| `ds4_gpu_release_q8_f16_cache` | REAL |
| `ds4_gpu_release_zero_prefix_prefill_mask_cache` | STUB |
| `ds4_gpu_set_decode_score_vec4` | REAL |
| `ds4_gpu_set_glm_model` | STUB |
| `ds4_gpu_set_quality` | REAL |
| `ds4_gpu_sort_i32_rows_asc_tensor` | STUB |

## model (8 real / 0 stub)

| function | status |
|---|---|
| `ds4_gpu_cache_model_range` | REAL |
| `ds4_gpu_q8_cache_suppressed` | REAL |
| `ds4_gpu_set_model_fd` | REAL |
| `ds4_gpu_set_model_fd_for_map` | REAL |
| `ds4_gpu_set_model_map` | REAL |
| `ds4_gpu_set_model_map_range` | REAL |
| `ds4_gpu_set_model_map_spans` | REAL |
| `ds4_gpu_set_q8_cache_suppressed` | REAL |

## moe (5 real / 19 stub)

| function | status |
|---|---|
| `ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor` | STUB |
| `ds4_gpu_glm_routed_moe_batch_tensor` | STUB |
| `ds4_gpu_glm_routed_moe_one_tensor` | STUB |
| `ds4_gpu_glm_router_select_batch_tensor` | STUB |
| `ds4_gpu_glm_router_select_tensor` | STUB |
| `ds4_gpu_hc_split_sinkhorn_tensor` | STUB |
| `ds4_gpu_moe_handoff_pack_tensor` | STUB |
| `ds4_gpu_routed_moe_batch_owned_tensor` | STUB |
| `ds4_gpu_routed_moe_batch_tensor` | VERIFIED |
| `ds4_gpu_routed_moe_one_owned_tensor` | STUB |
| `ds4_gpu_routed_moe_one_tensor` | VERIFIED |
| `ds4_gpu_routed_moe_owned_packed_combine_tensor` | STUB |
| `ds4_gpu_routed_moe_owned_slots_combine_rows_tensor` | STUB |
| `ds4_gpu_routed_moe_owned_slots_combine_tensor` | STUB |
| `ds4_gpu_routed_moe_set_selected_override` | STUB |
| `ds4_gpu_router_select_batch_tensor` | REAL |
| `ds4_gpu_router_select_tensor` | VERIFIED |
| `ds4_gpu_shared_down_hc_expand_add_q8_0_tensor` | STUB |
| `ds4_gpu_shared_down_hc_expand_owned_q8_0_tensor` | STUB |
| `ds4_gpu_shared_down_hc_expand_q8_0_tensor` | REAL |
| `ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor` | STUB |
| `ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor` | STUB |
| `ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor` | STUB |
| `ds4_gpu_shared_gate_up_swiglu_q8_0_tensor` | REAL |
| `ds4_gpu_shared_mid_swiglu_q8_0_decode_exact_tensor` | STUB |
| `ds4_gpu_shared_mid_swiglu_q8_0_tensor` | REAL |
| `ds4_gpu_swiglu_tensor` | REAL |

## norm (3 real / 4 stub)

| function | status |
|---|---|
| `ds4_gpu_add_rms_norm_weight_tensor` | STUB |
| `ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor` | VERIFIED |
| `ds4_gpu_dsv4_qkv_rms_norm_rows_tensor` | VERIFIED |
| `ds4_gpu_glm_kv_lora_rms_norm_tensor` | STUB |
| `ds4_gpu_glm_qkv_norm_store_compact_kv_tensor` | STUB |
| `ds4_gpu_hc_split_weighted_sum_norm_tensor` | REAL |
| `ds4_gpu_hc_weighted_sum_norm_tensor` | STUB |
| `ds4_gpu_head_rms_norm_rope_tail_tensor` | VERIFIED |
| `ds4_gpu_head_rms_norm_tensor` | VERIFIED |
| `ds4_gpu_rms_norm_plain_rows_tensor` | VERIFIED |
| `ds4_gpu_rms_norm_plain_tensor` | REAL |
| `ds4_gpu_rms_norm_weight_rows_tensor` | VERIFIED |
| `ds4_gpu_rms_norm_weight_tensor` | REAL |

## rope (1 real / 3 stub)

| function | status |
|---|---|
| `ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor` | REAL |
| `ds4_gpu_glm_indexer_rope_tail_tensor` | STUB |
| `ds4_gpu_glm_rope_tail_tensor` | STUB |
| `ds4_gpu_rope_tail_decode_rows_tensor` | STUB |
| `ds4_gpu_rope_tail_tensor` | VERIFIED |

## stream (17 real / 4 stub)

| function | status |
|---|---|
| `ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor` | STUB |
| `ds4_gpu_model_residency_skip` | STUB |
| `ds4_gpu_preload_q4_expert_tables` | REAL |
| `ds4_gpu_pro_q4_expert_table_auto_available` | REAL |
| `ds4_gpu_set_glm_streaming_prefill_full_layer` | STUB |
| `ds4_gpu_set_ssd_streaming` | REAL |
| `ds4_gpu_set_streaming_expert_cache_budget` | REAL |
| `ds4_gpu_set_streaming_expert_cache_expert_bytes` | REAL |
| `ds4_gpu_stream_expert_cache_begin_selected_load` | REAL |
| `ds4_gpu_stream_expert_cache_budget_for_expert_size` | REAL |
| `ds4_gpu_stream_expert_cache_configured_count` | REAL |
| `ds4_gpu_stream_expert_cache_current_count` | REAL |
| `ds4_gpu_stream_expert_cache_load_layer` | REAL |
| `ds4_gpu_stream_expert_cache_note_service_thread` | STUB |
| `ds4_gpu_stream_expert_cache_prepare_selected_batch` | REAL |
| `ds4_gpu_stream_expert_cache_release_layer_cache` | REAL |
| `ds4_gpu_stream_expert_cache_release_resident` | REAL |
| `ds4_gpu_stream_expert_cache_reset_route_hotness` | REAL |
| `ds4_gpu_stream_expert_cache_seed_experts` | REAL |
| `ds4_gpu_stream_expert_cache_seed_from_layer_selected` | REAL |
| `ds4_gpu_stream_expert_cache_seed_selected` | REAL |

## tensor (31 real / 1 stub)

| function | status |
|---|---|
| `ds4_gpu_device_cache_support_tensors` | REAL |
| `ds4_gpu_device_cache_tensors` | REAL |
| `ds4_gpu_lookup_cache` | REAL |
| `ds4_gpu_lookup_cache_device` | REAL |
| `ds4_gpu_lookup_cache_strict` | REAL |
| `ds4_gpu_register_model_map_no_copy` | REAL |
| `ds4_gpu_register_support_map` | REAL |
| `ds4_gpu_tensor_alloc` | REAL |
| `ds4_gpu_tensor_alloc_managed` | REAL |
| `ds4_gpu_tensor_alloc_managed_on` | REAL |
| `ds4_gpu_tensor_alloc_on` | REAL |
| `ds4_gpu_tensor_alloc_ptr_on` | REAL |
| `ds4_gpu_tensor_bytes` | REAL |
| `ds4_gpu_tensor_contents` | REAL |
| `ds4_gpu_tensor_copy` | REAL |
| `ds4_gpu_tensor_copy_async` | REAL |
| `ds4_gpu_tensor_copy_f32_to_f16` | REAL |
| `ds4_gpu_tensor_copy_xdev` | REAL |
| `ds4_gpu_tensor_copy_xdev3` | REAL |
| `ds4_gpu_tensor_copy_xdev3_default_dst` | REAL |
| `ds4_gpu_tensor_copy_xdev_default` | REAL |
| `ds4_gpu_tensor_copy_xdev_ordered` | REAL |
| `ds4_gpu_tensor_device` | REAL |
| `ds4_gpu_tensor_fill_f32` | REAL |
| `ds4_gpu_tensor_free` | REAL |
| `ds4_gpu_tensor_free_in_place` | REAL |
| `ds4_gpu_tensor_read` | REAL |
| `ds4_gpu_tensor_read_after_selected_event` | STUB |
| `ds4_gpu_tensor_view` | REAL |
| `ds4_gpu_tensor_wait_xdev` | REAL |
| `ds4_gpu_tensor_wait_xdev_default` | REAL |
| `ds4_gpu_tensor_write` | REAL |

## tp (0 real / 15 stub)

| function | status |
|---|---|
| `ds4_gpu_set_glm_mtp_verify_mode` | STUB |
| `ds4_gpu_tp_batch_gate_encode` | STUB |
| `ds4_gpu_tp_big_gate_encode` | STUB |
| `ds4_gpu_tp_big_gate_kick` | STUB |
| `ds4_gpu_tp_big_gate_wait` | STUB |
| `ds4_gpu_tp_failed` | STUB |
| `ds4_gpu_tp_gate_encode` | STUB |
| `ds4_gpu_tp_init` | STUB |
| `ds4_gpu_tp_keepalive_pause` | STUB |
| `ds4_gpu_tp_set_attn_head_split` | STUB |
| `ds4_gpu_tp_set_batch_exchange` | STUB |
| `ds4_gpu_tp_set_big_exchange` | STUB |
| `ds4_gpu_tp_set_session_batch_mode` | STUB |
| `ds4_gpu_tp_shutdown` | STUB |
| `ds4_gpu_tp_suspend_expert_sharding` | STUB |

TOTAL: 261 functions, 102 real, 128 stub
