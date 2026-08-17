# Decode fusion target

## What is now staged

`DS4_VULKAN_Q8_HC_ADD_FUSE=1` admits an exact artifact-native Wave64 consumer
(`matmul_q8_0_exec_hc_expand_add_wave64.comp`) for the common ratio-4 decode
shared-down shape (`4096 -> 4096`, one token, four HC streams). It computes
the shared Q8 down row, adds `routed_out`, and performs HC post in one
workgroup. When no immutable Q8 artifact is available, the legacy aligned
consumer remains an opt-in fallback; enabling this fusion must not evict or
bypass an execution artifact.

The arithmetic order is the existing order:

```text
shared = Q8-down reduction in ascending block order
combined = routed_out + shared
out_hc = combined * post + ascending comb/residual sum
```

The established path remains the default. The candidate removes the
`shared_out` write/read and one separate `hc_expand_add_split` dispatch, but
it still performs the ordinary Q8 activation quantization. Its defensible
upper bound is therefore the avoided 16 KiB intermediate traffic plus one
small dispatch, not 0.465 ms/layer; expect a low-tens-of-microseconds class
win unless the production trace proves otherwise. It is a correctness and
composition gate, not the larger mixed-Q2/Q8 fusion claim.

## The >=0.465 ms/layer fusion

The production FFN tail currently has this dataflow:

```text
routed IQ2 gate/up -> SwiGLU -> Q8 mid -> Q2 down/reduce -> routed_out
shared Q8 gate/up -> SwiGLU -> shared Q8 down -> shared_out
routed_out + shared_out -> HC expand
```

The single bounded fusion with enough leverage is an expert-output-row
shader that performs the last two projections together:

```text
one WG per two output rows:
  read routed Q8 mid and shared Q8 mid
  reduce six Q2 routed rows and one Q8 shared row
  add routed + shared in canonical order
  apply HC post/comb directly to out_hc
```

It removes the routed-out and shared-out global intermediates, their add
dispatch, and the standalone HC-expand dispatch. It must use the existing
Q2 `q2_block_dot_words` order and Q8 aligned/artifact block order, then use
the same `routed + shared` expression as `hc_expand_add_split`. The shader
needs separate Q2 and Q8 weight plane bindings (or the GGUF fallback pairs),
Q8 routed-mid and f32/Q8 shared-mid bindings, residual/split, and out_hc.

Host entry point: add
`ds4_gpu_routed_shared_down_hc_expand_tensor` beside
`ds4_gpu_shared_down_hc_expand_q8_0_tensor`, and call it in the single-GPU
ratio-4 branch before the existing routed/shared tail. Admit only:

* `IQ2_XXS` gate/up, `Q2_K` down;
* `4096` expert input, `2048` expert mid, `4096` output;
* `n_tokens == 1`, six selected experts, no TP/steering/debug/profile;
* both execution-artifact entries present, or both exact GGUF fallback
  descriptors present.

Keep the existing path as an immediate fallback. Add a synthetic exact test
that compares the fused output to the current route + shared + HC sequence,
including a nonzero routed output and nontrivial residual/comb values. Only
then run the sustained decode gate; this change is the first candidate in
this area that can plausibly remove at least 0.465 ms/layer because it joins
the Q2 routed reduction and Q8 shared down rather than merely deleting a
small float scratch copy.

## Current implementation candidate

`routed_moe_q2_shared_hc_exec.comp` is staged as the larger, artifact-only
variant. It is admitted only when both the routed Q2 and shared Q8 execution
artifacts are present and the caller sets:

```text
DS4_VULKAN_ROUTED_Q2_SHARED_HC_FUSE=1
DS4_VULKAN_REQUIRE_ROUTED_Q2_SHARED_HC_FUSE=1
```

The existing IQ2 gate/up path remains intact. After it produces the routed
Q8_K mid, the candidate quantizes shared mid once, then one 256-lane
workgroup per five output rows performs the exact six-rank Q2 reduction, the
shared Q8 artifact down projection, routed-first/shared-second addition, and
HC post. It retains the routed output write for diagnostics, but no later
dispatch consumes it. Artifact admission failure leaves the established path
available unless `REQUIRE` is set. This is a compile/static candidate only
until the exact full-model gate and sustained decode test run.

The fused tail removes the standalone Q2 down/reduce dispatch's intermediate
expert-output write/read and the subsequent shared-down projection plus HC
expand/add dispatches: the production tail is reduced to one dispatch, while
the existing quantize, IQ2 gate/up/SwiGLU, and shared gate/up dispatches remain.
The compatibility `routed_out` write is intentionally retained, so this is a
conservative fusion rather than a claim that every intermediate allocation is
gone. The `REQUIRE` admission path now fails closed if the Q2 or shared-Q8
execution artifact is unavailable, and its resource barrier includes the
GPU-produced mid, route IDs, residual HC state, and HC coefficients.
