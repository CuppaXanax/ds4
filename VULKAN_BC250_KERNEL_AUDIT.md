# Vulkan BC-250 Kernel Audit and Night Plan

Hydrated from the final kernel/backend audit in the ChatGPT conversation
`DS4 BC-250 Optimization` on 2026-08-12, then checked against the current
workspace at `pr-557-merge` commit `bbe1ed9`.

## Baseline verification

The checkout is the recorded 3.48 TPS baseline:

- branch: `pr-557-merge`
- `HEAD`: `bbe1ed9`
- remote: `origin/pr-557-merge`
- working tree at hydration time: clean
- recorded distributed result: 287.224 ms/token, 3.482 TPS
- deterministic artifact: 2,178 bytes
- deterministic SHA-256: `3fbf53f82bb25e37502ff64e11d660104d32618880fe07e06f021142b969b9e4`
- Vulkan suite at the recorded baseline: 79/79 passing

The 3.482 TPS result is specifically the **all-layer-batched** result. At the
hydrated baseline it was still gated by `DS4_VULKAN_BATCH_LAYER=1`; the manual
command that produced roughly 1.5--1.7 generation TPS did not set it. The gate
has now been removed so the qualified path is the Vulkan default.

The other variables in that manual command have different status:

| Variable | Current status at `bbe1ed9` |
|---|---|
| `DS4_VULKAN_Q8_PREQUANT=1` | Obsolete/no-op. Qualified prequantization was promoted to the automatic path in `34b5297`. |
| `DS4_VULKAN_Q8_ALIGNED=1` | Obsolete/no-op. The aligned artifact is selected automatically when eligible, with fallback. |
| `DS4_VULKAN_Q2_WORDS=1` | Obsolete/no-op. Q2 direct-word down decode is automatic for the eligible type. |
| `DS4_VULKAN_WEIGHT_BUDGET_GB=11` | Active and required for the intended resident-weight budget. |
| `DS4_VULKAN_BATCH_LAYER=1` | Removed. All-layer batching is now unconditional. |

No fleet launch flag is now required for layer batching.

## Interpreting the cold first prompt

The observed first/second prefill rates were 0.53 and 3.19 tokens/s. That is
consistent with a cold first-use penalty but is not yet a useful kernel
benchmark: the samples contain only 10 and 7 suffix tokens, so fixed first-use
cost dominates the first rate.

Plausible first-use costs include mapped-weight page residency, first-touch
temporary/KV allocation, and driver/cache warming. The second turn also reuses
the existing conversation KV and processes only its new suffix. This behavior
does not explain the steady generation regression; missing all-layer batching
does.

Before changing prefill code, measure at least three identical fresh-session
prompts with a meaningful token count, discard one explicit warm-up run, and
check the Vulkan timeline for steady-state `weight_upload` events. Any
`weight_upload` during warm generation is a stop-the-line finding.

## Hydrated audit

### 1. Aligned Q8 finishes with a serial lane-0 reduction

`vulkan/shaders/matmul_q8_0_aligned.comp` launches 256 lanes for an output row.
The lanes compute block dots in parallel, write shared memory, and then lane 0
serially sums as many as 256 block results while the other lanes wait.

A targeted Wave64 experiment should keep parallel block-dot calculation and use
a hierarchical final reduction: subgroup totals for four 64-lane waves, then a
small final reduction. Changing floating-point reduction order can change low
bits, so exact deterministic validation and full-model token validation are
mandatory.

This is not a repeat of rejected experiment `q8-wave64` (`ea13a81`). That branch
transposed aligned payload records; it did not replace this serial reduction.

### 2. Router selection is a one-invocation GPU program

`vulkan/shaders/router_select.comp` uses `layout(local_size_x = 1) in`. One GPU
invocation evaluates all 256 experts, performs the serial top-6 insertion, and
normalizes the result. The audited router stage was about 0.471 ms.

A later experiment can distribute probability evaluation and top-k selection
over Wave64 lanes. A sub-0.1 ms router stage is a reasonable experiment target,
not a qualification promise.

### 3. F16 matvec repeats the lane-0 pattern and uses FP64

`vulkan/shaders/matmul_f16.comp` accumulates per-lane work in `double`, stores
shared `float` partials, then lane 0 serially adds all 256 partials in `double`.
It also manually converts FP16 weights through branchy math.

Do not immediately reimplement this. Rejected experiment `f16-tree` (`7825a80`)
already changed F16 reduction behavior without a production gain. First recover
its precise A/B and determine whether its loss came from reduction structure,
occupancy, exactness constraints, or the tested production shape.

### 4. Simple dispatches receive universal compute barriers

`finish_simple_dispatch()` emits an all-shader-write to all-shader-read compute
memory barrier after every simple dispatch. It is safe but orders independent
work and may inhibit overlap.

This is a resource-hazard problem, not a blanket barrier-removal task. Record
read/write buffer ranges for consecutive dispatches, retain genuine RAW/WAW
ordering, narrow barriers to the affected resources, and omit barriers only for
proven-independent operations.

### 5. Routed MoE contains CPU-shaped GPU work

`vulkan/shaders/routed_moe.comp` includes repeated shared-memory tree barriers,
manual FP16 conversion, scalar quantized-weight extraction, and lane-0 scanning
and packing after parallel loads. The backend executes quantize, gate, up,
SwiGLU, requantize, down, and reduction as distinct operations with dependencies.

Do not restart routed fusion or cooperative routed MoE first. Both prior
experiments failed qualification. Work inside the known-good architecture and
measure one reduction/access-pattern change at a time.

### 6. Generic tensor memory is over-broadly host visible and coherent

Generic Vulkan tensors request random host access plus persistent mapping. Before
submission, the backend flushes every live tensor allocation in full; after a
fence, it invalidates every live allocation in full.

The longer-term correction is explicit allocation classes such as GPU-only,
host-upload, host-readback, and shared/persistent, with coherency operations only
for resources that require them.

### Verification landmines

- `ds4_gpu_argmax_tensor()` performs CPU argmax over mapped logits. Time it before
  deciding whether it matters.
- `ds4_gpu_matmul_f32_tensor()` is a CPU fallback. Normal decode must never enter
  it unnoticed.
- A weight-cache miss synchronously stages, submits, and waits. Warm decode should
  report zero misses/uploads.

## Sequential optimization ladder

Only one implementation milestone should be active at a time. Each milestone
gets its own branch/commit, same-binary cache-hot A/B, correctness gate, and an
explicit keep/revert decision before the next begins.

### Completed: promote qualified all-layer batching

The already-qualified 287.224 ms/token, 3.482 TPS all-layer batching path is now
the Vulkan default. The environment gate was removed rather than spending a
fresh fleet cycle proving the same opt-in.

### Milestone 1: aligned-Q8 hierarchical final reduction

Scope only `matmul_q8_0_aligned.comp` plus its focused tests/instrumentation. Do
not change payload layout, activation quantization, batching, or other kernels.

Qualification:

- focused production-shape Vulkan Q8 tests
- exact deterministic artifact/hash
- complete Vulkan suite
- cache-hot single-blade kernel/stage A/B
- cache-hot 12-blade generation A/B

Keep only a repeatable win with exact output. The useful target is movement in
several Q8-heavy stages together, not merely a faster synthetic kernel.

### Milestone 2: router and F16 reductions

Parallelize the currently single-invocation router over Wave64 and revisit the
F16 final reduction with the failed `f16-tree` evidence in hand. These are the
next CPU-shaped decode kernels, not optional candidates selected by rerolling a
leaderboard.

### Milestone 3: dependency-aware barriers

Replace universal compute barriers with resource-aware RAW/WAW ordering and omit
barriers between proven-independent dispatches.

### Milestone 4: routed-MoE Wave64 cleanup

Surgically replace shared-memory tree reductions, serial quantization packing,
and scalar access patterns inside the qualified routed architecture. Do not
restart the rejected fusion/cooperative designs wholesale.

### Milestone 5: GPU-only allocation classes

Split GPU-only, host-upload, host-readback, and persistent allocations, then stop
flushing and invalidating every live tensor around every submission.

## Agent operating model

Use one GPT-5.6 Sol subagent as the milestone owner at a time. Its job is to
inspect, patch, commit, and write the exact remote validation recipe. The primary
agent controls the fleet A/B, qualification decision, and integration. A Luna
subagent is appropriate for bounded log/table reduction, but it should not race
an independent code patch against the active milestone.

The loop is intentionally:

```text
scope -> patch -> commit/push -> fleet pull/build -> focused test -> exactness
      -> cache-hot A/B -> keep or revert -> next milestone
```
