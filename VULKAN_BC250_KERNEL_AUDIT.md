# Vulkan BC-250 Decode Audit

> Current as of 2026-08-16. This is the authoritative performance and
> qualification snapshot for the 24-CU BC-250 appliance. Historical bootstrap
> notes remain in `vulkan/RESUME.md`.

## Qualified baseline and integrated successor

The previously published baseline is `origin/pr-557-merge` at `4d46c92`
(`f85a909` code plus its qualification record). The integrated successor is
`86a0bcd`; it is exact and has passed the complete Vulkan suite, and is being
published to `pr-557-merge` as the new engineering baseline.

| Item | Qualified result |
|---|---:|
| Previous distributed generation | approximately 5.4-5.5 tokens/s |
| Worker-graph candidate | 5.27/5.27 tokens/s versus same-binary fallback 5.37/5.43; default-off |
| Representative Layer 4 GPU time | approximately 3.25 ms |
| Attention/projection group | approximately 1.77 ms/layer |
| Routed/shared MoE group | approximately 1.31 ms/layer |
| Complete GFX1013 Vulkan suite | 86/86 passing |
| Same-configuration 16-step artifact SHA-256 | `d10641803e804633726f63a128f4cdb266c5f55514ad59ee61e4c140cb8515aa` |

The SHA-256 above is the generated logprob artifact hash, not a binary hash.
The integrated path and its control produced the same 35,108-byte artifact.
Exact qualification also compares selected tokens, top-20 ordering, logits,
and logprobs. The complete GFX1013 suite remains 86/86 passing.

## What the current production path contains

The qualified `74a22cd..c519309` range includes:

- GPU-resident prefill layer scopes and decode command batching;
- a timeline-semaphore command ring with ring-safe descriptor, tensor, scratch,
  and cached-weight lifetimes;
- exact fast F16 projections and aligned Q8/BFE projections;
- reusable Q8 activation quantization for qualified paired projections;
- grouped and tiled Q8 attention-output paths;
- GPU-resident router selection;
- fused routed IQ2 gate/up plus SwiGLU to a Q8_K intermediate;
- fused Q2 down projection and selected-rank reduction;
- device-local pooled decode scratch;
- fused attention HC/inverse-RoPE paths;
- exact Wave64 routed arithmetic as the BC-250 default;
- non-perturbing layer, stage, dispatch, submission, and resource timelines.

The integrated `c36933b..86a0bcd` successor additionally includes:

- batched indexed-prefill Q8 rows;
- a 2-row by 4-token Q8 prefill kernel that reuses weights across tokens;
- an opt-in command/dependency chain for each four-layer distributed worker
  slice, with descriptor/scratch lifetimes extended through retirement;
- opt-in resource-specific RAW/WAW/WAR dependencies in place of unconditional
  post-dispatch barriers inside that slice;
- explicit flushes for mapped weight-staging allocations on non-coherent heaps;
- `ds4-slice-bench`, a target-only layers 4:7 prefill/decode loop for cheap
  single-blade iteration with output-hash and timeline checks.

The graph implementation is retained as exact enabling architecture but is
default-off: a same-binary comparison measured 5.27/5.27 tokens/s with it and
5.37/5.43 with the established path. Its short 17/12-token prompts are not a
meaningful prefill benchmark. Batched/token-tiled prefill remains independent.

Production fallbacks remain available:

```text
DS4_VULKAN_F16_MODE=exact
DS4_VULKAN_Q8_MODE=exact
DS4_VULKAN_ROUTED_WAVE64=0
DS4_VULKAN_ROUTED_MID_ONLY=0
DS4_VULKAN_ROUTED_DOWN_REDUCE=0
DS4_VULKAN_ATTN_INDEXED_WAVE64=0
DS4_VULKAN_ATTN_INDEXED_WAVE64_INV_ROPE=0
```

## Newly qualified indexed attention path

The indexed long-context path uses one Wave64 per 128-wide head, keeps online
attention accumulators in registers, and applies the 64-element inverse-RoPE
tail before writing the final head. It removes the standalone inverse-RoPE
dispatch and its global output round-trip.

Qualification on a BC-250:

- focused fallback/candidate output: byte-identical;
- canonical indexed attention plus standalone RoPE: byte-identical to fusion;
- same-command-buffer push-constant regression: passing;
- production-shaped focused time: `0.864 -> 0.824 ms` per invocation, about
  4.6% faster;
- non-BC devices retain the fallback unless explicitly enabled;
- explicit `=0` kill switches remain available.

This is a long-context indexed-attention improvement. It is not evidence of an
immediate short-position generation TPS increase.

## What the evidence rules out

### Warm paging is not the current decode limiter

A production worker trace contained 76 cached-weight uses and zero
`weight_upload` or `weight_cache_evict` events. Live workers used about 8.24 GiB
of GTT under the 11 GiB weight-cache budget. Each worker owns roughly four
layers, so the distributed routed working set fits after warm-up.

Increasing `DS4_VULKAN_WEIGHT_BUDGET_GB` to 14 or 15 GiB cannot improve a path
that already performs zero warm uploads. Whole-expert-blob residency remains a
memory-efficiency issue, but it is not proven steady-state traffic.

### Transport and CPU bookkeeping are not rate-setting

- worker-hop transport is approximately 0.1 ms;
- descriptor CPU and command recording costs are small relative to GPU work;
- measured GPU idle gaps are about 0.06 ms/layer;
- production router-selected IDs remain GPU-resident.

Fence waits overlap GPU execution and expose a serialized dependency chain,
but they do not account for the approximately 3.25 ms of timestamped GPU work.

### Resource-specific worker dependencies are exact but default-off

The opt-in four-layer worker slice tracks buffer reads and writes and emits
dependencies for actual hazards instead of a blanket barrier after every
simple dispatch. A plain-RMS metadata defect in the first integrated version
misdeclared both descriptor count and output binding; correcting it restored
the exact full-model artifact. A second full mapping audit found no remaining
dispatch ABI or RAW/WAW/WAR mismatch. The graph remains disabled because its
same-binary decode result was about 3% slower. Enable repair experiments with
both `DS4_VULKAN_WORKER_SLICE_BATCH=1` and
`DS4_VULKAN_WORKER_RESOURCE_HAZARDS=1`.

## Current diagnosis

The BC-250 same-allocation stream test reached approximately 246.8 GB/s, while
production decode realizes only about 62-64 GB/s of useful weight traffic.
Warm paging, transport, and empty GPU gaps are too small to explain the gap.

The remaining loss is inside the useful GPU graph:

- scalar quantized unpack and address work;
- reductions that leave lanes idle;
- activation vectors reread across output rows;
- quantize/project/transform intermediates written and immediately reread;
- generic GEMV decomposition instead of BC-250 Wave64 consumption layouts;
- conservative producer/consumer boundaries that prevent coordinated fusion.

Prefill now has a token-tiled Q8 path for the common 4096-input shape, but
grouped Q8 attention output still performs token/group gather and scatter
copies because those batch kernels require contiguous token rows. A direct
strided grouped-batch kernel is still needed; the 64 `group_copy` events
observed in a four-layer mixed trace were prefill, not decode.

## Performance target and required scale

The appliance target remains:

```text
attention/projections <= 0.60 ms/layer
routed/shared MoE    <= 0.40 ms/layer
total layer          approximately 1.0-1.2 ms
single-session       >= 10 TPS minimum, 20 TPS north star
```

Moving from about 3.25 ms to 1.1 ms requires removing roughly two-thirds of
current GPU layer time. No launch-only cleanup or isolated 20-microsecond
kernel win can close that gap.

Planning ranges, not promises:

| Rewrite family | Current group | Plausible layer saving | Why it matters |
|---|---:|---:|---|
| Coordinated attention projection/activation reuse | ~1.77 ms | 0.4-0.7 ms | Multiple projections reread and requantize related activations. |
| Routed/shared MoE dataflow and packed access | ~1.31 ms | 0.4-0.65 ms | Gate/up/down still spend most time on packed arithmetic and weight access. |
| Shared-down directly into HC expansion | included above | 0.08-0.20 ms | Removes one global intermediate and dependent dispatch. |
| Strided grouped/token-tiled Q8 prefill | prefill-only | potentially large prefill win | Removes 16 copies/layer and eight independent group projections. |

## Next implementation order

1. Build a layer-scoped Q8 activation producer consumed by multiple decode
   projections without re-quantization or premature retirement.
2. Design attention projection kernels around several output rows per
   workgroup, activation reuse, and prepacked Wave64 weight consumption.
3. Fuse shared-down directly into HC expansion while preserving the existing
   Q8 reduction and Sinkhorn accumulation order.
4. Rework routed IQ2/Q2 loads and dots around a BC-250 resident/prepacked
   representation; retain the exact rank and FP32 reduction order.
5. Extend the qualified token-tiled Q8 primitive into a strided grouped-batch
   prefill path so gather/scatter and per-group projection islands disappear.

Allocation and barrier refinements remain worthwhile only when a trace ties
them to material wall or GPU time. They are not ahead of these dataflow rewrites.

## Promotion policy and ROI gate

Every experiment has exactly one status:

- **production-qualified**: exact artifact, deployment safety, complete tests,
  and either a measured performance win or retained enabling architecture with
  no demonstrated material regression;
- **candidate**: static/compile or focused evidence only;
- **rejected**: exactness, safety, timeline, or performance gate failed.

The normal gate is:

```text
static compile/SPIR-V validation
-> one focused exactness test
-> one 16-step full-model artifact
-> one same-binary focused timing/TPS smoke test
-> promote or delete
```

Do not run benchmark matrices. Before implementation, a decode candidate should
have a credible path to at least 0.25-0.35 ms/layer or roughly 0.5 TPS. Smaller
ideas should be folded into a coordinated rewrite or deferred. Exact,
maintainable enabling work is retained when it composes with the planned graph;
a noise-sized isolated result limits further benchmarking but is not by itself
a reason to discard sound architecture. A synthetic kernel speedup is
insufficient unless it moves its production stage.
