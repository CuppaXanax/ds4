# BC-250 Vulkan: Architecture for 50 ms/token

> Status: architecture plan, not a performance claim.  This document is
> intentionally scoped to the existing 24-CU BC-250 distributed topology.
> It does not assume the 40-CU firmware unlock, tensor parallelism, MTP, or
> speculative decoding.

Date: 2026-08-17  
Reference baseline: `origin/pr-557-merge` at `8fb6bd9`  
Integrated streaming candidate: `4262c43` (not promoted and not a TPS result)

## Target and authoritative budget

The target is sustained end-to-end decode latency of at most 50 ms/token
(20 sustained tokens/s) on the existing 24-CU topology, with model outputs,
quantization semantics, routing, coordinator behavior, and layer placement
unchanged.

The strongest retained whole-token evidence is the four-layer worker slice:

| Quantity | Four layers | Per layer | 43-layer projection |
|---|---:|---:|---:|
| Worker wall time | ~16.5 ms | ~4.125 ms | ~177.4 ms |
| Timestamped GPU work | ~12.9 ms | ~3.225 ms | ~138.7 ms |
| Non-GPU residual | ~3.6 ms | ~0.900 ms | ~38.7 ms |

The 177.4 ms projection agrees with the archived 5.4-5.65 TPS range.  It is
not a kernel microbenchmark extrapolation.  A 50 ms token permits only
`50 / 43 = 1.16 ms` per layer, so the architecture must remove approximately
2.96 ms from each current 4.125 ms layer.

The retained GPU group accounting is:

| GPU group | Per layer | 43-layer cost |
|---|---:|---:|
| Attention and projections | ~1.77 ms | ~76.1 ms |
| Routed/shared MoE | ~1.31 ms | ~56.3 ms |
| Other GPU work | ~0.17 ms | ~7.3 ms |
| Total | ~3.25 ms | ~139.8 ms |

The known non-GPU terms are only about 0.214 ms/layer: descriptor work,
dispatch recording, flush/invalidate, submission CPU, and measured GPU idle.
The remaining approximately 0.686 ms/layer is not independently assigned,
which is about 29.5 ms/token.  Reported fence time overlaps useful GPU work;
it must not be added to the budget a second time.

The useful production weight stream is approximately 62-64 GB/s, compared
with approximately 246.8 GB/s for the same-allocation stream.  The retained
distinct-weight ledger is approximately 8.9 GiB/token:

```text
8.9 GiB / 64 GB/s    ~= 149 ms of traffic-equivalent time
8.9 GiB / 180 GB/s   ~=  53 ms
8.9 GiB / 200 GB/s   ~=  48 ms
8.9 GiB / 220 GB/s   ~=  43 ms
8.9 GiB / 246.8 GB/s ~=  39 ms
```

The latter two figures are bounds, not achieved results.  Two hundred GB/s is
an intermediate gate, not a sufficient final target: approximately 48 ms of
weight traffic leaves essentially no budget for quantized arithmetic,
attention state, graph overhead, transport, or the output head.  The credible
production target is at least 220 GB/s useful traffic or a materially smaller
execution byte count.

The hard 50 ms allocation is:

| Category | Token budget | Per layer |
|---|---:|---:|
| Weight stream plus quantized arithmetic | <=43.5 ms | <=1.01 ms |
| Attention/projection non-weight work | <=2.0 ms | <=0.047 ms |
| Routed/shared-MoE non-weight work | <=1.5 ms | <=0.035 ms |
| KV/output/other GPU work | <=0.5 ms | <=0.012 ms |
| CPU/Vulkan graph/fence overhead | <=2.0 ms | <=0.047 ms |
| Transport/coordinator handoff | <=0.5 ms | <=0.012 ms |
| Headroom | 0.0-0.5 ms | -- |

Transport is not a credible primary explanation: the retained worker-hop
measurement is approximately 0.08-0.13 ms/hop.  Warm decode also recorded 76
cached-weight uses with zero uploads and zero evictions.  Neither transport
nor warm paging can supply the missing 130 ms/token.

## Ranked architecture plan

These are deliberately large, coordinated changes.  Individual shader
polishes below the 0.465 ms/layer threshold are parked unless they are part of
one of these designs.

### 1. Packed execution artifacts for the decode weight stream

Make load-time execution layout, rather than GGUF layout, the Vulkan contract.
Build one immutable, blade-local arena with aligned payload and metadata planes
for Q8, IQ2, and Q2.  The artifact must be a lossless copy/reordering of the
source quantized values; it must not change dequantization or accumulation
semantics.

The required hardware shape is consecutive packed words for consecutive lanes,
with metadata separated from payloads and no per-lane 66/84-byte record stride.
The routed IQ2 gate/up representation is first because the retained production
shape reaches only approximately 55 GB/s in the isolated C ladder, while Q8
and Q-B reach approximately 147 and 226 GB/s respectively in the same family
of tests.  Those C results are diagnostic, not whole-token results.

The architecture-level gate is not “one IQ2 shader got faster.”  It is a
production useful-stream target of at least 220 GB/s across the decode-critical
weight families.  Reaching that region can remove well over 20 ms/token and is
the only single-topology change that can plausibly remove most of the current
139 ms GPU budget.

### 2. Persistent worker-slice execution graph

Represent a worker's layer slice as a bounded, persistent GPU command graph:

- persistent descriptors, scratch, and tensor lifetimes;
- indirect or pre-recorded dispatches where shapes are stable;
- resource-range dependencies instead of a blanket barrier after every helper;
- timeline-semaphore ordering between bounded graph chunks;
- one host wait at the worker-output boundary, not host participation at each
  internal dependency.

The graph must not be implemented as an unbounded mega-command-buffer.  Earlier
one-submit/cross-layer experiments were unsafe or regressed under the RADV
command-count bound.  The intended design is a small number of reusable graph
chunks with explicit resource access metadata.

The retained budget gives this change a concrete ROI gate:

```text
20 ms/token / 43 layers = 0.465 ms/layer required
unassigned residual      ~= 0.686 ms/layer available
```

If the missing residual is orchestration rather than hidden GPU work, this
change can remove approximately 20-30 ms/token.  A continuous timeline must
confirm that result; no claim is made until it does.

### 3. Weight-pass fusion with activation reuse

After the packed artifacts exist, change the layer dataflow so one activation
tile is consumed by all compatible projections before it is retired or
requantized.  The objective is to eliminate repeated activation quantization,
temporary global intermediates, and immediate write/read pairs while keeping
enough register headroom for occupancy.

The first coordinated passes are:

1. Q/QB/KV and attention-output projection families;
2. shared gate/up/SwiGLU/down;
3. routed IQ2 gate/up through Q8 intermediate and routed Q2 down/reduction.

The existing group ranges imply the required scale: attention/projection reuse
has a plausible 0.4-0.7 ms/layer range and routed/shared dataflow has a
0.4-0.65 ms/layer range.  These are planning ranges, not measurements.  The
combined pass must demonstrate at least 0.465 ms/layer of production saving to
qualify as a 20 ms/token architecture lever.  Do not add the ranges together
as an achieved forecast; their overlap must be measured on the same binary.

## Achieved substrate evidence

The integrated candidate through `4262c43` now losslessly repacks and consumes
all three hot quantized weight families:

- dense Q8 through `matmul_q8_0_exec`;
- routed IQ2 gate/up through the fused-mid execution variants;
- routed Q2 down/reduce through `routed_moe_down_reduce_q2_exec`.

Each artifact uses exact-size immutable device-local buffers with separately
addressable metadata and payload planes.  The host-side packed copy is released
after upload, the artifact replaces overlapping raw/aligned cache ranges, and
every descriptor plane is checked against `maxStorageBufferRange`.  The Q8,
IQ2, and Q2 registered GPU gates all passed on blade `.53`.  Each gate requires
the artifact path, invalidates or removes the raw source as appropriate, and
compares against the established reference path; the IQ2 and Q2 gates compare
raw and artifact results bit-for-bit.

A strict two-layer production-shaped startup also succeeded on `.53` with 66
shaders.  It prepared the worker's 4.21 GiB model span in 6.556 seconds, reported
the expected 7.19 GiB total planned footprint at 128K context, and did not
recreate the rejected multi-gigabyte host mirror or transient arena resize.
The blade was then restored to the published `8fb6bd9` LKG binary and its
verified hash.  The coordinator was not touched.

The same branch also closes two statically proven streaming defects discovered
after that hardware gate.  Execution-artifact startup now reports Q8/IQ2/Q2
hit, fallback, unsupported, and failure counts and fails closed when an
artifact family is explicitly required.  Command-ring dispatches are no longer
counted twice, changing a roughly 158-dispatch worker slice from about five
premature submissions toward the configured three plus the final output fence.
For common 4096-input Q8 projections, a validated 128-lane variant replaces the
256-lane workgroup in which half the invocations were permanently idle; larger
8192-input shapes retain the 256-lane variant.

These results prove the representation, GPU consumers, and startup memory
shape.  They are not a production layer timing, a full-model exactness result,
or a TPS claim.  `4262c43` remains a candidate branch and has not replaced the
`8fb6bd9` deployment baseline.

### Composed streaming infrastructure review

The execution-artifact substrate was composed with the persistent descriptor
cache and the resource hazard tracker on
`codex/execution-artifact-substrate`.  The combined Vulkan build passes with
72 compiled shader variants.  Descriptor sets remain live through their
command-ring retirement epoch, are recycled only after completion, and are
never rewritten while a submitted command buffer can reference them.  The
recording-generation key prevents reuse across command-buffer generations and
unsubmitted validation allocations are returned safely.

The hazard tracker is now the default inside a bounded worker-slice batch
after the complete 89-test GFX1013 suite passed with it enabled. Setting
`DS4_VULKAN_HAZARD_TRACKER=0` is the explicit kill switch; ordinary layer and
unbatched execution retain the established blanket-barrier path unless
explicitly opted in. Unknown shader
interfaces and allocation aliases are conservative; routed dispatches retain
their existing explicit input/output barriers.  This branch is therefore
passed the complete BC-250 Vulkan harness on `.53`: 89 tests passed and zero
failed with hazard tracking enabled.  The earlier router-select failure was a
test-lifetime defect: the Q2 artifact test freed its synthetic model without
retiring the backend model identity.  The test now retires that identity before
freeing it, and the sequenced full suite is clean.  This is a correctness gate,
not a production performance claim.

The branch also contains an opt-in production-shape fusion for routed Q2 down,
shared Q8 down, routed/shared addition, and HC post-processing. Static review
closed a fail-open admission bug and added explicit residual/split hazards.
A later call-graph audit found a more fundamental reachability defect: the
host, backend, and shader assumed a 4096-wide shared intermediate, while the
Flash model's shared intermediate is 2048 wide. The corrected path now admits
the real shared 2048->4096 matrix, consumes 64 Q8_0 source blocks/eight packed
artifact tiles, and retains the routed 4096->2048->4096 shape. GLSL, SPIR-V,
the C graph, and the C++ backend validate. It remains disabled until a
full-shape fused-vs-unfused HC comparison and full-model exactness gate pass.

### Persistent worker-slice scratch leases

The bounded worker slice now assigns hot decode temporaries stable backing
storage by `(layer, allocation sequence, memory class)`. Each layer therefore
keeps distinct Q8/routed scratch while command-ring segments are in flight,
avoiding unsafe same-size aliasing. Backing storage persists across tokens and
callers receive non-owning views, so the existing graph no longer returns and
reacquires those buffers on every token. Shape or call-order mismatches fail
safe to the established scratch allocator. This is a prerequisite for fully
persistent descriptors and re-record-free worker graphs; it is not claimed as
a standalone TPS result.

Exact descriptor bindings now persist with those stable buffers across worker
tokens. The cache is limited to 8192 sets per command context, applies only to
bounded worker slices, and can be disabled with
`DS4_VULKAN_PERSIST_DESCRIPTORS=0`. Every tensor, raw-weight, aligned-weight,
and execution-artifact destruction/eviction path invalidates descriptor keys
that reference the retiring `VkBuffer` before it is destroyed. This removes
per-token descriptor writes and establishes the immutable binding layer needed
by graph replay; measured descriptor overhead alone is too small for this to
be treated as the target latency win.

## Remaining qualification gates

For each materially different architecture candidate:

1. Compile and validate all affected SPIR-V and the Vulkan backend.
2. Prove the production predicate selects the intended artifact/graph.
3. Run the exact focused gate, recording every repeated timing rather than the
   best sample.
4. Run the complete GFX1013 Vulkan suite.
5. Run the established full-model exact artifact gate.
6. Run the sustained full-sequence end-to-end benchmark; no `-n`-limited run
   qualifies as TPS evidence.
7. Promote only reproducible end-to-end gains and delete rejected experiments.

The integrated artifact still requires full-model exactness and
production-shaped GPU timing.  The full packed stream and weight-pass fusion
require a no-upload warm decode trace and a continuous worker timeline.  Until
those gates pass, the only defensible statement is that the architecture is
implemented and GPU-correct on its focused gates, not that 50 ms or 20 TPS has
been achieved.

The runtime report now includes a Q8 coverage ledger keyed by input/output
shape. A production gate must show `fallbacks=0` for every reported hot shape;
the aggregate artifact flag alone is insufficient evidence because a single
large dense projection can dominate the remaining stream.

## Evidence limits

The retained artifacts do not contain a complete 43-layer continuous timeline,
final output-head timing, authoritative production dispatch/barrier counts, or
context-correlated KV migration measurements.  Any future budget that assigns
those milliseconds without collecting those events is an inference, not a
measurement.

## 2026-08-17 integrated 128K hardware gate

The first full-fleet gate exposed two concrete integration defects and one
architectural blocker:

- A fixed nine-element descriptor-layout binding array overflowed when the
  twelve-binding mixed routed/shared HC shader was registered. This crashed
  RADV in `radv_CreateDescriptorSetLayout` on every blade before model load.
  Commit `e71e76b` replaces it with an exact-size binding vector; startup was
  then verified on GFX1013.
- Strict Q8 decode-artifact enforcement incorrectly rejected the distinct
  token-batched prefill path before decode began. Commit `15066ff` scopes the
  fail-closed runtime requirement to `n_tok == 1` while retaining prefill
  fallback accounting.
- With those defects repaired, the 128K coordinator was OOM-killed during the
  first prompt. The kernel recorded a global OOM kill of `ds4`; startup had
  packed 45 artifacts while the process also planned 8.85 GiB resident model
  plus 2.98 GiB context. Therefore the current per-weight artifact buffers do
  not yet satisfy the architecture requirement that packed execution ranges
  replace, rather than coexist with, raw resident mappings. No TPS or
  correctness claim is valid from this run.

The fleet was restored to published LKG `8fb6bd9` (binary SHA-256
`f0d32af04aade31505ecb698514d703a2569c772fe1c98258e29d525455bb028`),
with workers running and the coordinator intentionally stopped. The next
candidate must use a single replacement weight arena or otherwise unmap raw
resident spans before packed artifacts are admitted at 128K.
