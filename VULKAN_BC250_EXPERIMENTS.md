# Vulkan BC-250 Experiment Ledger

> Current as of 2026-08-16. The production integration branch is
> `pr-557-merge`. Performance context and the forward plan are in
> [`VULKAN_BC250_KERNEL_AUDIT.md`](VULKAN_BC250_KERNEL_AUDIT.md).

## Current qualified deployment

Published baseline: `4d46c92` (`f85a909` code plus docs) on
`origin/pr-557-merge`. The `c36933b..a073023` integration remains off-main.

- previous baseline approximately 5.4-5.5 generation tokens/s;
- worker-graph candidate 5.27/5.27 generation tokens/s versus same-binary
  fallback 5.37/5.43; graph and hazard tracker are default-off;
- governor-on published `4d46c92` control: 5.65 generation tokens/s on the
  second warm prompt; this is the current promotion floor;
- approximately 3.25 ms representative Layer 4 GPU time;
- 86/86 complete GFX1013 Vulkan tests;
- same-configuration control/candidate 16-step artifact: 35,108 bytes, SHA-256
  `d10641803e804633726f63a128f4cdb266c5f55514ad59ee61e4c140cb8515aa`;
- exact fallbacks remain available for F16, Q8, routed Wave64/fusion, and
  indexed Wave64/inverse-RoPE paths;
- warm worker trace: 76 weight uses, zero uploads, zero evictions.

Cluster operating state: all 12 blades run the enabled upstream
`cyan-skillfish-governor-smu` v0.4.12 service with a conservative
1000-2000 MHz / 800-1000 mV curve. The `.42` pilot reduced idle power from
55.2 W to 43.2 W. Performance comparisons must record governor state and load
clock; pre-governor TPS figures are retained as history, not clock-matched A/Bs.

## Production-qualified work

| Commit/range | Result |
|---|---|
| `74a22cd..c519309` | Current BC-250 decode baseline: GPU-resident layer scopes/routing, command ring, pooled scratch, Q8 grouping/tiling, activation reuse, routed mid/down fusion, HC/inverse-RoPE fusion, and Wave64 routed arithmetic. |
| indexed Wave64 publication | Register-resident indexed attention plus fused inverse-RoPE. Byte-exact focused gate; `0.864 -> 0.824 ms` (~4.6%) on the production-shaped indexed path. Long-context scope only. |
| `c36933b..a073023` | Held integration: batched/token-tiled Q8 prefill; opt-in four-layer worker command chain, persistent slice descriptors/scratch, and exact resource hazards; mapped-staging flush correctness; two-phase slice tooling. Exact artifact; 86/86 suite. Not published because it did not clear the guarded decode baseline. |

The indexed path is enabled by default only for the qualified BC-250/subgroup64
predicate. Disable it with:

```text
DS4_VULKAN_ATTN_INDEXED_WAVE64=0
DS4_VULKAN_ATTN_INDEXED_WAVE64_INV_ROPE=0
```

## Rejected or closed experiments

These must not be rediscovered and promoted from architectural appeal alone.

| Experiment | Disposition |
|---|---|
| Q-B normalization/RoPE fusion | Exact, but production timeline regressed. |
| Q-B four-row Wave64/LDS reuse | Exact, but attention output regressed by about 0.10 ms. |
| Routed Q2 LDS input cache | Exact, but routed stage became slower. |
| Routed Q2 low-live-set streaming | Focused stage improved about 0.02 ms, but full-model tokens/artifact diverged. |
| Native DP4A / packed-dot paths | RADV/ACO did not lower the candidate to native integer-dot instructions on this device. Not qualified. |
| Exact i24 regroup | Generated code did not contain the intended native instructions. Rejected without TPS promotion. |
| Paired F16 projection shader | Exact; `5.43/5.41` versus `5.45/5.41` TPS was noise-sized. |
| Generic Q-cache optimization | Unreachable on the production indexed path. |
| Q8 rows4/q36 transfer | Exact, but lower occupancy/cache behavior made the production dispatch slower. |
| Q8/Q2 cosmetic unpack rearrangements | Rejected when they did not move the production stage materially. |

## Candidates, not production

| Candidate | Required proof |
|---|---|
| Direct strided grouped Q8 prefill | Extend the qualified token-tiled primitive to remove remaining group gather/scatter copies; one meaningful 4K prefill measurement. |
| Layer-scoped reusable Q8 activation producer | Exact quantized bytes and projection results, material reduction in production quantize dispatches/stage time. |
| Shared-down directly into HC expansion | Exact Q8 row reduction and HC accumulation order; at least a material stage reduction. |
| BC-250 prepacked routed weight representation | Same quantized values and exact accumulation order; production gate/up/down bandwidth improvement. |
| Whole-worker resource graph extensions | Extend the qualified four-layer hazard graph only where a timeline identifies remaining conservative drains. |

## Measurement discipline

- Verify that the production predicate actually activates.
- Require zero warm `weight_upload` events for decode measurements.
- Use same-binary fallbacks whenever possible.
- Run one exactness gate and one focused timing/TPS gate, not a matrix.
- Promote measured production-stage gains and retain exact, maintainable
  enabling architecture that composes with the next deliverable; delete
  abandoned worktrees and temporary patches immediately.
- Exactness is mandatory. A noise-sized isolated result stops extra benchmark
  cycles, but does not automatically discard non-regressing X+Y infrastructure.
