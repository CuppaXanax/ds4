# Vulkan BC-250 Experiment Ledger

> Current as of 2026-08-16. The production integration branch is
> `pr-557-merge`. Performance context and the forward plan are in
> [`VULKAN_BC250_KERNEL_AUDIT.md`](VULKAN_BC250_KERNEL_AUDIT.md).

## Current qualified deployment

Publication base: `c519309` (`origin/pr-557-merge`).

- approximately 5.4-5.5 generation tokens/s;
- approximately 3.25 ms representative Layer 4 GPU time;
- 83/83 complete GFX1013 Vulkan tests;
- exact 16-step artifact SHA-256:
  `5e31e01d847a5f1e409c4169e249ae187efe1c3827fd7009711c3679dcfe8023`;
- exact fallbacks remain available for F16, Q8, routed Wave64/fusion, and
  indexed Wave64/inverse-RoPE paths;
- warm worker trace: 76 weight uses, zero uploads, zero evictions.

## Production-qualified work

| Commit/range | Result |
|---|---|
| `74a22cd..c519309` | Current BC-250 decode baseline: GPU-resident layer scopes/routing, command ring, pooled scratch, Q8 grouping/tiling, activation reuse, routed mid/down fusion, HC/inverse-RoPE fusion, and Wave64 routed arithmetic. |
| indexed Wave64 publication | Register-resident indexed attention plus fused inverse-RoPE. Byte-exact focused gate; `0.864 -> 0.824 ms` (~4.6%) on the production-shaped indexed path. Long-context scope only. |

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
| Resource-aware barrier trackers | First version ordered barriers after consumers; corrected version still changed the exact artifact. |
| Generic Q-cache optimization | Unreachable on the production indexed path. |
| One-submit/cross-layer command buffers | Unsafe or regressed beyond the qualified RADV command-count bound. |
| Q8 rows4/q36 transfer | Exact, but lower occupancy/cache behavior made the production dispatch slower. |
| Q8/Q2 cosmetic unpack rearrangements | Rejected when they did not move the production stage materially. |

## Candidates, not production

| Candidate | Required proof |
|---|---|
| Strided grouped/token-tiled Q8 prefill | Compile/SPIR-V, byte-exact batch output, then one meaningful prefill A/B. Existing prototype evidence is static only. |
| Layer-scoped reusable Q8 activation producer | Exact quantized bytes and projection results, material reduction in production quantize dispatches/stage time. |
| Shared-down directly into HC expansion | Exact Q8 row reduction and HC accumulation order; at least a material stage reduction. |
| BC-250 prepacked routed weight representation | Same quantized values and exact accumulation order; production gate/up/down bandwidth improvement. |
| Complete resource access graph | Exact full-model artifact plus fewer production barriers. Descriptor overlap alone is insufficient. |

## Measurement discipline

- Verify that the production predicate actually activates.
- Require zero warm `weight_upload` events for decode measurements.
- Use same-binary fallbacks whenever possible.
- Run one exactness gate and one focused timing/TPS gate, not a matrix.
- Promote only production-stage gains; delete rejected worktrees and temporary
  patches immediately.
- Treat exactness as necessary but not sufficient: an exact, noise-sized, or
  slower candidate is rejected.
