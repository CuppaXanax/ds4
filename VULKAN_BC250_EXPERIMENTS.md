# Vulkan BC-250 Experiment Disposition

This ledger closes the `luna/*` optimization branches created during the 2026-08-11/12 BC-250 Vulkan work. The production integration branch is `pr-557-merge`.

The follow-up kernel/backend audit and sequential optimization plan are in
[`VULKAN_BC250_KERNEL_AUDIT.md`](VULKAN_BC250_KERNEL_AUDIT.md).

The deleted branch refs are recoverable from the local bundle:

- Path: `D:\cuppaxanax\github\bc-250-dbg\ds4-luna-branches-20260812.bundle`
- SHA-256: `59dc48fe7f5f93248f19442f76a21c1e4825b5d83fe0b23f65a12bacd70d667e`
- Contents: all 33 local `luna/*` refs before cleanup

## Retained Production Work

The following work is contained in `pr-557-merge` and remains in production history.

| Branch | Tip | Disposition |
|---|---|---|
| `all-layer-command-batching` | `51477a4` | Integrated. Batching for every normal Vulkan decode layer; 287.224 ms/token and 3.482 TPS on 12 BC-250 nodes. Promoted to the default after qualification. |
| `attention-output-batching` | `4bb509b` | Integrated. Default single-token attention-output batching with deferred resource retirement. |
| `default-vulkan-paths` | `34b5297` | Integrated. Promoted qualified Q8 prequant/aligned artifacts and Q2 direct-word decode. |
| `full-layer-timeline` | `6394d70` | Integrated as the basis of non-perturbing production layer timing. |
| `layer4-command-batching` | `aae1108` | Integrated. Proved two-span batching around selected-expert readback before widening to all layers. |
| `q2-word` | `e4289f7` | Integrated. Q2_K direct-word decode and production-shape coverage. |
| `q8-aligned-repair` | `96910c4` | Integrated through the qualified aligned artifact implementation. |
| `q8-layout-analysis` | `b8f9e24` | Integrated through reusable Q8 prequantization work. |
| `q8-prequant` | `17edb0b` | Integrated. Hardened the production Q8 prequant path. |
| `layer4-stage-leaderboard` | `da77d88` | Retained by cherry-pick as `40078a4`. Opt-in stage GPU totals on the all-layer-batched path. |

The following names pointed at already-integrated commits and carried no unique work: `attn-direct`, `mtp-throughput`, `q2-packed`, and `submit-graph`.

## Closed Experiments

These branches are intentionally not merged. Their results remain documented here and in the archived bundle.

| Branch | Tip | Reason closed |
|---|---|---|
| `attn-output-batch` | `7c1e9bd` | Rejected. The attempted batch kernel was about 3.4x slower; the later command-stream batching design replaced it. |
| `bc250-profile` | `9135159` | Diagnostic-only runtime counters; superseded by the non-perturbing timeline instrumentation. |
| `command-batching` | `21f79ed` | Superseded prototype. It added descriptor retirement but retained synchronous helper waits and did not solve temporary lifetime. |
| `f16-tree` | `7825a80` | Rejected after exactness/performance qualification; changed reduction behavior without a production gain. |
| `gpu-routing` | `9c1f485` | Rejected. Valid output but no repeatable gain because the controlling router synchronization remained. |
| `hc-split` | `2e1e450` | Unqualified isolated kernel experiment; not needed for the command-batching production result. |
| `occupancy-combined` | `ecd21cc` | Closed unqualified. Combined RMS/HC experiments never completed an exact-logit production A/B. |
| `profile-dot` | `4a6debe` | Diagnostic-only integer-dot capability instrumentation; no production behavior retained. |
| `q8-adaptive-wg` | `0741ebe` | Rejected during Q8 workgroup experiments; no material qualified stage improvement. |
| `q8-aligned` | `7b43404` | Superseded aligned-artifact prototype. The repaired implementation was integrated separately. |
| `q8-aligned-gpu-test` | `90f7007` | Superseded test branch used while qualifying aligned artifacts. Relevant coverage was retained in production tests. |
| `q8-packed` | `0c0aee6` | Rejected packed-Q8 experiment; no production improvement. |
| `q8-reuse` | `5828164` | Rejected. Reuse path did not produce a qualified gain. |
| `q8-rows4` | `3f95008` | Rejected rows-per-dispatch experiment; no qualified improvement. |
| `q8-wave64` | `ea13a81` | Rejected. Exact output, but averaged 632.361 ms/token, about 2.3% slower than its LKG. |
| `rms-subgroup` | `e37c8bc` | Closed unqualified subgroup RMS experiment; not part of the validated production path. |
| `routed-artifacts` | `324e9ff` | Rejected. Adaptive artifact layout preserved output but regressed to about 115.7 seconds/token due to destroyed record locality. |
| `routed-coop` | `87b5f07` | Rejected cooperative routed-MoE experiment; no qualified production gain. |
| `routed-fusion` | `fa6eefd` | Rejected fused routed path after correctness/performance qualification. |

## Final Qualified State

At cleanup, `pr-557-merge` contains:

- automatic Q8 activation prequantization and aligned Q8 artifacts
- direct-word Q2_K routed down decode
- default attention-output command batching
- default all-layer command batching
- non-perturbing Vulkan timeline and stage leaderboard instrumentation

Validated results:

- complete Vulkan suite: 79/79 passing
- deterministic artifact: 2,178 bytes
- SHA-256: `3fbf53f82bb25e37502ff64e11d660104d32618880fe07e06f021142b969b9e4`
- all-layer-batched distributed decode: approximately 287.224 ms/token, 3.482 TPS
