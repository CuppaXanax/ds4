# BC-250 decode engineering system

## Scoreboard

- Runtime restore point: `8fb6bd9` (operational reference only; its 4K
  performance is regressed and it is not a performance-qualified LKG)
- Canonical score: **2.41 sustained decode TPS / 414.94 ms per token**
- Canonical samples: `2.40`, `2.41`, `2.42` TPS
- Historical `a2fd02c` short-prompt samples: `5.58`, `5.50` TPS (diagnostic
  only; incompatible with the canonical 4K/128K denominator)
- User-observed interactive start: approximately 5.0 TPS followed by session
  degradation (open reproduction target)
- First milestone: 10 sustained TPS
- Target: 20 sustained TPS / 50 ms per token

The canonical score is the median `gen_steady_tps` from three 512-token,
greedy, non-EOS `ds4-bench` runs at a 4K frontier while allocating the real
128K context. It excludes the first decoded token. This is intentionally not
the interactive CLI's `-n 16` measurement.

## One-command workflow

First, the user captures a read-only identity audit of all 12 blades by running
`probe_fleet_identity.sh` through their fleet controller. The audit is valid
for 15 minutes and must show one clean, correctly configured worker per worker
blade, identical commits/binaries/environments, 128K context, 11 GiB weight
budget, and the reviewed shader count. The coordinator must be idle.

On the coordinator, after the user has deliberately handed it to the benchmark:

```bash
DS4_USER_APPROVED_BENCHMARK=1 \
  performance/bc250/capture_decode.sh \
  /tmp/bc250-lkg-run-1 /tmp/bc250-fleet-identity.txt
```

Repeat for runs 2 and 3, then score them from any machine with Python 3:

```bash
python3 performance/bc250/score.py \
  --baseline-run /tmp/bc250-lkg-run-1 \
  --baseline-run /tmp/bc250-lkg-run-2 \
  --baseline-run /tmp/bc250-lkg-run-3 \
  --output performance/bc250/evidence/lkg-score.json
```

Candidate scoring additionally requires three candidate runs, a control and
candidate exactness artifact, and a candidate-activation log:

```bash
python3 performance/bc250/score.py \
  --baseline-run BASELINE_1 --baseline-run BASELINE_2 --baseline-run BASELINE_3 \
  --candidate-run CANDIDATE_1 --candidate-run CANDIDATE_2 --candidate-run CANDIDATE_3 \
  --control-artifact CONTROL.json \
  --candidate-artifact CANDIDATE.json \
  --activation-evidence CANDIDATE_ACTIVATION.log \
  --output performance/bc250/evidence/current/verdict.json
```

`promote_eligible` still requires the user's decision. Hooks reject direct
runtime commits to `pr-557-merge` and reject every push without explicit human
approval.

For a candidate, the user creates one branch and names it explicitly when
checking the workspace:

```powershell
.\performance\bc250\check_workspace.ps1 -CandidateBranch perf/iq2-layout
```

The capture script scores the runtime commit named by `DS4_EXPECTED_COMMIT`.
This lets the baseline remain the pinned LKG even though the current branch has
an engineering-system-only descendant commit. A candidate sets the variable to
its own clean commit; if its build adds a shader, it also sets
`DS4_EXPECTED_SHADER_COUNT` to the reviewed count.

## Current technical lane

Treat the gap between the historical 5.0-5.58 TPS interactive results and the
2.41 TPS canonical score as a severe context-scaling regression until it is
removed. The historical `a2fd02c` gate used 17- and 12-token prompts and only
16 generated tokens. It is valid evidence of shallow-context performance, not
evidence of sustained performance after a session grows.

The first actionable long-context attribution is now known:

- Flash has 512-wide attention heads, a 128-row raw window, and compression
  ratio 4 on even layers from layer 2 onward.
- Vulkan's dense attention score array holds 1,024 rows. The runtime therefore
  switches a ratio-4 layer to indexed attention after 896 compressed rows. The
  first indexed token is approximately token 3,588.
- The indexed Wave64 shader promoted by `f85a909` requires `head_dim == 128`.
  The real Flash call passes `head_dim == 512`, so the promoted shader cannot
  activate for this model. Its focused exactness and timing tests also used
  128-wide heads.
- At the 4K frontier, the trace proves that the ratio-4 layer falls back to
  `attention_mixed_online`: 6.0346 ms for that dispatch alone. Indexer score
  and top-k add 1.7747 ms and 0.6060 ms.

The final timeline JSON must not be reported as one 18.50 ms layer. With
`DS4_VULKAN_TIMELINE_LAYER_NO_WAIT=1`, dispatch capture remains enabled after
the selected layer ends and stops only at token completion. The JSON therefore
contains two consecutive local layer bodies:

- target layer 2, ratio-4 indexed path, recording generations 366+367:
  13.3202 ms dispatch sum (11.4741 ms attention/indexer, 1.6191 ms MoE,
  0.2270 ms other);
- following layer 3, ratio-128 non-indexed path, generation 368: 5.1775 ms
  dispatch sum (3.0616 ms attention, 1.8929 ms MoE, 0.2231 ms other).

Their 18.4978 ms combined dispatch sum is useful as a consecutive even/odd
layer-pair sample, but it is neither one-layer latency nor a critical-path
measurement. Repeating that pair shape across layers 2-41 and adding layer 42
projects about 383 ms of GPU dispatch work before layers 0-1, output, runtime,
and transport. That projection is diagnostic, not a score, but it accounts for
roughly 92% of the measured 414.94 ms/token and identifies the context cliff as
the primary lane.

The 24-CU ceiling has not been proved. Do not run another `a2fd02c` versus
`8fb6bd9` context ladder: the retained trace already identifies a production
path that is both active and actionable. Do not add instrumentation unless it
is required to prove candidate activation or output correctness.

The current lane is fixed-topology, 24-CU Vulkan decode optimization. The first
candidate must make the real 512-wide, ratio-4 indexed-attention call use an
optimized implementation instead of `attention_mixed_online`, or replace that
fallback with a faster exact implementation. Change one causal mechanism, run
the existing correctness and activation gates, and then run the canonical score
gate. A passing candidate becomes the restore point; a failing candidate is
rolled back before the next dominant production dispatch is addressed.

Do not describe poor end-to-end performance as a 24-CU hardware ceiling merely
because GPU dispatches account for most token time. A ceiling claim requires
evidence that the dominant production path is limited by a measured device
resource envelope rather than shader selection, instruction count, occupancy,
dispatch structure, synchronization, or avoidable data movement. Proving a
ceiling does not itself authorize 40 CUs: changing the topology still requires
the user's explicit approval in the current turn.

Retained evidence:

- production useful traffic: roughly 62-64 GB/s;
- same-allocation stream: roughly 246.8 GB/s;
- routed IQ2 gate/up: roughly 54.9 GB/s;
- canonical routed shader: roughly 64 VGPR and 6,671 VALU instructions;
- warm uploads and evictions: zero.

Scheduler, governor, 40-CU, ROCm, prefill, and whole-model artifact work remain
out of scope unless the user changes the lane.

## Candidate record

Every candidate gets one short Markdown record in the ignored evidence
directory containing:

1. bottleneck and production call site;
2. predicted end-to-end delta;
3. changed causal unit;
4. activation proof;
5. compiler/occupancy evidence;
6. exactness artifact hashes;
7. three canonical scores;
8. verdict and cleanup state.

No new lane begins until the current candidate is promoted, rejected, or
explicitly parked by the user.
