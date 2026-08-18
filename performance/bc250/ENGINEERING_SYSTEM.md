# BC-250 decode engineering system

## Scoreboard

- Runtime restore point: `474aa3a` (exact, production-activated indexed-layer
  v2 checkpoint)
- Previous canonical score: **2.41 sustained decode TPS / 414.94 ms per token**
- Canonical samples: `2.40`, `2.41`, `2.42` TPS
- Pre-v2 checkpoint measurement on `86840a1`: **2.94 steady TPS** over 63
  steady tokens at 4K with 128K allocated context. It remains diagnostic; no
  canonical 3x512 score has yet been recorded for `474aa3a`.
- `474aa3a` is retained because it is exact, production-active, materially
  faster in its causal unit, and explicitly user-approved. This is not a claim
  that end-to-end TPS moved or that the 10/20 TPS milestones were reached.
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
budget, and the reviewed role-specific shader profile. The coordinator must be
idle.

The current BC-250 LKG is intentionally role-heterogeneous. Five legacy SPIR-V
artifacts on the coordinator differ byte-for-byte from the worker copies even
though their GLSL sources match. The coordinator and worker manifests in
`lkg.json` are therefore frozen and validated independently. Never normalize,
recompile, or overwrite either profile as a side effect of an unrelated shader
change.

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
runtime commits to `pr-557-merge` without that approval and reject every push
without separate explicit human approval.

When the user explicitly directs that a proven causal unit be checkpointed and
also declines the three-run repetition gate, record the honest verdict
`user_approved_checkpoint`. The record must bind the staged runtime diff,
bit-identical end-to-end artifacts, production activation, the retained
measurement, and the explicit reason the canonical score is deferred. This
advances the recoverable LKG without mislabeling the checkpoint as a 10/20 TPS
qualification.

Candidates do not get branches or worktrees. Build each candidate as a clean,
deterministic commit in a disposable checkout rooted at the exact runtime LKG.
Record its base commit, patch SHA-256, commit/tree, binaries, and shader
manifest in the ignored evidence directory. Set `DS4_EXPECTED_COMMIT` to that
disposable commit when capturing its three canonical runs; if its build adds a
shader, also set `DS4_EXPECTED_COORDINATOR_SHADER_COUNT`,
`DS4_EXPECTED_COORDINATOR_SHADER_MANIFEST`,
`DS4_EXPECTED_WORKER_SHADER_COUNT`, and
`DS4_EXPECTED_WORKER_SHADER_MANIFEST` to the reviewed values. After scoring,
delete the disposable checkout. A promoted patch is applied once to
`pr-557-merge`; no candidate ref survives.

## Current technical lane

Treat the gap between the historical 5.0-5.58 TPS interactive results and the
2.41 TPS canonical score as a severe context-scaling regression until it is
removed. The historical `a2fd02c` gate used 17- and 12-token prompts and only
16 generated tokens. It is valid evidence of shallow-context performance, not
evidence of sustained performance after a session grows.

The indexed-layer v2 candidate is closed and promoted:

- Flash has 512-wide attention heads, a 128-row raw window, and compression
  ratio 4 on even layers from layer 2 onward.
- Vulkan's dense attention score array holds 1,024 rows. The runtime therefore
  switches a ratio-4 layer to indexed attention after 896 compressed rows. The
  first indexed token is approximately token 3,588.
- `86840a1` supplied the exact production `head_dim == 512`, ratio-4 Wave64
  attention path. `474aa3a` adds the exact Wave64 selector plus the retained
  grouped-output-A, output-B, and F16 projection kernels and wires the selector
  into the real single-stream decode call site.
- Focused production-shape measurements were selector `3.086 -> 0.722 ms`,
  grouped output A `0.565 -> 0.290 ms`, output B `0.349 -> 0.305 ms`, and F16
  projection `0.685 -> 0.614 ms`. The final selector repetition was
  `3.058 -> 0.706 ms` (4.33x).
- The matched complete indexed-layer capture was `7.358880 -> 5.453640 ms`,
  saving `1.905240 ms` per indexed layer. A normal post-deployment repetition
  measured `5.451120 ms`: compressor/indexer `1.182240 ms`, selector score
  `0.355840 ms`, selector top-k `0.153600 ms`, indexed attention `0.703080 ms`,
  grouped output A `0.219720 ms`, and output B `0.296560 ms`.
- The first-token artifact remained bit-identical with SHA-256
  `a31e2d480caf26ff51b05d6aae8a4b2ed05db04478b006f223a8c34590e9f815`.
  Focused exactness, causal, 32K-row, disabled-fallback, and resource gates all
  passed; the new shaders use Wave64 and reported zero scratch/spills.
- A fresh 12/12 identity audit binds every blade to commit `474aa3a`, binary
  `608f831d...0c91`, 128K context, 11 GiB weight budget, 24 CUs, 70 shaders,
  clean source, one worker per worker blade, and the frozen role-specific
  shader manifests recorded in `lkg.json`.

Twenty indexed layers times the captured `1.905240 ms` saving projects about
`38.1 ms/token` recovered when all twenty paths are active. That is a causal
projection, not an end-to-end score. No canonical three-run 4K/512 result has
been recorded for `474aa3a`, so do not claim a TPS increase from this checkpoint.

The 24-CU ceiling has not been proved. Do not run another `a2fd02c` versus
`8fb6bd9` context ladder: the retained trace already identifies a production
path that is both active and actionable. Do not add instrumentation unless it
is required to prove candidate activation or output correctness.

The current lane is fixed-topology, 24-CU Vulkan decode optimization. The
complete indexed-layer causal unit is now promoted. The next highest-leverage
candidate is the routed MoE path: it is approximately `1.145 ms` in the
promoted indexed layer and applies across the model, while the remaining
compressor/indexer cost applies only to the twenty ratio-4 layers. Optimize one
integrated routed-MoE causal unit, preserve exact routing/output, prove the real
single-stream dispatch is active, and reuse the existing correctness/resource
gates before any canonical score.

Exact, production-active kernel wins do not get discarded merely because an
end-to-end TPS movement is unresolved or below run noise. Retain them unless
they cause output drift, instability/OOM/reset, unintended fallback, mixed
fleet identity, a material resource regression in another production shape,
or a matched canonical regression of at least 1%. Document deferred scoring
honestly and keep the last exact restore point recoverable.

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
7. three canonical scores, or the explicit user-approved reason they were
   deferred for an exact checkpoint;
8. verdict and cleanup state.

No new lane begins until the current candidate is promoted, rejected, or
explicitly parked by the user.
