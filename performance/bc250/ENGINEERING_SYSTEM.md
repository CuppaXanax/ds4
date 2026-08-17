# BC-250 decode engineering system

## Scoreboard

- Runtime LKG: `8fb6bd9`
- Canonical score: not yet recorded
- Historical short-run range: approximately 5.1-5.5 TPS
- First milestone: 10 sustained TPS
- Target: 20 sustained TPS / 50 ms per token

The first action is to record three LKG runs with the canonical harness. Until
that exists, no new optimization has a valid denominator.

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

The next decode lane is routed IQ2 gate/up, not fleet scheduling or a retain-all
artifact architecture.

Retained evidence:

- production useful traffic: roughly 62-64 GB/s;
- same-allocation stream: roughly 246.8 GB/s;
- routed IQ2 gate/up: roughly 54.9 GB/s;
- canonical routed shader: roughly 64 VGPR and 6,671 VALU instructions;
- steady decode GPU idle: roughly 0.06-0.10 ms/layer;
- warm uploads and evictions: zero.

The next candidate must therefore change routed IQ2's instruction/data layout
at its real production shape. Before implementation it must prove generated
ISA, Wave64 behavior, occupancy/live state, exact accumulation semantics, and
a bounded memory representation. Scheduler, governor, 40-CU, ROCm, prefill,
and whole-model artifact work are out of scope unless the user changes the
lane.

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
