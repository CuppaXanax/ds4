# BC-250 performance engineering contract

This repository is an appliance-performance project. The current objective is
**20 sustained decode tokens/s** on the existing 12-blade, 24-CU BC-250
topology, without MTP, speculative decoding, topology changes, quantization
changes, or output drift.

These rules are mandatory for every agent and override any urge to “keep
trying” after a failed gate.

## Authority and workspace

- The user owns the coordinator and all fleet mutations. Do not stop, start,
  rebuild, deploy, reboot, or reconfigure any blade without explicit approval
  in the current turn.
- Do not create subagents unless the user explicitly requests them.
- Do not create a branch or worktree unless the user explicitly requests it.
- Keep at most one candidate branch. Never create per-hypothesis branches.
- Never commit runtime changes directly to `pr-557-merge` without explicit
  user approval and qualifying evidence from `performance/bc250/score.py`.
  User-requested engineering-system-only commits are permitted. Those paths
  are `AGENTS.md`, `.gitattributes`, `.gitignore`, `.githooks/`, and
  `performance/bc250/`.
- Never push any ref without explicit user approval in the current turn.
- Never set `DS4_USER_APPROVED_PROMOTION` or `DS4_USER_APPROVED_PUSH`; those
  variables are human authorization gates.
- Do not store experiment patches, generated logs, or benchmark output in the
  source tree. Use `performance/bc250/evidence/`, which is ignored.

## Sources of truth

- Runtime LKG and fixed benchmark configuration:
  `performance/bc250/lkg.json`
- Canonical score and promotion verdict: `performance/bc250/score.py`
- Coordinator-side capture: `performance/bc250/capture_decode.sh`
- Read-only fleet identity gate: `performance/bc250/probe_fleet_identity.sh`
- Process and current technical lane:
  `performance/bc250/ENGINEERING_SYSTEM.md`

Historical chat numbers, `-n 16` results, microbenchmarks, focused shader
timings, and unbracketed runs are diagnostic only. They are never the project
score.

## Required loop

Before changing runtime code:

1. Run `performance/bc250/check_workspace.ps1`.
2. Ensure three canonical LKG runs have been scored.
3. Write one falsifiable bottleneck hypothesis and its predicted end-to-end
   effect. If the expected effect is below 2%, treat it as a diagnostic or ask
   the user before implementing it.
4. Prove the production call site, model shape, selected shader, and fallback
   behavior before timing the candidate.
5. For allocations or repacking, write a peak-residency ledger covering raw
   weights, packed weights, KV/context, staging, CPU packing, scratch, and a
   driver reserve. An artifact-backed tensor must have exactly one steady-state
   owner.

For a GPU candidate, retain evidence for:

- generated ISA or compiler statistics;
- subgroup size and workgroup geometry;
- VGPR, SGPR, LDS, scratch/spill, and occupancy information when available;
- exact production dimensions and dispatch reachability;
- full-model exactness;
- candidate activation and zero unexpected fallback;
- three canonical sustained decode runs.

Change one causal unit per scored candidate. Do not combine unrelated kernel,
scheduler, allocator, and deployment changes in one performance verdict.

A scored runtime candidate must be a clean commit on the single user-created
candidate branch. The user supplies that branch name with
`check_workspace.ps1 -CandidateBranch NAME`; agents do not create it
themselves.

## Promotion and stopping

- `promote_eligible` means only that the automated evidence gate passed. The
  user still decides whether to promote.
- `hold` is not permission to merge. It means the result needs a user decision.
- Any exactness failure, unexpected fallback, mixed fleet identity, OOM, GPU
  reset, or canonical regression ends the candidate immediately.
- After two invalid gates, stop and report the invariant that failed. Do not
  pivot into a new architecture in the same turn.
- A user request to stop ends cluster and mutation activity immediately. Emit
  one verified handoff; never produce repeated goal-status messages.
