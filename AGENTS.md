# BC-250 performance engineering contract

This repository is an appliance-performance project. The current objective is
**20 sustained decode tokens/s** on the existing 12-blade, 24-CU BC-250
topology, without MTP, speculative decoding, topology changes, quantization
changes, or output drift.

The 24-CU topology is a hard experimental invariant, not merely the current
inventory. Do not enable, test, stage, or deploy the BC-250 40-CU unlock until
the user explicitly changes this invariant in the current turn. Kernel and ROCm
work must prove the 24-CU ceiling first.

These rules are mandatory for every agent and override any urge to “keep
trying” after a failed gate.

## Authority and workspace

- The user owns the coordinator and all fleet mutations. Do not stop, start,
  rebuild, deploy, reboot, or reconfigure any blade without explicit approval
  in the current turn.
- Do not create subagents unless the user explicitly requests them.
- Do not create candidate branches or worktrees. Runtime candidates are clean,
  disposable checkouts under `/tmp`, identified by the LKG base commit, exact
  patch SHA-256, candidate commit/tree, binaries, and role-specific shader
  manifests. Delete
  the checkout after scoring or rejection; retain the ignored evidence record.
- The only local branches are `main` and the runtime LKG branch declared in
  `performance/bc250/lkg.json`.
- Never commit runtime changes directly to `pr-557-merge` without explicit
  user approval and qualifying evidence from `performance/bc250/score.py`.
  User-requested engineering-system-only commits are permitted. Those paths
  are `AGENTS.md`, `.gitattributes`, `.gitignore`, `.githooks/`, and
  `performance/bc250/`.
- If the user explicitly orders an exact, production-activated causal unit to
  be retained while deferring the canonical repetition gate, the hook may
  accept a `user_approved_checkpoint` evidence record. It must bind the exact
  staged runtime diff, matching end-to-end artifacts, production activation,
  and the reason the full score was deferred. This is an LKG checkpoint, not a
  claim that the 10/20 TPS milestone was met.
- Never push any ref without explicit user approval in the current turn.
- Never infer promotion or push approval. An agent may set
  `DS4_USER_APPROVED_PROMOTION=1` only when the user explicitly approved
  promotion of the exact scored candidate in the current turn. Never set
  `DS4_USER_APPROVED_PUSH`; the user must separately authorize every push.
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

A scored runtime candidate must be a clean deterministic commit in a disposable
checkout rooted at the exact runtime LKG. Its evidence must bind the LKG base,
patch hash, candidate commit/tree, binaries, role-specific shader manifests, exactness
artifacts, activation proof, and scores. It never gets a persistent branch.
After a `promote_eligible` verdict and explicit user approval, apply that exact
patch once to the runtime LKG branch, commit it, update `lkg.json`, and delete
the disposable checkout. Do not ask the user to name internal Git objects.

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
