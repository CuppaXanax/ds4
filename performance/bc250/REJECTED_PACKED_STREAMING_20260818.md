# Rejected packed-streaming candidate (2026-08-18)

This branch is a forensic and iteration reference. **Do not merge it as a
whole.** The exact measured runtime is commit
`19af69c152c991604202bc9a3752e4f4e167e311`; this documentation commit is not
part of the measured binary.

## Canonical result

The candidate ran uniformly on the existing 12-blade, 24-CU BC-250 topology
with the canonical 4096-token prompt, 512 generated tokens,
`ctx_alloc=128000`, 32-bit distributed activations, an 11 GiB Vulkan weight
budget, MTP/speculation disabled, and packed Q8/IQ2/Q2 execution required.

| Metric | Result |
|---|---:|
| Prefill | 29.77 TPS |
| Decode | 0.78 TPS |
| Steady decode | 0.78 TPS |
| First-token latency | 2176.229 ms |

The repaired LKG reference is median 3.89 steady TPS (257.1 ms/token). This
candidate is approximately 1282.1 ms/token: a 79.95% throughput regression and
about 1025 ms/token of added latency. It was rejected and never promoted.

There was no OOM, raw-weight fallback, GPU reset, device loss, or worker loss
during the completed benchmark. The official multi-token semantic gate was
not run after this performance rejection.

## What is defensible

The last runtime commit fixes a real production reachability bug. Packed Q8
execution was limited to `n_tok == 1`; a 4096-token prefill therefore attempted
to use aligned/raw weights after sole-owner startup had released their raw
pages. The later `raw KV batch store failed` diagnostic masked that earlier
Q/KV projection failure.

`19af69c` makes the existing packed Q8 consumer eligible for batched tokens and
makes required-artifact mode fail closed at every token count. On a real
BC-250, a 4096-token focused test zeroed the raw source after artifact creation,
hit `matmul_q8_0_exec`, and matched its CPU reference exactly:

```text
[PASS] matmul_q8_0_execution_artifact
harness: 1 passed, 0 failed
```

That repair allowed canonical prefill to complete. It does not explain the
decode regression because one-token decode already selected the packed Q8
artifact before `19af69c`.

## Commit chain

```text
474aa3a  repaired runtime LKG / indexed selector v2
b0edfb4  integrated streaming runtime and strict decode artifacts
6f53628  packed artifacts become intended sole owners
2e7d8c2  packed IQ2 reachability during prefill
d46204a  incorrect scratch-lifetime diagnosis (did not fix production failure)
19af69c  actual batched packed-Q8 reachability repair; measured runtime
```

Exact measured identities:

- Tree: `28932a5c79dceb373abcfcf330e3d50792832e81`
- `ds4` SHA-256:
  `3004b24fa8bd3c5fd6a7e093917b56ef6fc6907e6f939db73083555a01967412`
- `ds4-bench` SHA-256:
  `2c479c58bbf8151006033ccc7a4a5fe6b3d903c6396095bd68203492c05c3b7a`
- Uniform shader count / manifest: `79` /
  `de864f1358b0aea9ed68330a494847892b4ebe2eead1cf35fa151f4ce5241169`
- Unit patch SHA-256 (`d46204a..19af69c`, fleet-native Git):
  `c067e5eef7eff12a3e538226ed0f832e1fe52688ccbcd9cac5bfbc15e02ab2af`
- Aggregate patch SHA-256 (`474aa3a..19af69c`, fleet-native Git):
  `93ee4eb20b33889ea131680717250e09cfe2fb81708c6a03e998aa0969375764`

## How to iterate

Do not optimize the aggregate candidate in place or infer that packed
residency improved throughput. Start from exact `474aa3a` and evaluate the
runtime units independently. The immediate question is which packed decode
consumer or whole-slice scheduling change adds roughly 1025 ms/token.

The leading suspect is the execution-artifact decode family—especially
`matmul_q8_0_exec_wave64`—because artifact construction, ownership, and
exactness were proved while production decode throughput was not. Compare the
packed Q8, packed IQ2 gate/up, packed Q2 down/reduce, and whole-slice command
scope separately against LKG. Only retain a unit that passes official semantics
and improves the canonical score.
