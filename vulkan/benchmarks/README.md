# Benchmark A — resident weight-stream roofline

`roofline_a` builds a one-layer Flash byte ledger directly from GGUF tensor
metadata, then runs a checksum-only Vulkan shader over the same file-backed
ranges used by the production weight cache. It includes every non-expert
`blk.<layer>.*.weight` tensor once and replaces the three routed expert stacks
with the selected expert IDs (default top-6 `0..5`): all IQ2 gate rows, all IQ2
up rows, then Q2 down rows. Activations and KV are excluded from the
denominator.

Build on the BC-250 host:

```sh
make vulkan-roofline-a
```

Run with a resident weight budget large enough for the selected spans:

```sh
env DS4_VULKAN_WEIGHT_BUDGET_GB=11 \
  ./vulkan-roofline-a /models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --layer 6 --experts 0,1,2,3,4,5 --repeats 20
```

The first-pass `warm upload bytes/ns` proves residency. The timed result uses
GPU timestamp queries only and reports useful bytes/pass, dispatch count,
checksum, effective decimal GB/s, and GiB/s. `--dense NAME` can restrict the
non-expert portion to one Q8_0 tensor for a narrower dense-projection run.
