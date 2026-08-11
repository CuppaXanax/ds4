# Agent Notes

`ds4.c` is a DeepSeek V4 Flash specific inference engine. It is not a generic
GGUF runner. The goal is a small, readable, high-performance C codebase with
Objective-C only where Metal requires it and Metal kernels under `metal/`.

## Goals

- Keep the production path as whole-model Metal graph inference.
- Always make sure that the SSD streaming, CUDA, distributed inference, Metal default inference are not affected by fixes to other parts of the code.
- Keep model loading mmap-backed for the Metal default case; do not eagerly copy the full GGUF. Keep the model loading for SSD streaming of routed experts explicit: allocated buffers, fast reads from disk, always try to hide loading of missing routed experts by loading them while performing the inference of the shared expert and routed experts already in RAM. Always try to hide loading of layers for prefill in SSD streaming mode using the inference time of the current layer as the next one is loaded.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV checkpoints.

## Quality Rules

- Keep the implementation small, sharp, easy to understand. Try to write elegant code in a state of grace. Don't settle for the first thing that comes to mind, try to find the most minimal and better working design. Don't introduce slop: very fragile code that just patches specific cases, dead code, useless code and code ways more complicated of how it should be.
- Comment important inference code where the model mechanics, cache lifetime, memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are fine when they validate the one release path.
- Do not introduce C++.

## Safety

- Avoid large CPU inference runs on macOS; the CPU path has previously exposed kernel VM failures with very large mappings.
- Do not run multiple huge model processes concurrently. The instance lock is intentional.

## Layout

- `ds4.c`: model loading, tokenizer, CPU reference code, Metal graph scheduling,
  sessions, disk-cache payload serialization.
- `ds4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy.
- `ds4_metal.m`: Objective-C Metal runtime and kernel wrappers.
- `metal/*.metal`: compute kernels.
- `tests/`: unit and live integration tests.
- `misc/`: ignored notes, experiments, and old planning material.

This list is not complete, check the files for more info.

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when a
model and Metal are available. Use live server tests only when intentionally
testing the API surface.

At every major change where one of the following could be affected, make sure to:

1. Test the normal Metal path and that speed is still at the level it was.
2. Test the SSD streaming path.
3. Test the distributed inference if it could be affected, but ask the user before doing so.
4. Check if CUDA could be broken after the change, and ask the user to give you access to the CUDA machine to actually test if everything is still fine.

## Vulkan backend (vulkan/)

The Vulkan backend lives under `vulkan/` and is written in C++ (the engine
itself stays C). It is a single translation unit, `vulkan/vulkan_backend.cpp`,
which includes `_impl_gen.cpp` (no-op stubs) at the end.

### Contract and stubs

- The engine requires every `ds4_gpu_*` symbol declared in `ds4_gpu.h` /
  `ds4_gpu_mgpu.h` to be defined. Functions not implemented in the backend
  are generated as no-op stubs by `python3 vulkan/gen_impl.py`.
- **After adding/removing a `ds4_gpu_*` definition in `vulkan_backend.cpp`,
  ALWAYS rerun `python3 vulkan/gen_impl.py`** or the build will have a
  duplicate definition (the stub must be removed from `_impl_gen.cpp`).

### Weight streaming (the memory model)

- The GPU heap is ~47 GiB; the DeepSeek V4 Flash model is ~81 GiB, so the
  model is streamed: `cache_model_range` only records metadata,
  `ensure_weight(offset, needed_bytes)` uploads a weight range from the model
  mmap into a VMA buffer on first use, with LRU eviction against
  `DS4_VULKAN_WEIGHT_BUDGET_GB` (default 40).
- **`needed_bytes` MUST be the correct byte size of the weight**: f16 → 2
  bytes/element, q8_0 → 34 bytes per 32-element block (GGUF: f16 scale +
  32×int8, matching ds4.c), f32 → 4 bytes/element.
  Getting this wrong corrupts the descriptor range and the math.
- Never read past the end of the file-backed mmap (SIGBUS): clamp the upload
  size to `model_size - offset`.
- Eviction must never destroy a buffer still referenced by the command buffer
  being recorded (RADV crashes at submit). `ensure_weight` tags entries with
  the current command-buffer generation (`g_vk.cmd_gen`, bumped in
  `begin_cmd`) and skips current-generation entries when evicting.

### Command buffer semantics (the engine contract)

- The engine wraps GPU work as: `ds4_gpu_begin_commands()` → kernels →
  `ds4_gpu_end_commands()` (submit+wait) per token; some paths submit per
  layer.
- **Mid-decode, the engine calls `ds4_gpu_commit_and_wait_selected_readback`
  and `ds4_gpu_flush_commands` and then KEEPS RECORDING kernels without a new
  `begin_commands`** (Metal semantics: the next encoder is implicit). The
  backend therefore **re-begins a fresh command buffer after submit** in those
  two functions. Never finalize a command buffer twice.
- A kernel test that dispatches a GPU shader MUST wrap the call in
  `begin_commands`/`end_commands`; otherwise the commands are never submitted
  and the output tensor keeps stale memory.

### RADV (mesa 26.0.2) quirks

- `subgroupAdd` on `double` is unreliable → use a plain workgroup reduction
  (shared array + barrier + lane 0 sum).
- Descriptor buffer ranges must be exact; out-of-range accesses can crash
  RADV at command-stream finalize.
- `vkGetMemoryHostPointerPropertiesEXT` (VK_EXT_external_memory_host)
  **segfaults on this driver for any pointer** — do not use external host
  memory.
- Weight upload goes host → staging → `vkCmdCopyBuffer` → device (the
  destination is device-local even on the iGPU).

### Shaders

- GLSL sources in `vulkan/shaders/*.comp`, SPIR-V in `vulkan/shaders/spv/`.
  Compile with `glslangValidator -V --target-env vulkan1.2 <name>.comp -o
  spv/<name>.spv`. The backend loads the listed shaders in
  `load_all_shaders()` (name + push-constant size); add new shaders there.
- f16 weights are stored as IEEE halves packed two-per-uint32; the shader
  must extract the low half (even element) or high half (odd element).

### Kernel tests

- Harness: `vulkan/tests/harness.cpp` + `vulkan/tests/tests.cpp`. Each test
  is its own file `vulkan/tests/tests/t_<name>.cpp` registering
  `REGISTER_TEST(name, fn)` and returning 0 on PASS. Tests build synthetic
  tensors, call the real kernel, and compare against an inline CPU reference.
- Run: `./run-kernel-tests.sh` — **requires GPU (/dev/dri) access; agents and
  subagents cannot run it, only compile**. Results go to
  `vulkan/tests/results.txt`; verified kernels go in the `VERIFIED` set of
  `vulkan/kernel_status.py` and the table in `vulkan/STATUS.md` is
  regenerated with `python3 vulkan/kernel_status.py >> vulkan/STATUS.md`
  (after truncating the old table).
- Harness compile command (no GPU needed):
  `g++ -O2 -g -std=c++17 -pthread -I. -Ivulkan -Ivulkan/include -DDS4_VULKAN_BUILD vulkan/tests/harness.cpp vulkan/tests/tests.cpp vulkan/tests/tests/*.cpp vulkan/vulkan_backend.cpp vulkan/q8_aligned_artifact.cpp -lm -pthread -lvulkan -o vulkan-tests`
- `ds4_gpu_add_tensor` is a CPU-hosted implementation (host add on mapped
  memory), NOT a GPU dispatch — do not assume "real" means GPU.

### Verification workflow (kernel-by-kernel)

1. Subagent implements the kernel (GPU shader preferred; trivial ops may be
   backend-hosted like `add_tensor`) and its test with a CPU reference.
2. Compile-check (no GPU). 3. User runs `./run-kernel-tests.sh`. 4. On PASS,
   add the name to `VERIFIED` in `kernel_status.py` and update `STATUS.md`.

