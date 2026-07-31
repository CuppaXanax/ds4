# llama.cpp Vulkan Q8_0 Quantized MatMul — Complete Reference

Source tree: `ggml/src/ggml-vulkan/vulkan-shaders/` and `ggml/src/ggml-vulkan/ggml-vulkan.cpp`

---

## File Index

| File | Purpose |
|------|---------|
| `mul_mmq.comp` | Main GLSL compute shader for Q8_0 × Q8_1 matmul |
| `mul_mmq_funcs.glsl` | Block load/store/dot-product functions per quant type |
| `mul_mmq_shmem_types.glsl` | Shared memory cache structs per quant type |
| `types.glsl` | All block struct definitions (Q8_0, Q8_1, K-quants, IQuants) |
| `vulkan-shaders-gen.cpp` | Compiles all shader variants with specialization constants |
| `ggml-vulkan.cpp` | C++ dispatch, pipeline creation, push constants |
| `dequant_q8_0.comp` | Simple Q8_0 → F32 dequant-only shader |
| `flash_attn_mmq_funcs.glsl` | MMQ patterns reused in flash attention K-side loads |

---

## Q8_0 Type Definition (`types.glsl`)

```glsl
// Q8_0: 32 int8 values + fp16 scale, packed for efficient 2-byte loads
#define QUANT_K_Q8_0 32
#define QUANT_R_Q8_0 1

// CPU-side / naive buffer layout
struct block_q8_0 {
    float16_t d;        // block scale, 2 bytes
    int8_t qs[32];      // 32 quantized int8 values, 32 bytes
};

// Packed for aligned 2-byte loads (int16 pairs)
struct block_q8_0_packed16 {
    float16_t d;        // block scale, 2 bytes
    int16_t qs[32/2];   // 16 × int16 = 32 bytes, each int16 = 2 adjacent int8
};

// When compiled with -DDATA_A_Q8_0:
#define A_TYPE block_q8_0
#define A_TYPE_PACKED16 block_q8_0_packed16
```

## Q8_1 (B-side) Packed Type (`types.glsl`)

```glsl
// Q8_1: int8 + scale pair (d, s), used as the B-side operand
struct block_q8_1 {
    f16vec2 ds;         // (d, s) = (absolute scale, Q8_0's zero-point equivalent)
    int8_t qs[32];      // 32 int8 values
};

// 4 blocks packed together for 128-byte alignment
struct block_q8_1_x4_packed128 {
    f16vec2 ds[4];      // 4 scale pairs (16 bytes)
    ivec4 qs[8];        // 8 × ivec4 = 32 int32 = 128 int8 (128 bytes)
};
//
// Total: 144 bytes per 4 blocks = 36 bytes per block
```

---

## Shared Memory Cache Types (`mul_mmq_shmem_types.glsl`)

```glsl
// --- A-side cache (Q8_0 variant) ---
// QUANT_R_MMQ = 1 for Q8_0
struct block_a_cache {
    int32_t qs[32/4];   // 8 int32, each packing 4 × int8 via pack32(i16vec2)
    float dm;            // single scale factor (d), promoted to fp32
};

// --- B-side cache (same for all quant types) ---
// BK = 32, so BK/4 = 8 int32
struct block_b_cache {
    int32_t qs[8];      // 8 int32, each packing 4 × int8
    FLOAT_TYPEV2 ds;    // (d, s) scale pair from Q8_1
};
```

---

## Q8_0 Block Loading Functions (`mul_mmq_funcs.glsl`, lines 115–145)

### Global → Shared Memory

```glsl
void block_a_to_shmem(const uint buf_ib, const uint ib, const uint iqs) {
    // iqs = 0..7 (one int32 per invocation, 8 invocations × 4 int8 = 32 int8)
    // Load 2 adjacent int8 as int16, then pack 2 int16 into one int32
    buf_a[buf_ib].qs[iqs] = pack32(i16vec2(
        data_a_packed16[ib].qs[iqs * 2],
        data_a_packed16[ib].qs[iqs * 2 + 1]
    ));
    // Only thread iqs==0 loads the scale
    if (iqs == 0) {
        buf_a[buf_ib].dm = FLOAT_TYPE(data_a_packed16[ib].d);
    }
}
```

Key detail: `LOAD_VEC_A = 4 * QUANT_R_MMQ = 4` for Q8_0. Each invocation loads
`4 × int32 = 16 int8` worth of data per outer loop iteration. With `BK=32`,
there are `BK / LOAD_VEC_A = 8` invocations per A row.

### Shared Memory → Registers

```glsl
void block_a_to_registers(const uint reg_ib, const uint buf_ib) {
    cache_a[reg_ib].dm = buf_a[buf_ib].dm;
    // Copy all 8 int32 from shmem to register
    for (uint iqs = 0; iqs < 8; iqs++) {
        cache_a[reg_ib].qs[iqs] = buf_a[buf_ib].qs[iqs];
    }
}
```

### B-side Block Loading (shared across all quant types)

```glsl
void block_b_to_shmem(const uint buf_ib, const uint ib, const uint iqs,
                      const bool is_in_bounds) {
    if (is_in_bounds) {
        // Packed 4-block structure: ib_outer = ib/4, ib_inner = ib%4
        const uint ib_outer = ib / 4;
        const uint ib_inner = ib % 4;
        // Thread iqs==0 loads the scale pair
        if (iqs == 0) {
            buf_b[buf_ib].ds = FLOAT_TYPEV2(data_b[ib_outer].ds[ib_inner]);
        }
        // Load 4 × int32 = 16 int8 via ivec4
        const ivec4 values = data_b[ib_outer].qs[ib_inner * 2 + iqs];
        buf_b[buf_ib].qs[iqs * 4    ] = values.x;
        buf_b[buf_ib].qs[iqs * 4 + 1] = values.y;
        buf_b[buf_ib].qs[iqs * 4 + 2] = values.z;
        buf_b[buf_ib].qs[iqs * 4 + 3] = values.w;
    } else {
        // Out-of-bounds: zero fill
        buf_b[buf_ib].ds = FLOAT_TYPEV2(0.0f);
        buf_b[buf_ib].qs[iqs * 4    ] = 0;
        // ...
    }
}
```

### B Register Cache

```glsl
void block_b_to_registers(const uint ib) {
    cache_b.ds = buf_b[ib].ds;
    for (uint iqs = 0; iqs < BK / 4; iqs++) {  // BK=32 → 8 iterations
        cache_b.qs[iqs] = buf_b[ib].qs[iqs];
    }
}
```

---

## Q8_0 Integer Dot Product (`mul_mmq_funcs.glsl`, lines 134–144)

```glsl
ACC_TYPE mmq_dot_product(const uint ib_a) {
    int32_t q_sum = 0;
    // BK=32 → BK/4 = 8 int32 packed values
    for (uint iqs = 0; iqs < 8; iqs++) {
        const int32_t qs_a = cache_a[ib_a].qs[iqs];  // 4 × int8 from A
        const int32_t qs_b = cache_b.qs[iqs];        // 4 × int8 from B
        // GL_EXT_integer_dot_product built-in: dotPacked4x8EXT(a, b)
        // Computes dp4a(a, b) = sum of 4 signed int8 multiplications,
        // returns int32.  Maps to hardware DP4a instruction.
        q_sum += dotPacked4x8EXT(qs_a, qs_b);
    }
    // Q8_0 has no min (unlike Q4_0 which subtracts 8.0*d_b*s_b).
    // Q8_0 = d_a * d_b * q_sum
    return ACC_TYPE(float(q_sum) * float(cache_a[ib_a].dm) * float(cache_b.ds.x));
}
```

---

## Main Shader Dispatch (`mul_mmq.comp`)

### Binding and Push Constants

```glsl
#version 450
#extension GL_EXT_integer_dot_product : require

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer A { A_TYPE data_a[]; };
// For Q8_0, A is accessed via packed16 alias
layout(binding = 0) readonly buffer A_PACKED16 { A_TYPE_PACKED16 data_a_packed16[]; };
layout(binding = 1) readonly buffer B { block_q8_1_x4_packed128 data_b[]; };
layout(binding = 2) writeonly buffer D { D_TYPE data_d[]; };

layout(push_constant) uniform parameter {
    uint M, N, K;
    uint stride_a, stride_b, stride_d;
    uint batch_stride_a, batch_stride_b, batch_stride_d;
    uint base_work_group_z, num_batches, k_split;
    uint ne02, ne12, broadcast2, broadcast3;
} p;

// Specialization constants
layout(constant_id = 0) const uint BLOCK_SIZE = 64;
layout(constant_id = 1) const uint BM = 64;
layout(constant_id = 2) const uint BN = 64;
// BK hardcoded to 32
layout(constant_id = 4) const uint WM = 32;
layout(constant_id = 5) const uint WN = 32;
layout(constant_id = 6) const uint WMITER = 2;
layout(constant_id = 7) const uint TM = 4;
layout(constant_id = 8) const uint TN = 2;
layout(constant_id = 10) const uint WARP = 32;

#define BK 32
#define BK_STEP 4
```

### main() — Full Dispatch Flow

```glsl
void main() {
    // --- Step 1: Compute tile indices ---
    const uint ic = gl_WorkGroupID.y;                     // N tile
    const uint batch_idx = gl_WorkGroupID.z + p.base_work_group_z;

    // Broadcasting support
    const uint i13 = batch_idx / p.ne12;
    const uint i12 = batch_idx % p.ne12;
    const uint i03 = i13 / p.broadcast3;
    const uint i02 = i12 / p.broadcast2;
    const uint batch_idx_a = i03 * p.ne02 + i02;

    const uint blocks_m = (p.M + BM - 1) / BM;
    const uint ir = gl_WorkGroupID.x % blocks_m;          // M tile
    const uint ik = gl_WorkGroupID.x / blocks_m;          // K split

    // --- Step 2: Compute warp/thread positions ---
    const uint WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
    const uint WSUBM = WM / WMITER;
    const uint WSUBN = WN / WNITER;
    const uint warp_i = gl_LocalInvocationID.x / WARP;
    const uint tiw = gl_LocalInvocationID.x % WARP;
    const uint tiwr = tiw % (WSUBM / TM);
    const uint tiwc = tiw / (WSUBM / TM);
    const uint warp_r = warp_i % (BM / WM);
    const uint warp_c = warp_i / (BM / WM);

    // Loader positions
    const uint loadr_a = gl_LocalInvocationID.x % (BK / 4);
    const uint loadc_a = gl_LocalInvocationID.x / (BK / 4);
    const uint loadr_b = gl_LocalInvocationID.x % (BK / 16);
    const uint loadc_b = gl_LocalInvocationID.x / (BK / 16);
    const uint loadstride_a = BLOCK_SIZE * 4 / BK;        // 64*4/32 = 8
    const uint loadstride_b = BLOCK_SIZE * 16 / BK;       // 64*16/32 = 32

    // --- Step 3: Initialize accumulators ---
    ACC_TYPE sums[WMITER * TM * WNITER * TN];
    for (uint i = 0; i < WMITER*TM*WNITER*TN; i++) sums[i] = ACC_TYPE(0.0f);

    // --- Step 4: Main K-loop (outer tiling) ---
    const uint start_k = ik * p.k_split;
    const uint end_k = min(p.K, (ik + 1) * p.k_split);
    uint pos_a_ib = batch_idx_a * (p.batch_stride_a / BK)
                  + (ir * BM * p.stride_a + start_k) / BK;
    uint pos_b_ib = (batch_idx * p.batch_stride_b
                  + ic * BN * p.stride_b + start_k) / BK;

    for (uint block = start_k; block < end_k; block += BK * BK_STEP) {
        // Phase 1: Load A tile from global → shmem
        for (uint l = 0; loadc_a + l < BM; l += loadstride_a) {
            uint buf_ib = loadc_a + l;
            uint ib = pos_a_ib + buf_ib * p.stride_a / BK;
            for (int ks = 0; ks < BK_STEP; ks++) {
                block_a_to_shmem(ks * BM + buf_ib, ib + ks, loadr_a);
            }
        }
        // Phase 2: Load B tile from global → shmem
        for (uint l = 0; loadc_b + l < BN; l += loadstride_b) {
            uint buf_ib = loadc_b + l;
            uint ib = pos_b_ib + buf_ib * p.stride_b / BK;
            for (int ks = 0; ks < BK_STEP; ks++) {
                block_b_to_shmem(ks * BN + buf_ib, ib + ks, loadr_b,
                                 block + ks * BK < end_k);
            }
        }
        barrier();

        pos_a_ib += BK_STEP;
        pos_b_ib += BK_STEP;

        // Phase 3: Inner K-step dot products
        for (uint k_step = 0; k_step < BK_STEP; k_step++) {
            // Load A from shmem → registers
            for (uint wsir = 0; wsir < WMITER; wsir++) {
                for (uint cr = 0; cr < TM; cr++) {
                    uint reg_ib = wsir * TM + cr;
                    uint buf_ib = warp_r * WM + wsir * WSUBM + tiwr * TM + cr;
                    block_a_to_registers(reg_ib, k_step * BM + buf_ib);
                }
            }
            // Iterate over B columns, compute dot products
            for (uint wsic = 0; wsic < WNITER; wsic++) {
                for (uint cc = 0; cc < TN; cc++) {
                    uint ib = k_step * BN + warp_c * WN
                            + wsic * WSUBN + tiwc * TN + cc;
                    block_b_to_registers(ib);
                    for (uint wsir = 0; wsir < WMITER; wsir++) {
                        for (uint cr = 0; cr < TM; cr++) {
                            uint cache_a_idx = wsir * TM + cr;
                            uint sums_idx = (wsic * TN + cc) * (WMITER * TM)
                                          + wsir * TM + cr;
                            sums[sums_idx] += mmq_dot_product(cache_a_idx);
                        }
                    }
                }
            }
        }
        barrier();
    }

    // --- Step 5: Write results ---
    const uint dr = ir * BM + warp_r * WM;
    const uint dc = ic * BN + warp_c * WN;
    const uint offsets = batch_idx * p.batch_stride_d + ik * p.batch_stride_d * p.num_batches;

    for (uint wsic = 0; wsic < WNITER; wsic++) {
        for (uint wsir = 0; wsir < WMITER; wsir++) {
            uint dr_warp = dr + wsir * WSUBM + tiwr * TM;
            uint dc_warp = dc + wsic * WSUBN + tiwc * TN;
            for (uint cc = 0; cc < TN; cc++) {
                for (uint cr = 0; cr < TM; cr++) {
                    uint sums_idx = (wsic * TN + cc) * WMITER * TM + wsir * TM + cr;
                    if (dr_warp + cr < p.M && dc_warp + cc < p.N) {
                        data_d[offsets + (dc_warp + cc) * p.stride_d + dr_warp + cr]
                            = D_TYPE(sums[sums_idx].x);
                    }
                }
            }
        }
    }
}
```

---

## C++ Dispatch Code

### Push Constant Struct (`ggml-vulkan.cpp`, line 1084)

```cpp
struct vk_mat_mat_push_constants {
    uint32_t M, N, K;
    uint32_t stride_a, stride_b, stride_d;
    uint32_t batch_stride_a, batch_stride_b, batch_stride_d;
    uint32_t base_work_group_z;
    uint32_t num_batches;
    uint32_t k_split;
    uint32_t ne02, ne12, broadcast2, broadcast3;
    uint32_t padded_N;
};
```

### Core Dispatch Function (`ggml-vulkan.cpp`, line 7628)

```cpp
template <typename T>
static void ggml_vk_dispatch_pipeline(
    ggml_backend_vk_context* ctx, vk_context& subctx,
    vk_pipeline& pipeline,
    std::initializer_list<vk::DescriptorBufferInfo> const& descriptor_buffer_infos,
    const T &push_constants,
    std::array<uint32_t, 3> elements)
{
    const uint32_t wg0 = CEIL_DIV(elements[0], pipeline->wg_denoms[0]);
    const uint32_t wg1 = CEIL_DIV(elements[1], pipeline->wg_denoms[1]);
    const uint32_t wg2 = CEIL_DIV(elements[2], pipeline->wg_denoms[2]);

    GGML_ASSERT(wg0 <= device->limits.maxComputeWorkGroupCount[0]);
    // ... (same for wg1, wg2)
    GGML_ASSERT(descriptor_buffer_infos.size() == pipeline->parameter_count);
    GGML_ASSERT(push_constant_size(push_constants) == pipeline->push_constant_size);

    // Get next descriptor set from pool
    vk::DescriptorSet& descriptor_set = ctx->descriptor_sets[ctx->descriptor_set_idx++];

    // Write all storage buffer bindings at once
    vk::WriteDescriptorSet write_descriptor_set{
        descriptor_set, 0, 0, pipeline->parameter_count,
        vk::DescriptorType::eStorageBuffer, nullptr,
        descriptor_buffer_infos.begin()
    };
    ctx->device->device.updateDescriptorSets({ write_descriptor_set }, {});

    // Push constants → make visible to shader
    subctx->s->buffer->buf.pushConstants(
        pipeline->layout,
        vk::ShaderStageFlagBits::eCompute,
        0,
        push_constant_size(push_constants),
        push_constant_data(push_constants));

    // Bind pipeline and descriptor set
    subctx->s->buffer->buf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline->pipeline);
    subctx->s->buffer->buf.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute,
        pipeline->layout, 0, { descriptor_set }, {});

    // DISPATCH — returns immediately, GPU executes asynchronously
    subctx->s->buffer->buf.dispatch(wg0, wg1, wg2);
}
```

### MatMul Dispatcher (non-MoE, line 8273)

```cpp
static void ggml_vk_matmul(
    ctx, subctx, pipeline,
    vk_subbuffer&& a, vk_subbuffer&& b, vk_subbuffer&& d,
    vk_subbuffer&& split_k_buffer,
    uint32_t m, uint32_t n, uint32_t k,
    uint32_t stride_a, uint32_t stride_b, uint32_t stride_d,
    uint32_t batch_stride_a, uint32_t batch_stride_b, uint32_t batch_stride_d,
    uint32_t split_k, uint32_t batch,
    uint32_t ne02, uint32_t ne12, uint32_t broadcast2, uint32_t broadcast3,
    uint32_t padded_n)
{
    if (split_k == 1) {
        // Simple dispatch — no split-K
        ggml_pipeline_request_descriptor_sets(ctx, pipeline,
            CEIL_DIV(batch, maxComputeWorkGroupCount[2]));

        uint32_t base_work_group_z = 0;
        while (base_work_group_z < batch) {
            uint32_t groups_z = std::min(batch - base_work_group_z,
                ctx->device->properties.limits.maxComputeWorkGroupCount[2]);

            const vk_mat_mat_push_constants pc = {
                m, n, k, stride_a, stride_b, stride_d,
                batch_stride_a, batch_stride_b, batch_stride_d,
                base_work_group_z, batch, k,
                ne02, ne12, broadcast2, broadcast3, padded_n
            };
            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
                { a, b, d },     // 3 storage buffer descriptors
                pc,
                { m, n, groups_z });  // elements → dispatch wg0, wg1, wg2
            base_work_group_z += groups_z;
        }
    } else {
        // Split-K dispatch: each workgroup handles part of K dimension
        // Then a reduction pass combines the partial sums
        uint32_t k_split = CEIL_DIV(k, split_k);
        k_split = ROUNDUP_POW2(k_split, 256);
        // ... dispatch with k_split in push constants ...
        // ... then run pipeline_matmul_split_k_reduce ...
    }
}
```

### Pipeline Selection for the Q8_0 Integer-Dot Path (`ggml-vulkan.cpp`, line 9600)

```cpp
// Key logic in ggml_vk_mul_mat:
bool quantize_y = device->integer_dot_product &&
                  src1->type == GGML_TYPE_F32 &&
                  ggml_is_contiguous(src1) && (ne11 * ne10) % 4 == 0;

// Try Q8_1 integer-dot path first
vk_matmul_pipeline mmp = quantize_y
    ? ggml_vk_get_mul_mat_mat_id_pipeline(ctx, src0->type, GGML_TYPE_Q8_1, prec)
    : nullptr;

if (mmp == nullptr) {
    // Fall back: dequant src0 to f16, then do f16 matmul
    mmp = ggml_vk_get_mul_mat_mat_id_pipeline(
        ctx, src0->type, y_non_contig ? f16_type : src1->type, prec);
    quantize_y = false;
}

// Pipeline availability is checked through separate "int" flags
// ggml_vk_guess_matmul_id_pipeline:
bool is_q8_1 = (src1_type == GGML_TYPE_Q8_1);
bool mm_l = is_q8_1 ? device->mul_mat_id_l_int[src0_type]
                    : device->mul_mat_id_l[src0_type];
bool mm_m = is_q8_1 ? device->mul_mat_id_m_int[src0_type]
                    : device->mul_mat_id_m[src0_type];
bool mm_s = is_q8_1 ? device->mul_mat_id_s_int[src0_type]
                    : device->mul_mat_id_s[src0_type];
```

### Workgroup Denoms (tuning parameters, line 3968)

```cpp
l_mmq_wg_denoms = { 128, 128, 1 };   // large: M=128, N=128 per dispatch element
m_mmq_wg_denoms = { 64,  64,  1 };   // medium: M=64, N=64
s_mmq_wg_denoms = { 32,  32,  1 };   // small: M=32, N=32
```

These are the `wg_denoms` used to compute `CEIL_DIV(elements[i], wg_denoms[i])`
for the dispatch call. The denoms correspond to the BM/BN workgroup tile sizes.

---

## Shader Compilation Pipeline (`vulkan-shaders-gen.cpp`)

The Q8_0 integer-dot variant is compiled as:

```cpp
string_to_spv("matmul_q8_0_q8_1", "mul_mmq.comp",
    merge_maps(base_dict, float_type_dict,
        {{"DATA_A_Q8_0", "1"}, {"D_TYPE", "float"}}),
    fp16, coopmat, coopmat2, f16acc);
```

This produces shaders named `matmul_q8_0_q8_1_l`, `_m`, `_s`, and their
aligned variants `_aligned_l`, `_aligned_m`, `_aligned_s`.

The "int" suffix in pipeline availability flags (`mul_mat_id_l_int[type]`)
refers to shader names ending in `_q8_1` rather than `_f16` or `_f32`.
They're compiled only when `GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT` is defined,
which depends on `GL_EXT_integer_dot_product` support in glslang/shaderc.

---

## Empty Bounds Handling

When the K dimension bound is reached early (partial tile at trailing edge),
`block_b_to_shmem` zero-fills the out-of-bounds region rather than branching:

```glsl
if (!is_in_bounds) {
    buf_b[buf_ib].ds = FLOAT_TYPEV2(0.0f);
    buf_b[buf_ib].qs[iqs * 4    ] = 0;
    buf_b[buf_ib].qs[iqs * 4 + 1] = 0;
    buf_b[buf_ib].qs[iqs * 4 + 2] = 0;
    buf_b[buf_ib].qs[iqs * 4 + 3] = 0;
}
```

Zero int8 × anything = 0, so out-of-bounds B elements contribute nothing
to the dot product. This avoids warp divergence from an `if/return` at
the K-boundary — all threads stay converged.

---

## Dequant-Only Shader (`dequant_q8_0.comp`)

For the simpler `get_rows` operation that just dequantizes Q8_0 → F32:

```glsl
#version 450
#include "dequant_head.glsl"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0) readonly buffer A { block_q8_0 data_a[]; };
layout(binding = 1) writeonly buffer D { D_TYPE data_b[]; };

void main() {
    const uint i = gl_WorkGroupID.x * 4 + gl_LocalInvocationID.x / 64;
    const uint ib = 32 * i + (gl_LocalInvocationID.x % 32);

    if (ib >= p.nel / 32) return;

    const float d = float(data_a[ib].d);
    const uint b_idx = 1024*i + 16*(gl_LocalInvocationID.x/32) + (gl_LocalInvocationID.x%32);
    const uint q_idx = 16*(gl_LocalInvocationID.x/32);

    for (uint l = 0; l < 16; l += 2) {
        data_b[b_idx + l    ] = D_TYPE(d * data_a[ib].qs[q_idx + l]);
        data_b[b_idx + l + 1] = D_TYPE(d * data_a[ib].qs[q_idx + l + 1]);
    }
}
```

256 threads per workgroup, each processing 4 blocks of 32 elements.

---

## Key Takeaways for Implementing a Vulkan Backend

1. **A-side is quantized, B-side is always Q8_1.** On-device quantization
   (F32 → Q8_1) happens as a pre-pass before the matmul.

2. **Integer dot product via `GL_EXT_integer_dot_product`.** The
   `dotPacked4x8EXT()` built-in maps to hardware DP4a. Requires SPIR-V
   capability `DotProductInput4x8Bit` / `DotProduct`.

3. **Aligned packed structs for efficient loads.** Block structs are
   explicitly packed with alignment-friendly sizes (16-byte for Q8_1_x4,
   2-byte int16 pairs for Q8_0).

4. **Triple-level tiling (workgroup → warp → register).** 64 threads per
   workgroup, split into 2 warps of 32. WM=32, WMITER=2, TM=4 means each
   warp handles 8 output rows of A tiles × 4 unrolls.

5. **Push constants carry tile dimensions plus batch metadata.** The same
   push constant struct serves all quant types.

6. **Three pipeline variants per type (large/medium/small).** Selected at
   runtime based on matrix M × N dimensions. Variant with "int" suffix
   for the integer-dot Q8_1 path.

7. **Split-K support for large inner dimensions.** When split_k > 1,
   the K dimension is divided into workgroups, each writes partial sums,
   then a reduction kernel combines them.
