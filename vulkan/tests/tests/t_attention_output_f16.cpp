/* Kernel test: ds4_gpu_attention_output_q8_batch_f16_tensor (host-side
 * Q8_0 grouped attention output projection with f16 output).
 *
 * Identical math to ds4_gpu_attention_output_q8_batch_tensor, but the final
 * out_dim-wide projection of every token is stored as IEEE f16
 * (2 bytes/element) in out_h (the engine's g->batch_q_half fast path).  The
 * CPU reference below replicates ds4.c layer_grouped_out_batch +
 * matmul_q8_0_batch and rounds every output value through the same half
 * conversion (f32 -> f16 -> f32), validating the storage layout, the math,
 * and the half rounding all at once.
 *
 * The kernel is CPU-hosted in the Vulkan backend: no begin/end_commands are
 * needed, the tensors are plain host-mapped buffers.
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <vector>

/* ---- f16 helpers (same decode/encode as ds4.c f32_to_f16/f16_to_f32) ---- */
static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t e    = (x >> 23) & 0xffu;
    uint32_t m    = x & 0x7fffffu;
    if (e == 0xffu) return (uint16_t)(sign | 0x7c00u | (m ? 0x200u : 0u));
    if (e == 0u)    return (uint16_t)sign;
    int32_t e16 = (int32_t)e - 127 + 15;
    if (e16 >= 31)  return (uint16_t)(sign | 0x7c00u);
    if (e16 <= 0) {
        uint32_t shift = 126u - e;
        uint32_t m16 = (0x800000u | m) >> shift;
        return (uint16_t)(sign | m16);
    }
    return (uint16_t)(sign | ((uint32_t)e16 << 10) | (m >> 13));
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e    = (h >> 10) & 0x1fu;
    uint32_t m    = h & 0x3ffu;
    uint32_t bits;
    if (e == 0u) {
        if (m == 0u) bits = sign;
        else {
            uint32_t mm = m, p = 0u;
            while (!(mm & 0x400u) && p < 10u) { mm <<= 1; p++; }
            bits = sign | ((113u - p) << 23) | ((mm & 0x3ffu) << 13);
        }
    } else if (e == 31u) {
        bits = sign | 0x7f800000u | (m << 13);
    } else {
        bits = sign | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

/* ---- Q8_0 quantize + dot (identical to ds4.c quantize_q8_0_activation /
 *      dot_q8_0_row and to the Vulkan backend helpers) ---- */
static void quant_q8_0(const float *x, int8_t *xq, float *scale, uint64_t n) {
    const uint64_t blocks = (n + 31u) / 32u;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = n - i0 < 32u ? n - i0 : 32u;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) {
            const float ax = std::fabsf(x[i0 + i]);
            if (ax > amax) amax = ax;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        scale[b] = d;
        for (uint64_t i = 0; i < bn; i++) {
            int v = (int)lrintf(x[i0 + i] * id);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            xq[i0 + i] = (int8_t)v;
        }
        for (uint64_t i = bn; i < 32u && i0 + i < blocks * 32u; i++) xq[i0 + i] = 0;
    }
}

static float dot_q8_0(const uint8_t *row, const int8_t *xq, const float *xscale,
                      uint64_t in_dim, uint64_t blocks) {
    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t scale_bits;
        std::memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
        const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
        const uint64_t i0 = b * 32u;
        const uint64_t n = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        int32_t sum = 0;
        for (uint64_t i = 0; i < n; i++) sum += (int32_t)qs[i] * (int32_t)xq[i0 + i];
        acc += f16_to_f32(scale_bits) * xscale[b] * (float)sum;
    }
    return acc;
}

/* Quantize one weight row into GGUF Q8_0 storage: {f16 scale, 32 int8} per
 * 32-element block. */
static void make_q8_0_row(uint8_t *dst, const float *vals, uint64_t n) {
    const uint64_t blocks = (n + 31u) / 32u;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = n - i0 < 32u ? n - i0 : 32u;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) {
            const float ax = std::fabsf(vals[i0 + i]);
            if (ax > amax) amax = ax;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        uint16_t dbits = f32_to_f16(d);
        std::memcpy(dst + b * 34u, &dbits, sizeof(dbits));
        int8_t *qs = (int8_t *)(dst + b * 34u + 2u);
        for (uint64_t i = 0; i < bn; i++) {
            int v = (int)lrintf(vals[i0 + i] * id);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            qs[i] = (int8_t)v;
        }
        for (uint64_t i = bn; i < 32u; i++) qs[i] = 0;
    }
}

/* Deterministic pseudo-random float in [-1, 1). */
static float lcg_float(uint64_t *s) {
    *s = *s * 6364136223846793005ull + 1442695040888963407ull;
    return (float)((double)((*s >> 33) & 0x7fffffffu) / (double)0x40000000u) * 2.0f - 1.0f;
}

/* ---- attention_output_q8_batch_f16: two-stage batch projection, f16 out ---- */
static int test_attention_output_q8_batch_f16(void) {
    const uint32_t n_groups  = 2;
    const uint64_t group_dim = 64;
    const uint64_t rank      = 16;
    const uint32_t n_tokens  = 3;
    const uint64_t out_dim   = 24;
    const uint64_t low_dim   = (uint64_t)n_groups * rank;  /* 32 */

    const uint64_t blocks_a = (group_dim + 31u) / 32u;      /* 2 */
    const uint64_t row_a_bytes = blocks_a * 34u;
    const uint64_t out_a_bytes = low_dim * row_a_bytes;
    const uint64_t blocks_b = (low_dim + 31u) / 32u;        /* 1 */
    const uint64_t row_b_bytes = blocks_b * 34u;
    const uint64_t out_b_bytes = out_dim * row_b_bytes;
    const uint64_t header = 16;
    const uint64_t out_a_offset = header;
    const uint64_t out_b_offset = out_a_offset + out_a_bytes;
    const uint64_t model_size = out_b_offset + out_b_bytes;

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(n_tokens * n_groups * group_dim * sizeof(float));
    ds4_gpu_tensor *low   = ds4_gpu_tensor_alloc(n_tokens * low_dim * sizeof(float));
    ds4_gpu_tensor *out_h = ds4_gpu_tensor_alloc(n_tokens * out_dim * sizeof(uint16_t));
    if (!heads || !low || !out_h) {
        if (heads) ds4_gpu_tensor_free(heads);
        if (low) ds4_gpu_tensor_free(low);
        if (out_h) ds4_gpu_tensor_free(out_h);
        return 1;
    }

    unsigned char *model = (unsigned char *)malloc(model_size);
    if (!model) {
        ds4_gpu_tensor_free(heads); ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(out_h);
        return 1;
    }
    std::memset(model, 0xAA, header);

    uint64_t seed = 0x0ddc0ffee1234567ull;
    std::vector<float> wrow;
    wrow.resize(group_dim > low_dim ? group_dim : low_dim);
    for (uint64_t r = 0; r < low_dim; r++) {
        for (uint64_t i = 0; i < group_dim; i++) wrow[i] = lcg_float(&seed) * 0.5f;
        make_q8_0_row(model + out_a_offset + r * row_a_bytes, wrow.data(), group_dim);
    }
    for (uint64_t o = 0; o < out_dim; o++) {
        for (uint64_t i = 0; i < low_dim; i++) wrow[i] = lcg_float(&seed) * 0.5f;
        make_q8_0_row(model + out_b_offset + o * row_b_bytes, wrow.data(), low_dim);
    }
    if (ds4_gpu_set_model_map(model, model_size) == 0) {
        free(model);
        ds4_gpu_tensor_free(heads); ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(out_h);
        return 1;
    }

    /* Known heads activation (n_tokens x n_groups x group_dim). */
    std::vector<float> hv((size_t)n_tokens * n_groups * group_dim);
    for (uint64_t i = 0; i < hv.size(); i++) hv[i] = lcg_float(&seed) * 0.25f;
    if (ds4_gpu_tensor_write(heads, 0, hv.data(), hv.size() * sizeof(float)) == 0) {
        free(model);
        ds4_gpu_tensor_free(heads); ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(out_h);
        return 1;
    }

    /* Reference (exact ds4.c layer_grouped_out_batch order). */
    std::vector<float> ref_low((size_t)n_tokens * low_dim);
    std::vector<float> ref_out((size_t)n_tokens * out_dim);
    std::vector<int8_t> xq(n_groups * blocks_a * 32u);
    std::vector<float> xscale(n_groups * blocks_a);
    std::vector<int8_t> bq(blocks_b * 32u);
    std::vector<float> bscale(blocks_b);
    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *ht = hv.data() + (size_t)t * n_groups * group_dim;
        for (uint32_t g = 0; g < n_groups; g++)
            quant_q8_0(ht + (uint64_t)g * group_dim, xq.data() + g * blocks_a * 32u,
                       xscale.data() + g * blocks_a, group_dim);
        float *rl = ref_low.data() + (size_t)t * low_dim;
        for (uint64_t idx = 0; idx < low_dim; idx++) {
            const uint64_t g = idx / rank;
            rl[idx] = dot_q8_0(model + out_a_offset + idx * row_a_bytes,
                               xq.data() + g * blocks_a * 32u,
                               xscale.data() + g * blocks_a,
                               group_dim, blocks_a);
        }
        quant_q8_0(rl, bq.data(), bscale.data(), low_dim);
        float *ro = ref_out.data() + (size_t)t * out_dim;
        for (uint64_t o = 0; o < out_dim; o++)
            ro[o] = dot_q8_0(model + out_b_offset + o * row_b_bytes,
                             bq.data(), bscale.data(), low_dim, blocks_b);
    }

    const int rc = ds4_gpu_attention_output_q8_batch_f16_tensor(
        out_h, low, model, model_size, out_a_offset, out_b_offset,
        group_dim, rank, n_groups, out_dim, heads, n_tokens) == 0 ? 0 : 1;

    free(model);
    ds4_gpu_tensor_free(out_h);
    ds4_gpu_tensor_free(low);
    ds4_gpu_tensor_free(heads);
    return rc;
}
REGISTER_TEST(attention_output_q8_batch_f16, test_attention_output_q8_batch_f16);
