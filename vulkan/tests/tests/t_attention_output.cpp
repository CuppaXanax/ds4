/* Kernel tests: ds4_gpu_attention_output_low_q8_tensor and
 * ds4_gpu_attention_output_q8_batch_tensor (host-side Q8_0 projections).
 *
 * The attention output projection is grouped:
 *   stage A: low[g*rank + r]  = dot(out_a row g*rank + r, q8(heads[g]))
 *   stage B: out[o]           = dot(out_b row o,       q8(low))
 * where q8(x) is the GGUF Q8_0 activation quantization (scale = amax/127,
 * q = clamp(lrintf(x*127/amax))) and each weight row is stored as GGUF Q8_0
 * blocks of {f16 scale, 32 x int8} (34 bytes/block).
 *
 * The reference below replicates ds4.c matvec_q8_0_grouped_rows +
 * matmul_q8_0_batch exactly (quantized activation dot products, not raw
 * float dequant), so it validates the storage layout AND the math.
 *
 * Both functions are CPU-hosted in the Vulkan backend: no begin/end_commands
 * are needed, the tensors are plain host-mapped buffers.
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <vector>

/* ---- f16 helpers (same decode as ds4.c f16_to_f32) ---- */
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
    return (float)((double)((*s >> 33) & 0x7fffffffu) / (double)0x80000000u) * 2.0f - 1.0f;
}

/* ---- attention_output_low_q8: single-token stage A ---- */
static int test_attention_output_low_q8(void) {
    const uint32_t n_groups  = 2;
    const uint64_t group_dim = 64;          /* multiple of 32 */
    const uint64_t rank      = 16;
    const uint64_t low_dim   = (uint64_t)n_groups * rank;  /* 32 */

    const uint64_t blocks_a = (group_dim + 31u) / 32u;      /* 2 */
    const uint64_t row_a_bytes = blocks_a * 34u;
    const uint64_t out_a_bytes = low_dim * row_a_bytes;
    const uint64_t header = 16;
    const uint64_t model_size = header + out_a_bytes;
    const uint64_t out_a_offset = header;

    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(n_groups * group_dim * sizeof(float));
    ds4_gpu_tensor *low   = ds4_gpu_tensor_alloc(low_dim * sizeof(float));
    if (!heads || !low) {
        if (heads) ds4_gpu_tensor_free(heads);
        if (low) ds4_gpu_tensor_free(low);
        return 1;
    }

    unsigned char *model = (unsigned char *)malloc(model_size);
    if (!model) { ds4_gpu_tensor_free(heads); ds4_gpu_tensor_free(low); return 1; }
    std::memset(model, 0xAA, header);

    /* Synthetic out_a rows (low_dim rows of group_dim f32 values -> Q8_0). */
    uint64_t seed = 0x123456789abcdef0ull;
    std::vector<float> wrow(group_dim);
    for (uint64_t r = 0; r < low_dim; r++) {
        for (uint64_t i = 0; i < group_dim; i++) wrow[i] = lcg_float(&seed) * 0.5f;
        make_q8_0_row(model + out_a_offset + r * row_a_bytes, wrow.data(), group_dim);
    }
    if (ds4_gpu_set_model_map(model, model_size) == 0) {
        free(model); ds4_gpu_tensor_free(heads); ds4_gpu_tensor_free(low); return 1;
    }

    /* Known heads activation. */
    float hv[n_groups * group_dim];
    for (uint64_t i = 0; i < n_groups * group_dim; i++) hv[i] = lcg_float(&seed) * 0.25f;
    if (ds4_gpu_tensor_write(heads, 0, hv, sizeof(hv)) == 0) {
        free(model); ds4_gpu_tensor_free(heads); ds4_gpu_tensor_free(low); return 1;
    }

    /* Reference: quantize each group slice, then dot with out_a rows. */
    std::vector<int8_t> xq(n_groups * blocks_a * 32u);
    std::vector<float> xscale(n_groups * blocks_a);
    for (uint32_t g = 0; g < n_groups; g++)
        quant_q8_0(hv + g * group_dim, xq.data() + g * blocks_a * 32u,
                   xscale.data() + g * blocks_a, group_dim);
    std::vector<float> ref(low_dim);
    for (uint64_t idx = 0; idx < low_dim; idx++) {
        const uint64_t g = idx / rank;
        ref[idx] = dot_q8_0(model + out_a_offset + idx * row_a_bytes,
                            xq.data() + g * blocks_a * 32u,
                            xscale.data() + g * blocks_a,
                            group_dim, blocks_a);
    }

    int rc = 1;
    if (ds4_gpu_attention_output_low_q8_tensor(low, model, model_size, out_a_offset,
                                               group_dim, rank, n_groups, heads) != 0) {
        float outv[low_dim];
        if (ds4_gpu_tensor_read(low, 0, outv, sizeof(outv)) != 0) {
            rc = 0;
            for (uint64_t i = 0; i < low_dim; i++) {
                if (!(std::fabsf(outv[i] - ref[i]) <= 1e-3f)) {
                    fprintf(stderr, "--- attention_output_low_q8 mismatch idx=%llu got=%.6f want=%.6f\n",
                            (unsigned long long)i, outv[i], ref[i]);
                    rc = 1;
                }
            }
        }
    }

    free(model);
    ds4_gpu_tensor_free(low);
    ds4_gpu_tensor_free(heads);
    return rc;
}
REGISTER_TEST(attention_output_low_q8, test_attention_output_low_q8);

/* ---- attention_output_q8_batch: two-stage batch projection ---- */
static int test_attention_output_q8_batch(void) {
    const bool production_shape = getenv("DS4_TEST_PRODUCTION_SHAPE") != nullptr;
    const uint32_t n_groups  = production_shape ? 8u : 2u;
    const uint64_t group_dim = production_shape ? 4096u : 64u;
    const uint64_t rank      = production_shape ? 1024u : 16u;
    const uint32_t n_tokens  = production_shape ? 1u : 3u;
    const uint64_t out_dim   = production_shape ? 4096u : 24u;
    const uint64_t low_dim   = (uint64_t)n_groups * rank;

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
    ds4_gpu_tensor *out   = ds4_gpu_tensor_alloc(n_tokens * out_dim * sizeof(float));
    ds4_gpu_tensor *gt    = ds4_gpu_tensor_alloc(64);
    ds4_gpu_tensor *lt    = ds4_gpu_tensor_alloc(64);
    if (!heads || !low || !out || !gt || !lt) {
        if (heads) ds4_gpu_tensor_free(heads);
        if (low) ds4_gpu_tensor_free(low);
        if (out) ds4_gpu_tensor_free(out);
        if (gt) ds4_gpu_tensor_free(gt);
        if (lt) ds4_gpu_tensor_free(lt);
        return 1;
    }

    unsigned char *model = (unsigned char *)malloc(model_size);
    if (!model) {
        ds4_gpu_tensor_free(heads); ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(gt); ds4_gpu_tensor_free(lt);
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
        ds4_gpu_tensor_free(heads); ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(gt); ds4_gpu_tensor_free(lt);
        return 1;
    }

    /* Known heads activation (n_tokens x n_groups x group_dim). */
    std::vector<float> hv((size_t)n_tokens * n_groups * group_dim);
    for (uint64_t i = 0; i < hv.size(); i++) hv[i] = lcg_float(&seed) * 0.25f;
    if (ds4_gpu_tensor_write(heads, 0, hv.data(), hv.size() * sizeof(float)) == 0) {
        free(model);
        ds4_gpu_tensor_free(heads); ds4_gpu_tensor_free(low); ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(gt); ds4_gpu_tensor_free(lt);
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

    int rc = 1;
    int gpu_ok = 1;
    if (production_shape)
        gpu_ok = ds4_gpu_attention_output_q8_batch_tensor(out, low, gt, lt,
                                                          model, model_size,
                                                          out_a_offset, out_b_offset,
                                                          group_dim, rank, n_groups, out_dim,
                                                          heads, n_tokens);
    if (gpu_ok && ds4_gpu_attention_output_q8_batch_tensor(out, low, gt, lt,
                                                 model, model_size,
                                                 out_a_offset, out_b_offset,
                                                 group_dim, rank, n_groups, out_dim,
                                                 heads, n_tokens) != 0) {
        std::vector<float> outv((size_t)n_tokens * out_dim);
        std::vector<float> lowv((size_t)n_tokens * low_dim);
        if (ds4_gpu_tensor_read(out, 0, outv.data(), outv.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(low, 0, lowv.data(), lowv.size() * sizeof(float)) != 0) {
            rc = 0;
            for (uint64_t t = 0; t < n_tokens && rc == 0; t++) {
                for (uint64_t o = 0; o < out_dim; o++) {
                    float got = outv[(size_t)t * out_dim + o];
                    float want = ref_out[(size_t)t * out_dim + o];
                    if (!(std::fabsf(got - want) <= 1e-3f)) {
                        fprintf(stderr, "--- attention_output_q8_batch out[t=%llu][o=%llu] got=%.6f want=%.6f\n",
                                (unsigned long long)t, (unsigned long long)o, got, want);
                        rc = 1;
                    }
                }
                for (uint64_t i = 0; i < low_dim; i++) {
                    float got = lowv[(size_t)t * low_dim + i];
                    float want = ref_low[(size_t)t * low_dim + i];
                    if (!(std::fabsf(got - want) <= 1e-3f)) {
                        fprintf(stderr, "--- attention_output_q8_batch low[t=%llu][i=%llu] got=%.6f want=%.6f\n",
                                (unsigned long long)t, (unsigned long long)i, got, want);
                        rc = 1;
                    }
                }
            }
        }
    }

    free(model);
    ds4_gpu_tensor_free(lt);
    ds4_gpu_tensor_free(gt);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(low);
    ds4_gpu_tensor_free(heads);
    return rc;
}
REGISTER_TEST(attention_output_q8_batch, test_attention_output_q8_batch);
