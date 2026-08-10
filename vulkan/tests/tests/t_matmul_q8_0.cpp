/* Kernel test: ds4_gpu_matmul_q8_0_tensor (real GPU dispatch of matmul_q8_0).
 *
 * out[t][o] = sum_i x[t][i] * W_q8_0[o][i]
 *
 * Weights are stored in the GGUF Q8_0 layout used by ds4.c / ds4_test.c:
 * per 32-element block, { f16 scale (2 bytes), int8 qs[32] (32 bytes) } =
 * 34 bytes per block.  The model buffer mimics a GGUF-style file: a junk
 * header, then the raw Q8_0 weights at weight_offset.
 *
 * weight_offset is 4096 (not 16): the backend's weight cache persists across
 * harness tests, so a range starting at 16 would alias t_matmul_f16's cached
 * entry and produce wrong data (or a RADV descriptor-range crash).
 *
 * The CPU reference quantizes activations per 32-value block before taking
 * integer dots, matching ds4.c matvec_q8_0.
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>

/* f32 -> IEEE half (round toward zero; test values are normal-range). */
static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t e    = (x >> 23) & 0xffu;
    uint32_t m    = x & 0x7fffffu;
    if (e == 0xffu) return (uint16_t)(sign | 0x7c00u | (m ? 0x200u : 0u));
    if (e == 0u)    return (uint16_t)sign;              /* ±0 / flush subnormal */
    int32_t e16 = (int32_t)e - 127 + 15;
    if (e16 >= 31)  return (uint16_t)(sign | 0x7c00u);  /* ±inf */
    if (e16 <= 0) {                                     /* subnormal f16 */
        uint32_t shift = 126u - e;
        uint32_t m16 = (0x800000u | m) >> shift;
        return (uint16_t)(sign | m16);
    }
    return (uint16_t)(sign | ((uint32_t)e16 << 10) | (m >> 13));
}

/* IEEE half -> f32 (exact decode, same semantics as the shader). */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e    = (h >> 10) & 0x1fu;
    uint32_t m    = h & 0x3ffu;
    uint32_t bits;
    if (e == 0u) {
        if (m == 0u) bits = sign;                       /* ±0 */
        else {                                          /* subnormal */
            uint32_t mm = m, p = 0u;
            while (!(mm & 0x400u) && p < 10u) { mm <<= 1; p++; }
            bits = sign | ((113u - p) << 23) | ((mm & 0x3ffu) << 13);
        }
    } else if (e == 31u) {
        bits = sign | 0x7f800000u | (m << 13);          /* ±inf/nan */
    } else {
        bits = sign | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static int test_matmul_q8_0(void) {
    const uint64_t in_dim    = 4096;                /* Flash Q_A width: 128 blocks */
    const uint64_t out_dim   = 8;
    const uint64_t n_tok     = 2;
    const uint64_t n_blocks  = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = n_blocks * 34u;      /* GGUF Q8_0: 34 B/block */
    const uint64_t w_bytes   = out_dim * row_bytes;
    const uint64_t header    = 4096;                /* avoid t_matmul_f16's cache @16 */
    const uint64_t model_size = header + w_bytes;
    const uint64_t weight_offset = header;

    ds4_gpu_tensor *x   = ds4_gpu_tensor_alloc(n_tok * in_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(n_tok * out_dim * sizeof(float));
    if (!x || !out) {
        if (x) ds4_gpu_tensor_free(x);
        if (out) ds4_gpu_tensor_free(out);
        return 1;
    }

    /* Synthetic "model file": junk header + Q8_0 weight matrix. */
    unsigned char *model = (unsigned char*)malloc(model_size);
    if (!model) { ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(out); return 1; }
    std::memset(model, 0xAA, header);
    for (uint64_t o = 0; o < out_dim; o++) {
        uint8_t *row = model + header + o * row_bytes;
        for (uint64_t b = 0; b < n_blocks; b++) {
            const float scale = (float)(o + 1) * 0.25f;   /* f16-representable */
            const uint16_t sb = f32_to_f16(scale);
            std::memcpy(row + b * 34u, &sb, sizeof(sb));
            int8_t *qs = (int8_t *)(row + b * 34u + 2u);
            for (uint32_t i = 0; i < 32; i++) {
                const uint32_t k = (uint32_t)(b * 32u + i);
                qs[i] = (int8_t)(int)((((uint32_t)o + 1u) * 31u +
                                       k * 17u + ((uint32_t)o ^ k) * 7u) % 65u) - 32;
            }
        }
    }
    if (ds4_gpu_set_model_map(model, model_size) == 0) {
        free(model); ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(out); return 1;
    }

    /* Known input activations (f32). */
    float xv[n_tok * in_dim];
    for (uint64_t t = 0; t < n_tok; t++) {
        for (uint64_t i = 0; i < in_dim; i++) {
            const int value = (int)((i * 37u + t * 19u) % 257u) - 128;
            xv[t * in_dim + i] = (float)value * 0.00075f;
        }
    }
    if (ds4_gpu_tensor_write(x, 0, xv, sizeof(xv)) == 0) {
        free(model); ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(out); return 1;
    }

    /* CPU reference: blockwise Q8_0 activation quantization followed by the
     * scaled integer dot used by ds4.c dot_q8_0_row. */
    float ref[n_tok * out_dim];
    for (uint64_t t = 0; t < n_tok; t++) {
        int8_t xq[in_dim];
        float xscale[n_blocks];
        for (uint64_t b = 0; b < n_blocks; b++) {
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32; i++)
                amax = std::fmaxf(amax, std::fabsf(xv[t * in_dim + b * 32u + i]));
            const float d = amax / 127.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            xscale[b] = d;
            for (uint32_t i = 0; i < 32; i++) {
                int q = (int)std::lrintf(xv[t * in_dim + b * 32u + i] * id);
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                xq[b * 32u + i] = (int8_t)q;
            }
        }
        for (uint64_t o = 0; o < out_dim; o++) {
            const uint8_t *row = model + header + o * row_bytes;
            float acc = 0.0f;
            for (uint64_t b = 0; b < n_blocks; b++) {
                uint16_t scale_bits;
                std::memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
                const float scale = f16_to_f32(scale_bits);
                const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
                int32_t dot = 0;
                for (uint32_t i = 0; i < 32; i++)
                    dot += (int32_t)qs[i] * (int32_t)xq[b * 32u + i];
                acc += scale * xscale[b] * (float)dot;
            }
            ref[t * out_dim + o] = acc;
        }
    }

    int rc = 1;
    if (ds4_gpu_begin_commands() != 0 &&
        ds4_gpu_matmul_q8_0_tensor(out, model, model_size, weight_offset,
                                   in_dim, out_dim, x, n_tok) != 0 &&
        ds4_gpu_end_commands() != 0) {
        ds4_gpu_synchronize();              /* wait for GPU before reading */
        float outv[n_tok * out_dim];
        if (ds4_gpu_tensor_read(out, 0, outv, sizeof(outv)) != 0) {
            fprintf(stderr, "--- matmul_q8_0 diagnostic ---\n");
            for (uint64_t o = 0; o < out_dim; o++) {
                const uint8_t *row = model + header + o * row_bytes;
                uint16_t scale_bits;
                std::memcpy(&scale_bits, row, sizeof(scale_bits));
                fprintf(stderr, "W[%llu] scale=%.3f qs0[0..7]: ",
                        (unsigned long long)o, f16_to_f32(scale_bits));
                const int8_t *qs = (const int8_t *)(row + 2u);
                for (uint32_t i = 0; i < 8; i++)
                    fprintf(stderr, "%d ", (int)qs[i]);
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "X[0]: ");
            for (uint64_t i = 0; i < 64; i++)
                fprintf(stderr, "%.3f ", xv[i]);
            fprintf(stderr, "\n");
            for (uint64_t t = 0; t < n_tok; t++) {
                for (uint64_t o = 0; o < out_dim; o++) {
                    fprintf(stderr, "out[%llu][%llu] got=%.6f want=%.6f %s\n",
                            (unsigned long long)t, (unsigned long long)o,
                            outv[t * out_dim + o], ref[t * out_dim + o],
                            std::fabsf(outv[t * out_dim + o] - ref[t * out_dim + o]) <= 1e-6f ? "ok" : "MISMATCH");
                }
            }
            rc = 0;
            for (uint64_t t = 0; t < n_tok && rc == 0; t++) {
                for (uint64_t o = 0; o < out_dim; o++) {
                    float got = outv[t * out_dim + o];
                    float want = ref[t * out_dim + o];
                    if (!(std::fabsf(got - want) <= 1e-6f)) rc = 1;
                }
            }
        }
    }

    free(model);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    return rc;
}
REGISTER_TEST(matmul_q8_0, test_matmul_q8_0);

static int test_matmul_q8_0_rejects_unbounded_batch(void) {
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(float));
    if (!x || !out) {
        if (x) ds4_gpu_tensor_free(x);
        if (out) ds4_gpu_tensor_free(out);
        return 1;
    }
    const int result = ds4_gpu_matmul_q8_0_tensor(
        out, nullptr, 0, 0, 1, 32768, x, 9);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    return result == 0 ? 0 : 1;
}
REGISTER_TEST(matmul_q8_0_rejects_unbounded_batch,
              test_matmul_q8_0_rejects_unbounded_batch);
