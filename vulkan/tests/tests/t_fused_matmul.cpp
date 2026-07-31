/* Kernel tests for the fused Q8_0 matmul family (host-side in the backend).
 *
 * Covered kernels:
 *
 *   ds4_gpu_matmul_q8_0_pair_tensor        (W0 @ x, W1 @ x, two projections)
 *   ds4_gpu_matmul_q8_0_hc_expand_tensor   (block_out = W @ x, then
 *                                           hc_post_one(block_out, residual,
 *                                           post/comb from split))
 *
 * Both implementations are host-side CPU loops (like ds4_gpu_add_tensor and
 * the rest of the HC family), so the tests do NOT wrap the calls in
 * begin/end_commands.  The matmul reference dequantizes the GGUF Q8_0
 * layout exactly like the verified matmul_q8_0 kernel (f16 block scale,
 * int8 quants, raw f32 activations, double accumulation); the HC reference
 * is hc_post_one copied verbatim from ds4.c.
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

/* Deterministic synthetic value: in-range, alternating signs. */
static float synth_float(uint32_t i, float step) {
    float f = (float)(i + 1) * step + 0.0625f * (float)(i % 5);
    if ((i & 1u) != 0) f = -f;
    return f;
}

static int check_f32(const char *what, const float *got, const float *want,
                     uint64_t n, float tol) {
    int bad = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (!(std::fabsf(got[i] - want[i]) <= tol)) {
            if (bad < 8)
                fprintf(stderr, "--- %s mismatch[%llu]: got %.6f want %.6f\n",
                        what, (unsigned long long)i, got[i], want[i]);
            bad++;
        }
    }
    if (bad) fprintf(stderr, "--- %s: %d mismatches\n", what, bad);
    return bad ? 1 : 0;
}

/* Build the synthetic "model file": junk header + Q8_0 matrices.
 * Each matrix row is n_blocks * 34 bytes (f16 scale + 32 x int8). */
static unsigned char *make_q8_0_model(uint64_t header, uint64_t out_dim0,
                                      uint64_t out_dim1, uint64_t n_blocks,
                                      uint64_t *model_size,
                                      uint64_t *off0, uint64_t *off1) {
    const uint64_t row_bytes = n_blocks * 34u;
    *off0 = header;
    *off1 = header + out_dim0 * row_bytes;
    const uint64_t total = header + (out_dim0 + out_dim1) * row_bytes;
    unsigned char *model = (unsigned char *)malloc(total);
    if (!model) return nullptr;
    std::memset(model, 0xAA, header);
    uint64_t off = *off0;
    for (uint64_t w = 0; w < 2; w++) {
        const uint64_t out_dim = w == 0 ? out_dim0 : out_dim1;
        for (uint64_t o = 0; o < out_dim; o++) {
            uint8_t *row = model + off + o * row_bytes;
            for (uint64_t b = 0; b < n_blocks; b++) {
                const float scale = (float)(o + 1 + w * 2) * 0.25f;
                const uint16_t sb = f32_to_f16(scale);
                std::memcpy(row + b * 34u, &sb, sizeof(sb));
                int8_t *qs = (int8_t *)(row + b * 34u + 2u);
                for (uint32_t i = 0; i < 32; i++) {
                    const uint32_t k = (uint32_t)(b * 32u + i);
                    qs[i] = (int8_t)(int)((((uint32_t)o + 1u + w * 5u) * 31u +
                                           k * 17u + ((uint32_t)o ^ k) * 7u) % 65u) - 32;
                }
            }
        }
        off = *off1;
    }
    *model_size = total;
    return model;
}

/* CPU reference for one Q8_0 row x vector: same math as the backend /
 * matmul_q8_0 shader (f16 scale, int8 quants, double accumulation). */
static float ref_q8_0_dot(const uint8_t *row, const float *xp,
                          uint64_t in_dim, uint64_t n_blocks) {
    double acc = 0.0;
    for (uint64_t b = 0; b < n_blocks; b++) {
        uint16_t scale_bits;
        std::memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
        const float scale = f16_to_f32(scale_bits);
        const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
        const uint64_t i0 = b * 32u;
        const uint64_t n = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        for (uint64_t i = 0; i < n; i++)
            acc += (double)scale * (double)qs[i] * (double)xp[i0 + i];
    }
    return (float)acc;
}

/* ---------- matmul_q8_0_pair ---------- */

static int test_matmul_q8_0_pair(void) {
    const uint64_t in_dim  = 64;                  /* 2 x 32-element blocks */
    const uint64_t out0_dim = 6;
    const uint64_t out1_dim = 5;
    const uint64_t n_tok    = 2;
    const uint64_t n_blocks = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = n_blocks * 34u;
    const uint64_t header = 4096;                 /* avoid other tests' cache @16 */

    uint64_t model_size = 0, off0 = 0, off1 = 0;
    unsigned char *model = make_q8_0_model(header, out0_dim, out1_dim,
                                           n_blocks, &model_size, &off0, &off1);
    if (!model) return 1;
    if (ds4_gpu_set_model_map(model, model_size) == 0) {
        free(model); return 1;
    }

    ds4_gpu_tensor *x    = ds4_gpu_tensor_alloc(n_tok * in_dim * sizeof(float));
    ds4_gpu_tensor *out0 = ds4_gpu_tensor_alloc(n_tok * out0_dim * sizeof(float));
    ds4_gpu_tensor *out1 = ds4_gpu_tensor_alloc(n_tok * out1_dim * sizeof(float));
    if (!x || !out0 || !out1) {
        if (x) ds4_gpu_tensor_free(x);
        if (out0) ds4_gpu_tensor_free(out0);
        if (out1) ds4_gpu_tensor_free(out1);
        free(model); return 1;
    }

    std::vector<float> xv(n_tok * in_dim);
    for (uint64_t t = 0; t < n_tok; t++)
        for (uint64_t i = 0; i < in_dim; i++)
            xv[t * in_dim + i] = synth_float((uint32_t)(t * 64 + i), 0.125f);
    if (ds4_gpu_tensor_write(x, 0, xv.data(), xv.size() * sizeof(float)) == 0) {
        ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(out0); ds4_gpu_tensor_free(out1);
        free(model); return 1;
    }

    /* CPU reference. */
    std::vector<float> ref0(n_tok * out0_dim), ref1(n_tok * out1_dim);
    for (uint64_t t = 0; t < n_tok; t++)
        for (uint64_t o = 0; o < out0_dim; o++)
            ref0[t * out0_dim + o] =
                ref_q8_0_dot(model + off0 + o * row_bytes,
                             &xv[t * in_dim], in_dim, n_blocks);
    for (uint64_t t = 0; t < n_tok; t++)
        for (uint64_t o = 0; o < out1_dim; o++)
            ref1[t * out1_dim + o] =
                ref_q8_0_dot(model + off1 + o * row_bytes,
                             &xv[t * in_dim], in_dim, n_blocks);

    int rc = 1;
    if (ds4_gpu_matmul_q8_0_pair_tensor(out0, out1, model, model_size,
                                        off0, off1, in_dim,
                                        out0_dim, out1_dim, x, n_tok) != 0) {
        std::vector<float> got0(n_tok * out0_dim), got1(n_tok * out1_dim);
        if (ds4_gpu_tensor_read(out0, 0, got0.data(), got0.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(out1, 0, got1.data(), got1.size() * sizeof(float)) != 0) {
            rc = check_f32("matmul_q8_0_pair/out0", got0.data(), ref0.data(),
                           n_tok * out0_dim, 1e-2f);
            if (rc == 0)
                rc = check_f32("matmul_q8_0_pair/out1", got1.data(), ref1.data(),
                               n_tok * out1_dim, 1e-2f);
        }
    }
    /* Safety: out-of-range weight offset must fail cleanly. */
    if (rc == 0 &&
        ds4_gpu_matmul_q8_0_pair_tensor(out0, out1, model, model_size,
                                        model_size + 1, off1, in_dim,
                                        out0_dim, out1_dim, x, n_tok) != 0) {
        fprintf(stderr, "--- matmul_q8_0_pair: out-of-range offset accepted\n");
        rc = 1;
    }

    ds4_gpu_tensor_free(out1); ds4_gpu_tensor_free(out0); ds4_gpu_tensor_free(x);
    free(model);
    return rc;
}
REGISTER_TEST(matmul_q8_0_pair, test_matmul_q8_0_pair);

/* ---------- matmul_q8_0_hc_expand ---------- */

static int test_matmul_q8_0_hc_expand(void) {
    const uint32_t n_hc = 4, n_embd = 8;
    const uint64_t in_dim = 64;
    const uint64_t out_dim = n_embd;              /* block_out width */
    const uint64_t n_blocks = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = n_blocks * 34u;
    const uint64_t header = 4096;
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t hc_dim = (uint64_t)n_hc * n_embd;

    const uint64_t model_size = header + out_dim * row_bytes;
    unsigned char *model = (unsigned char *)malloc(model_size);
    if (!model) return 1;
    std::memset(model, 0xAA, header);
    for (uint64_t o = 0; o < out_dim; o++) {
        uint8_t *row = model + header + o * row_bytes;
        for (uint64_t b = 0; b < n_blocks; b++) {
            const float scale = (float)(o + 1) * 0.25f;
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
    if (ds4_gpu_set_model_map(model, model_size) == 0) { free(model); return 1; }

    ds4_gpu_tensor *out_hc      = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    ds4_gpu_tensor *block_out   = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    ds4_gpu_tensor *x           = ds4_gpu_tensor_alloc(in_dim * sizeof(float));
    ds4_gpu_tensor *residual_hc = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    ds4_gpu_tensor *split       = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    if (!out_hc || !block_out || !x || !residual_hc || !split) {
        if (out_hc) ds4_gpu_tensor_free(out_hc);
        if (block_out) ds4_gpu_tensor_free(block_out);
        if (x) ds4_gpu_tensor_free(x);
        if (residual_hc) ds4_gpu_tensor_free(residual_hc);
        if (split) ds4_gpu_tensor_free(split);
        free(model); return 1;
    }

    std::vector<float> xv(in_dim), rh(hc_dim), sp(mix_hc);
    for (uint64_t i = 0; i < in_dim; i++) xv[i] = synth_float((uint32_t)i, 0.20f);
    for (uint32_t i = 0; i < (uint32_t)hc_dim; i++) rh[i] = synth_float(i, 0.15f);
    for (uint32_t i = 0; i < (uint32_t)mix_hc; i++) sp[i] = synth_float(i, 0.08f);
    if (ds4_gpu_tensor_write(x, 0, xv.data(), xv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(residual_hc, 0, rh.data(), rh.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(split, 0, sp.data(), sp.size() * sizeof(float)) == 0) {
        ds4_gpu_tensor_free(out_hc); ds4_gpu_tensor_free(block_out);
        ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(residual_hc);
        ds4_gpu_tensor_free(split); free(model); return 1;
    }

    /* Reference: block = W @ x (same dequant math), then hc_post_one with
     * post = split[n_hc .. 2*n_hc), comb = split[2*n_hc .. + n_hc*n_hc)
     * addressed [dst + src*n_hc] (ds4.c hc_post_one). */
    std::vector<float> want_block(out_dim), want_hc(hc_dim);
    for (uint64_t o = 0; o < out_dim; o++)
        want_block[o] = ref_q8_0_dot(model + header + o * row_bytes,
                                     xv.data(), in_dim, n_blocks);
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        for (uint32_t d = 0; d < n_embd; d++) {
            float acc = want_block[d] * sp[n_hc + dst];
            for (uint32_t src = 0; src < n_hc; src++)
                acc += sp[2u * n_hc + dst + src * n_hc] * rh[(uint64_t)src * n_embd + d];
            want_hc[(uint64_t)dst * n_embd + d] = acc;
        }
    }

    int rc = 1;
    if (ds4_gpu_matmul_q8_0_hc_expand_tensor(out_hc, block_out, model, model_size,
                                             header, in_dim, out_dim, x,
                                             residual_hc, split, n_embd, n_hc) != 0) {
        std::vector<float> got_block(out_dim), got_hc(hc_dim);
        if (ds4_gpu_tensor_read(block_out, 0, got_block.data(),
                                got_block.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(out_hc, 0, got_hc.data(),
                                got_hc.size() * sizeof(float)) != 0) {
            rc = check_f32("matmul_q8_0_hc_expand/block", got_block.data(),
                           want_block.data(), out_dim, 1e-2f);
            if (rc == 0)
                rc = check_f32("matmul_q8_0_hc_expand/out_hc", got_hc.data(),
                               want_hc.data(), hc_dim, 1e-2f);
        }
    }
    /* Safety: out-of-range weight offset must fail cleanly. */
    if (rc == 0 &&
        ds4_gpu_matmul_q8_0_hc_expand_tensor(out_hc, block_out, model, model_size,
                                             model_size + 1, in_dim, out_dim, x,
                                             residual_hc, split, n_embd, n_hc) != 0) {
        fprintf(stderr, "--- matmul_q8_0_hc_expand: out-of-range offset accepted\n");
        rc = 1;
    }

    ds4_gpu_tensor_free(split); ds4_gpu_tensor_free(residual_hc);
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(block_out);
    ds4_gpu_tensor_free(out_hc);
    free(model);
    return rc;
}
REGISTER_TEST(matmul_q8_0_hc_expand, test_matmul_q8_0_hc_expand);
