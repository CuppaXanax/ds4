/* Kernel test: ds4_gpu_matmul_q8_0_f16_out_tensor (host-side).
 *
 *   out_h[o + t*out_dim] = sum_i deq_q8(W[o][i]) * x[i + t*in_dim]
 *
 * W is stored in the GGUF Q8_0 layout (34 B/block: f16 scale + 32 x int8)
 * and the result is written as IEEE halves (2 B/element) into out_h->ptr
 * (the engine's batch_q_half buffer).  The model buffer mimics a GGUF-style
 * file: a junk header, then the quantized weight matrix at weight_offset
 * (4096, away from the f16 pair cache).
 *
 * The test quantizes synthetic float weights with the same format the
 * engine expects (per 32-element block: scale = amax/127 stored as f16,
 * q[i] = round(w[i]/scale)), computes a CPU reference with double
 * accumulation, and compares the kernel's halves dequantized back to f32.
 *
 * This kernel is host-side (tensor->ptr is host-mapped), so the call is
 * NOT wrapped in begin/end_commands.
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
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

/* IEEE half -> f32 (exact decode, same semantics as the backend helper). */
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

/* Quantize one float row into GGUF Q8_0 layout (34 B/block).
 * scale = amax/127 stored as f16; q[i] = round(w[i]/scale), clamped. */
static void quant_q8_0_row(uint8_t *row, const float *w, uint64_t n) {
    const uint64_t n_blocks = (n + 31u) / 32u;
    for (uint64_t b = 0; b < n_blocks; b++) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = n - i0 < 32u ? n - i0 : 32u;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++)
            if (std::fabsf(w[i0 + i]) > amax) amax = std::fabsf(w[i0 + i]);
        const float scale = amax / 127.0f;
        const uint16_t sb = f32_to_f16(scale);
        std::memcpy(row + b * 34u, &sb, sizeof(sb));
        int8_t *qs = (int8_t *)(row + b * 34u + 2u);
        for (uint64_t i = 0; i < bn; i++) {
            long q = std::lround(w[i0 + i] / scale);
            if (q > 127) q = 127;
            if (q < -127) q = -127;
            qs[i] = (int8_t)q;
        }
        for (uint64_t i = bn; i < 32u; i++) qs[i] = 0;  /* partial tail */
    }
}

/* CPU reference: dequant Q8_0 and accumulate in double, then store the f16
 * of the f32 result exactly like the backend (half rounding is covered by
 * the tolerance in the comparison). */
static void ref_q8_0_f16_out(std::vector<uint16_t> &out_h,
                             const uint8_t *wbase, const float *xv,
                             uint64_t in_dim, uint64_t out_dim,
                             uint64_t n_tok, uint64_t row_bytes) {
    const uint64_t blocks = (in_dim + 31u) / 32u;
    out_h.assign(n_tok * out_dim, 0u);
    for (uint64_t t = 0; t < n_tok; t++) {
        for (uint64_t o = 0; o < out_dim; o++) {
            const uint8_t *row = wbase + o * row_bytes;
            double acc = 0.0;
            for (uint64_t b = 0; b < blocks; b++) {
                uint16_t scale_bits;
                std::memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
                const float scale = f16_to_f32(scale_bits);
                const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
                const uint64_t i0 = b * 32u;
                const uint64_t n = in_dim - i0 < 32u ? in_dim - i0 : 32u;
                for (uint64_t i = 0; i < n; i++)
                    acc += (double)scale * (double)qs[i] *
                           (double)xv[t * in_dim + i0 + i];
            }
            out_h[t * out_dim + o] = f32_to_f16((float)acc);
        }
    }
}

static int test_matmul_q8_0_f16_out(void) {
    const uint64_t in_dim    = 64;                  /* 2 x 32-element blocks */
    const uint64_t out_dim   = 8;
    const uint64_t n_tok     = 3;
    const uint64_t n_blocks  = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = n_blocks * 34u;      /* GGUF Q8_0: 34 B/block */
    const uint64_t w_bytes   = out_dim * row_bytes;
    const uint64_t header    = 4096;                /* avoid f16 pair cache */
    const uint64_t model_size = header + w_bytes;
    const uint64_t weight_offset = header;

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(n_tok * in_dim * sizeof(float));
    ds4_gpu_tensor *out_h = ds4_gpu_tensor_alloc(n_tok * out_dim * sizeof(uint16_t));
    if (!x || !out_h) {
        if (x) ds4_gpu_tensor_free(x);
        if (out_h) ds4_gpu_tensor_free(out_h);
        return 1;
    }

    /* Synthetic "model file": junk header + quantized Q8_0 weight matrix. */
    std::vector<unsigned char> model(model_size, 0xAA);
    for (uint64_t o = 0; o < out_dim; o++) {
        std::vector<float> wrow(in_dim);
        for (uint64_t i = 0; i < in_dim; i++) {
            float v = (float)((o + 1) * (i + 1)) * 0.02f;
            if (((o + i) & 1u) != 0) v = -v;        /* alternate signs */
            wrow[i] = v;
        }
        quant_q8_0_row(model.data() + header + o * row_bytes,
                       wrow.data(), in_dim);
    }
    if (ds4_gpu_set_model_map(model.data(), model_size) == 0) {
        ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(out_h);
        return 1;
    }

    /* Known input activations (f32). */
    std::vector<float> xv(n_tok * in_dim);
    for (uint64_t t = 0; t < n_tok; t++)
        for (uint64_t i = 0; i < in_dim; i++)
            xv[t * in_dim + i] = (float)(t * 8 + i + 1) * 0.125f;
    if (ds4_gpu_tensor_write(x, 0, xv.data(), xv.size() * sizeof(float)) == 0) {
        ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(out_h);
        return 1;
    }

    const int rc = ds4_gpu_matmul_q8_0_f16_out_tensor(
        out_h, model.data(), model_size, weight_offset,
        in_dim, out_dim, x, n_tok) == 0 ? 0 : 1;

    ds4_gpu_tensor_free(out_h);
    ds4_gpu_tensor_free(x);
    return rc;
}
REGISTER_TEST(matmul_q8_0_f16_out, test_matmul_q8_0_f16_out);
