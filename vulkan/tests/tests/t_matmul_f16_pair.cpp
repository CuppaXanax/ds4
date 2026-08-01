/* Kernel test: ds4_gpu_matmul_f16_pair_tensor (host-side).
 *
 *   out_a[o + t*out_dim] = sum_i deq_f16(Wa[o][i]) * x[i + t*in_dim]
 *   out_b[o + t*out_dim] = sum_i deq_f16(Wb[o][i]) * x[i + t*in_dim]
 *
 * Both weight matrices are IEEE halves (2 B/element, row-major
 * [out_dim][in_dim]) stored in the "model" buffer as [junk header][Wa][Wb];
 * outputs are f32.  This is the engine's paired compressor / indexer
 * KV+score projection (n_tok may be > 1, e.g. the ratio-4 prefill tail).
 *
 * The CPU reference decodes the stored f16 weights and accumulates in
 * double exactly like the backend helper, so the comparison only needs to
 * absorb float rounding.
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

/* CPU reference: paired f16 matmul, double accumulation. */
static void ref_f16_pair(std::vector<float> &out_a, std::vector<float> &out_b,
                         const uint16_t *wa, const uint16_t *wb,
                         const float *xv, uint64_t in_dim, uint64_t out_dim,
                         uint64_t n_tok) {
    out_a.assign(n_tok * out_dim, 0.0f);
    out_b.assign(n_tok * out_dim, 0.0f);
    for (uint64_t t = 0; t < n_tok; t++) {
        for (uint64_t o = 0; o < out_dim; o++) {
            const uint16_t *ra = wa + o * in_dim;
            const uint16_t *rb = wb + o * in_dim;
            double acca = 0.0, accb = 0.0;
            for (uint64_t i = 0; i < in_dim; i++) {
                acca += (double)f16_to_f32(ra[i]) * (double)xv[t * in_dim + i];
                accb += (double)f16_to_f32(rb[i]) * (double)xv[t * in_dim + i];
            }
            out_a[t * out_dim + o] = (float)acca;
            out_b[t * out_dim + o] = (float)accb;
        }
    }
}

static int test_matmul_f16_pair(void) {
    const uint64_t in_dim  = 32;
    const uint64_t out_dim = 8;
    const uint64_t n_tok   = 3;                  /* multi-token, like the tail */
    const uint64_t w_bytes = in_dim * out_dim * 2u;   /* f16: 2 B/elem */
    const uint64_t header  = 16;                 /* mimic file header */
    const uint64_t model_size = header + 2 * w_bytes;
    const uint64_t wa_offset = header;
    const uint64_t wb_offset = header + w_bytes;

    ds4_gpu_tensor *x     = ds4_gpu_tensor_alloc(n_tok * in_dim * sizeof(float));
    ds4_gpu_tensor *out_a = ds4_gpu_tensor_alloc(n_tok * out_dim * sizeof(float));
    ds4_gpu_tensor *out_b = ds4_gpu_tensor_alloc(n_tok * out_dim * sizeof(float));
    if (!x || !out_a || !out_b) {
        if (x) ds4_gpu_tensor_free(x);
        if (out_a) ds4_gpu_tensor_free(out_a);
        if (out_b) ds4_gpu_tensor_free(out_b);
        return 1;
    }

    /* Synthetic "model file": junk header + two f16 weight matrices. */
    std::vector<unsigned char> model(model_size, 0xAA);
    uint16_t *wa = (uint16_t *)(model.data() + wa_offset);
    uint16_t *wb = (uint16_t *)(model.data() + wb_offset);
    for (uint64_t o = 0; o < out_dim; o++) {
        for (uint64_t i = 0; i < in_dim; i++) {
            float v = (float)((o + 1) * (i + 1)) * 0.01f;
            if (((o + i) & 1u) != 0) v = -v;            /* alternate signs */
            wa[o * in_dim + i] = f32_to_f16(v);
            float s = (float)((o + 1) * (i + 1)) * 0.0075f + 0.25f;
            if (((o + i * 2u) & 1u) != 0) s = -s;       /* different signs */
            wb[o * in_dim + i] = f32_to_f16(s);
        }
    }
    if (ds4_gpu_set_model_map(model.data(), model_size) == 0) {
        ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(out_a); ds4_gpu_tensor_free(out_b);
        return 1;
    }

    /* Known input activations (f32). */
    std::vector<float> xv(n_tok * in_dim);
    for (uint64_t t = 0; t < n_tok; t++)
        for (uint64_t i = 0; i < in_dim; i++) {
            float f = (float)(t * 16 + i + 1) * 0.375f + 0.125f * (float)(i % 7);
            if ((i & 1u) != 0) f = -f;
            xv[t * in_dim + i] = f;
        }
    if (ds4_gpu_tensor_write(x, 0, xv.data(), xv.size() * sizeof(float)) == 0) {
        ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(out_a); ds4_gpu_tensor_free(out_b);
        return 1;
    }

    std::vector<float> refa, refb;
    ref_f16_pair(refa, refb, wa, wb, xv.data(), in_dim, out_dim, n_tok);

    int rc = 1;
    if (ds4_gpu_matmul_f16_pair_tensor(out_a, out_b, model.data(), model_size,
                                       wa_offset, wb_offset, in_dim, out_dim,
                                       x, n_tok) != 1) {
        fprintf(stderr, "--- matmul_f16_pair: kernel returned != 1\n");
    } else {
        std::vector<float> gota(n_tok * out_dim), gotb(n_tok * out_dim);
        if (ds4_gpu_tensor_read(out_a, 0, gota.data(), gota.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(out_b, 0, gotb.data(), gotb.size() * sizeof(float)) != 0) {
            rc = 0;
            for (uint64_t t = 0; t < n_tok && rc == 0; t++) {
                for (uint64_t o = 0; o < out_dim; o++) {
                    const float ga = gota[t * out_dim + o];
                    const float gb = gotb[t * out_dim + o];
                    const float wa_ = refa[t * out_dim + o];
                    const float wb_ = refb[t * out_dim + o];
                    if (!(std::fabsf(ga - wa_) <= 1e-3f * (1.0f + std::fabsf(wa_)))) {
                        fprintf(stderr,
                                "--- matmul_f16_pair A[%llu][%llu]: "
                                "got=%.6f want=%.6f\n",
                                (unsigned long long)t, (unsigned long long)o,
                                ga, wa_);
                        rc = 1;
                    } else if (!(std::fabsf(gb - wb_) <= 1e-3f * (1.0f + std::fabsf(wb_)))) {
                        fprintf(stderr,
                                "--- matmul_f16_pair B[%llu][%llu]: "
                                "got=%.6f want=%.6f\n",
                                (unsigned long long)t, (unsigned long long)o,
                                gb, wb_);
                        rc = 1;
                    }
                }
            }
        }
    }

    /* Bounds / safety: invalid args must be rejected with -1. */
    if (ds4_gpu_matmul_f16_pair_tensor(out_a, out_b, model.data(), model_size,
                                       model_size + 4, wb_offset, in_dim,
                                       out_dim, x, n_tok) != -1) {
        fprintf(stderr, "--- matmul_f16_pair: out-of-range A offset not rejected\n");
        rc = 1;
    }
    if (ds4_gpu_matmul_f16_pair_tensor(out_a, out_b, model.data(), model_size,
                                       wa_offset, wb_offset, in_dim, out_dim,
                                       x, 0) != -1) {
        fprintf(stderr, "--- matmul_f16_pair: n_tok=0 not rejected\n");
        rc = 1;
    }
    if (ds4_gpu_matmul_f16_pair_tensor(out_a, out_b, nullptr, 0,
                                       wa_offset, wb_offset, in_dim, out_dim,
                                       x, n_tok) != -1) {
        fprintf(stderr, "--- matmul_f16_pair: null model_map not rejected\n");
        rc = 1;
    }

    ds4_gpu_tensor_free(out_b);
    ds4_gpu_tensor_free(out_a);
    ds4_gpu_tensor_free(x);
    return rc;
}
REGISTER_TEST(matmul_f16_pair, test_matmul_f16_pair);
