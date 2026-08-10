/* Kernel test for ds4_gpu_hc_expand_split_half_tensor.
 *
 * Host-side kernel (like ds4_gpu_add_tensor), so the call is NOT wrapped in
 * begin/end_commands.  The sublayer block arrives in f16 (2 B/element):
 *   out_hc[h*n_embd + i] = split[n_hc + h] * deq_f16(block_out_h[i])
 *                          + residual_hc[h*n_embd + i]
 * with split laid out as the sinkhorn buffer [pre | post | comb]; only the
 * post gates split[n_hc + h] participate (batch fast path, matching
 * ds4_gpu_hc_expand_split_tensor).
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <vector>

static const uint32_t kN_HC   = 3;
static const uint32_t kN_EMBD = 8;
static const uint32_t kMixHC  = 2 * kN_HC + kN_HC * kN_HC; /* 2*n_hc + n_hc*n_hc */

/* ---------- helpers ---------- */

/* IEEE-754 f32 -> f16, mirrors ds4_float_to_half in the backend so the test
 * half tensors round-trip exactly like the production path. */
static uint16_t f32_to_f16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) {
        if (((bits >> 23) & 0xffu) == 0xffu && mant != 0)
            return (uint16_t)(sign | 0x7e00u);
        return (uint16_t)(sign | 0x7c00u);
    }

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half & 1u))) half++;
    return (uint16_t)half;
}

/* IEEE-754 f16 -> f32, mirrors ds4_half_to_float in the backend. */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t expo = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (expo == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            int e = -14;
            uint32_t m = mant;
            while ((m & 0x400u) == 0) { m <<= 1; e--; }
            bits = sign | (uint32_t)(e + 127) << 23 | (m & 0x3ffu) << 13;
        }
    } else if (expo == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | (expo + 112u) << 23 | mant << 13;
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static int check_f32(const char *what, const float *got, const float *want, uint32_t n) {
    int bad = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (!(std::fabsf(got[i] - want[i]) <= 1e-3f)) {
            if (bad < 8)
                fprintf(stderr, "--- %s mismatch[%u]: got %.6f want %.6f\n",
                        what, i, got[i], want[i]);
            bad++;
        }
    }
    if (bad) fprintf(stderr, "--- %s: %d mismatches\n", what, bad);
    return bad ? 1 : 0;
}

/* Deterministic synthetic value: in-range, alternating signs. */
static float synth_float(uint32_t i, float step) {
    float f = (float)(i + 1) * step + 0.0625f * (float)(i % 5);
    if ((i & 1u) != 0) f = -f;
    return f;
}

/* ---------- hc_expand_split_half ---------- */

static int test_hc_expand_split_half(void) {
    const uint32_t n_hc = kN_HC, n_embd = kN_EMBD;
    const uint64_t hc_dim = (uint64_t)n_hc * n_embd;
    ds4_gpu_tensor *out_hc      = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    ds4_gpu_tensor *block_out_h = ds4_gpu_tensor_alloc((uint64_t)n_embd * sizeof(uint16_t));
    ds4_gpu_tensor *residual    = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    ds4_gpu_tensor *split       = ds4_gpu_tensor_alloc((uint64_t)kMixHC * sizeof(float));
    if (!out_hc || !block_out_h || !residual || !split) {
        if (out_hc) ds4_gpu_tensor_free(out_hc);
        if (block_out_h) ds4_gpu_tensor_free(block_out_h);
        if (residual) ds4_gpu_tensor_free(residual);
        if (split) ds4_gpu_tensor_free(split);
        return 1;
    }
    std::vector<uint16_t> boh(n_embd);
    std::vector<float> rh((size_t)hc_dim), sp(kMixHC);
    for (uint32_t i = 0; i < n_embd; i++) boh[i] = f32_to_f16(synth_float(i, 0.25f));
    for (uint32_t i = 0; i < (uint32_t)hc_dim; i++) rh[i] = synth_float(i, 0.20f);
    for (uint32_t i = 0; i < kMixHC; i++)           sp[i] = synth_float(i, 0.08f);
    if (!ds4_gpu_tensor_write(block_out_h, 0, boh.data(), boh.size() * sizeof(uint16_t)) ||
        !ds4_gpu_tensor_write(residual, 0, rh.data(), rh.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(split, 0, sp.data(), sp.size() * sizeof(float))) {
        ds4_gpu_tensor_free(out_hc); ds4_gpu_tensor_free(block_out_h);
        ds4_gpu_tensor_free(residual); ds4_gpu_tensor_free(split);
        return 1;
    }
    /* Reference: hc_post_one with an f16 block and full combine matrix. */
    std::vector<float> want((size_t)hc_dim);
    const float *post = sp.data() + n_hc;
    const float *comb = post + n_hc;
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        for (uint32_t i = 0; i < n_embd; i++) {
            float acc = post[dst] * f16_to_f32(boh[i]);
            for (uint32_t src = 0; src < n_hc; src++)
                acc += comb[dst + src * n_hc] * rh[(uint64_t)src * n_embd + i];
            want[(uint64_t)dst * n_embd + i] = acc;
        }
    }

    int rc = 1;
    if (ds4_gpu_hc_expand_split_half_tensor(out_hc, block_out_h, residual, split,
                                            n_embd, n_hc) != 0) {
        std::vector<float> got((size_t)hc_dim);
        if (ds4_gpu_tensor_read(out_hc, 0, got.data(), got.size() * sizeof(float)) != 0)
            rc = check_f32("hc_expand_split_half", got.data(), want.data(), (uint32_t)hc_dim);
    }
    ds4_gpu_tensor_free(out_hc); ds4_gpu_tensor_free(block_out_h);
    ds4_gpu_tensor_free(residual); ds4_gpu_tensor_free(split);
    return rc;
}
REGISTER_TEST(hc_expand_split_half, test_hc_expand_split_half);
