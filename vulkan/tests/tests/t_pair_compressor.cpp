/* Kernel test: ds4_gpu_matmul_f16_pair_compressor_store_tensor.
 *
 * Fused host-side pair:
 *   out_kv[o]    = sum_i W_kv[o][i] * x[i]      (f16 weights, [width][in_dim])
 *   out_score[o] = sum_i W_score[o][i] * x[i]
 *   state_kv / state_score row = out_kv / out_score + APE(phase = pos % ratio)
 *
 * with the rolling compressor state store at
 *   dst_row = (ratio == 4) ? ratio + pos % ratio : pos % ratio
 * exactly like ds4_gpu_compressor_store_batch_tensor / the CUDA
 * compressor_store_kernel.  The implementation is host-side in the
 * backend (operates on tensor->ptr directly), so this test does NOT wrap
 * the call in begin/end_commands.
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

/* f32 -> IEEE half (round-to-nearest-even; test values are normal-range). */
static uint16_t ref_f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
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
        if (((bits >> 23) & 0xffu) == 0xffu && mant != 0) return (uint16_t)(sign | 0x7e00u);
        return (uint16_t)(sign | 0x7c00u);
    }
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half & 1u))) half++;
    return (uint16_t)half;
}

/* IEEE half -> f32 (exact decode). */
static float ref_f16_to_f32(uint16_t h) {
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
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static int check_f32(const char *what, const float *got, const float *want, uint32_t n) {
    int bad = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float g = got[i], w = want[i];
        bool ok;
        if (std::isnan(g) && std::isnan(w)) ok = true;
        else if (std::isinf(g) || std::isinf(w)) ok = (g == w);
        else ok = std::fabsf(g - w) <= 1e-3f;
        if (!ok) {
            if (bad < 8)
                fprintf(stderr, "--- %s mismatch[%u]: got %.6f want %.6f\n",
                        what, i, g, w);
            bad++;
        }
    }
    if (bad) fprintf(stderr, "--- %s: %d mismatches\n", what, bad);
    return bad ? 1 : 0;
}

/* Deterministic synthetic values with alternating signs. */
static void synth_vals(float *v, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        float f = (float)(i + 1) * 0.375f + 0.125f * (float)(i % 7);
        if ((i & 1u) != 0) f = -f;
        v[i] = f;
    }
}

/* Read one APE scalar from the model (ape_type 0 = f32, 1 = f16). */
static float ref_ape_val(const unsigned char *model, uint64_t ape_offset,
                         uint32_t ape_type, uint64_t idx) {
    if (ape_type == 1u) {
        uint16_t h;
        memcpy(&h, model + ape_offset + idx * 2, 2);
        return ref_f16_to_f32(h);
    }
    float v;
    memcpy(&v, model + ape_offset + idx * 4, 4);
    return v;
}

/* Synthetic model: [junk header][W_kv f16][W_score f16][APE]. */
struct PairModel {
    std::vector<unsigned char> data;
    uint64_t kv_off;
    uint64_t score_off;
    uint64_t ape_off;
};

static PairModel build_pair_model(uint64_t in_dim, uint32_t width,
                                  uint32_t ratio, uint32_t ape_type) {
    const uint64_t w_bytes = in_dim * (uint64_t)width * 2u; /* f16 */
    const uint64_t elem = ape_type == 1u ? 2u : 4u;
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem;
    const uint64_t header = 16;
    PairModel m;
    m.data.assign(header + 2 * w_bytes + ape_bytes, 0x00);
    m.kv_off = header;
    m.score_off = header + w_bytes;
    m.ape_off = header + 2 * w_bytes;
    uint16_t *wkv = (uint16_t *)(m.data.data() + m.kv_off);
    uint16_t *wsc = (uint16_t *)(m.data.data() + m.score_off);
    for (uint64_t o = 0; o < width; o++) {
        for (uint64_t i = 0; i < in_dim; i++) {
            float v = (float)((o + 1) * (i + 1)) * 0.01f;
            if (((o + i) & 1u) != 0) v = -v;
            wkv[o * in_dim + i] = ref_f32_to_f16(v);
            float s = (float)((o + 1) * (i + 1)) * 0.0075f + 0.25f;
            if (((o + i * 2u) & 1u) != 0) s = -s;
            wsc[o * in_dim + i] = ref_f32_to_f16(s);
        }
    }
    for (uint32_t ph = 0; ph < ratio; ph++) {
        for (uint32_t j = 0; j < width; j++) {
            const float v = 0.5f * (float)(ph + 1) + 0.125f * (float)(j % 5);
            if (ape_type == 1u) {
                const uint16_t h = ref_f32_to_f16(v);
                memcpy(m.data.data() + m.ape_off + ((uint64_t)ph * width + j) * 2, &h, 2);
            } else {
                memcpy(m.data.data() + m.ape_off + ((uint64_t)ph * width + j) * 4, &v, 4);
            }
        }
    }
    return m;
}

/* CPU reference: paired f16 matvec + rolling state store (one token). */
static void ref_pair_store(std::vector<float> &okv, std::vector<float> &osc,
                           std::vector<float> &skv, std::vector<float> &ssc,
                           const unsigned char *model, uint64_t kv_off, uint64_t score_off,
                           uint64_t ape_off, uint32_t ape_type,
                           uint64_t in_dim, uint32_t width, uint32_t ratio,
                           uint32_t pos, const float *x) {
    const uint16_t *wkv = (const uint16_t *)(model + kv_off);
    const uint16_t *wsc = (const uint16_t *)(model + score_off);
    for (uint32_t o = 0; o < width; o++) {
        double akv = 0.0, asc = 0.0;
        for (uint64_t i = 0; i < in_dim; i++) {
            akv += (double)ref_f16_to_f32(wkv[o * in_dim + i]) * (double)x[i];
            asc += (double)ref_f16_to_f32(wsc[o * in_dim + i]) * (double)x[i];
        }
        okv[o] = (float)akv;
        osc[o] = (float)asc;
    }
    const uint32_t pos_mod = pos % ratio;
    const uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
    for (uint32_t j = 0; j < width; j++) {
        skv[(uint64_t)dst_row * width + j] = okv[j];
        ssc[(uint64_t)dst_row * width + j] = osc[j] +
            ref_ape_val(model, ape_off, ape_type, (uint64_t)pos_mod * width + j);
    }
}

static int run_pair_case(const char *what, uint64_t in_dim, uint32_t head_dim,
                         uint32_t ratio, uint32_t pos, uint32_t ape_type) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    PairModel m = build_pair_model(in_dim, width, ratio, ape_type);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(in_dim * sizeof(float));
    ds4_gpu_tensor *okv = ds4_gpu_tensor_alloc((uint64_t)width * sizeof(float));
    ds4_gpu_tensor *osc = ds4_gpu_tensor_alloc((uint64_t)width * sizeof(float));
    ds4_gpu_tensor *skv = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *ssc = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    if (!x || !okv || !osc || !skv || !ssc) {
        if (x) ds4_gpu_tensor_free(x);
        if (okv) ds4_gpu_tensor_free(okv);
        if (osc) ds4_gpu_tensor_free(osc);
        if (skv) ds4_gpu_tensor_free(skv);
        if (ssc) ds4_gpu_tensor_free(ssc);
        return 1;
    }
    std::vector<float> xv(in_dim);
    synth_vals(xv.data(), xv.size());
    if (ds4_gpu_tensor_write(x, 0, xv.data(), xv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_fill_f32(okv, 777.0f, width) == 0 ||
        ds4_gpu_tensor_fill_f32(osc, 777.0f, width) == 0 ||
        ds4_gpu_tensor_fill_f32(skv, 123.0f, (uint64_t)state_rows * width) == 0 ||
        ds4_gpu_tensor_fill_f32(ssc, -123.0f, (uint64_t)state_rows * width) == 0) {
        ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(okv); ds4_gpu_tensor_free(osc);
        ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc);
        return 1;
    }
    if (ds4_gpu_set_model_map(m.data.data(), m.data.size()) == 0) {
        ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(okv); ds4_gpu_tensor_free(osc);
        ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc);
        return 1;
    }

    std::vector<float> wokv(width), wosc(width);
    std::vector<float> wskv((uint64_t)state_rows * width, 123.0f);
    std::vector<float> wssc((uint64_t)state_rows * width, -123.0f);
    ref_pair_store(wokv, wosc, wskv, wssc, m.data.data(), m.kv_off, m.score_off,
                   m.ape_off, ape_type, in_dim, width, ratio, pos, xv.data());

    int rc = 1;
    if (ds4_gpu_matmul_f16_pair_compressor_store_tensor(
            okv, osc, skv, ssc, m.data.data(), m.data.size(),
            m.kv_off, m.score_off, m.ape_off, ape_type,
            in_dim, width, x, ratio, pos) != 1) {
        fprintf(stderr, "--- %s: fused store returned != 1\n", what);
    } else {
        std::vector<float> gokv(width), gosc(width);
        std::vector<float> gskv((uint64_t)state_rows * width), gssc((uint64_t)state_rows * width);
        if (ds4_gpu_tensor_read(okv, 0, gokv.data(), gokv.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(osc, 0, gosc.data(), gosc.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(skv, 0, gskv.data(), gskv.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(ssc, 0, gssc.data(), gssc.size() * sizeof(float)) != 0) {
            if (check_f32(what, gokv.data(), wokv.data(), width) != 0) rc = 1;
            else if (check_f32("pair/out_score", gosc.data(), wosc.data(), width) != 0) rc = 1;
            else if (check_f32("pair/state_kv", gskv.data(), wskv.data(), (uint32_t)gskv.size()) != 0) rc = 1;
            else if (check_f32("pair/state_score", gssc.data(), wssc.data(), (uint32_t)gssc.size()) != 0) rc = 1;
            else rc = 0;
        }
    }
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(okv); ds4_gpu_tensor_free(osc);
    ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc);
    return rc;
}

static int test_matmul_f16_pair_compressor_store(void) {
    int rc = 0;
    /* ratio 4: window goes to the second lane (rows [4, 8)); f32 APE. */
    if (run_pair_case("pair/ratio4_f32ape", 32, 8, 4, 3, 0) != 0) rc = 1;
    /* ratio != 4 with f16 APE: row = pos % ratio. */
    if (run_pair_case("pair/ratio2_f16ape", 24, 8, 2, 1, 1) != 0) rc = 1;
    /* pos at a ratio boundary (pos % ratio == 0). */
    if (run_pair_case("pair/ratio4_pos0", 16, 8, 4, 4, 0) != 0) rc = 1;
    return rc;
}
REGISTER_TEST(matmul_f16_pair_compressor_store, test_matmul_f16_pair_compressor_store);

/* ---- bounds / safety: -1 on undersized tensors or out-of-range offsets ---- */

static int test_matmul_f16_pair_compressor_store_bounds(void) {
    const uint64_t in_dim = 16;
    const uint32_t head_dim = 8, ratio = 4, pos = 1, ape_type = 0;
    const uint32_t width = 2 * head_dim;
    const uint32_t state_rows = 2 * ratio;
    PairModel m = build_pair_model(in_dim, width, ratio, ape_type);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(in_dim * sizeof(float));
    ds4_gpu_tensor *okv = ds4_gpu_tensor_alloc((uint64_t)width * sizeof(float));
    ds4_gpu_tensor *osc = ds4_gpu_tensor_alloc((uint64_t)width * sizeof(float));
    ds4_gpu_tensor *skv_ok = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *ssc_ok = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *skv_small = ds4_gpu_tensor_alloc((uint64_t)(state_rows - 1) * width * sizeof(float));
    ds4_gpu_tensor *okv_small = ds4_gpu_tensor_alloc((uint64_t)(width - 1) * sizeof(float));
    if (!x || !okv || !osc || !skv_ok || !ssc_ok || !skv_small || !okv_small) {
        if (x) ds4_gpu_tensor_free(x);
        if (okv) ds4_gpu_tensor_free(okv);
        if (osc) ds4_gpu_tensor_free(osc);
        if (skv_ok) ds4_gpu_tensor_free(skv_ok);
        if (ssc_ok) ds4_gpu_tensor_free(ssc_ok);
        if (skv_small) ds4_gpu_tensor_free(skv_small);
        if (okv_small) ds4_gpu_tensor_free(okv_small);
        return 1;
    }
    int rc = 0;
    const int good = ds4_gpu_matmul_f16_pair_compressor_store_tensor(
        okv, osc, skv_ok, ssc_ok, m.data.data(), m.data.size(),
        m.kv_off, m.score_off, m.ape_off, ape_type, in_dim, width, x, ratio, pos);
    if (good != 1) {
        fprintf(stderr, "--- bounds: happy path returned %d, expected 1\n", good);
        rc = 1;
    }
    if (ds4_gpu_matmul_f16_pair_compressor_store_tensor(
            okv, osc, skv_small, ssc_ok, m.data.data(), m.data.size(),
            m.kv_off, m.score_off, m.ape_off, ape_type, in_dim, width, x, ratio, pos) != -1) {
        fprintf(stderr, "--- bounds: undersized state_kv not rejected\n");
        rc = 1;
    }
    if (ds4_gpu_matmul_f16_pair_compressor_store_tensor(
            okv_small, osc, skv_ok, ssc_ok, m.data.data(), m.data.size(),
            m.kv_off, m.score_off, m.ape_off, ape_type, in_dim, width, x, ratio, pos) != -1) {
        fprintf(stderr, "--- bounds: undersized out_kv not rejected\n");
        rc = 1;
    }
    /* Weight offset past the end of the model. */
    if (ds4_gpu_matmul_f16_pair_compressor_store_tensor(
            okv, osc, skv_ok, ssc_ok, m.data.data(), m.data.size(),
            m.data.size() + 4, m.score_off, m.ape_off, ape_type,
            in_dim, width, x, ratio, pos) != -1) {
        fprintf(stderr, "--- bounds: out-of-range weight offset not rejected\n");
        rc = 1;
    }
    /* APE offset past the end of the model. */
    if (ds4_gpu_matmul_f16_pair_compressor_store_tensor(
            okv, osc, skv_ok, ssc_ok, m.data.data(), m.data.size(),
            m.kv_off, m.score_off, m.data.size() + 4, ape_type,
            in_dim, width, x, ratio, pos) != -1) {
        fprintf(stderr, "--- bounds: out-of-range APE offset not rejected\n");
        rc = 1;
    }
    /* Null model pointer. */
    if (ds4_gpu_matmul_f16_pair_compressor_store_tensor(
            okv, osc, skv_ok, ssc_ok, nullptr, 0,
            m.kv_off, m.score_off, m.ape_off, ape_type,
            in_dim, width, x, ratio, pos) != -1) {
        fprintf(stderr, "--- bounds: null model_map not rejected\n");
        rc = 1;
    }
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(okv); ds4_gpu_tensor_free(osc);
    ds4_gpu_tensor_free(skv_ok); ds4_gpu_tensor_free(ssc_ok);
    ds4_gpu_tensor_free(skv_small); ds4_gpu_tensor_free(okv_small);
    return rc;
}
REGISTER_TEST(matmul_f16_pair_compressor_store_bounds, test_matmul_f16_pair_compressor_store_bounds);
