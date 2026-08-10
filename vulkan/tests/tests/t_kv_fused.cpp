/*
 * Fused decode KV kernels (host-side in the backend):
 *
 *   ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor
 *       (fused Q/KV RMSNorm with weights + RoPE tail on KV, one call)
 *   ds4_gpu_kv_fp8_store_raw_tensor
 *       (FP8 E4M3FN non-RoPE round trip in place + F16 raw-cache ring store)
 *
 * Both are host-side in the backend (they operate on tensor->ptr directly),
 * so the tests do NOT wrap the calls in begin/end_commands.
 *
 * The CPU references are copied verbatim from ds4.c (rms_norm_weight,
 * rope_tail_ext_inplace, dsv4_fp8_kv_quantize_row_inplace_cpu,
 * kv_cache_push_raw) — the same math the CUDA fused kernels
 * (dsv4_qkv_rms_norm_rows_kv_rope_kernel, fp8_kv_quantize_store_rows_kernel)
 * implement.
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const float kPi = 3.14159265358979323846f;

/* ---------- ds4.c rope_yarn helpers (copied verbatim) ---------- */

static float rope_yarn_ramp(float low, float high, int i0) {
    const float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

static float rope_yarn_corr_dim(int n_dims, uint64_t n_ctx_orig, float n_rot, float base) {
    return (float)n_dims * logf((float)n_ctx_orig / (n_rot * 2.0f * kPi)) / (2.0f * logf(base));
}

static void rope_yarn_corr_dims(int n_dims, uint64_t n_ctx_orig, float freq_base,
                                float beta_fast, float beta_slow, float dims[2]) {
    const float start = floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    const float end = ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = fmaxf(0.0f, start);
    dims[1] = fminf((float)(n_dims - 1), end);
}

/* ---------- ds4.c f32/f16 helpers (copied verbatim) ---------- */

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

static float ref_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t expo = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (expo == 0) {
        if (mant == 0) bits = sign;
        else {
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

/* ---------- ds4.c E4M3FN helpers (copied verbatim) ---------- */

static float ref_e4m3fn_value(int i) {
    static const float exp_scale[16] = {
        0.0f, 0.015625f, 0.03125f, 0.0625f,
        0.125f, 0.25f, 0.5f, 1.0f,
        2.0f, 4.0f, 8.0f, 16.0f,
        32.0f, 64.0f, 128.0f, 256.0f,
    };
    const int exp = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0
        ? (float)mant * 0.001953125f
        : (1.0f + (float)mant * 0.125f) * exp_scale[exp];
}

static float ref_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);
    int lo = 0;
    int hi = 126;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (ref_e4m3fn_value(mid) <= ax) lo = mid; else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        const float best_diff = fabsf(ax - ref_e4m3fn_value(best));
        const float next_diff = fabsf(ax - ref_e4m3fn_value(best + 1));
        if (next_diff < best_diff ||
            (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) best++;
    }
    return sign * ref_e4m3fn_value(best);
}

/* ---------- CPU references (copied from ds4.c) ---------- */

/* rms_norm_weight (per row, learned scale). */
static void ref_rms_norm_weight(float *out, const float *x, const float *weight,
                                uint64_t n, float eps) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint64_t i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
}

/* RoPE on the tail of every head (ds4.c rope_tail_ext_inplace). */
static void ref_rope_tail(float *x, uint32_t n_tok, uint32_t n_head,
                          uint32_t head_dim, uint32_t n_rot, uint32_t pos,
                          uint64_t n_ctx_orig, float freq_base, float freq_scale,
                          float ext_factor, float attn_factor, float beta_fast,
                          float beta_slow, bool inverse) {
    const uint32_t n_nope = head_dim - n_rot;
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    const float sin_sign = inverse ? -1.0f : 1.0f;
    float corr_dims[2] = { 0.0f, 0.0f };
    if (ext_factor != 0.0f) {
        rope_yarn_corr_dims((int)n_rot, n_ctx_orig, freq_base, beta_fast, beta_slow,
                            corr_dims);
    }
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t h = 0; h < n_head; h++) {
            float *tail = x + ((uint64_t)t * n_head + h) * head_dim + n_nope;
            float theta_extrap = (float)(pos + t);
            for (uint32_t i = 0; i < n_rot; i += 2) {
                const float theta_interp = freq_scale * theta_extrap;
                float theta = theta_interp;
                float mscale = attn_factor;
                if (ext_factor != 0.0f) {
                    const float ramp_mix =
                        rope_yarn_ramp(corr_dims[0], corr_dims[1], (int)i) * ext_factor;
                    theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                    mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
                }
                const float c = cosf(theta) * mscale;
                const float s = sin_sign * sinf(theta) * mscale;
                const float x0 = tail[i + 0];
                const float x1 = tail[i + 1];
                tail[i + 0] = x0 * c - x1 * s;
                tail[i + 1] = x0 * s + x1 * c;
                theta_extrap *= theta_scale;
            }
        }
    }
}

/* ds4.c dsv4_fp8_kv_quantize_row_inplace_cpu (per row). */
static void ref_fp8_kv_quantize_row(float *x, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t off = 0; off < n_nope; off += 64) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < 64; i++) {
            const float av = fabsf(x[off + i]);
            if (av > amax) amax = av;
        }
        if (amax < 1.0e-4f) amax = 1.0e-4f;
        const float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 448.0f)));
        for (uint32_t i = 0; i < 64; i++) {
            float v = x[off + i] / scale;
            if (v > 448.0f) v = 448.0f;
            if (v < -448.0f) v = -448.0f;
            x[off + i] = ref_e4m3fn_dequant(v) * scale;
        }
    }
}

/* ---------- helpers ---------- */

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

/* Deterministic synthetic row-major float data, in-range values. */
static void synth_f32(float *v, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        float f = (float)(i + 1) * 0.375f + 0.125f * (float)(i % 7);
        if ((i & 1u) != 0) f = -f;
        v[i] = f;
    }
}

/* ---------- dsv4_qkv_rms_norm_rows_kv_rope ---------- */

static int test_qkv_rms_norm_rows_kv_rope(void) {
    /* kv row = kv_n_head * kv_head_dim floats; rope rotates the last n_rot
     * channels of each head, the head prefix stays at the normed value. */
    const uint32_t qn = 16, kv_n_head = 2, kv_head_dim = 24, n_rot = 8;
    const uint32_t kvn = kv_n_head * kv_head_dim;
    const uint32_t rows = 3;
    const uint32_t pos0 = 5;
    const float eps = 1e-5f;
    const uint64_t header = 64;
    const uint64_t qw_off = header;
    const uint64_t kvw_off = qw_off + (uint64_t)qn * sizeof(float);
    const uint64_t model_size = kvw_off + (uint64_t)kvn * sizeof(float);

    std::vector<unsigned char> model(model_size, 0xAA);
    float *qw = (float *)(model.data() + qw_off);
    float *kvw = (float *)(model.data() + kvw_off);
    for (uint32_t i = 0; i < qn; i++) qw[i] = 0.5f + 0.0625f * (float)(i + 1);
    for (uint32_t i = 0; i < kvn; i++) kvw[i] = 0.75f + 0.03125f * (float)(i + 1);
    if (ds4_gpu_set_model_map(model.data(), model_size) == 0) return 1;

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc((uint64_t)rows * qn * sizeof(float));
    ds4_gpu_tensor *qo = ds4_gpu_tensor_alloc((uint64_t)rows * qn * sizeof(float));
    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc((uint64_t)rows * kvn * sizeof(float));
    ds4_gpu_tensor *kvo = ds4_gpu_tensor_alloc((uint64_t)rows * kvn * sizeof(float));
    if (!q || !qo || !kv || !kvo) {
        if (q) ds4_gpu_tensor_free(q);
        if (qo) ds4_gpu_tensor_free(qo);
        if (kv) ds4_gpu_tensor_free(kv);
        if (kvo) ds4_gpu_tensor_free(kvo);
        return 1;
    }
    std::vector<float> qv((uint64_t)rows * qn), kvv((uint64_t)rows * kvn);
    synth_f32(qv.data(), qv.size());
    synth_f32(kvv.data(), kvv.size());
    if (ds4_gpu_tensor_write(q, 0, qv.data(), qv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(kv, 0, kvv.data(), kvv.size() * sizeof(float)) == 0) {
        ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(qo);
        ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(kvo);
        return 1;
    }

    /* Reference: RMSNorm with weight on every q row, then the same on every
     * kv row, then the RoPE tail on the kv rows (head-major inside a row). */
    std::vector<float> want_q((uint64_t)rows * qn), want_kv((uint64_t)rows * kvn);
    for (uint32_t r = 0; r < rows; r++) {
        ref_rms_norm_weight(want_q.data() + (uint64_t)r * qn,
                            qv.data() + (uint64_t)r * qn, qw, qn, eps);
        ref_rms_norm_weight(want_kv.data() + (uint64_t)r * kvn,
                            kvv.data() + (uint64_t)r * kvn, kvw, kvn, eps);
    }
    ref_rope_tail(want_kv.data(), rows, kv_n_head, kv_head_dim, n_rot, pos0,
                  0, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, false);

    int rc = 1;
    if (ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
            qo, q, model.data(), model_size, qw_off, qn,
            kvo, kv, kvw_off, kvn, rows,
            kv_n_head, kv_head_dim, n_rot, pos0, 0, false,
            10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, eps) != 0) {
        std::vector<float> got_q((uint64_t)rows * qn), got_kv((uint64_t)rows * kvn);
        if (ds4_gpu_tensor_read(qo, 0, got_q.data(), got_q.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(kvo, 0, got_kv.data(), got_kv.size() * sizeof(float)) != 0) {
            if (check_f32("qkv_rms_kv_rope/q", got_q.data(), want_q.data(),
                          (uint32_t)got_q.size()) == 0)
                rc = check_f32("qkv_rms_kv_rope/kv", got_kv.data(), want_kv.data(),
                               (uint32_t)got_kv.size());
        }
    }
    ds4_gpu_tensor_free(kvo);
    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(qo);
    ds4_gpu_tensor_free(q);
    return rc;
}
REGISTER_TEST(dsv4_qkv_rms_norm_rows_kv_rope, test_qkv_rms_norm_rows_kv_rope);

/* ---------- kv_fp8_store_raw ---------- */

static int test_kv_fp8_store_raw(void) {
    /* Model-like: head_dim 128, n_rot 64 → n_nope 64 = one full 64-block;
     * raw_cap 5 and row 12 exercise the ring wrap (12 % 5 = 2). */
    const uint32_t head_dim = 128, n_rot = 64, raw_cap = 5, row = 12;
    const uint32_t dst_row = row % raw_cap;
    const uint64_t cache_elems = (uint64_t)raw_cap * head_dim;

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc((uint64_t)head_dim * sizeof(float));
    ds4_gpu_tensor *cache = ds4_gpu_tensor_alloc(cache_elems * sizeof(float));
    if (!kv || !cache) {
        if (kv) ds4_gpu_tensor_free(kv);
        if (cache) ds4_gpu_tensor_free(cache);
        return 1;
    }
    std::vector<float> kvv(head_dim);
    synth_f32(kvv.data(), kvv.size());
    /* Scale into a fp8-friendly range (same as the fp8 quantize test). */
    for (uint32_t i = 0; i < head_dim; i++) kvv[i] *= 0.25f;
    if (ds4_gpu_tensor_write(kv, 0, kvv.data(), kvv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_fill_f32(cache, 0.0f, cache_elems) == 0) {
        ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(cache);
        return 1;
    }

    /* Reference: fp8 round trip on the non-RoPE part in place, then the
     * whole row stored at dst_row with the f16 round trip. */
    std::vector<float> quant(kvv);
    ref_fp8_kv_quantize_row(quant.data(), head_dim, n_rot);
    std::vector<float> want(cache_elems, 0.0f);
    for (uint32_t d = 0; d < head_dim; d++)
        want[(uint64_t)dst_row * head_dim + d] =
            ref_f16_to_f32(ref_f32_to_f16(quant[d]));

    int rc = 1;
    if (ds4_gpu_kv_fp8_store_raw_tensor(kv, cache, raw_cap, row, head_dim,
                                        n_rot) != 0) {
        std::vector<float> got(cache_elems);
        std::vector<float> got_kv(head_dim);
        if (ds4_gpu_tensor_read(cache, 0, got.data(), got.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(kv, 0, got_kv.data(), got_kv.size() * sizeof(float)) != 0) {
            if (check_f32("kv_fp8_store_raw/cache", got.data(), want.data(),
                          (uint32_t)cache_elems) == 0)
                /* The kernel also leaves the fp8-quantized row in kv. */
                rc = check_f32("kv_fp8_store_raw/kv", got_kv.data(), quant.data(),
                               head_dim);
        }
    }
    ds4_gpu_tensor_free(cache);
    ds4_gpu_tensor_free(kv);
    return rc;
}
REGISTER_TEST(kv_fp8_store_raw, test_kv_fp8_store_raw);
