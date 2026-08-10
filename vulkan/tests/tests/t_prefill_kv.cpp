/* Kernel tests for the prefill / KV-cache pipeline:
 *
 *   ds4_gpu_dsv4_qkv_rms_norm_rows_tensor   (fused Q+KV RMSNorm, weights)
 *   ds4_gpu_dsv4_fp8_kv_quantize_tensor     (E4M3FN round trip on non-RoPE part)
 *   ds4_gpu_store_raw_kv_batch_tensor       (ring-store batch KV with f16 round trip)
 *   ds4_gpu_attention_prefill_raw_heads_tensor (causal windowed raw attention)
 *
 * The Vulkan backend dispatches these operations and synchronizes before
 * returning, so the tests can keep the direct API shape and read back here.
 * The CPU references are copied verbatim from ds4.c (rms_norm_weight,
 * dsv4_fp8_kv_quantize_row_inplace_cpu, kv_cache_push_raw f16 round trip,
 * layer_attention_prefix_batch_worker raw attention), which is the same
 * math the CUDA backend kernels implement.  These four build the raw KV
 * cache that decode attention reads, so drift here is a correctness bug.
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

/* ---------- CPU references (copied from ds4.c) ---------- */

/* ds4.c f32_to_f16 (round-to-nearest-even). */
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

/* ds4.c f16_to_f32. */
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

/* ds4.c dsv4_e4m3fn_value_cpu. */
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

/* ds4.c dsv4_e4m3fn_dequant_cpu. */
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

/* ds4.c dsv4_fp8_kv_quantize_row_inplace_cpu (per row). */
static void ref_fp8_kv_quantize_row(float *x, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t n_nope = head_dim - n_rot;
    for (uint32_t off = 0; off < n_nope; off += 64) {
        const uint32_t valid = (n_nope - off < 64u) ? (n_nope - off) : 64u;
        float amax = 0.0f;
        for (uint32_t i = 0; i < valid; i++) {
            const float av = fabsf(x[off + i]);
            if (av > amax) amax = av;
        }
        if (amax < 1.0e-4f) amax = 1.0e-4f;
        const float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 448.0f)));
        for (uint32_t i = 0; i < valid; i++) {
            float v = x[off + i] / scale;
            if (v > 448.0f) v = 448.0f;
            if (v < -448.0f) v = -448.0f;
            x[off + i] = ref_e4m3fn_dequant(v) * scale;
        }
    }
}

/* ds4.c rms_norm_weight (per row). */
static void ref_rms_norm_weight(float *out, const float *x, const float *weight,
                                uint64_t n, float eps) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint64_t i = 0; i < n; i++) out[i] = x[i] * scale * weight[i];
}

/* Raw part of ds4.c layer_attention_prefix_batch_worker: causal windowed
 * attention over a contiguous chunk of raw KV rows with the sink prior. */
static void ref_prefill_raw_heads(float *heads, const float *sinks, const float *q,
                                  const float *raw_kv, uint32_t n_tokens,
                                  uint32_t window, uint32_t n_head,
                                  uint32_t head_dim) {
    const float kq_scale = 1.0f / sqrtf((float)head_dim);
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t raw_count = t + 1 < window ? t + 1 : window;
        const uint32_t raw_start = t + 1 - raw_count;
        for (uint32_t h = 0; h < n_head; h++) {
            const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
            float max_score = sinks[h];
            std::vector<float> score(raw_count);
            for (uint32_t r = 0; r < raw_count; r++) {
                const float *kv = raw_kv + (uint64_t)(raw_start + r) * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
                score[r] = dot * kq_scale;
                if (score[r] > max_score) max_score = score[r];
            }
            float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
            std::memset(oh, 0, (size_t)head_dim * sizeof(float));
            float denom = expf(sinks[h] - max_score);
            for (uint32_t r = 0; r < raw_count; r++) {
                const float w = expf(score[r] - max_score);
                const float *kv = raw_kv + (uint64_t)(raw_start + r) * head_dim;
                denom += w;
                for (uint32_t d = 0; d < head_dim; d++) oh[d] += w * kv[d];
            }
            const float inv = 1.0f / denom;
            for (uint32_t d = 0; d < head_dim; d++) oh[d] *= inv;
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

/* ---------- dsv4_qkv_rms_norm_rows ---------- */

static int test_qkv_rms_norm_rows(void) {
    const uint32_t qn = 16, kvn = 24, rows = 4;
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
    if (ds4_gpu_tensor_write(q, 0, qv.data(), qv.size() * sizeof(float)) == 0) {
        ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(qo);
        ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(kvo);
        return 1;
    }
    if (ds4_gpu_tensor_write(kv, 0, kvv.data(), kvv.size() * sizeof(float)) == 0) {
        ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(qo);
        ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(kvo);
        return 1;
    }

    std::vector<float> qw_ok(qn), kvw_ok(kvn);
    for (uint32_t i = 0; i < qn; i++) qw_ok[i] = qw[i];
    for (uint32_t i = 0; i < kvn; i++) kvw_ok[i] = kvw[i];

    int rc = 1;
    if (ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(qo, q, model.data(), model_size,
                                              qw_off, qn, kvo, kv, kvw_off,
                                              kvn, rows, eps) != 0) {
        std::vector<float> want_q((uint64_t)rows * qn), want_kv((uint64_t)rows * kvn);
        std::vector<float> got_q((uint64_t)rows * qn), got_kv((uint64_t)rows * kvn);
        for (uint32_t r = 0; r < rows; r++) {
            ref_rms_norm_weight(want_q.data() + (uint64_t)r * qn,
                                qv.data() + (uint64_t)r * qn, qw_ok.data(), qn, eps);
            ref_rms_norm_weight(want_kv.data() + (uint64_t)r * kvn,
                                kvv.data() + (uint64_t)r * kvn, kvw_ok.data(), kvn, eps);
        }
        if (ds4_gpu_tensor_read(qo, 0, got_q.data(), got_q.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(kvo, 0, got_kv.data(), got_kv.size() * sizeof(float)) != 0) {
            if (check_f32("qkv_rms/q", got_q.data(), want_q.data(), (uint32_t)got_q.size()) == 0)
                rc = check_f32("qkv_rms/kv", got_kv.data(), want_kv.data(), (uint32_t)got_kv.size());
        }
    }
    ds4_gpu_tensor_free(kvo);
    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(qo);
    ds4_gpu_tensor_free(q);
    return rc;
}
REGISTER_TEST(dsv4_qkv_rms_norm_rows, test_qkv_rms_norm_rows);

/* ---------- dsv4_fp8_kv_quantize ---------- */

static int fp8_quantize_one(const char *what, uint32_t n_tok, uint32_t head_dim,
                            uint32_t n_rot) {
    const uint64_t elems = (uint64_t)n_tok * head_dim;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(elems * sizeof(float));
    if (!x) return 1;
    std::vector<float> xv(elems);
    synth_f32(xv.data(), elems);
    /* Scale into a fp8-friendly range so the quantized values stay small
     * but still exercise every table bucket. */
    for (uint64_t i = 0; i < elems; i++) xv[i] *= 0.25f;
    if (ds4_gpu_tensor_write(x, 0, xv.data(), xv.size() * sizeof(float)) == 0) {
        ds4_gpu_tensor_free(x); return 1;
    }

    std::vector<float> want(xv);
    for (uint32_t t = 0; t < n_tok; t++)
        ref_fp8_kv_quantize_row(want.data() + (uint64_t)t * head_dim, head_dim, n_rot);

    int rc = 1;
    if (ds4_gpu_dsv4_fp8_kv_quantize_tensor(x, n_tok, head_dim, n_rot) != 0) {
        std::vector<float> got(elems);
        if (ds4_gpu_tensor_read(x, 0, got.data(), got.size() * sizeof(float)) != 0)
            rc = check_f32(what, got.data(), want.data(), (uint32_t)elems);
    }
    ds4_gpu_tensor_free(x);
    return rc;
}

static int test_fp8_kv_quantize(void) {
    /* Model-like: head_dim 512, n_rot 64 → n_nope 448 = 7 full 64-blocks. */
    if (fp8_quantize_one("fp8_kv_quantize/model", 3, 512, 64) != 0) return 1;
    /* Two full blocks. */
    if (fp8_quantize_one("fp8_kv_quantize/blocks", 2, 160, 32) != 0) return 1;
    /* Partial final 64-value block must be quantized without touching RoPE data. */
    if (fp8_quantize_one("fp8_kv_quantize/tail", 2, 150, 32) != 0) return 1;
    return 0;
}
REGISTER_TEST(dsv4_fp8_kv_quantize, test_fp8_kv_quantize);

/* ---------- store_raw_kv_batch ---------- */

static int test_store_raw_kv_batch(void) {
    const uint32_t raw_cap = 8, head_dim = 16, n_tokens = 5, pos0 = 6;
    /* pos0 + n_tokens = 11 > raw_cap, so rows wrap: 6,7,0,1,2. */
    ds4_gpu_tensor *cache = ds4_gpu_tensor_alloc((uint64_t)raw_cap * head_dim * sizeof(float));
    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc((uint64_t)n_tokens * head_dim * sizeof(float));
    if (!cache || !kv) {
        if (cache) ds4_gpu_tensor_free(cache);
        if (kv) ds4_gpu_tensor_free(kv);
        return 1;
    }
    if (ds4_gpu_tensor_fill_f32(cache, 0.0f, (uint64_t)raw_cap * head_dim) == 0) {
        ds4_gpu_tensor_free(cache); ds4_gpu_tensor_free(kv); return 1;
    }
    std::vector<float> kvv((uint64_t)n_tokens * head_dim);
    synth_f32(kvv.data(), kvv.size());
    if (ds4_gpu_tensor_write(kv, 0, kvv.data(), kvv.size() * sizeof(float)) == 0) {
        ds4_gpu_tensor_free(cache); ds4_gpu_tensor_free(kv); return 1;
    }

    std::vector<float> want((uint64_t)raw_cap * head_dim, 0.0f);
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t row = (pos0 + t) % raw_cap;
        for (uint32_t d = 0; d < head_dim; d++)
            want[(uint64_t)row * head_dim + d] =
                ref_f16_to_f32(ref_f32_to_f16(kvv[(uint64_t)t * head_dim + d]));
    }

    int rc = 1;
    if (ds4_gpu_store_raw_kv_batch_tensor(cache, kv, raw_cap, pos0, n_tokens,
                                          head_dim) != 0) {
        std::vector<float> got((uint64_t)raw_cap * head_dim);
        if (ds4_gpu_tensor_read(cache, 0, got.data(), got.size() * sizeof(float)) != 0)
            rc = check_f32("store_raw_kv_batch", got.data(), want.data(),
                           (uint32_t)got.size());
    }
    ds4_gpu_tensor_free(kv);
    ds4_gpu_tensor_free(cache);
    return rc;
}
REGISTER_TEST(store_raw_kv_batch, test_store_raw_kv_batch);

/* ---------- attention_prefill_raw_heads ---------- */

static int test_attention_prefill_raw_heads(void) {
    const uint32_t n_tokens = 32, window = 8, n_head = 4, head_dim = 32;
    const uint64_t head_elems = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t kv_elems = (uint64_t)n_tokens * head_dim;
    const uint64_t sink_off = 8;
    const uint64_t model_size = sink_off + (uint64_t)n_head * sizeof(float);

    std::vector<unsigned char> model(model_size, 0x00);
    float *sinks = (float *)(model.data() + sink_off);
    for (uint32_t h = 0; h < n_head; h++) sinks[h] = 0.25f + 0.1f * (float)h;
    if (ds4_gpu_set_model_map(model.data(), model_size) == 0) return 1;

    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(head_elems * sizeof(float));
    ds4_gpu_tensor *raw_kv = ds4_gpu_tensor_alloc(kv_elems * sizeof(float));
    ds4_gpu_tensor *heads = ds4_gpu_tensor_alloc(head_elems * sizeof(float));
    if (!q || !raw_kv || !heads) {
        if (q) ds4_gpu_tensor_free(q);
        if (raw_kv) ds4_gpu_tensor_free(raw_kv);
        if (heads) ds4_gpu_tensor_free(heads);
        return 1;
    }
    std::vector<float> qv(head_elems), kvv(kv_elems);
    synth_f32(qv.data(), qv.size());
    synth_f32(kvv.data(), kvv.size());
    if (ds4_gpu_tensor_write(q, 0, qv.data(), qv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(raw_kv, 0, kvv.data(), kvv.size() * sizeof(float)) == 0) {
        ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(raw_kv); ds4_gpu_tensor_free(heads);
        return 1;
    }

    std::vector<float> want(head_elems);
    ref_prefill_raw_heads(want.data(), sinks, qv.data(), kvv.data(), n_tokens,
                          window, n_head, head_dim);

    int rc = 1;
    if (ds4_gpu_attention_prefill_raw_heads_tensor(heads, model.data(), model_size,
                                                   sink_off, q, raw_kv, n_tokens,
                                                   window, n_head, head_dim) != 0) {
        std::vector<float> got(head_elems);
        if (ds4_gpu_tensor_read(heads, 0, got.data(), got.size() * sizeof(float)) != 0)
            rc = check_f32("attention_prefill_raw_heads", got.data(), want.data(),
                           (uint32_t)head_elems);
    }
    ds4_gpu_tensor_free(heads);
    ds4_gpu_tensor_free(raw_kv);
    ds4_gpu_tensor_free(q);
    return rc;
}
REGISTER_TEST(attention_prefill_raw_heads, test_attention_prefill_raw_heads);
