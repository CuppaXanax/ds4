/* Kernel tests for the DS4 KV compressor group:
 *
 *   ds4_gpu_compressor_store_batch_tensor
 *   ds4_gpu_compressor_update_tensor
 *   ds4_gpu_compressor_prefill_tensor
 *
 * All three are host-side in the backend (they operate on tensor->ptr
 * directly), so the tests do NOT wrap the calls in begin/end_commands.
 * The CPU references are derived from ds4.c (compressor_decode_one,
 * compressor_pool_decode_state, rms_norm_weight, rope_tail_ext_inplace,
 * dsv4_fp8_kv_quantize_row_inplace_cpu) and match the CUDA production
 * kernels (compressor_store_kernel / compressor_update_pool_kernel /
 * compressor_prefill_pool_kernel / compressor_shift_ratio4_kernel) that
 * the backend replicates.
 *
 * State layout (both engine and backend):
 *   width      = (ratio == 4 ? 2 : 1) * head_dim
 *   state_rows = (ratio == 4 ? 2 : 1) * ratio
 *   state_kv / state_score : [state_rows][width] f32
 *   comp_cache             : [n_comp][head_dim] f32 (the exact layout
 *                            consumed by ds4_gpu_attention_decode_heads_tensor)
 * Unused state rows are kv = 0, score = -INFINITY (ds4.c
 * compressor_finish_prefill_state_cpu convention).
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include "../../ds4_gpu_mgpu.h"   /* complete ds4_gpu_tensor for stack views */
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const float kPi = 3.14159265358979323846f;

/* ---------- small helpers ---------- */

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

/* Deterministic synthetic values: alternating signs, in-range so softmax /
 * rms / rope never blow up. */
static void synth_vals(float *v, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        float f = (float)(i + 1) * 0.375f + 0.125f * (float)(i % 7);
        if ((i & 1u) != 0) f = -f;
        v[i] = f;
    }
}

/* f16 round trip helpers (ds4.c f32_to_f16 / f16_to_f32). */
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

/* ---------- model map builder (APE + RMS norm weight) ---------- */

struct CompModel {
    std::vector<unsigned char> data;
    uint64_t ape_offset;
    uint64_t norm_offset;
};

/* APE is laid out as [ratio][width] f32/f16 (flat index = phase * width + j),
 * exactly the layout the CUDA kernels and the GGUF column-major tensor use. */
static CompModel build_comp_model(uint32_t head_dim, uint32_t ratio, uint32_t ape_type) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint64_t elem = ape_type == 1u ? 2u : 4u;
    const uint64_t header = 16;
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    CompModel m;
    m.data.assign(header + ape_bytes + norm_bytes, 0x00);
    m.ape_offset = header;
    m.norm_offset = header + ape_bytes;
    for (uint32_t ph = 0; ph < ratio; ph++) {
        for (uint32_t j = 0; j < width; j++) {
            const float v = 0.5f * (float)(ph + 1) + 0.125f * (float)(j % 5);
            if (ape_type == 1u) {
                const uint16_t h = ref_f32_to_f16(v);
                memcpy(m.data.data() + m.ape_offset + ((uint64_t)ph * width + j) * 2, &h, 2);
            } else {
                memcpy(m.data.data() + m.ape_offset + ((uint64_t)ph * width + j) * 4, &v, 4);
            }
        }
    }
    float *nw = (float *)(m.data.data() + m.norm_offset);
    for (uint32_t i = 0; i < head_dim; i++) nw[i] = 0.5f + 0.0625f * (float)(i + 1);
    return m;
}

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

/* ---------- CPU references (from ds4.c / CUDA kernels) ---------- */

/* compressor_store_kernel (host version). */
static void ref_store_batch(std::vector<float> &skv, std::vector<float> &ssc,
                            const std::vector<float> &kv, const std::vector<float> &sc,
                            const unsigned char *model, uint64_t ape_offset, uint32_t ape_type,
                            uint32_t head_dim, uint32_t ratio, uint32_t pos0, uint32_t n_tokens) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t pos_mod = (pos0 + t) % ratio;
        const uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
        for (uint32_t j = 0; j < width; j++) {
            skv[(uint64_t)dst_row * width + j] = kv[(uint64_t)t * width + j];
            ssc[(uint64_t)dst_row * width + j] = sc[(uint64_t)t * width + j] +
                ref_ape_val(model, ape_offset, ape_type, (uint64_t)pos_mod * width + j);
        }
    }
}

/* ds4.c compressor_pool_decode_state (identical for finite scores to the
 * CUDA compressor_update_pool_kernel). */
static void ref_pool_state(float *out, const float *skv, const float *ssc,
                           uint32_t head_dim, uint32_t ratio) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    for (uint32_t j = 0; j < head_dim; j++) {
        float max_score = -INFINITY;
        if (ratio == 4u) {
            for (uint32_t r = 0; r < 4; r++) {
                const float sp = ssc[(uint64_t)r * width + j];
                const float sc = ssc[(uint64_t)(4u + r) * width + head_dim + j];
                if (sp > max_score) max_score = sp;
                if (sc > max_score) max_score = sc;
            }
        } else {
            for (uint32_t r = 0; r < ratio; r++) {
                const float s = ssc[(uint64_t)r * width + j];
                if (s > max_score) max_score = s;
            }
        }
        if (max_score <= -INFINITY * 0.5f) { out[j] = 0.0f; continue; }
        float denom = 0.0f, sum = 0.0f;
        if (ratio == 4u) {
            for (uint32_t r = 0; r < 4; r++) {
                const float wp = expf(ssc[(uint64_t)r * width + j] - max_score);
                const float wc = expf(ssc[(uint64_t)(4u + r) * width + head_dim + j] - max_score);
                denom += wp + wc;
                sum += wp * skv[(uint64_t)r * width + j];
                sum += wc * skv[(uint64_t)(4u + r) * width + head_dim + j];
            }
        } else {
            for (uint32_t r = 0; r < ratio; r++) {
                const float w = expf(ssc[(uint64_t)r * width + j] - max_score);
                denom += w;
                sum += w * skv[(uint64_t)r * width + j];
            }
        }
        out[j] = denom > 0.0f ? sum / denom : 0.0f;
    }
}

/* ds4.c rms_norm_weight (rows version). */
static void ref_rms_norm_weight_rows(float *x, const float *w, uint32_t n,
                                     uint32_t rows, float eps) {
    for (uint32_t r = 0; r < rows; r++) {
        double ss = 0.0;
        for (uint32_t i = 0; i < n; i++) ss += (double)x[(uint64_t)r * n + i] * x[(uint64_t)r * n + i];
        const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
        for (uint32_t i = 0; i < n; i++) x[(uint64_t)r * n + i] *= scale * w[i];
    }
}

/* ds4.c rope_yarn helpers + rope_tail_ext_inplace (single row). */
static float ref_yarn_ramp(float low, float high, int i0) {
    const float y = ((float)(i0 / 2) - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

static void ref_rope_tail_row(float *x, uint32_t head_dim, uint32_t n_rot, uint32_t pos,
                              uint64_t n_ctx_orig, float freq_base, float freq_scale,
                              float ext_factor, float attn_factor, float beta_fast,
                              float beta_slow) {
    const uint32_t n_nope = head_dim - n_rot;
    const float theta_scale = powf(freq_base, -2.0f / (float)n_rot);
    float corr_dims[2] = { 0.0f, 0.0f };
    if (ext_factor != 0.0f) {
        const float start = floorf((float)n_rot * logf((float)n_ctx_orig /
            (beta_fast * 2.0f * kPi)) / (2.0f * logf(freq_base)));
        const float end = ceilf((float)n_rot * logf((float)n_ctx_orig /
            (beta_slow * 2.0f * kPi)) / (2.0f * logf(freq_base)));
        corr_dims[0] = fmaxf(0.0f, start);
        corr_dims[1] = fminf((float)(n_rot - 1), end);
    }
    float *tail = x + n_nope;
    float theta_extrap = (float)pos;
    for (uint32_t i = 0; i < n_rot; i += 2) {
        const float theta_interp = freq_scale * theta_extrap;
        float theta = theta_interp;
        float mscale = attn_factor;
        if (ext_factor != 0.0f) {
            const float ramp_mix = ref_yarn_ramp(corr_dims[0], corr_dims[1], (int)i) * ext_factor;
            theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
        }
        const float c = cosf(theta) * mscale;
        const float s = sinf(theta) * mscale;
        const float x0 = tail[i + 0];
        const float x1 = tail[i + 1];
        tail[i + 0] = x0 * c - x1 * s;
        tail[i + 1] = x0 * s + x1 * c;
        theta_extrap *= theta_scale;
    }
}

/* compressor_prefill_pool_kernel (host version, replay = 0). */
static void ref_prefill_pool(std::vector<float> &comp, const std::vector<float> &kv,
                             const std::vector<float> &sc, const unsigned char *model,
                             uint64_t ape_offset, uint32_t ape_type, uint32_t head_dim,
                             uint32_t ratio, uint32_t pos0, uint32_t n_comp) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    for (uint32_t c = 0; c < n_comp; c++) {
        for (uint32_t d = 0; d < head_dim; d++) {
            float vals[8], scores[8];
            uint32_t n_cand = 0;
            float max_s = -INFINITY;
            if (ratio == 4u) {
                if (c > 0) {
                    const uint32_t base = (c - 1u) * ratio;
                    for (uint32_t r = 0; r < 4; r++) {
                        const uint32_t t = base + r;
                        const float ape = ref_ape_val(model, ape_offset, ape_type,
                            (uint64_t)((pos0 + t) % ratio) * width + d);
                        vals[n_cand] = kv[(uint64_t)t * width + d];
                        scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
                        if (scores[n_cand] > max_s) max_s = scores[n_cand];
                        n_cand++;
                    }
                }
                const uint32_t base = c * ratio;
                for (uint32_t r = 0; r < 4; r++) {
                    const uint32_t t = base + r;
                    const float ape = ref_ape_val(model, ape_offset, ape_type,
                        (uint64_t)((pos0 + t) % ratio) * width + head_dim + d);
                    vals[n_cand] = kv[(uint64_t)t * width + head_dim + d];
                    scores[n_cand] = sc[(uint64_t)t * width + head_dim + d] + ape;
                    if (scores[n_cand] > max_s) max_s = scores[n_cand];
                    n_cand++;
                }
            } else {
                const uint32_t base = c * ratio;
                for (uint32_t r = 0; r < ratio; r++) {
                    const uint32_t t = base + r;
                    const float ape = ref_ape_val(model, ape_offset, ape_type,
                        (uint64_t)((pos0 + t) % ratio) * width + d);
                    vals[n_cand] = kv[(uint64_t)t * width + d];
                    scores[n_cand] = sc[(uint64_t)t * width + d] + ape;
                    if (scores[n_cand] > max_s) max_s = scores[n_cand];
                    n_cand++;
                }
            }
            float den = 0.0f, acc = 0.0f;
            for (uint32_t i = 0; i < n_cand; i++) {
                const float w = expf(scores[i] - max_s);
                den += w;
                acc += vals[i] * w;
            }
            comp[(uint64_t)c * head_dim + d] = den != 0.0f ? acc / den : 0.0f;
        }
    }
}

/* compressor_shift_ratio4_kernel (host version). */
static void ref_shift_ratio4(std::vector<float> &skv, std::vector<float> &ssc,
                             uint32_t width) {
    for (uint32_t i = 0; i < 4u * width; i++) {
        const float v = skv[4u * width + i];
        const float s = ssc[4u * width + i];
        skv[i] = v;
        ssc[i] = s;
        skv[4u * width + i] = v;
        ssc[4u * width + i] = s;
    }
}

/* One full compressor_update_tensor step (store + optional emit), matching
 * ds4.c compressor_decode_one / the CUDA update wrapper. */
static void ref_update_one(std::vector<float> &skv, std::vector<float> &ssc,
                           std::vector<float> &comp, const float *kv_cur, const float *sc_cur,
                           const unsigned char *model, uint64_t ape_offset, uint32_t ape_type,
                           uint64_t norm_offset, uint32_t head_dim, uint32_t ratio,
                           uint32_t pos, uint32_t comp_row, uint32_t n_rot,
                           uint32_t n_ctx_orig, float freq_base, float freq_scale,
                           float ext_factor, float attn_factor, float beta_fast,
                           float beta_slow, float rms_eps) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t pos_mod = pos % ratio;
    const uint32_t row = ratio == 4u ? ratio + pos_mod : pos_mod;
    const bool emit = ((pos + 1u) % ratio) == 0u;
    for (uint32_t j = 0; j < width; j++) {
        skv[(uint64_t)row * width + j] = kv_cur[j];
        ssc[(uint64_t)row * width + j] = sc_cur[j] +
            ref_ape_val(model, ape_offset, ape_type, (uint64_t)pos_mod * width + j);
    }
    if (!emit) return;
    std::vector<float> pooled(head_dim);
    ref_pool_state(pooled.data(), skv.data(), ssc.data(), head_dim, ratio);
    float *crow = comp.data() + (uint64_t)comp_row * head_dim;
    for (uint32_t i = 0; i < head_dim; i++) crow[i] = pooled[i];
    const float *nw = (const float *)(model + norm_offset);
    ref_rms_norm_weight_rows(crow, nw, head_dim, 1, rms_eps);
    ref_rope_tail_row(crow, head_dim, n_rot, pos + 1u - ratio, n_ctx_orig, freq_base,
                      freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
    if (ratio == 4u) ref_shift_ratio4(skv, ssc, width);
}

/* ds4.c dsv4_e4m3fn_value_cpu / dequant / fp8_kv_quantize_row_inplace_cpu. */
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

/* ---------- compressor_store_batch ---------- */

static int test_compressor_store_batch(void) {
    const uint32_t head_dim = 8, ratio = 4, pos0 = 3, n_tokens = 5;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    CompModel m = build_comp_model(head_dim, ratio, 0); /* f32 APE */

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc((uint64_t)n_tokens * width * sizeof(float));
    ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc((uint64_t)n_tokens * width * sizeof(float));
    ds4_gpu_tensor *skv = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *ssc = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    if (!kv || !sc || !skv || !ssc) {
        if (kv) ds4_gpu_tensor_free(kv);
        if (sc) ds4_gpu_tensor_free(sc);
        if (skv) ds4_gpu_tensor_free(skv);
        if (ssc) ds4_gpu_tensor_free(ssc);
        return 1;
    }
    std::vector<float> kvv((uint64_t)n_tokens * width), scv((uint64_t)n_tokens * width);
    synth_vals(kvv.data(), kvv.size());
    synth_vals(scv.data(), scv.size());
    if (ds4_gpu_tensor_write(kv, 0, kvv.data(), kvv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(sc, 0, scv.data(), scv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_fill_f32(skv, 123.0f, (uint64_t)state_rows * width) == 0 ||
        ds4_gpu_tensor_fill_f32(ssc, 123.0f, (uint64_t)state_rows * width) == 0) {
        ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(sc);
        ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc);
        return 1;
    }

    std::vector<float> wskv((uint64_t)state_rows * width, 123.0f);
    std::vector<float> wssc((uint64_t)state_rows * width, 123.0f);
    ref_store_batch(wskv, wssc, kvv, scv, m.data.data(), m.ape_offset, 0,
                    head_dim, ratio, pos0, n_tokens);

    int rc = 1;
    if (ds4_gpu_compressor_store_batch_tensor(
            kv, sc, skv, ssc, m.data.data(), m.data.size(), m.ape_offset, 0,
            head_dim, ratio, pos0, n_tokens) != 0) {
        std::vector<float> gkv((uint64_t)state_rows * width), gsc((uint64_t)state_rows * width);
        if (ds4_gpu_tensor_read(skv, 0, gkv.data(), gkv.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(ssc, 0, gsc.data(), gsc.size() * sizeof(float)) != 0) {
            if (check_f32("compressor_store_batch/kv", gkv.data(), wskv.data(),
                          (uint32_t)gkv.size()) != 0) rc = 1;
            else if (check_f32("compressor_store_batch/sc", gsc.data(), wssc.data(),
                               (uint32_t)gsc.size()) != 0) rc = 1;
            else rc = 0;
        }
    }
    ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(sc);
    ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc);
    return rc;
}
REGISTER_TEST(compressor_store_batch, test_compressor_store_batch);

/* f16 APE path (ape_type == 1) and ratio != 4. */
static int test_compressor_store_batch_f16(void) {
    const uint32_t head_dim = 8, ratio = 2, pos0 = 1, n_tokens = 2;
    const uint32_t width = head_dim;
    const uint32_t state_rows = ratio;
    CompModel m = build_comp_model(head_dim, ratio, 1); /* f16 APE */

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc((uint64_t)n_tokens * width * sizeof(float));
    ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc((uint64_t)n_tokens * width * sizeof(float));
    ds4_gpu_tensor *skv = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *ssc = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    if (!kv || !sc || !skv || !ssc) {
        if (kv) ds4_gpu_tensor_free(kv);
        if (sc) ds4_gpu_tensor_free(sc);
        if (skv) ds4_gpu_tensor_free(skv);
        if (ssc) ds4_gpu_tensor_free(ssc);
        return 1;
    }
    std::vector<float> kvv((uint64_t)n_tokens * width), scv((uint64_t)n_tokens * width);
    synth_vals(kvv.data(), kvv.size());
    synth_vals(scv.data(), scv.size());
    if (ds4_gpu_tensor_write(kv, 0, kvv.data(), kvv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(sc, 0, scv.data(), scv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_fill_f32(skv, -1.0f, (uint64_t)state_rows * width) == 0 ||
        ds4_gpu_tensor_fill_f32(ssc, -1.0f, (uint64_t)state_rows * width) == 0) {
        ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(sc);
        ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc);
        return 1;
    }

    std::vector<float> wskv((uint64_t)state_rows * width, -1.0f);
    std::vector<float> wssc((uint64_t)state_rows * width, -1.0f);
    ref_store_batch(wskv, wssc, kvv, scv, m.data.data(), m.ape_offset, 1,
                    head_dim, ratio, pos0, n_tokens);

    int rc = 1;
    if (ds4_gpu_compressor_store_batch_tensor(
            kv, sc, skv, ssc, m.data.data(), m.data.size(), m.ape_offset, 1,
            head_dim, ratio, pos0, n_tokens) != 0) {
        std::vector<float> gkv((uint64_t)state_rows * width), gsc((uint64_t)state_rows * width);
        if (ds4_gpu_tensor_read(skv, 0, gkv.data(), gkv.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(ssc, 0, gsc.data(), gsc.size() * sizeof(float)) != 0) {
            if (check_f32("compressor_store_batch_f16/kv", gkv.data(), wskv.data(),
                          (uint32_t)gkv.size()) != 0) rc = 1;
            else if (check_f32("compressor_store_batch_f16/sc", gsc.data(), wssc.data(),
                               (uint32_t)gsc.size()) != 0) rc = 1;
            else rc = 0;
        }
    }
    ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(sc);
    ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc);
    return rc;
}
REGISTER_TEST(compressor_store_batch_f16, test_compressor_store_batch_f16);

/* ---------- compressor_update ---------- */

static const uint32_t kUDim = 8;
static const uint32_t kURot = 4;
static const float kUFreqBase = 10000.0f;
static const float kUFreqScale = 1.0f;
static const float kUExtFactor = 0.0f;
static const float kUAttnFactor = 1.0f;
static const float kUBetaFast = 32.0f;
static const float kUBetaSlow = 1.0f;
static const float kURmsEps = 1e-5f;

/* Drive a full streaming sequence with the backend and compare against the
 * CPU reference.  Returns 0 on PASS. */
static int update_sequence(const char *what, uint32_t head_dim, uint32_t ratio,
                           uint32_t n_steps, uint32_t n_comp_rows) {
    (void)what;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    CompModel m = build_comp_model(head_dim, ratio, 0);

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc((uint64_t)n_steps * width * sizeof(float));
    ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc((uint64_t)n_steps * width * sizeof(float));
    ds4_gpu_tensor *skv = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *ssc = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc((uint64_t)n_comp_rows * head_dim * sizeof(float));
    if (!kv || !sc || !skv || !ssc || !comp) {
        if (kv) ds4_gpu_tensor_free(kv);
        if (sc) ds4_gpu_tensor_free(sc);
        if (skv) ds4_gpu_tensor_free(skv);
        if (ssc) ds4_gpu_tensor_free(ssc);
        if (comp) ds4_gpu_tensor_free(comp);
        return 1;
    }
    std::vector<float> kvv((uint64_t)n_steps * width), scv((uint64_t)n_steps * width);
    synth_vals(kvv.data(), kvv.size());
    synth_vals(scv.data(), scv.size());
    if (ds4_gpu_tensor_write(kv, 0, kvv.data(), kvv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(sc, 0, scv.data(), scv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_fill_f32(skv, 0.0f, (uint64_t)state_rows * width) == 0 ||
        ds4_gpu_tensor_fill_f32(ssc, -INFINITY, (uint64_t)state_rows * width) == 0 ||
        ds4_gpu_tensor_fill_f32(comp, -7.0f, (uint64_t)n_comp_rows * head_dim) == 0) {
        ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(sc);
        ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc); ds4_gpu_tensor_free(comp);
        return 1;
    }

    std::vector<float> wskv((uint64_t)state_rows * width, 0.0f);
    std::vector<float> wssc((uint64_t)state_rows * width, -INFINITY);
    std::vector<float> wcomp((uint64_t)n_comp_rows * head_dim, -7.0f);

    int rc = 1;
    uint32_t comp_row = 0;
    int ok = 1;
    for (uint32_t pos = 0; pos < n_steps; pos++) {
        ds4_gpu_tensor kv_view, sc_view;
        kv_view.ptr = (char *)kv->ptr + (uint64_t)pos * width * sizeof(float);
        kv_view.bytes = (uint64_t)width * sizeof(float);
        kv_view.owner = 0;
        kv_view.device_id = kv->device_id;
        sc_view.ptr = (char *)sc->ptr + (uint64_t)pos * width * sizeof(float);
        sc_view.bytes = (uint64_t)width * sizeof(float);
        sc_view.owner = 0;
        sc_view.device_id = sc->device_id;
        if (!ds4_gpu_compressor_update_tensor(
                &kv_view, &sc_view, skv, ssc, comp, m.data.data(), m.data.size(),
                m.ape_offset, 0, m.norm_offset, 0,
                head_dim, ratio, pos, comp_row,
                kURot, 0, kUFreqBase, kUFreqScale, kUExtFactor,
                kUAttnFactor, kUBetaFast, kUBetaSlow, kURmsEps, false)) {
            ok = 0;
            break;
        }
        ref_update_one(wskv, wssc, wcomp, kvv.data() + (uint64_t)pos * width,
                       scv.data() + (uint64_t)pos * width, m.data.data(), m.ape_offset, 0,
                       m.norm_offset, head_dim, ratio, pos, comp_row,
                       kURot, 0, kUFreqBase, kUFreqScale, kUExtFactor,
                       kUAttnFactor, kUBetaFast, kUBetaSlow, kURmsEps);
        if (((pos + 1u) % ratio) == 0u) comp_row++;
    }
    if (ok) {
        std::vector<float> gkv((uint64_t)state_rows * width), gsc((uint64_t)state_rows * width);
        std::vector<float> gc((uint64_t)n_comp_rows * head_dim);
        if (ds4_gpu_tensor_read(skv, 0, gkv.data(), gkv.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(ssc, 0, gsc.data(), gsc.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(comp, 0, gc.data(), gc.size() * sizeof(float)) != 0) {
            if (check_f32("update/comp", gc.data(), wcomp.data(), (uint32_t)gc.size()) != 0) rc = 1;
            else if (check_f32("update/state_kv", gkv.data(), wskv.data(), (uint32_t)gkv.size()) != 0) rc = 1;
            else if (check_f32("update/state_score", gsc.data(), wssc.data(), (uint32_t)gsc.size()) != 0) rc = 1;
            else rc = 0;
        }
    }
    ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(sc);
    ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc); ds4_gpu_tensor_free(comp);
    return rc;
}

static int test_compressor_update_emit_ratio4(void) {
    /* 8 tokens at ratio 4 → emits at pos 3 and 7; rope positions 0 and 4. */
    return update_sequence("update/ratio4", kUDim, 4, 8, 2);
}
REGISTER_TEST(compressor_update_emit_ratio4, test_compressor_update_emit_ratio4);

static int test_compressor_update_emit_ratio2(void) {
    /* 4 tokens at ratio 2 → emits at pos 1 and 3; rope positions 0 and 2. */
    return update_sequence("update/ratio2", kUDim, 2, 4, 2);
}
REGISTER_TEST(compressor_update_emit_ratio2, test_compressor_update_emit_ratio2);

/* Non-boundary update: only the state row is written, comp_cache untouched. */
static int test_compressor_update_non_emit(void) {
    const uint32_t head_dim = 8, ratio = 4, pos = 2;
    const uint32_t width = 16, state_rows = 8;
    CompModel m = build_comp_model(head_dim, ratio, 0);

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(4ull * width * sizeof(float));
    ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc(4ull * width * sizeof(float));
    ds4_gpu_tensor *skv = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *ssc = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc((uint64_t)head_dim * sizeof(float));
    if (!kv || !sc || !skv || !ssc || !comp) {
        if (kv) ds4_gpu_tensor_free(kv);
        if (sc) ds4_gpu_tensor_free(sc);
        if (skv) ds4_gpu_tensor_free(skv);
        if (ssc) ds4_gpu_tensor_free(ssc);
        if (comp) ds4_gpu_tensor_free(comp);
        return 1;
    }
    std::vector<float> kvv(width), scv(width);
    synth_vals(kvv.data(), kvv.size());
    synth_vals(scv.data(), scv.size());
    if (ds4_gpu_tensor_write(kv, 0, kvv.data(), kvv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(sc, 0, scv.data(), scv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_fill_f32(skv, 0.0f, (uint64_t)state_rows * width) == 0 ||
        ds4_gpu_tensor_fill_f32(ssc, -INFINITY, (uint64_t)state_rows * width) == 0 ||
        ds4_gpu_tensor_fill_f32(comp, 42.0f, head_dim) == 0) {
        ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(sc);
        ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc); ds4_gpu_tensor_free(comp);
        return 1;
    }

    std::vector<float> wskv((uint64_t)state_rows * width, 0.0f);
    std::vector<float> wssc((uint64_t)state_rows * width, -INFINITY);
    std::vector<float> wcomp_empty; /* never dereferenced (non-emit step) */
    ref_update_one(wskv, wssc, wcomp_empty, kvv.data(), scv.data(),
                   m.data.data(), m.ape_offset, 0, m.norm_offset, head_dim, ratio,
                   pos, 0, kURot, 0, kUFreqBase, kUFreqScale, kUExtFactor,
                   kUAttnFactor, kUBetaFast, kUBetaSlow, kURmsEps);

    int rc = 1;
    if (ds4_gpu_compressor_update_tensor(
            kv, sc, skv, ssc, comp, m.data.data(), m.data.size(),
            m.ape_offset, 0, m.norm_offset, 0, head_dim, ratio, pos, 0,
            kURot, 0, kUFreqBase, kUFreqScale, kUExtFactor,
            kUAttnFactor, kUBetaFast, kUBetaSlow, kURmsEps, false) != 0) {
        std::vector<float> gkv((uint64_t)state_rows * width), gsc((uint64_t)state_rows * width);
        std::vector<float> gc(head_dim);
        if (ds4_gpu_tensor_read(skv, 0, gkv.data(), gkv.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(ssc, 0, gsc.data(), gsc.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(comp, 0, gc.data(), gc.size() * sizeof(float)) != 0) {
            std::vector<float> wcomp(head_dim, 42.0f);
            if (check_f32("update_non_emit/comp_untouched", gc.data(), wcomp.data(),
                          head_dim) != 0) rc = 1;
            else if (check_f32("update_non_emit/state_kv", gkv.data(), wskv.data(),
                               (uint32_t)gkv.size()) != 0) rc = 1;
            else if (check_f32("update_non_emit/state_score", gsc.data(), wssc.data(),
                               (uint32_t)gsc.size()) != 0) rc = 1;
            else rc = 0;
        }
    }
    ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(sc);
    ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc); ds4_gpu_tensor_free(comp);
    return rc;
}
REGISTER_TEST(compressor_update_non_emit, test_compressor_update_non_emit);

/* state_already_stored=true: the caller pre-populated the state with a batch
 * store; update must only pool/emit and shift without re-storing. */
static int test_compressor_update_state_already_stored(void) {
    const uint32_t head_dim = 8, ratio = 4, pos = 3;
    const uint32_t width = 16, state_rows = 8;
    CompModel m = build_comp_model(head_dim, ratio, 0);

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc((uint64_t)4 * width * sizeof(float));
    ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc((uint64_t)4 * width * sizeof(float));
    ds4_gpu_tensor *skv = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *ssc = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc((uint64_t)head_dim * sizeof(float));
    if (!kv || !sc || !skv || !ssc || !comp) {
        if (kv) ds4_gpu_tensor_free(kv);
        if (sc) ds4_gpu_tensor_free(sc);
        if (skv) ds4_gpu_tensor_free(skv);
        if (ssc) ds4_gpu_tensor_free(ssc);
        if (comp) ds4_gpu_tensor_free(comp);
        return 1;
    }
    std::vector<float> kvv(4ull * width), scv(4ull * width);
    synth_vals(kvv.data(), kvv.size());
    synth_vals(scv.data(), scv.size());
    if (ds4_gpu_tensor_write(kv, 0, kvv.data(), kvv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(sc, 0, scv.data(), scv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_fill_f32(skv, 0.0f, (uint64_t)state_rows * width) == 0 ||
        ds4_gpu_tensor_fill_f32(ssc, -INFINITY, (uint64_t)state_rows * width) == 0 ||
        ds4_gpu_tensor_fill_f32(comp, 0.0f, head_dim) == 0) {
        ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(sc);
        ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc); ds4_gpu_tensor_free(comp);
        return 1;
    }
    if (ds4_gpu_compressor_store_batch_tensor(
            kv, sc, skv, ssc, m.data.data(), m.data.size(), m.ape_offset, 0,
            head_dim, ratio, 0, 4) == 0) {
        ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(sc);
        ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc); ds4_gpu_tensor_free(comp);
        return 1;
    }

    /* Reference: state = batch store of tokens 0..3, then update at pos 3
     * with state_already_stored (no store, just emit + shift). */
    std::vector<float> wskv((uint64_t)state_rows * width, 0.0f);
    std::vector<float> wssc((uint64_t)state_rows * width, -INFINITY);
    ref_store_batch(wskv, wssc, kvv, scv, m.data.data(), m.ape_offset, 0,
                    head_dim, ratio, 0, 4);
    std::vector<float> wcomp(head_dim, 0.0f);
    ref_update_one(wskv, wssc, wcomp, kvv.data() + 3ull * width,
                   scv.data() + 3ull * width, m.data.data(), m.ape_offset, 0,
                   m.norm_offset, head_dim, ratio, pos, 0,
                   kURot, 0, kUFreqBase, kUFreqScale, kUExtFactor,
                   kUAttnFactor, kUBetaFast, kUBetaSlow, kURmsEps);

    int rc = 1;
    if (ds4_gpu_compressor_update_tensor(
            kv, sc, skv, ssc, comp, m.data.data(), m.data.size(),
            m.ape_offset, 0, m.norm_offset, 0, head_dim, ratio, pos, 0,
            kURot, 0, kUFreqBase, kUFreqScale, kUExtFactor,
            kUAttnFactor, kUBetaFast, kUBetaSlow, kURmsEps, true) != 0) {
        std::vector<float> gkv((uint64_t)state_rows * width), gsc((uint64_t)state_rows * width);
        std::vector<float> gc(head_dim);
        if (ds4_gpu_tensor_read(skv, 0, gkv.data(), gkv.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(ssc, 0, gsc.data(), gsc.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(comp, 0, gc.data(), gc.size() * sizeof(float)) != 0) {
            if (check_f32("update_state_already_stored/comp", gc.data(), wcomp.data(),
                          head_dim) != 0) rc = 1;
            else if (check_f32("update_state_already_stored/state_kv", gkv.data(), wskv.data(),
                               (uint32_t)gkv.size()) != 0) rc = 1;
            else if (check_f32("update_state_already_stored/state_score", gsc.data(), wssc.data(),
                               (uint32_t)gsc.size()) != 0) rc = 1;
            else rc = 0;
        }
    }
    ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(sc);
    ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc); ds4_gpu_tensor_free(comp);
    return rc;
}
REGISTER_TEST(compressor_update_state_already_stored, test_compressor_update_state_already_stored);

/* ---------- compressor_prefill ---------- */

static int prefill_one(const char *what, uint32_t head_dim, uint32_t ratio,
                       uint32_t pos0, uint32_t n_tokens, uint32_t n_rot,
                       bool quantize_fp8) {
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    CompModel m = build_comp_model(head_dim, ratio, 0);

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc((uint64_t)n_tokens * width * sizeof(float));
    ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc((uint64_t)n_tokens * width * sizeof(float));
    ds4_gpu_tensor *skv = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *ssc = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc((uint64_t)(n_comp ? n_comp : 1) * head_dim * sizeof(float));
    if (!kv || !sc || !skv || !ssc || !comp) {
        if (kv) ds4_gpu_tensor_free(kv);
        if (sc) ds4_gpu_tensor_free(sc);
        if (skv) ds4_gpu_tensor_free(skv);
        if (ssc) ds4_gpu_tensor_free(ssc);
        if (comp) ds4_gpu_tensor_free(comp);
        return 1;
    }
    std::vector<float> kvv((uint64_t)n_tokens * width), scv((uint64_t)n_tokens * width);
    synth_vals(kvv.data(), kvv.size());
    synth_vals(scv.data(), scv.size());
    if (ds4_gpu_tensor_write(kv, 0, kvv.data(), kvv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(sc, 0, scv.data(), scv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_fill_f32(skv, 0.0f, (uint64_t)state_rows * width) == 0 ||
        ds4_gpu_tensor_fill_f32(ssc, -INFINITY, (uint64_t)state_rows * width) == 0 ||
        ds4_gpu_tensor_fill_f32(comp, 0.0f, (uint64_t)(n_comp ? n_comp : 1) * head_dim) == 0) {
        ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(sc);
        ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc); ds4_gpu_tensor_free(comp);
        return 1;
    }

    /* CPU reference.  The prefill state rows are filled with
     * compressor_set_rows semantics (phase from (pos0 + src), contiguous dst
     * rows), which differs from store_batch's modulo row mapping. */
    std::vector<float> wskv((uint64_t)state_rows * width, 0.0f);
    std::vector<float> wssc((uint64_t)state_rows * width, -INFINITY);
    const uint32_t cutoff = n_comp * ratio;
    const uint32_t rem = n_tokens - cutoff;
    if (ratio == 4u) {
        if (cutoff >= ratio) {
            const uint32_t src0 = cutoff - ratio;
            for (uint32_t r = 0; r < ratio; r++) {
                const uint32_t src = src0 + r;
                const uint32_t phase = (pos0 + src) % ratio;
                for (uint32_t j = 0; j < width; j++) {
                    wskv[(uint64_t)r * width + j] = kvv[(uint64_t)src * width + j];
                    wssc[(uint64_t)r * width + j] = scv[(uint64_t)src * width + j] +
                        ref_ape_val(m.data.data(), m.ape_offset, 0, (uint64_t)phase * width + j);
                }
            }
        }
        if (rem != 0) {
            for (uint32_t r = 0; r < rem; r++) {
                const uint32_t src = cutoff + r;
                const uint32_t phase = (pos0 + src) % ratio;
                for (uint32_t j = 0; j < width; j++) {
                    wskv[(uint64_t)(ratio + r) * width + j] = kvv[(uint64_t)src * width + j];
                    wssc[(uint64_t)(ratio + r) * width + j] = scv[(uint64_t)src * width + j] +
                        ref_ape_val(m.data.data(), m.ape_offset, 0, (uint64_t)phase * width + j);
                }
            }
        }
    } else if (rem != 0) {
        for (uint32_t r = 0; r < rem; r++) {
            const uint32_t src = cutoff + r;
            const uint32_t phase = (pos0 + src) % ratio;
            for (uint32_t j = 0; j < width; j++) {
                wskv[(uint64_t)r * width + j] = kvv[(uint64_t)src * width + j];
                wssc[(uint64_t)r * width + j] = scv[(uint64_t)src * width + j] +
                    ref_ape_val(m.data.data(), m.ape_offset, 0, (uint64_t)phase * width + j);
            }
        }
    }
    std::vector<float> wcomp((uint64_t)(n_comp ? n_comp : 1) * head_dim, 0.0f);
    if (n_comp) {
        ref_prefill_pool(wcomp, kvv, scv, m.data.data(), m.ape_offset, 0,
                         head_dim, ratio, pos0, n_comp);
        const float *nw = (const float *)(m.data.data() + m.norm_offset);
        ref_rms_norm_weight_rows(wcomp.data(), nw, head_dim, n_comp, kURmsEps);
        if (n_rot) {
            for (uint32_t c = 0; c < n_comp; c++)
                ref_rope_tail_row(wcomp.data() + (uint64_t)c * head_dim, head_dim, n_rot,
                                  pos0 + c * ratio, 0, kUFreqBase, kUFreqScale,
                                  kUExtFactor, kUAttnFactor, kUBetaFast, kUBetaSlow);
        }
        if (quantize_fp8) {
            for (uint32_t c = 0; c < n_comp; c++)
                ref_fp8_kv_quantize_row(wcomp.data() + (uint64_t)c * head_dim, head_dim, n_rot);
        }
    }

    int rc = 1;
    if (ds4_gpu_compressor_prefill_tensor(
            comp, skv, ssc, kv, sc, m.data.data(), m.data.size(),
            m.ape_offset, 0, m.norm_offset, 0,
            head_dim, ratio, pos0, n_tokens, n_rot, 0, quantize_fp8,
            kUFreqBase, kUFreqScale, kUExtFactor, kUAttnFactor,
            kUBetaFast, kUBetaSlow, kURmsEps) != 0) {
        std::vector<float> gkv((uint64_t)state_rows * width), gsc((uint64_t)state_rows * width);
        std::vector<float> gc((uint64_t)(n_comp ? n_comp : 1) * head_dim);
        if (ds4_gpu_tensor_read(skv, 0, gkv.data(), gkv.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(ssc, 0, gsc.data(), gsc.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(comp, 0, gc.data(), gc.size() * sizeof(float)) != 0) {
            if (n_comp && check_f32(what, gc.data(), wcomp.data(), n_comp * head_dim) != 0) rc = 1;
            else if (check_f32("prefill/state_kv", gkv.data(), wskv.data(),
                               (uint32_t)gkv.size()) != 0) rc = 1;
            else if (check_f32("prefill/state_score", gsc.data(), wssc.data(),
                               (uint32_t)gsc.size()) != 0) rc = 1;
            else rc = 0;
        }
    }
    ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(sc);
    ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc); ds4_gpu_tensor_free(comp);
    return rc;
}

static int test_compressor_prefill_ratio4(void) {
    /* Two full windows + 3 leftover tokens; rope positions 1 and 5. */
    return prefill_one("prefill/ratio4", 8, 4, 1, 11, 4, false);
}
REGISTER_TEST(compressor_prefill_ratio4, test_compressor_prefill_ratio4);

static int test_compressor_prefill_ratio2(void) {
    /* Three full windows + 1 leftover token, no rope (n_rot = 0). */
    return prefill_one("prefill/ratio2", 8, 2, 2, 7, 0, false);
}
REGISTER_TEST(compressor_prefill_ratio2, test_compressor_prefill_ratio2);

static int test_compressor_prefill_fp8(void) {
    /* head_dim 80 / n_rot 16 → n_nope 64 (one full fp8 block); ratio 2 with
     * 3 windows + 1 leftover so the rolling state is non-trivial. */
    return prefill_one("prefill/fp8", 80, 2, 0, 7, 16, true);
}
REGISTER_TEST(compressor_prefill_fp8, test_compressor_prefill_fp8);

/* ---------- bounds / safety ---------- */

static int test_compressor_bounds(void) {
    const uint32_t head_dim = 8, ratio = 4;
    const uint32_t width = 16, state_rows = 8;
    CompModel m = build_comp_model(head_dim, ratio, 0);

    ds4_gpu_tensor *kv = ds4_gpu_tensor_alloc(4ull * width * sizeof(float));
    ds4_gpu_tensor *sc = ds4_gpu_tensor_alloc(4ull * width * sizeof(float));
    ds4_gpu_tensor *skv_small = ds4_gpu_tensor_alloc((uint64_t)(state_rows - 1) * width * sizeof(float));
    ds4_gpu_tensor *skv = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *ssc = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *comp_small = ds4_gpu_tensor_alloc(4); /* too small for any emit */
    ds4_gpu_tensor *comp = ds4_gpu_tensor_alloc((uint64_t)head_dim * sizeof(float));
    if (!kv || !sc || !skv_small || !skv || !ssc || !comp_small || !comp) {
        if (kv) ds4_gpu_tensor_free(kv);
        if (sc) ds4_gpu_tensor_free(sc);
        if (skv_small) ds4_gpu_tensor_free(skv_small);
        if (skv) ds4_gpu_tensor_free(skv);
        if (ssc) ds4_gpu_tensor_free(ssc);
        if (comp_small) ds4_gpu_tensor_free(comp_small);
        if (comp) ds4_gpu_tensor_free(comp);
        return 1;
    }
    int rc = 0;

    /* store_batch with undersized state → 0. */
    if (ds4_gpu_compressor_store_batch_tensor(
            kv, sc, skv_small, ssc, m.data.data(), m.data.size(), m.ape_offset, 0,
            head_dim, ratio, 0, 1) != 0) {
        fprintf(stderr, "--- bounds: store_batch accepted undersized state\n");
        rc = 1;
    }
    /* update at an emit boundary with undersized comp_cache → 0. */
    if (ds4_gpu_compressor_update_tensor(
            kv, sc, skv, ssc, comp_small, m.data.data(), m.data.size(),
            m.ape_offset, 0, m.norm_offset, 0, head_dim, ratio, 3, 0,
            kURot, 0, kUFreqBase, kUFreqScale, kUExtFactor,
            kUAttnFactor, kUBetaFast, kUBetaSlow, kURmsEps, false) != 0) {
        fprintf(stderr, "--- bounds: update accepted undersized comp_cache at emit\n");
        rc = 1;
    }
    /* update at a non-emit boundary with tiny comp_cache → 1 (no write). */
    if (ds4_gpu_compressor_update_tensor(
            kv, sc, skv, ssc, comp_small, m.data.data(), m.data.size(),
            m.ape_offset, 0, m.norm_offset, 0, head_dim, ratio, 0, 0,
            kURot, 0, kUFreqBase, kUFreqScale, kUExtFactor,
            kUAttnFactor, kUBetaFast, kUBetaSlow, kURmsEps, false) == 0) {
        fprintf(stderr, "--- bounds: update failed on non-emit with small comp_cache\n");
        rc = 1;
    }
    /* prefill with undersized comp_cache (n_comp > 0) → 0. */
    if (ds4_gpu_compressor_prefill_tensor(
            comp_small, skv, ssc, kv, sc, m.data.data(), m.data.size(),
            m.ape_offset, 0, m.norm_offset, 0, head_dim, ratio, 0, 4,
            kURot, 0, false, kUFreqBase, kUFreqScale, kUExtFactor,
            kUAttnFactor, kUBetaFast, kUBetaSlow, kURmsEps) != 0) {
        fprintf(stderr, "--- bounds: prefill accepted undersized comp_cache\n");
        rc = 1;
    }
    /* prefill with n_comp == 0 (only leftover state rows) → 1 even with a
     * 4-byte comp_cache, and the leftover state rows are populated. */
    if (ds4_gpu_compressor_prefill_tensor(
            comp_small, skv, ssc, kv, sc, m.data.data(), m.data.size(),
            m.ape_offset, 0, m.norm_offset, 0, head_dim, ratio, 0, 2,
            kURot, 0, false, kUFreqBase, kUFreqScale, kUExtFactor,
            kUAttnFactor, kUBetaFast, kUBetaSlow, kURmsEps) == 0) {
        fprintf(stderr, "--- bounds: prefill failed on n_comp==0 path\n");
        rc = 1;
    }
    /* prefill with a broken APE range (offset past model end) → 0. */
    if (ds4_gpu_compressor_prefill_tensor(
            comp, skv, ssc, kv, sc, m.data.data(), m.data.size(),
            m.data.size() + 8, 0, m.norm_offset, 0, head_dim, ratio, 0, 4,
            kURot, 0, false, kUFreqBase, kUFreqScale, kUExtFactor,
            kUAttnFactor, kUBetaFast, kUBetaSlow, kURmsEps) != 0) {
        fprintf(stderr, "--- bounds: prefill accepted out-of-range APE offset\n");
        rc = 1;
    }

    ds4_gpu_tensor_free(kv); ds4_gpu_tensor_free(sc);
    ds4_gpu_tensor_free(skv_small); ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc);
    ds4_gpu_tensor_free(comp_small); ds4_gpu_tensor_free(comp);
    return rc;
}
REGISTER_TEST(compressor_bounds, test_compressor_bounds);
