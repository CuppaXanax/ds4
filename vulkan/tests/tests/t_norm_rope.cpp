/* Kernel tests for the norm/RoPE family:
 *
 *   ds4_gpu_rms_norm_plain_rows_tensor
 *   ds4_gpu_rms_norm_weight_rows_tensor
 *   ds4_gpu_rope_tail_tensor
 *   ds4_gpu_head_rms_norm_tensor
 *   ds4_gpu_head_rms_norm_rope_tail_tensor
 *
 * All five are host-side in the backend (they operate on tensor->ptr
 * directly), so the tests do NOT wrap the calls in begin/end_commands.
 * The CPU reference is copied verbatim from ds4.c (rms_norm_no_weight,
 * rms_norm_weight, head_rms_norm_inplace, rope_tail_ext_inplace), so this
 * validates the exact eps / layout / RoPE frequency semantics of the
 * engine's CPU path.
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>

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

/* ---------- CPU references (copied from ds4.c) ---------- */

/* rms_norm without a learned scale, rows version (ds4.c rms_norm_no_weight). */
static void ref_rms_norm_plain_rows(float *out, const float *x, uint32_t n,
                                    uint32_t rows, float eps) {
    for (uint32_t r = 0; r < rows; r++) {
        const float *xr = x + (uint64_t)r * n;
        double ss = 0.0;
        for (uint32_t i = 0; i < n; i++) ss += (double)xr[i] * xr[i];
        const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
        for (uint32_t i = 0; i < n; i++) out[(uint64_t)r * n + i] = xr[i] * scale;
    }
}

/* rms_norm with a learned per-channel scale (ds4.c rms_norm_weight). */
static void ref_rms_norm_weight_rows(float *out, const float *x, const float *w,
                                     uint32_t n, uint32_t rows, float eps) {
    for (uint32_t r = 0; r < rows; r++) {
        const float *xr = x + (uint64_t)r * n;
        double ss = 0.0;
        for (uint32_t i = 0; i < n; i++) ss += (double)xr[i] * xr[i];
        const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
        for (uint32_t i = 0; i < n; i++) out[(uint64_t)r * n + i] = xr[i] * scale * w[i];
    }
}

/* Per-head normalization of a [n_tok][n_head][head_dim] tensor
 * (ds4.c head_rms_norm_inplace applied per token). */
static void ref_head_rms_norm(float *x, uint32_t n_tok, uint32_t n_head,
                              uint32_t head_dim, float eps) {
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t h = 0; h < n_head; h++) {
            float *head = x + ((uint64_t)t * n_head + h) * head_dim;
            double ss = 0.0;
            for (uint32_t i = 0; i < head_dim; i++) ss += (double)head[i] * head[i];
            const float scale = 1.0f / sqrtf((float)(ss / (double)head_dim) + eps);
            for (uint32_t i = 0; i < head_dim; i++) head[i] *= scale;
        }
    }
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
            float theta_extrap = (float)pos;
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

/* Deterministic synthetic row-major tensor: alternating signs, in-range
 * values so rms/rope never blow up. */
static void synth_row_major(float *v, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        float f = (float)(i + 1) * 0.375f + 0.125f * (float)(i % 7);
        if ((i & 1u) != 0) f = -f;
        v[i] = f;
    }
}

/* ---------- rms_norm_plain_rows ---------- */

static int test_rms_norm_plain_rows(void) {
    const uint32_t n = 16, rows = 4;
    const float eps = 1e-5f;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n * rows * sizeof(float));
    ds4_gpu_tensor *o = ds4_gpu_tensor_alloc((uint64_t)n * rows * sizeof(float));
    if (!x || !o) {
        if (x) ds4_gpu_tensor_free(x);
        if (o) ds4_gpu_tensor_free(o);
        return 1;
    }
    float xv[n * rows];
    synth_row_major(xv, n * rows);
    if (ds4_gpu_tensor_write(x, 0, xv, sizeof(xv)) == 0) {
        ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(o); return 1;
    }
    float want[n * rows];
    ref_rms_norm_plain_rows(want, xv, n, rows, eps);

    int rc = 1;
    if (ds4_gpu_rms_norm_plain_rows_tensor(o, x, n, rows, eps) != 0) {
        float got[n * rows];
        if (ds4_gpu_tensor_read(o, 0, got, sizeof(got)) != 0)
            rc = check_f32("rms_norm_plain_rows", got, want, n * rows);
    }
    ds4_gpu_tensor_free(o);
    ds4_gpu_tensor_free(x);
    return rc;
}
REGISTER_TEST(rms_norm_plain_rows, test_rms_norm_plain_rows);

/* ---------- rms_norm_weight_rows ---------- */

static int test_rms_norm_weight_rows(void) {
    const uint32_t n = 16, rows = 3;
    const float eps = 1e-5f;
    const uint64_t header = 16;
    const uint64_t model_size = header + (uint64_t)n * sizeof(float);
    unsigned char *model = (unsigned char *)malloc(model_size);
    if (!model) return 1;
    std::memset(model, 0xAA, header);
    float *w = (float *)(model + header);
    for (uint32_t i = 0; i < n; i++) w[i] = 0.5f + 0.0625f * (float)(i + 1); /* positive, varied */
    if (ds4_gpu_set_model_map(model, model_size) == 0) { free(model); return 1; }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)n * rows * sizeof(float));
    ds4_gpu_tensor *o = ds4_gpu_tensor_alloc((uint64_t)n * rows * sizeof(float));
    if (!x || !o) {
        if (x) ds4_gpu_tensor_free(x);
        if (o) ds4_gpu_tensor_free(o);
        free(model);
        return 1;
    }
    float xv[n * rows];
    synth_row_major(xv, n * rows);
    if (ds4_gpu_tensor_write(x, 0, xv, sizeof(xv)) == 0) {
        ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(o); free(model);
        return 1;
    }
    float want[n * rows];
    ref_rms_norm_weight_rows(want, xv, w, n, rows, eps);

    int rc = 1;
    if (ds4_gpu_rms_norm_weight_rows_tensor(o, x, model, model_size, header,
                                            n, rows, eps) != 0) {
        float got[n * rows];
        if (ds4_gpu_tensor_read(o, 0, got, sizeof(got)) != 0)
            rc = check_f32("rms_norm_weight_rows", got, want, n * rows);
    }
    ds4_gpu_tensor_free(o);
    ds4_gpu_tensor_free(x);
    free(model);
    return rc;
}
REGISTER_TEST(rms_norm_weight_rows, test_rms_norm_weight_rows);

/* ---------- rope_tail ---------- */

/* Run one rope_tail scenario against the ds4.c reference.  The synthetic
 * input is written to a fresh tensor; the same buffer is normalized by the
 * reference so the prefix (n_nope dims) is checked to stay untouched too. */
static int rope_tail_one(const char *what, uint32_t n_tok, uint32_t n_head,
                         uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
                         uint64_t n_ctx_orig, bool inverse, float freq_base,
                         float freq_scale, float ext_factor, float attn_factor,
                         float beta_fast, float beta_slow) {
    const uint64_t elems = (uint64_t)n_tok * n_head * head_dim;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(elems * sizeof(float));
    if (!x) return 1;
    float *xv = (float *)malloc(elems * sizeof(float));
    if (!xv) { ds4_gpu_tensor_free(x); return 1; }
    synth_row_major(xv, elems);
    if (ds4_gpu_tensor_write(x, 0, xv, elems * sizeof(float)) == 0) {
        free(xv); ds4_gpu_tensor_free(x); return 1;
    }
    /* Reference runs on a private copy. */
    float *want = (float *)malloc(elems * sizeof(float));
    if (!want) { free(xv); ds4_gpu_tensor_free(x); return 1; }
    std::memcpy(want, xv, elems * sizeof(float));
    ref_rope_tail(want, n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig,
                  freq_base, freq_scale, ext_factor, attn_factor, beta_fast,
                  beta_slow, inverse);

    int rc = 1;
    if (ds4_gpu_rope_tail_tensor(x, n_tok, n_head, head_dim, n_rot, pos0,
                                 (uint32_t)n_ctx_orig, inverse, freq_base,
                                 freq_scale, ext_factor, attn_factor, beta_fast,
                                 beta_slow) != 0) {
        float *got = (float *)malloc(elems * sizeof(float));
        if (!got) { free(want); free(xv); ds4_gpu_tensor_free(x); return 1; }
        if (ds4_gpu_tensor_read(x, 0, got, elems * sizeof(float)) != 0)
            rc = check_f32(what, got, want, (uint32_t)elems);
        free(got);
    }
    free(want);
    free(xv);
    ds4_gpu_tensor_free(x);
    return rc;
}

static int test_rope_tail(void) {
    /* (a) Dense-layer shape: head_dim == n_rot, no YaRN (ext_factor 0). */
    if (rope_tail_one("rope_tail/dense-fwd", 2, 3, 32, 32, 5, 0, false,
                      10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f) != 0) return 1;
    /* (b) Same shape, inverse rotation must undo the forward one. */
    if (rope_tail_one("rope_tail/dense-inv", 2, 3, 32, 32, 5, 0, true,
                      10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f) != 0) return 1;
    /* (c) Compressed-layer shape: n_rot < head_dim, so the first
     *     n_nope = head_dim - n_rot dims of every head must stay intact. */
    if (rope_tail_one("rope_tail/comp-fwd", 2, 2, 64, 32, 123, 0, false,
                      10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f) != 0) return 1;
    /* (d) Same compressed shape with inverse + long position. */
    if (rope_tail_one("rope_tail/comp-inv", 2, 2, 64, 32, 1234, 0, true,
                      10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f) != 0) return 1;
    /* (e) YaRN extrapolation/interpolation: ext_factor != 0 with
     *     n_ctx_orig > 0; exercises corr_dims + ramp + mscale paths. */
    if (rope_tail_one("rope_tail/yarn", 1, 2, 64, 32, 9000, 16384, false,
                      10000.0f, 0.25f, 1.0f, 1.0f, 32.0f, 1.0f) != 0) return 1;
    /* (f) YaRN inverse. */
    if (rope_tail_one("rope_tail/yarn-inv", 1, 2, 64, 32, 9000, 16384, true,
                      10000.0f, 0.25f, 1.0f, 1.0f, 32.0f, 1.0f) != 0) return 1;
    /* (g) Non-standard base / scale / attn_factor. */
    if (rope_tail_one("rope_tail/custom", 1, 1, 16, 16, 77, 0, false,
                      500000.0f, 0.5f, 0.0f, 2.0f, 32.0f, 1.0f) != 0) return 1;
    return 0;
}
REGISTER_TEST(rope_tail, test_rope_tail);

/* ---------- head_rms_norm ---------- */

static int test_head_rms_norm(void) {
    const uint32_t n_tok = 2, n_head = 3, head_dim = 8;
    const float eps = 1e-5f;
    const uint64_t elems = (uint64_t)n_tok * n_head * head_dim;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(elems * sizeof(float));
    if (!x) return 1;
    float xv[elems];
    synth_row_major(xv, elems);
    if (ds4_gpu_tensor_write(x, 0, xv, sizeof(xv)) == 0) {
        ds4_gpu_tensor_free(x); return 1;
    }
    float want[elems];
    std::memcpy(want, xv, sizeof(xv));
    ref_head_rms_norm(want, n_tok, n_head, head_dim, eps);

    int rc = 1;
    if (ds4_gpu_head_rms_norm_tensor(x, n_tok, n_head, head_dim, eps) != 0) {
        float got[elems];
        if (ds4_gpu_tensor_read(x, 0, got, sizeof(got)) != 0)
            rc = check_f32("head_rms_norm", got, want, (uint32_t)elems);
    }
    ds4_gpu_tensor_free(x);
    return rc;
}
REGISTER_TEST(head_rms_norm, test_head_rms_norm);

/* ---------- head_rms_norm_rope_tail (fused) ---------- */

static int test_head_rms_norm_rope_tail(void) {
    const uint32_t n_tok = 2, n_head = 2, head_dim = 64, n_rot = 32, pos0 = 17;
    const float eps = 1e-5f;
    const uint64_t elems = (uint64_t)n_tok * n_head * head_dim;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(elems * sizeof(float));
    if (!x) return 1;
    float xv[elems];
    synth_row_major(xv, elems);
    if (ds4_gpu_tensor_write(x, 0, xv, sizeof(xv)) == 0) {
        ds4_gpu_tensor_free(x); return 1;
    }
    float want[elems];
    std::memcpy(want, xv, sizeof(xv));
    ref_head_rms_norm(want, n_tok, n_head, head_dim, eps);
    ref_rope_tail(want, n_tok, n_head, head_dim, n_rot, pos0, 0,
                  10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, false);

    int rc = 1;
    if (ds4_gpu_head_rms_norm_rope_tail_tensor(x, n_tok, n_head, head_dim,
                                               n_rot, pos0, 0, false, 10000.0f,
                                               1.0f, 0.0f, 1.0f, 32.0f, 1.0f,
                                               eps) != 0) {
        float got[elems];
        if (ds4_gpu_tensor_read(x, 0, got, sizeof(got)) != 0)
            rc = check_f32("head_rms_norm_rope_tail", got, want, (uint32_t)elems);
    }
    ds4_gpu_tensor_free(x);
    return rc;
}
REGISTER_TEST(head_rms_norm_rope_tail, test_head_rms_norm_rope_tail);

static int test_attn_q_b_fused_unavailable(void) {
    return ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
               nullptr, nullptr, nullptr, 0, 0, 0, 0, nullptr,
               0, 0, 0, 0, 0, 0, false,
               10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f, 1e-5f) == 0
        ? 0 : 1;
}
REGISTER_TEST(attn_q_b_fused_unavailable, test_attn_q_b_fused_unavailable);
