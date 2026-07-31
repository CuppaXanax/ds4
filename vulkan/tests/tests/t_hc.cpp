/* Kernel tests for the Hyper-Connection (HC) family.
 *
 * Covered kernels (all host-side in the backend, like ds4_gpu_add_tensor,
 * so the tests do NOT wrap the calls in begin/end_commands):
 *
 *   ds4_gpu_hc_expand_tensor                    (new: hc_post_one)
 *   ds4_gpu_hc_expand_add_tensor                (new: hc_post_one + block_add)
 *   ds4_gpu_hc_split_weighted_sum_norm_tensor   (new: sinkhorn + sum + rmsnorm)
 *   ds4_gpu_hc_weighted_sum_tensor              (new: hc_weighted_sum_one)
 *   ds4_gpu_output_hc_weights_tensor            (new: output_hc_head_one)
 *   ds4_gpu_hc_expand_split_tensor              (existing real)
 *   ds4_gpu_hc_split_weighted_sum_tensor        (existing real, sinkhorn path)
 *   ds4_gpu_hc_weighted_sum_split_tensor        (existing real)
 *
 * The CPU references are copied verbatim from ds4.c (hc_post_one,
 * hc_weighted_sum_one, hc_split_sinkhorn_one, output_hc_head_one /
 * sigmoid_stable, rms_norm_weight) so this validates the exact layout
 * (n_embd per HC stream, n_hc streams, sinkhorn_iters for the split) and
 * eps semantics of the engine's CPU path.
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static const uint32_t kN_HC   = 4;
static const uint32_t kN_EMBD = 8;
static const uint32_t kMixHC  = 2 * kN_HC + kN_HC * kN_HC; /* 2*n_hc + n_hc*n_hc */

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

/* Deterministic synthetic value: in-range, alternating signs. */
static float synth_float(uint32_t i, float step) {
    float f = (float)(i + 1) * step + 0.0625f * (float)(i % 5);
    if ((i & 1u) != 0) f = -f;
    return f;
}

/* ---------- CPU references (copied from ds4.c) ---------- */

static float ref_sigmoid_stable(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    } else {
        const float e = expf(x);
        return e / (1.0f + e);
    }
}

/* ds4.c hc_split_sinkhorn_one.  split holds 2*n_hc + n_hc*n_hc floats. */
static void ref_hc_split_sinkhorn(float *split, const float *mix,
                                  const float *scale, const float *base,
                                  uint32_t n_hc, uint32_t iters, float eps) {
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];
    std::vector<float> c((size_t)n_hc * n_hc);

    for (uint32_t i = 0; i < n_hc; i++) {
        const float z = mix[i] * pre_scale + base[i];
        split[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }
    for (uint32_t i = 0; i < n_hc; i++) {
        const float z = mix[n_hc + i] * post_scale + base[n_hc + i];
        split[n_hc + i] = 2.0f / (1.0f + expf(-z));
    }

    for (uint32_t dst = 0; dst < n_hc; dst++) {
        float row_max = -INFINITY;
        for (uint32_t src = 0; src < n_hc; src++) {
            const size_t idx = (size_t)src + (size_t)dst * n_hc;
            const size_t off = 2ull * n_hc + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            if (v > row_max) row_max = v;
        }
        float row_sum = 0.0f;
        for (uint32_t src = 0; src < n_hc; src++) {
            const size_t idx = (size_t)src + (size_t)dst * n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }
        const float inv = 1.0f / row_sum;
        for (uint32_t src = 0; src < n_hc; src++) {
            const size_t idx = (size_t)src + (size_t)dst * n_hc;
            c[idx] = c[idx] * inv + eps;
        }
    }
    for (uint32_t src = 0; src < n_hc; src++) {
        float sum = 0.0f;
        for (uint32_t dst = 0; dst < n_hc; dst++) sum += c[(size_t)src + (size_t)dst * n_hc];
        const float inv = 1.0f / (sum + eps);
        for (uint32_t dst = 0; dst < n_hc; dst++) c[(size_t)src + (size_t)dst * n_hc] *= inv;
    }
    for (uint32_t iter = 1; iter < iters; iter++) {
        for (uint32_t dst = 0; dst < n_hc; dst++) {
            float sum = 0.0f;
            for (uint32_t src = 0; src < n_hc; src++) sum += c[(size_t)src + (size_t)dst * n_hc];
            const float inv = 1.0f / (sum + eps);
            for (uint32_t src = 0; src < n_hc; src++) c[(size_t)src + (size_t)dst * n_hc] *= inv;
        }
        for (uint32_t src = 0; src < n_hc; src++) {
            float sum = 0.0f;
            for (uint32_t dst = 0; dst < n_hc; dst++) sum += c[(size_t)src + (size_t)dst * n_hc];
            const float inv = 1.0f / (sum + eps);
            for (uint32_t dst = 0; dst < n_hc; dst++) c[(size_t)src + (size_t)dst * n_hc] *= inv;
        }
    }
    for (size_t i = 0; i < (size_t)n_hc * n_hc; i++) split[2ull * n_hc + i] = c[i];
}

/* ds4.c hc_weighted_sum_one. */
static void ref_hc_weighted_sum(float *out, const float *x, const float *w,
                                uint32_t n_embd, uint32_t n_hc) {
    for (uint32_t d = 0; d < n_embd; d++) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < n_hc; h++)
            acc += x[(uint64_t)h * n_embd + d] * w[h];
        out[d] = acc;
    }
}

/* ds4.c hc_post_one. */
static void ref_hc_post_one(float *out_hc, const float *block_out,
                            const float *residual_hc, const float *post,
                            const float *comb, uint32_t n_embd, uint32_t n_hc) {
    for (uint32_t dst = 0; dst < n_hc; dst++) {
        for (uint32_t d = 0; d < n_embd; d++) {
            float acc = block_out[d] * post[dst];
            for (uint32_t src = 0; src < n_hc; src++)
                acc += comb[(size_t)dst + (size_t)src * n_hc] *
                       residual_hc[(uint64_t)src * n_embd + d];
            out_hc[(uint64_t)dst * n_embd + d] = acc;
        }
    }
}

/* ds4.c rms_norm_weight (single row). */
static void ref_rms_norm_weight(float *out, const float *x, const float *w,
                                uint32_t n, float eps) {
    double ss = 0.0;
    for (uint32_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint32_t i = 0; i < n; i++) out[i] = x[i] * scale * w[i];
}

/* ---------- hc_expand (new) ---------- */

static int test_hc_expand(void) {
    const uint32_t n_hc = kN_HC, n_embd = kN_EMBD;
    const uint64_t hc_dim = (uint64_t)n_hc * n_embd;
    ds4_gpu_tensor *out_hc    = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    ds4_gpu_tensor *block_out = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    ds4_gpu_tensor *residual  = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    ds4_gpu_tensor *post      = ds4_gpu_tensor_alloc(n_hc * sizeof(float));
    ds4_gpu_tensor *comb      = ds4_gpu_tensor_alloc((uint64_t)n_hc * n_hc * sizeof(float));
    if (!out_hc || !block_out || !residual || !post || !comb) {
        if (out_hc) ds4_gpu_tensor_free(out_hc);
        if (block_out) ds4_gpu_tensor_free(block_out);
        if (residual) ds4_gpu_tensor_free(residual);
        if (post) ds4_gpu_tensor_free(post);
        if (comb) ds4_gpu_tensor_free(comb);
        return 1;
    }
    std::vector<float> bo(n_embd), rh((size_t)hc_dim), po(n_hc), co((size_t)n_hc * n_hc);
    for (uint32_t i = 0; i < n_embd; i++) bo[i] = synth_float(i, 0.25f);
    for (uint32_t i = 0; i < n_hc; i++)   po[i] = synth_float(i, 0.15f);
    for (uint32_t i = 0; i < (uint32_t)hc_dim; i++) rh[i] = synth_float(i, 0.20f);
    for (uint32_t i = 0; i < n_hc * n_hc; i++) co[i] = synth_float(i, 0.05f);
    if (!ds4_gpu_tensor_write(block_out, 0, bo.data(), bo.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(residual, 0, rh.data(), rh.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(post, 0, po.data(), po.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(comb, 0, co.data(), co.size() * sizeof(float))) {
        ds4_gpu_tensor_free(out_hc); ds4_gpu_tensor_free(block_out);
        ds4_gpu_tensor_free(residual); ds4_gpu_tensor_free(post); ds4_gpu_tensor_free(comb);
        return 1;
    }
    std::vector<float> want((size_t)hc_dim);
    ref_hc_post_one(want.data(), bo.data(), rh.data(), po.data(), co.data(), n_embd, n_hc);

    int rc = 1;
    if (ds4_gpu_hc_expand_tensor(out_hc, block_out, residual, post, comb,
                                 n_embd, n_hc) != 0) {
        std::vector<float> got((size_t)hc_dim);
        if (ds4_gpu_tensor_read(out_hc, 0, got.data(), got.size() * sizeof(float)) != 0)
            rc = check_f32("hc_expand", got.data(), want.data(), (uint32_t)hc_dim);
    }
    ds4_gpu_tensor_free(out_hc); ds4_gpu_tensor_free(block_out);
    ds4_gpu_tensor_free(residual); ds4_gpu_tensor_free(post); ds4_gpu_tensor_free(comb);
    return rc;
}
REGISTER_TEST(hc_expand, test_hc_expand);

/* ---------- hc_expand_add (new) ---------- */

static int test_hc_expand_add(void) {
    const uint32_t n_hc = kN_HC, n_embd = kN_EMBD;
    const uint64_t hc_dim = (uint64_t)n_hc * n_embd;
    ds4_gpu_tensor *out_hc    = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    ds4_gpu_tensor *block_out = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    ds4_gpu_tensor *block_add = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    ds4_gpu_tensor *residual  = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    ds4_gpu_tensor *post      = ds4_gpu_tensor_alloc(n_hc * sizeof(float));
    ds4_gpu_tensor *comb      = ds4_gpu_tensor_alloc((uint64_t)n_hc * n_hc * sizeof(float));
    if (!out_hc || !block_out || !block_add || !residual || !post || !comb) {
        if (out_hc) ds4_gpu_tensor_free(out_hc);
        if (block_out) ds4_gpu_tensor_free(block_out);
        if (block_add) ds4_gpu_tensor_free(block_add);
        if (residual) ds4_gpu_tensor_free(residual);
        if (post) ds4_gpu_tensor_free(post);
        if (comb) ds4_gpu_tensor_free(comb);
        return 1;
    }
    std::vector<float> bo(n_embd), ba(n_embd), rh((size_t)hc_dim), po(n_hc), co((size_t)n_hc * n_hc);
    for (uint32_t i = 0; i < n_embd; i++) { bo[i] = synth_float(i, 0.25f); ba[i] = synth_float(i + 3, 0.10f); }
    for (uint32_t i = 0; i < n_hc; i++)   po[i] = synth_float(i, 0.15f);
    for (uint32_t i = 0; i < (uint32_t)hc_dim; i++) rh[i] = synth_float(i, 0.20f);
    for (uint32_t i = 0; i < n_hc * n_hc; i++) co[i] = synth_float(i, 0.05f);
    if (!ds4_gpu_tensor_write(block_out, 0, bo.data(), bo.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(block_add, 0, ba.data(), ba.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(residual, 0, rh.data(), rh.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(post, 0, po.data(), po.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(comb, 0, co.data(), co.size() * sizeof(float))) {
        ds4_gpu_tensor_free(out_hc); ds4_gpu_tensor_free(block_out);
        ds4_gpu_tensor_free(block_add); ds4_gpu_tensor_free(residual);
        ds4_gpu_tensor_free(post); ds4_gpu_tensor_free(comb);
        return 1;
    }
    /* Reference: hc_post_one with block_out+block_add injected. */
    std::vector<float> want((size_t)hc_dim);
    for (uint32_t dst = 0; dst < n_hc; dst++)
        for (uint32_t d = 0; d < n_embd; d++) {
            float acc = (bo[d] + ba[d]) * po[dst];
            for (uint32_t src = 0; src < n_hc; src++)
                acc += co[(size_t)dst + (size_t)src * n_hc] * rh[(uint64_t)src * n_embd + d];
            want[(uint64_t)dst * n_embd + d] = acc;
        }

    int rc = 1;
    if (ds4_gpu_hc_expand_add_tensor(out_hc, block_out, block_add, residual, post, comb,
                                     n_embd, n_hc) != 0) {
        std::vector<float> got((size_t)hc_dim);
        if (ds4_gpu_tensor_read(out_hc, 0, got.data(), got.size() * sizeof(float)) != 0)
            rc = check_f32("hc_expand_add", got.data(), want.data(), (uint32_t)hc_dim);
    }
    ds4_gpu_tensor_free(out_hc); ds4_gpu_tensor_free(block_out);
    ds4_gpu_tensor_free(block_add); ds4_gpu_tensor_free(residual);
    ds4_gpu_tensor_free(post); ds4_gpu_tensor_free(comb);
    return rc;
}
REGISTER_TEST(hc_expand_add, test_hc_expand_add);

/* ---------- hc_weighted_sum (new) ---------- */

static int test_hc_weighted_sum(void) {
    const uint32_t n_hc = kN_HC, n_embd = kN_EMBD;
    const uint64_t hc_dim = (uint64_t)n_hc * n_embd;
    ds4_gpu_tensor *out      = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    ds4_gpu_tensor *residual = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    /* Engine style: `weights` is the full sinkhorn split buffer; only the
     * first n_hc entries (pre weights) participate. */
    ds4_gpu_tensor *weights  = ds4_gpu_tensor_alloc((uint64_t)kMixHC * sizeof(float));
    if (!out || !residual || !weights) {
        if (out) ds4_gpu_tensor_free(out);
        if (residual) ds4_gpu_tensor_free(residual);
        if (weights) ds4_gpu_tensor_free(weights);
        return 1;
    }
    std::vector<float> rh((size_t)hc_dim), w(kMixHC);
    for (uint32_t i = 0; i < (uint32_t)hc_dim; i++) rh[i] = synth_float(i, 0.20f);
    for (uint32_t i = 0; i < kMixHC; i++)           w[i] = synth_float(i, 0.08f);
    if (!ds4_gpu_tensor_write(residual, 0, rh.data(), rh.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(weights, 0, w.data(), w.size() * sizeof(float))) {
        ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(residual); ds4_gpu_tensor_free(weights);
        return 1;
    }
    std::vector<float> want(n_embd);
    ref_hc_weighted_sum(want.data(), rh.data(), w.data(), n_embd, n_hc);

    int rc = 1;
    if (ds4_gpu_hc_weighted_sum_tensor(out, residual, weights, n_embd, n_hc) != 0) {
        std::vector<float> got(n_embd);
        if (ds4_gpu_tensor_read(out, 0, got.data(), got.size() * sizeof(float)) != 0)
            rc = check_f32("hc_weighted_sum", got.data(), want.data(), n_embd);
    }
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(residual); ds4_gpu_tensor_free(weights);
    return rc;
}
REGISTER_TEST(hc_weighted_sum, test_hc_weighted_sum);

/* ---------- output_hc_weights (new) ---------- */

static int test_output_hc_weights(void) {
    const uint32_t n_hc = kN_HC;
    const uint64_t header = 16;
    const uint64_t scale_off = header;                    /* 1 f32 */
    const uint64_t base_off = scale_off + sizeof(float);  /* n_hc f32 */
    const uint64_t model_size = base_off + (uint64_t)n_hc * sizeof(float);
    unsigned char *model = (unsigned char *)malloc(model_size);
    if (!model) return 1;
    std::memset(model, 0xAA, header);
    float *scale = (float *)(model + scale_off);
    float *base = (float *)(model + base_off);
    scale[0] = 0.75f;
    for (uint32_t i = 0; i < n_hc; i++) base[i] = synth_float(i, 0.25f);
    if (ds4_gpu_set_model_map(model, model_size) == 0) { free(model); return 1; }

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(n_hc * sizeof(float));
    ds4_gpu_tensor *pre = ds4_gpu_tensor_alloc(n_hc * sizeof(float));
    if (!out || !pre) {
        if (out) ds4_gpu_tensor_free(out);
        if (pre) ds4_gpu_tensor_free(pre);
        free(model);
        return 1;
    }
    std::vector<float> prv(n_hc);
    for (uint32_t i = 0; i < n_hc; i++) prv[i] = synth_float(i, 0.30f);
    if (!ds4_gpu_tensor_write(pre, 0, prv.data(), prv.size() * sizeof(float))) {
        ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(pre); free(model);
        return 1;
    }
    const float eps = 1.0e-6f;
    std::vector<float> want(n_hc);
    for (uint32_t i = 0; i < n_hc; i++)
        want[i] = ref_sigmoid_stable(prv[i] * scale[0] + base[i]) + eps;

    int rc = 1;
    if (ds4_gpu_output_hc_weights_tensor(out, pre, model, model_size,
                                         scale_off, base_off, n_hc, eps) != 0) {
        std::vector<float> got(n_hc);
        if (ds4_gpu_tensor_read(out, 0, got.data(), got.size() * sizeof(float)) != 0)
            rc = check_f32("output_hc_weights", got.data(), want.data(), n_hc);
    }
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(pre);
    free(model);
    return rc;
}
REGISTER_TEST(output_hc_weights, test_output_hc_weights);

/* ---------- hc_split_weighted_sum_norm (new) ---------- */

static int test_hc_split_weighted_sum_norm(void) {
    const uint32_t n_hc = kN_HC, n_embd = kN_EMBD;
    const uint64_t hc_dim = (uint64_t)n_hc * n_embd;
    const uint64_t header = 16;
    const uint64_t scale_off = header;                              /* 3 f32 */
    const uint64_t base_off = scale_off + 3ull * sizeof(float);     /* kMixHC f32 */
    const uint64_t norm_off = base_off + (uint64_t)kMixHC * sizeof(float);
    const uint64_t model_size = norm_off + (uint64_t)n_embd * sizeof(float);
    unsigned char *model = (unsigned char *)malloc(model_size);
    if (!model) return 1;
    std::memset(model, 0xAA, header);
    float *scale = (float *)(model + scale_off);
    float *base = (float *)(model + base_off);
    float *norm_w = (float *)(model + norm_off);
    scale[0] = 0.60f; scale[1] = 0.80f; scale[2] = 0.50f;
    for (uint32_t i = 0; i < kMixHC; i++) base[i] = synth_float(i, 0.12f);
    for (uint32_t i = 0; i < n_embd; i++) norm_w[i] = 0.5f + 0.0625f * (float)(i + 1);
    if (ds4_gpu_set_model_map(model, model_size) == 0) { free(model); return 1; }

    ds4_gpu_tensor *out      = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    ds4_gpu_tensor *norm_out = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    ds4_gpu_tensor *split    = ds4_gpu_tensor_alloc((uint64_t)kMixHC * sizeof(float));
    ds4_gpu_tensor *mix      = ds4_gpu_tensor_alloc((uint64_t)kMixHC * sizeof(float));
    ds4_gpu_tensor *residual = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    if (!out || !norm_out || !split || !mix || !residual) {
        if (out) ds4_gpu_tensor_free(out);
        if (norm_out) ds4_gpu_tensor_free(norm_out);
        if (split) ds4_gpu_tensor_free(split);
        if (mix) ds4_gpu_tensor_free(mix);
        if (residual) ds4_gpu_tensor_free(residual);
        free(model);
        return 1;
    }
    std::vector<float> mixv(kMixHC), rh((size_t)hc_dim);
    for (uint32_t i = 0; i < kMixHC; i++) mixv[i] = synth_float(i, 0.10f);
    for (uint32_t i = 0; i < (uint32_t)hc_dim; i++) rh[i] = synth_float(i, 0.20f);
    if (!ds4_gpu_tensor_write(mix, 0, mixv.data(), mixv.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(residual, 0, rh.data(), rh.size() * sizeof(float))) {
        ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(norm_out);
        ds4_gpu_tensor_free(split); ds4_gpu_tensor_free(mix); ds4_gpu_tensor_free(residual);
        free(model);
        return 1;
    }
    const float eps = 1.0e-6f, norm_eps = 1.0e-5f;
    const uint32_t sinkhorn_iters = 6;
    std::vector<float> want_split(kMixHC), want_out(n_embd), want_norm(n_embd);
    ref_hc_split_sinkhorn(want_split.data(), mixv.data(), scale, base,
                          n_hc, sinkhorn_iters, eps);
    ref_hc_weighted_sum(want_out.data(), rh.data(), want_split.data(), n_embd, n_hc);
    ref_rms_norm_weight(want_norm.data(), want_out.data(), norm_w, n_embd, norm_eps);

    int rc = 1;
    if (ds4_gpu_hc_split_weighted_sum_norm_tensor(
            out, norm_out, split, mix, residual, model, model_size,
            scale_off, base_off, norm_off, n_embd, n_hc, sinkhorn_iters,
            eps, norm_eps) != 0) {
        std::vector<float> got_split(kMixHC), got_out(n_embd), got_norm(n_embd);
        if (ds4_gpu_tensor_read(split, 0, got_split.data(), got_split.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(out, 0, got_out.data(), got_out.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(norm_out, 0, got_norm.data(), got_norm.size() * sizeof(float)) != 0) {
            rc = check_f32("hc_split_weighted_sum_norm/split", got_split.data(),
                           want_split.data(), kMixHC);
            if (rc == 0)
                rc = check_f32("hc_split_weighted_sum_norm/out", got_out.data(),
                               want_out.data(), n_embd);
            if (rc == 0)
                rc = check_f32("hc_split_weighted_sum_norm/norm", got_norm.data(),
                               want_norm.data(), n_embd);
        }
    }
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(norm_out);
    ds4_gpu_tensor_free(split); ds4_gpu_tensor_free(mix); ds4_gpu_tensor_free(residual);
    free(model);
    return rc;
}
REGISTER_TEST(hc_split_weighted_sum_norm, test_hc_split_weighted_sum_norm);

/* ---------- hc_expand_split (existing real) ---------- */

static int test_hc_expand_split(void) {
    const uint32_t n_hc = kN_HC, n_embd = kN_EMBD;
    const uint64_t hc_dim = (uint64_t)n_hc * n_embd;
    ds4_gpu_tensor *out_hc    = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    ds4_gpu_tensor *block_out = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    ds4_gpu_tensor *residual  = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    ds4_gpu_tensor *split     = ds4_gpu_tensor_alloc((uint64_t)kMixHC * sizeof(float));
    if (!out_hc || !block_out || !residual || !split) {
        if (out_hc) ds4_gpu_tensor_free(out_hc);
        if (block_out) ds4_gpu_tensor_free(block_out);
        if (residual) ds4_gpu_tensor_free(residual);
        if (split) ds4_gpu_tensor_free(split);
        return 1;
    }
    std::vector<float> bo(n_embd), rh((size_t)hc_dim), sp(kMixHC);
    for (uint32_t i = 0; i < n_embd; i++) bo[i] = synth_float(i, 0.25f);
    for (uint32_t i = 0; i < (uint32_t)hc_dim; i++) rh[i] = synth_float(i, 0.20f);
    for (uint32_t i = 0; i < kMixHC; i++)           sp[i] = synth_float(i, 0.08f);
    if (!ds4_gpu_tensor_write(block_out, 0, bo.data(), bo.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(residual, 0, rh.data(), rh.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(split, 0, sp.data(), sp.size() * sizeof(float))) {
        ds4_gpu_tensor_free(out_hc); ds4_gpu_tensor_free(block_out);
        ds4_gpu_tensor_free(residual); ds4_gpu_tensor_free(split);
        return 1;
    }
    /* Post gates live at split[n_hc + h]; no combine mixing (batch fast path). */
    std::vector<float> want((size_t)hc_dim);
    for (uint32_t h = 0; h < n_hc; h++) {
        const float w = sp[n_hc + h];
        for (uint32_t i = 0; i < n_embd; i++)
            want[(uint64_t)h * n_embd + i] = w * bo[i] + rh[(uint64_t)h * n_embd + i];
    }
    int rc = 1;
    if (ds4_gpu_hc_expand_split_tensor(out_hc, block_out, residual, split,
                                       n_embd, n_hc) != 0) {
        std::vector<float> got((size_t)hc_dim);
        if (ds4_gpu_tensor_read(out_hc, 0, got.data(), got.size() * sizeof(float)) != 0)
            rc = check_f32("hc_expand_split", got.data(), want.data(), (uint32_t)hc_dim);
    }
    ds4_gpu_tensor_free(out_hc); ds4_gpu_tensor_free(block_out);
    ds4_gpu_tensor_free(residual); ds4_gpu_tensor_free(split);
    return rc;
}
REGISTER_TEST(hc_expand_split, test_hc_expand_split);

/* ---------- hc_weighted_sum_split (existing real) ---------- */

static int test_hc_weighted_sum_split(void) {
    const uint32_t n_hc = kN_HC, n_embd = kN_EMBD;
    const uint64_t hc_dim = (uint64_t)n_hc * n_embd;
    ds4_gpu_tensor *out      = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    ds4_gpu_tensor *residual = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    ds4_gpu_tensor *split    = ds4_gpu_tensor_alloc((uint64_t)kMixHC * sizeof(float));
    if (!out || !residual || !split) {
        if (out) ds4_gpu_tensor_free(out);
        if (residual) ds4_gpu_tensor_free(residual);
        if (split) ds4_gpu_tensor_free(split);
        return 1;
    }
    std::vector<float> rh((size_t)hc_dim), sp(kMixHC);
    for (uint32_t i = 0; i < (uint32_t)hc_dim; i++) rh[i] = synth_float(i, 0.20f);
    for (uint32_t i = 0; i < kMixHC; i++)           sp[i] = synth_float(i, 0.08f);
    if (!ds4_gpu_tensor_write(residual, 0, rh.data(), rh.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(split, 0, sp.data(), sp.size() * sizeof(float))) {
        ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(residual); ds4_gpu_tensor_free(split);
        return 1;
    }
    std::vector<float> want(n_embd);
    ref_hc_weighted_sum(want.data(), rh.data(), sp.data(), n_embd, n_hc);

    int rc = 1;
    if (ds4_gpu_hc_weighted_sum_split_tensor(out, residual, split, n_embd, n_hc) != 0) {
        std::vector<float> got(n_embd);
        if (ds4_gpu_tensor_read(out, 0, got.data(), got.size() * sizeof(float)) != 0)
            rc = check_f32("hc_weighted_sum_split", got.data(), want.data(), n_embd);
    }
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(residual); ds4_gpu_tensor_free(split);
    return rc;
}
REGISTER_TEST(hc_weighted_sum_split, test_hc_weighted_sum_split);

/* ---------- hc_split_weighted_sum (existing real, sinkhorn path) ---------- */

static int test_hc_split_weighted_sum(void) {
    const uint32_t n_hc = kN_HC, n_embd = kN_EMBD;
    const uint64_t hc_dim = (uint64_t)n_hc * n_embd;
    const uint64_t header = 16;
    const uint64_t scale_off = header;                          /* 3 f32 */
    const uint64_t base_off = scale_off + 3ull * sizeof(float); /* kMixHC f32 */
    const uint64_t model_size = base_off + (uint64_t)kMixHC * sizeof(float);
    unsigned char *model = (unsigned char *)malloc(model_size);
    if (!model) return 1;
    std::memset(model, 0xAA, header);
    float *scale = (float *)(model + scale_off);
    float *base = (float *)(model + base_off);
    scale[0] = 0.60f; scale[1] = 0.80f; scale[2] = 0.50f;
    for (uint32_t i = 0; i < kMixHC; i++) base[i] = synth_float(i, 0.12f);
    if (ds4_gpu_set_model_map(model, model_size) == 0) { free(model); return 1; }

    ds4_gpu_tensor *out      = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    ds4_gpu_tensor *split    = ds4_gpu_tensor_alloc((uint64_t)kMixHC * sizeof(float));
    ds4_gpu_tensor *mix      = ds4_gpu_tensor_alloc((uint64_t)kMixHC * sizeof(float));
    ds4_gpu_tensor *residual = ds4_gpu_tensor_alloc(hc_dim * sizeof(float));
    if (!out || !split || !mix || !residual) {
        if (out) ds4_gpu_tensor_free(out);
        if (split) ds4_gpu_tensor_free(split);
        if (mix) ds4_gpu_tensor_free(mix);
        if (residual) ds4_gpu_tensor_free(residual);
        free(model);
        return 1;
    }
    std::vector<float> mixv(kMixHC), rh((size_t)hc_dim);
    for (uint32_t i = 0; i < kMixHC; i++) mixv[i] = synth_float(i, 0.10f);
    for (uint32_t i = 0; i < (uint32_t)hc_dim; i++) rh[i] = synth_float(i, 0.20f);
    if (!ds4_gpu_tensor_write(mix, 0, mixv.data(), mixv.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(residual, 0, rh.data(), rh.size() * sizeof(float))) {
        ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(split);
        ds4_gpu_tensor_free(mix); ds4_gpu_tensor_free(residual);
        free(model);
        return 1;
    }
    const float eps = 1.0e-6f;
    const uint32_t sinkhorn_iters = 6;
    std::vector<float> want_split(kMixHC), want_out(n_embd);
    ref_hc_split_sinkhorn(want_split.data(), mixv.data(), scale, base,
                          n_hc, sinkhorn_iters, eps);
    ref_hc_weighted_sum(want_out.data(), rh.data(), want_split.data(), n_embd, n_hc);

    int rc = 1;
    if (ds4_gpu_hc_split_weighted_sum_tensor(
            out, split, mix, residual, model, model_size,
            scale_off, base_off, n_embd, n_hc, sinkhorn_iters, eps) != 0) {
        std::vector<float> got_split(kMixHC), got_out(n_embd);
        if (ds4_gpu_tensor_read(split, 0, got_split.data(), got_split.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(out, 0, got_out.data(), got_out.size() * sizeof(float)) != 0) {
            rc = check_f32("hc_split_weighted_sum/split", got_split.data(),
                           want_split.data(), kMixHC);
            if (rc == 0)
                rc = check_f32("hc_split_weighted_sum/out", got_out.data(),
                               want_out.data(), n_embd);
        }
    }
    ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(split);
    ds4_gpu_tensor_free(mix); ds4_gpu_tensor_free(residual);
    free(model);
    return rc;
}
REGISTER_TEST(hc_split_weighted_sum, test_hc_split_weighted_sum);
