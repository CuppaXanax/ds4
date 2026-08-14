/* Kernel test: ds4_gpu_attention_decode_heads_tensor (Vulkan decode
 * attention).
 *
 * Single-token causal decode attention over the raw ring cache plus the
 * compressed (MLA) cache.  The backend synchronizes the dispatch before
 * returning, so no begin_commands/end_commands are needed to read back the output; the
 * reference below replicates the ds4.c layer_attention_mixed_one math:
 *   scale = 1/sqrt(head_dim)
 *   score_i = dot(q_head, kv_i) * scale, raw rows in ring order
 *             (raw_start + i) % raw_cap, comp rows after raw rows
 *   per-head sink prior sinks[h] joins the softmax denominator
 *   comp_mask (use_mask != 0) is an additive bias per comp row
 *             (0.0 = allowed, -inf = masked, rows <= -1e20 excluded)
 *   heads[h] = sum(exp(score - max) * v) / denom
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>

/* IEEE half -> float (same decode the kernel uses for comp_kv_f16). */
static float half_to_float(uint16_t h) {
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
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint16_t float_to_half(float v) {
    uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t e = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t m = bits & 0x7fffffu;
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);           /* inf */
    if (e < -10) return (uint16_t)sign;                       /* zero */
    if (e <= 0) {                                             /* subnormal */
        const uint32_t shift = (uint32_t)(14 - e);
        m |= 0x800000u;                                       /* implicit 1 */
        uint32_t half_m = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (half_m & 1u))) half_m++;
        return (uint16_t)(sign | half_m);
    }
    uint32_t half_m = m >> 13;                                /* normal */
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half_m & 1u))) half_m++;
    if (half_m >= 0x400u) { half_m = 0; e++; }                /* carry */
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | (uint32_t)e << 10 | half_m);
}

/* Inline CPU reference.  raw rows in ring order, then comp rows; comp_mask
 * is an additive bias when use_mask != 0 (<= -1e20 -> masked). */
static void ref_attention_decode(
        float *heads, uint32_t n_head, uint32_t head_dim,
        const float *sinks,
        const float *q,
        const float *raw_kv, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        const float *comp_kv, uint32_t n_comp,
        const float *comp_mask, uint32_t use_mask)
{
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const uint32_t n_total = n_raw + n_comp;
    std::vector<float> score(n_total);

    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q + (uint64_t)h * head_dim;
        const float sink = sinks[h];
        float max_score = sink;

        for (uint32_t r = 0; r < n_raw; r++) {
            const uint32_t row = (raw_start + r) % raw_cap;
            const float *kv = raw_kv + (uint64_t)row * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
            score[r] = dot * scale;
            if (score[r] > max_score) max_score = score[r];
        }
        for (uint32_t c = 0; c < n_comp; c++) {
            const uint32_t idx = n_raw + c;
            const float add = use_mask ? comp_mask[c] : 0.0f;
            if (add <= -1.0e20f) { score[idx] = -INFINITY; continue; }
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
            score[idx] = dot * scale + add;
            if (score[idx] > max_score) max_score = score[idx];
        }

        float *oh = heads + (uint64_t)h * head_dim;
        for (uint32_t d = 0; d < head_dim; d++) oh[d] = 0.0f;
        float denom = std::exp(sink - max_score);

        for (uint32_t r = 0; r < n_raw; r++) {
            const uint32_t row = (raw_start + r) % raw_cap;
            const float *kv = raw_kv + (uint64_t)row * head_dim;
            const float w = std::exp(score[r] - max_score);
            denom += w;
            for (uint32_t d = 0; d < head_dim; d++) oh[d] += w * kv[d];
        }
        for (uint32_t c = 0; c < n_comp; c++) {
            const uint32_t idx = n_raw + c;
            if (score[idx] <= -1.0e20f) continue;
            const float *kv = comp_kv + (uint64_t)c * head_dim;
            const float w = std::exp(score[idx] - max_score);
            denom += w;
            for (uint32_t d = 0; d < head_dim; d++) oh[d] += w * kv[d];
        }
        const float inv = 1.0f / denom;
        for (uint32_t d = 0; d < head_dim; d++) oh[d] *= inv;
    }
}

static void ref_attention_decode_raw_batch(
        float *heads, uint32_t n_tokens, uint32_t pos0, uint32_t n_raw,
        uint32_t raw_cap, uint32_t raw_start, uint32_t window,
        uint32_t n_head, uint32_t head_dim, const float *sinks,
        const float *q, const float *raw_kv)
{
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const uint32_t base_abs = pos0 + n_tokens - n_raw;
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t end_abs = pos0 + t;
        const uint32_t candidate = end_abs + 1u > window ? end_abs + 1u - window : 0u;
        const uint32_t first_abs = std::max(base_abs, candidate);
        const uint32_t first = raw_start + first_abs - base_abs;
        const uint32_t count = end_abs >= first_abs ? end_abs - first_abs + 1u : 0u;
        for (uint32_t h = 0; h < n_head; h++) {
            const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
            float max_score = sinks[h];
            std::vector<float> score(count);
            for (uint32_t r = 0; r < count; r++) {
                const uint32_t row = (first + r) % raw_cap;
                const float *kv = raw_kv + (uint64_t)row * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
                score[r] = dot * scale;
                if (score[r] > max_score) max_score = score[r];
            }
            float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
            std::fill(oh, oh + head_dim, 0.0f);
            float denom = std::exp(sinks[h] - max_score);
            for (uint32_t r = 0; r < count; r++) {
                const uint32_t row = (first + r) % raw_cap;
                const float *kv = raw_kv + (uint64_t)row * head_dim;
                const float weight = std::exp(score[r] - max_score);
                denom += weight;
                for (uint32_t d = 0; d < head_dim; d++) oh[d] += weight * kv[d];
            }
            for (uint32_t d = 0; d < head_dim; d++) oh[d] /= denom;
        }
    }
}

/* Runs one decode-attention case end-to-end.
 * comp_f32 == NULL means n_comp = 0.  mask == NULL / use_mask == 0 means no
 * mask.  Returns 0 on PASS. */
static int run_attention_case(
        uint32_t n_head, uint32_t head_dim, uint32_t n_raw, uint32_t raw_cap,
        uint32_t raw_start,
        const float *raw_ring,              /* raw_cap * head_dim floats */
        const float *q,                     /* n_head * head_dim floats */
        const float *sinks,                 /* n_head floats */
        const float *comp_f32, uint32_t n_comp,
        const uint16_t *comp_f16,           /* used when comp_kv_f16 != 0 */
        uint32_t comp_kv_f16,
        const float *mask, uint32_t use_mask,
        const char *label)
{
    int rc = 1;
    ds4_gpu_tensor *heads_t = ds4_gpu_tensor_alloc((uint64_t)n_head * head_dim * sizeof(float));
    ds4_gpu_tensor *q_t     = ds4_gpu_tensor_alloc((uint64_t)n_head * head_dim * sizeof(float));
    ds4_gpu_tensor *raw_t   = ds4_gpu_tensor_alloc((uint64_t)raw_cap * head_dim * sizeof(float));
    ds4_gpu_tensor *comp_t  = n_comp ? ds4_gpu_tensor_alloc(
            (uint64_t)n_comp * head_dim * (comp_kv_f16 ? sizeof(uint16_t) : sizeof(float))) : nullptr;
    ds4_gpu_tensor *mask_t  = (use_mask && n_comp) ? ds4_gpu_tensor_alloc(
            (uint64_t)n_comp * sizeof(float)) : nullptr;
    if (!heads_t || !q_t || !raw_t || (n_comp && !comp_t) || (use_mask && n_comp && !mask_t)) {
        if (heads_t) ds4_gpu_tensor_free(heads_t);
        if (q_t)     ds4_gpu_tensor_free(q_t);
        if (raw_t)   ds4_gpu_tensor_free(raw_t);
        if (comp_t)  ds4_gpu_tensor_free(comp_t);
        if (mask_t)  ds4_gpu_tensor_free(mask_t);
        return 1;
    }

    bool ok =
        ds4_gpu_tensor_write(q_t,   0, q,       (uint64_t)n_head * head_dim * sizeof(float)) != 0 &&
        ds4_gpu_tensor_write(raw_t, 0, raw_ring, (uint64_t)raw_cap * head_dim * sizeof(float)) != 0;
    if (n_comp) {
        ok = ok && ds4_gpu_tensor_write(comp_t, 0, comp_kv_f16 ? (const void *)comp_f16
                                                               : (const void *)comp_f32,
                                        (uint64_t)n_comp * head_dim *
                                            (comp_kv_f16 ? sizeof(uint16_t) : sizeof(float))) != 0;
    }
    if (use_mask && n_comp) {
        ok = ok && ds4_gpu_tensor_write(mask_t, 0, mask, (uint64_t)n_comp * sizeof(float)) != 0;
    }

    /* Synthetic model buffer: junk header + f32 sinks at sinks_offset. */
    const uint64_t sinks_offset = 16;
    const uint64_t model_size = sinks_offset + (uint64_t)n_head * sizeof(float);
    unsigned char *model = (unsigned char *)std::malloc(model_size);
    if (!model) { ok = false; }
    if (ok) {
        std::memset(model, 0xAA, (size_t)sinks_offset);
        std::memcpy(model + sinks_offset, sinks, (size_t)n_head * sizeof(float));
        ok = ds4_gpu_set_model_map(model, model_size) != 0;
    }

    if (ok) {
        const int ret = ds4_gpu_attention_decode_heads_tensor(
                heads_t, model, model_size, sinks_offset, q_t, raw_t,
                n_raw, raw_cap, raw_start,
                comp_t, comp_kv_f16, n_comp, mask_t, use_mask,
                n_head, head_dim);
        if (ret == 0) {
            fprintf(stderr, "attention_decode[%s]: kernel returned error\n", label);
            ok = false;
        } else {
            std::vector<float> got((uint64_t)n_head * head_dim);
            ok = ds4_gpu_tensor_read(heads_t, 0, got.data(),
                                     (uint64_t)n_head * head_dim * sizeof(float)) != 0;
            if (ok) {
                /* Reference from the same data the kernel saw: for f16 comp
                 * rows, decode the stored halfs first. */
                std::vector<float> comp_ref;
                const float *comp_use = comp_f32;
                if (n_comp && comp_kv_f16) {
                    comp_ref.resize((uint64_t)n_comp * head_dim);
                    for (uint32_t i = 0; i < (uint32_t)comp_ref.size(); i++)
                        comp_ref[i] = half_to_float(comp_f16[i]);
                    comp_use = comp_ref.data();
                }
                std::vector<float> ref((uint64_t)n_head * head_dim);
                ref_attention_decode(ref.data(), n_head, head_dim, sinks, q,
                                     raw_ring, n_raw, raw_cap, raw_start,
                                     comp_use, n_comp, mask, use_mask);
                for (uint32_t i = 0; i < (uint32_t)ref.size() && ok; i++) {
                    if (!(std::fabsf(got[i] - ref[i]) <= 1e-3f)) {
                        fprintf(stderr, "attention_decode[%s]: head[%u] got %.6f want %.6f\n",
                                label, i, got[i], ref[i]);
                        ok = false;
                    }
                }
            }
        }
    }

    if (model) std::free(model);
    ds4_gpu_tensor_free(heads_t);
    ds4_gpu_tensor_free(q_t);
    ds4_gpu_tensor_free(raw_t);
    if (comp_t) ds4_gpu_tensor_free(comp_t);
    if (mask_t) ds4_gpu_tensor_free(mask_t);
    return ok ? 0 : 1;
}

static int test_attention_decode(void) {
    const uint32_t n_head = 2, head_dim = 4, n_raw = 3, raw_cap = 4;

    /* q: two heads. */
    float q[n_head * head_dim] = {
         0.5f, -1.0f,  1.5f,  0.25f,      /* head 0 */
         1.0f,  0.5f, -0.5f,  2.0f,       /* head 1 */
    };
    /* Per-head sink priors (model buffer). */
    float sinks[n_head] = { 0.0f, 0.1f };

    /* Raw ring, raw_cap = 4 rows.  Base case raw_start = 0: the three
     * chronological raw rows are rows 0, 1, 2 (row 3 unused filler). */
    float raw_ring0[raw_cap * head_dim] = {
         1.0f,  0.0f,  0.5f, -0.5f,       /* row 0 */
         0.0f,  1.0f,  0.25f, 0.75f,      /* row 1 */
        -0.5f,  0.5f,  1.0f,  0.0f,       /* row 2 */
         9.9f,  9.9f,  9.9f,  9.9f,       /* row 3 (unused) */
    };
    /* Wrapped layout raw_start = 2: chronological rows are 2, 3, 0. */
    float raw_ring2[raw_cap * head_dim] = {
         1.0f,  0.0f,  0.5f, -0.5f,       /* row 0 (3rd raw row) */
         9.9f,  9.9f,  9.9f,  9.9f,       /* row 1 (unused) */
         0.0f,  1.0f,  0.25f, 0.75f,      /* row 2 (1st raw row) */
        -0.5f,  0.5f,  1.0f,  0.0f,       /* row 3 (2nd raw row) */
    };

    /* Compressed rows (MLA cache). */
    const uint32_t n_comp = 2;
    float comp_f32[n_comp * head_dim] = {
         0.2f, -0.1f,  0.8f,  0.3f,       /* comp 0 */
         0.6f,  0.4f, -0.2f,  0.1f,       /* comp 1 */
    };
    uint16_t comp_f16[n_comp * head_dim];
    for (uint32_t i = 0; i < n_comp * head_dim; i++)
        comp_f16[i] = float_to_half(comp_f32[i]);

    int rc = 0;

    /* Case 1: raw-only base case (n_comp = 0), raw_start = 0. */
    rc |= run_attention_case(n_head, head_dim, n_raw, raw_cap, 0,
                             raw_ring0, q, sinks, nullptr, 0, nullptr, 0,
                             nullptr, 0, "raw-f32-start0");

    /* Case 2: wrapped circular layout, raw_start = 2 (rows 2, 3, 0). */
    rc |= run_attention_case(n_head, head_dim, n_raw, raw_cap, 2,
                             raw_ring2, q, sinks, nullptr, 0, nullptr, 0,
                             nullptr, 0, "raw-f32-start2-wrap");

    /* Case 3: mixed raw + comp f32, no mask. */
    rc |= run_attention_case(n_head, head_dim, n_raw, raw_cap, 0,
                             raw_ring0, q, sinks, comp_f32, n_comp, nullptr, 0,
                             nullptr, 0, "mixed-f32-nomask");

    /* Case 4: mixed raw + comp f16, no mask (exercises the f16 decode). */
    rc |= run_attention_case(n_head, head_dim, n_raw, raw_cap, 0,
                             raw_ring0, q, sinks, comp_f32, n_comp, comp_f16, 1,
                             nullptr, 0, "mixed-f16-nomask");

    /* Case 5: comp mask (0.0 allowed, -inf masked) excludes comp 1. */
    float mask[n_comp] = { 0.0f, -INFINITY };
    rc |= run_attention_case(n_head, head_dim, n_raw, raw_cap, 0,
                             raw_ring0, q, sinks, comp_f32, n_comp, nullptr, 0,
                             mask, 1, "mixed-f32-masked");

    /* Case 6: error path - null pointers must return 0. */
    if (ds4_gpu_attention_decode_heads_tensor(nullptr, nullptr, 0, 0, nullptr,
                                              nullptr, n_raw, raw_cap, 0,
                                              nullptr, 0, 0, nullptr, 0,
                                              n_head, head_dim) != 0) {
        fprintf(stderr, "attention_decode: null-pointer call should return 0\n");
        rc = 1;
    }

    return rc;
}
REGISTER_TEST(attention_decode, test_attention_decode);

static int test_attention_decode_indexed_boundary(void) {
    const uint32_t n_head = 2, head_dim = 4;
    const uint32_t n_raw = 128, raw_cap = 128, raw_start = 0;
    const uint32_t n_comp = 897, top_k = 512;
    const uint32_t pos0 = 3588, window = 128, ratio = 4;
    const uint64_t head_elems = (uint64_t)n_head * head_dim;
    const uint64_t raw_elems = (uint64_t)raw_cap * head_dim;
    const uint64_t comp_elems = (uint64_t)n_comp * head_dim;

    std::vector<float> q(head_elems);
    std::vector<float> raw(raw_elems);
    std::vector<float> comp(comp_elems);
    std::vector<uint32_t> selected(top_k);
    float sinks[n_head] = {0.05f, -0.1f};
    for (uint64_t i = 0; i < head_elems; i++)
        q[i] = 0.07f * (float)((i * 5u) % 13u) - 0.3f;
    for (uint64_t i = 0; i < raw_elems; i++)
        raw[i] = 0.03f * (float)((i * 7u) % 19u) - 0.25f;
    for (uint64_t i = 0; i < comp_elems; i++)
        comp[i] = 0.02f * (float)((i * 11u) % 23u) - 0.2f;
    for (uint32_t i = 0; i < top_k; i++) selected[i] = i;

    ds4_gpu_tensor *heads_t = ds4_gpu_tensor_alloc(head_elems * sizeof(float));
    ds4_gpu_tensor *q_t = ds4_gpu_tensor_alloc(head_elems * sizeof(float));
    ds4_gpu_tensor *raw_t = ds4_gpu_tensor_alloc(raw_elems * sizeof(float));
    ds4_gpu_tensor *comp_t = ds4_gpu_tensor_alloc(comp_elems * sizeof(float));
    ds4_gpu_tensor *selected_t = ds4_gpu_tensor_alloc((uint64_t)top_k * sizeof(uint32_t));
    const uint64_t sinks_offset = 16;
    const uint64_t model_size = sinks_offset + sizeof(sinks);
    unsigned char *model = (unsigned char *)std::malloc(model_size);
    bool ok = heads_t && q_t && raw_t && comp_t && selected_t && model;
    if (ok) {
        std::memset(model, 0xAA, (size_t)sinks_offset);
        std::memcpy(model + sinks_offset, sinks, sizeof(sinks));
        ok = ds4_gpu_set_model_map(model, model_size) != 0 &&
             ds4_gpu_tensor_write(q_t, 0, q.data(), head_elems * sizeof(float)) != 0 &&
             ds4_gpu_tensor_write(raw_t, 0, raw.data(), raw_elems * sizeof(float)) != 0 &&
             ds4_gpu_tensor_write(comp_t, 0, comp.data(), comp_elems * sizeof(float)) != 0 &&
             ds4_gpu_tensor_write(selected_t, 0, selected.data(),
                                  (uint64_t)top_k * sizeof(uint32_t)) != 0;
    }
    if (ok && ds4_gpu_attention_decode_heads_tensor(
                      heads_t, model, model_size, sinks_offset, q_t, raw_t,
                      n_raw, raw_cap, raw_start, comp_t, 0, n_comp, nullptr, 0,
                      n_head, head_dim) != 0) {
        fprintf(stderr, "attention_decode_indexed_boundary: dense path accepted 1025 rows\n");
        ok = false;
    }
    if (ok) {
        ok = ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                 heads_t, model, model_size, sinks_offset, q_t, raw_t, comp_t, 0,
                 selected_t, 1, pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
                 window, ratio, n_head, head_dim) != 0;
    }
    std::vector<float> got(head_elems);
    if (ok) {
        ok = ds4_gpu_tensor_read(heads_t, 0, got.data(), head_elems * sizeof(float)) != 0;
    }
    if (ok) {
        std::vector<float> selected_comp((uint64_t)top_k * head_dim);
        for (uint32_t row = 0; row < top_k; row++) {
            std::copy_n(comp.data() + (uint64_t)selected[row] * head_dim,
                        head_dim,
                        selected_comp.data() + (uint64_t)row * head_dim);
        }
        std::vector<float> ref(head_elems);
        ref_attention_decode(ref.data(), n_head, head_dim, sinks, q.data(),
                             raw.data(), n_raw, raw_cap, raw_start,
                             selected_comp.data(), top_k, nullptr, 0);
        for (uint32_t i = 0; i < head_elems && ok; i++) {
            if (std::fabs(got[i] - ref[i]) > 2e-3f) {
                fprintf(stderr,
                        "attention_decode_indexed_boundary: head[%u] got %.6f want %.6f\n",
                        i, got[i], ref[i]);
                ok = false;
            }
        }
    }

    std::free(model);
    if (selected_t) ds4_gpu_tensor_free(selected_t);
    if (comp_t) ds4_gpu_tensor_free(comp_t);
    if (raw_t) ds4_gpu_tensor_free(raw_t);
    if (q_t) ds4_gpu_tensor_free(q_t);
    if (heads_t) ds4_gpu_tensor_free(heads_t);
    return ok ? 0 : 1;
}
REGISTER_TEST(attention_decode_indexed_boundary, test_attention_decode_indexed_boundary);

static int test_attention_decode_raw_batch(void) {
    const uint32_t n_tokens = 3, pos0 = 7, n_raw = 5, raw_cap = 8;
    const uint32_t raw_start = 6, window = 3, n_head = 2, head_dim = 4;
    const uint64_t head_elems = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t raw_elems = (uint64_t)raw_cap * head_dim;
    float sinks[n_head] = {0.05f, -0.1f};
    float q[head_elems] = {
        0.5f, -1.0f, 1.5f, 0.25f, 1.0f, 0.5f, -0.5f, 2.0f,
        -0.25f, 0.75f, 1.25f, -1.5f, 0.8f, -0.4f, 0.6f, 1.1f,
        1.2f, 0.3f, -0.7f, 0.9f, -0.6f, 1.4f, 0.2f, -0.8f,
    };
    float raw[raw_elems];
    for (uint32_t row = 0; row < raw_cap; row++)
        for (uint32_t d = 0; d < head_dim; d++)
            raw[(uint64_t)row * head_dim + d] =
                0.11f * (float)(row + 1u) - 0.07f * (float)(d + 1u);

    ds4_gpu_tensor *heads_t = ds4_gpu_tensor_alloc(head_elems * sizeof(float));
    ds4_gpu_tensor *q_t = ds4_gpu_tensor_alloc(head_elems * sizeof(float));
    ds4_gpu_tensor *raw_t = ds4_gpu_tensor_alloc(raw_elems * sizeof(float));
    const uint64_t sinks_offset = 16;
    const uint64_t model_size = sinks_offset + sizeof(sinks);
    unsigned char *model = (unsigned char *)std::malloc(model_size);
    if (!heads_t || !q_t || !raw_t || !model) {
        if (heads_t) ds4_gpu_tensor_free(heads_t);
        if (q_t) ds4_gpu_tensor_free(q_t);
        if (raw_t) ds4_gpu_tensor_free(raw_t);
        std::free(model);
        return 1;
    }
    std::memset(model, 0xAA, (size_t)sinks_offset);
    std::memcpy(model + sinks_offset, sinks, sizeof(sinks));
    bool ok = ds4_gpu_tensor_write(q_t, 0, q, sizeof(q)) != 0 &&
              ds4_gpu_tensor_write(raw_t, 0, raw, sizeof(raw)) != 0 &&
              ds4_gpu_set_model_map(model, model_size) != 0;
    std::vector<float> want(head_elems);
    if (ok) {
        ok = ds4_gpu_attention_decode_raw_batch_heads_tensor(
                 heads_t, model, model_size, sinks_offset, q_t, raw_t,
                 n_tokens, pos0, n_raw, raw_cap, raw_start, window,
                 n_head, head_dim) != 0;
    }
    if (ok) {
        std::vector<float> got(head_elems);
        ok = ds4_gpu_tensor_read(heads_t, 0, got.data(), sizeof(float) * head_elems) != 0;
        ref_attention_decode_raw_batch(want.data(), n_tokens, pos0, n_raw,
                                       raw_cap, raw_start, window, n_head,
                                       head_dim, sinks, q, raw);
        for (uint32_t i = 0; ok && i < head_elems; i++) {
            if (!(std::fabsf(got[i] - want[i]) <= 1e-3f)) {
                fprintf(stderr, "attention_decode_raw_batch: [%u] got %.6f want %.6f\n",
                        i, got[i], want[i]);
                ok = false;
            }
        }
    }
    std::free(model);
    ds4_gpu_tensor_free(raw_t);
    ds4_gpu_tensor_free(q_t);
    ds4_gpu_tensor_free(heads_t);
    return ok ? 0 : 1;
}
REGISTER_TEST(attention_decode_raw_batch, test_attention_decode_raw_batch);
