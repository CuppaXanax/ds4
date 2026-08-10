/* Kernel test: ds4_gpu_attention_prefill_static_mixed_heads_tensor
 * (host-side mixed prefill attention).
 *
 * Prefill attention over the mixed key set: the raw batch KV rows of the
 * current chunk (MLA-style one row per token, K and V share the row) plus
 * the compressed MLA rows of the earlier prefix, with a per-head sink prior
 * and a causal mask.  The kernel is CPU-hosted over the host-mapped tensor
 * memory (like ds4_gpu_attention_prefill_raw_heads_tensor), so no
 * begin_commands/end_commands are needed.
 *
 * Semantics (replicated from the ROCm
 * attention_static_mixed_heads8_online_kernel and the verified raw/decode
 * attention kernels):
 *   raw_count = (window != 0 && t+1 > window) ? window : t+1
 *   raw_start = t+1 - raw_count
 *   comp_count = (n_comp != 0 && ratio != 0) ? min((t+1)/ratio, n_comp) : 0
 *   scale = 1/sqrt(head_dim)
 *   score(raw r)  = dot(q_head, raw_kv[raw_start+r]) * scale
 *   score(comp c) = dot(q_head, comp_kv[c]) * scale
 *   max starts at the sink prior sinks[h]
 *   heads[t][h] = sum(exp(score - max) * v) / (exp(sinks[h]-max) + sum exp(...))
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

/* Inline CPU reference: raw rows in chronological window order, then the
 * first comp_count compressed rows; comp_kv is already dequantized f32. */
static void ref_prefill_static_mixed(
        float *heads, uint32_t n_tokens,
        const float *sinks, const float *q, const float *raw_kv,
        const float *comp_kv, uint32_t n_comp,
        uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim)
{
    const float scale = 1.0f / std::sqrt((float)head_dim);
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t raw_count = (window != 0 && t + 1u > window) ? window : t + 1u;
        const uint32_t raw_start = t + 1u - raw_count;
        uint32_t comp_count = 0;
        if (n_comp != 0 && ratio != 0) {
            comp_count = (t + 1u) / ratio;
            if (comp_count > n_comp) comp_count = n_comp;
        }
        std::vector<float> score(raw_count + comp_count);

        for (uint32_t h = 0; h < n_head; h++) {
            const float *qh = q + ((uint64_t)t * n_head + h) * head_dim;
            float max_score = sinks[h];
            for (uint32_t r = 0; r < raw_count; r++) {
                const float *kv = raw_kv + (uint64_t)(raw_start + r) * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
                score[r] = dot * scale;
                if (score[r] > max_score) max_score = score[r];
            }
            for (uint32_t c = 0; c < comp_count; c++) {
                const float *kv = comp_kv + (uint64_t)c * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
                score[raw_count + c] = dot * scale;
                if (score[raw_count + c] > max_score) max_score = score[raw_count + c];
            }

            float *oh = heads + ((uint64_t)t * n_head + h) * head_dim;
            std::memset(oh, 0, (size_t)head_dim * sizeof(float));
            float denom = std::exp(sinks[h] - max_score);
            for (uint32_t r = 0; r < raw_count; r++) {
                const float w = std::exp(score[r] - max_score);
                const float *kv = raw_kv + (uint64_t)(raw_start + r) * head_dim;
                denom += w;
                for (uint32_t d = 0; d < head_dim; d++) oh[d] += w * kv[d];
            }
            for (uint32_t c = 0; c < comp_count; c++) {
                const float w = std::exp(score[raw_count + c] - max_score);
                const float *kv = comp_kv + (uint64_t)c * head_dim;
                denom += w;
                for (uint32_t d = 0; d < head_dim; d++) oh[d] += w * kv[d];
            }
            const float inv = 1.0f / denom;
            for (uint32_t d = 0; d < head_dim; d++) oh[d] *= inv;
        }
    }
}

/* Runs one static-mixed prefill case.  Returns 0 on PASS. */
static int run_prefill_case(
        uint32_t n_tokens, uint32_t n_comp, uint32_t n_head, uint32_t head_dim,
        uint32_t window, uint32_t ratio,
        const float *sinks,
        const float *q,
        const float *raw_kv,        /* n_tokens * head_dim floats */
        const float *comp_f32,      /* n_comp * head_dim floats */
        const uint16_t *comp_f16,   /* used when comp_kv_f16 != 0 */
        uint32_t comp_kv_f16,
        const char *label)
{
    int rc = 1;
    const uint64_t head_elems = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t kv_elems = (uint64_t)n_tokens * head_dim;
    const uint64_t comp_elems = (uint64_t)n_comp * head_dim;

    ds4_gpu_tensor *heads_t  = ds4_gpu_tensor_alloc(head_elems * sizeof(float));
    ds4_gpu_tensor *q_t      = ds4_gpu_tensor_alloc(head_elems * sizeof(float));
    ds4_gpu_tensor *raw_t    = ds4_gpu_tensor_alloc(kv_elems * sizeof(float));
    ds4_gpu_tensor *comp_t   = n_comp ? ds4_gpu_tensor_alloc(
            comp_elems * (comp_kv_f16 ? sizeof(uint16_t) : sizeof(float))) : nullptr;
    if (!heads_t || !q_t || !raw_t || (n_comp && !comp_t)) {
        if (heads_t) ds4_gpu_tensor_free(heads_t);
        if (q_t)     ds4_gpu_tensor_free(q_t);
        if (raw_t)   ds4_gpu_tensor_free(raw_t);
        if (comp_t)  ds4_gpu_tensor_free(comp_t);
        return 1;
    }

    bool ok =
        ds4_gpu_tensor_write(q_t,   0, q,     head_elems * sizeof(float)) != 0 &&
        ds4_gpu_tensor_write(raw_t, 0, raw_kv, kv_elems * sizeof(float)) != 0;
    if (n_comp) {
        ok = ok && ds4_gpu_tensor_write(comp_t, 0,
                                        comp_kv_f16 ? (const void *)comp_f16
                                                    : (const void *)comp_f32,
                                        comp_elems * (comp_kv_f16 ? sizeof(uint16_t)
                                                                  : sizeof(float))) != 0;
    }

    /* Synthetic model buffer: junk header + f32 sinks at sinks_offset. */
    const uint64_t sinks_offset = 16;
    const uint64_t model_size = sinks_offset + (uint64_t)n_head * sizeof(float);
    unsigned char *model = (unsigned char *)std::malloc(model_size);
    if (!model) ok = false;
    if (ok) {
        std::memset(model, 0xAA, (size_t)sinks_offset);
        std::memcpy(model + sinks_offset, sinks, (size_t)n_head * sizeof(float));
        ok = ds4_gpu_set_model_map(model, model_size) != 0;
    }

    if (ok) {
        const int ret = ds4_gpu_attention_prefill_static_mixed_heads_tensor(
                heads_t, model, model_size, sinks_offset, q_t, raw_t, comp_t,
                comp_kv_f16, n_tokens, n_comp, window, ratio, n_head, head_dim);
        if (ret == 0) {
            fprintf(stderr, "attention_prefill_static_mixed[%s]: kernel returned error\n", label);
            ok = false;
        } else {
            std::vector<float> got(head_elems);
            ok = ds4_gpu_tensor_read(heads_t, 0, got.data(),
                                     head_elems * sizeof(float)) != 0;
            if (ok) {
                /* Reference from the data the kernel saw: for f16 comp rows,
                 * decode the stored halfs first. */
                std::vector<float> comp_ref;
                const float *comp_use = comp_f32;
                if (n_comp && comp_kv_f16) {
                    comp_ref.resize(comp_elems);
                    for (uint64_t i = 0; i < comp_elems; i++)
                        comp_ref[i] = half_to_float(comp_f16[i]);
                    comp_use = comp_ref.data();
                }
                std::vector<float> ref(head_elems);
                ref_prefill_static_mixed(ref.data(), n_tokens, sinks, q, raw_kv,
                                         comp_use, n_comp, window, ratio,
                                         n_head, head_dim);
                for (uint64_t i = 0; i < head_elems && ok; i++) {
                    if (!(std::fabsf(got[i] - ref[i]) <= 1e-3f)) {
                        fprintf(stderr, "attention_prefill_static_mixed[%s]: head[%llu] got %.6f want %.6f\n",
                                label, (unsigned long long)i, got[i], ref[i]);
                        ok = false;
                    }
                }
            }
        }
        if (ok && ds4_gpu_attention_prefill_static_mixed_heads_tensor(
                heads_t, model, model_size, sinks_offset, q_t, raw_t, comp_t,
                1, n_tokens, n_comp, window, ratio, n_head, 3) != 0) {
            fprintf(stderr, "attention_prefill_static_mixed[%s]: odd F16 element count was accepted\n", label);
            ok = false;
        }
        if (ok && ds4_gpu_attention_prefill_static_mixed_heads_tensor(
                heads_t, model, model_size, sinks_offset, q_t, raw_t, comp_t,
                comp_kv_f16, n_tokens, 4097, window, ratio, n_head, head_dim) != 0) {
            fprintf(stderr, "attention_prefill_static_mixed[%s]: n_comp > 4096 was accepted\n", label);
            ok = false;
        }
    }

    if (model) std::free(model);
    ds4_gpu_tensor_free(heads_t);
    ds4_gpu_tensor_free(q_t);
    ds4_gpu_tensor_free(raw_t);
    if (comp_t) ds4_gpu_tensor_free(comp_t);
    return ok ? 0 : 1;
}

static int test_attention_prefill_static_mixed(void) {
    const uint32_t n_tokens = 4, n_comp = 1, n_head = 2, head_dim = 4;
    const uint32_t window = 3, ratio = 2;
    const uint64_t head_elems = (uint64_t)n_tokens * n_head * head_dim;
    const uint64_t kv_elems = (uint64_t)n_tokens * head_dim;

    /* q: 4 tokens x 2 heads. */
    std::vector<float> qv(head_elems);
    for (uint64_t i = 0; i < head_elems; i++)
        qv[i] = 0.25f * (float)((int32_t)(i % 7) - 3) + 0.1f * (float)(int32_t)(i / 7);
    /* raw KV: 4 MLA rows (one per token), K and V share the row. */
    std::vector<float> kvv(kv_elems);
    for (uint64_t i = 0; i < kv_elems; i++)
        kvv[i] = 0.5f * (float)((int32_t)(i % 5) - 2) - 0.05f * (float)(int32_t)(i / 5);
    /* Per-head sink priors (model buffer). */
    float sinks[n_head] = { 0.1f, -0.05f };

    /* Compressed rows (MLA cache): one comp row for token 0. */
    float comp_f32[n_comp * head_dim] = {
         0.8f, -0.3f,  0.4f,  0.9f,
    };
    uint16_t comp_f16[n_comp * head_dim];
    for (uint32_t i = 0; i < n_comp * head_dim; i++)
        comp_f16[i] = float_to_half(comp_f32[i]);

    int rc = 0;

    /* Case 1: mixed raw + comp, comp stored as f16 (exercises half decode).
     * window=3, ratio=2: raw rows for t are [max(0,t-2)..t]; comp rows are
     * visible once (t+1)/ratio >= 1, i.e. from t=1 on. */
    rc |= run_prefill_case(n_tokens, n_comp, n_head, head_dim, window, ratio,
                           sinks, qv.data(), kvv.data(), comp_f32, comp_f16, 1,
                           "mixed-f16-window3-ratio2");

    /* Case 2: same data with comp stored as f32 (f32 storage path). */
    rc |= run_prefill_case(n_tokens, n_comp, n_head, head_dim, window, ratio,
                           sinks, qv.data(), kvv.data(), comp_f32, nullptr, 0,
                           "mixed-f32-window3-ratio2");

    /* Case 3: wider window (full causal) and bigger ratio: comp rows only
     * appear later in the chunk.  window=8 (> n_tokens), ratio=3:
     * comp visible from t=2 on. */
    rc |= run_prefill_case(n_tokens, n_comp, n_head, head_dim, 8, 3,
                           sinks, qv.data(), kvv.data(), comp_f32, comp_f16, 1,
                           "mixed-f16-window8-ratio3");

    /* Case 4: ratio=0 disables compressed rows entirely (raw-only), even
     * though n_comp != 0 (comp_kv must still be a valid buffer). */
    rc |= run_prefill_case(n_tokens, n_comp, n_head, head_dim, window, 0,
                           sinks, qv.data(), kvv.data(), comp_f32, comp_f16, 1,
                           "mixed-ratio0");

    /* A chunk-sized case keeps the static path honest beyond the tiny
     * correctness vectors while remaining cheap for the kernel harness. */
    {
        const uint32_t long_tokens = 32, long_comp = 20;
        std::vector<float> long_q((uint64_t)long_tokens * n_head * head_dim);
        std::vector<float> long_raw((uint64_t)long_tokens * head_dim);
        std::vector<float> long_comp_data((uint64_t)long_comp * head_dim);
        std::vector<uint16_t> long_comp_half((uint64_t)long_comp * head_dim);
        for (size_t i = 0; i < long_q.size(); i++)
            long_q[i] = 0.01f * (float)((int)((i * 17) % 101) - 50);
        for (size_t i = 0; i < long_raw.size(); i++)
            long_raw[i] = 0.02f * (float)((int)((i * 13) % 73) - 36);
        for (size_t i = 0; i < long_comp_data.size(); i++) {
            long_comp_data[i] = 0.015f * (float)((int)((i * 11) % 89) - 44);
            long_comp_half[i] = float_to_half(long_comp_data[i]);
        }
        rc |= run_prefill_case(long_tokens, long_comp, n_head, head_dim, 16, 4,
                               sinks, long_q.data(), long_raw.data(),
                               long_comp_data.data(), long_comp_half.data(), 1,
                               "static-32-token-mixed");
    }

    /* Case 5: error path - null pointers must return 0. */
    if (ds4_gpu_attention_prefill_static_mixed_heads_tensor(
            nullptr, nullptr, 0, 0, nullptr, nullptr, nullptr, 0,
            n_tokens, n_comp, window, ratio, n_head, head_dim) != 0) {
        fprintf(stderr, "attention_prefill_static_mixed: null-pointer call should return 0\n");
        rc = 1;
    }

    return rc;
}
REGISTER_TEST(attention_prefill_static_mixed_heads, test_attention_prefill_static_mixed);
