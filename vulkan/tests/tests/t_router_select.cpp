/* Kernel test: ds4_gpu_router_select_tensor (host-side MoE router).
 *
 * Single-token MoE router: biased logits -> full softmax (probs), top-k
 * expert ids sorted by logit desc (selected), and softmax-over-chosen
 * weights scaled by expert_weight_scale.  The kernel is CPU-hosted over
 * the host-mapped tensor memory (like ds4_gpu_add_tensor), so no
 * begin_commands/end_commands are needed to read back the output; the
 * reference below is computed inline in double precision.
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>

/* Full softmax over one logit row (double precision reference). */
static void ref_softmax_all(const float *l, uint32_t n, double *p) {
    double maxv = (double)l[0];
    for (uint32_t i = 1; i < n; i++) maxv = fmax(maxv, (double)l[i]);
    if (std::isinf(maxv) && maxv < 0.0) {            /* all -inf: undefined */
        for (uint32_t i = 0; i < n; i++) p[i] = 1.0 / n;
        return;
    }
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) sum += std::exp((double)l[i] - maxv);
    for (uint32_t i = 0; i < n; i++) p[i] = std::exp((double)l[i] - maxv) / sum;
}

/* Top-k of a logit row: ids sorted by logit desc, tie -> lower id. */
static void ref_topk(const float *l, uint32_t n, uint32_t k, int32_t *sel) {
    std::vector<int32_t> idx(n);
    for (uint32_t i = 0; i < n; i++) idx[i] = (int32_t)i;
    std::stable_sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) {
        if (l[a] != l[b]) return l[a] > l[b];
        return a < b;
    });
    for (uint32_t i = 0; i < k; i++) sel[i] = idx[i];
}

/* Reference weights: softmax over the CHOSEN logits * scale; all -inf
 * chosen -> fallback 1/k (no scale), matching the kernel contract. */
static void ref_weights(const float *l, const int32_t *sel, uint32_t k,
                        double scale, float *w) {
    double maxv = (double)l[sel[0]];
    for (uint32_t i = 1; i < k; i++) maxv = fmax(maxv, (double)l[sel[i]]);
    if (std::isinf(maxv) && maxv < 0.0) {
        for (uint32_t i = 0; i < k; i++) w[i] = 1.0f / k;
        return;
    }
    double sum = 0.0;
    for (uint32_t i = 0; i < k; i++) sum += std::exp((double)l[sel[i]] - maxv);
    for (uint32_t i = 0; i < k; i++)
        w[i] = (float)(std::exp((double)l[sel[i]] - maxv) / sum * scale);
}

/* Runs one router case end-to-end and compares against the CPU reference.
 * Decode supplies one row of n_expert logits; token is the vocabulary token
 * ID and must not be interpreted as a row offset.
 * `bias` (may be NULL) is stored in a synthetic model buffer registered
 * with ds4_gpu_set_model_map, mimicking the engine's model mmap. */
static int run_router_case(const float *logits, uint32_t token,
                           const float *bias, float scale,
                           uint32_t n_expert, uint32_t n_expert_used,
                           uint32_t n_expert_groups, uint32_t n_group_used,
                           bool has_bias, bool hash_mode, const char *label)
{
    const uint64_t logits_bytes = (uint64_t)n_expert * sizeof(float);
    ds4_gpu_tensor *sel_t = ds4_gpu_tensor_alloc((uint64_t)n_expert_used * sizeof(int32_t));
    ds4_gpu_tensor *w_t   = ds4_gpu_tensor_alloc((uint64_t)n_expert_used * sizeof(float));
    ds4_gpu_tensor *p_t   = ds4_gpu_tensor_alloc((uint64_t)n_expert * sizeof(float));
    ds4_gpu_tensor *l_t   = ds4_gpu_tensor_alloc(logits_bytes);
    auto free_tensors = [&]() {
        if (sel_t) ds4_gpu_tensor_free(sel_t);
        if (w_t)   ds4_gpu_tensor_free(w_t);
        if (p_t)   ds4_gpu_tensor_free(p_t);
        if (l_t)   ds4_gpu_tensor_free(l_t);
    };
    if (!sel_t || !w_t || !p_t || !l_t) { free_tensors(); return 1; }
    if (ds4_gpu_tensor_write(l_t, 0, logits, logits_bytes) == 0) { free_tensors(); return 1; }

    /* Synthetic model buffer: junk header + f32 bias at bias_offset. */
    unsigned char *model = nullptr;
    uint64_t model_size = 0, bias_offset = 0;
    if (has_bias && bias) {
        bias_offset = 16;
        model_size = bias_offset + (uint64_t)n_expert * sizeof(float);
        model = (unsigned char*)malloc(model_size);
        if (!model) { free_tensors(); return 1; }
        std::memset(model, 0xAA, bias_offset);
        std::memcpy(model + bias_offset, bias, (uint64_t)n_expert * sizeof(float));
        if (ds4_gpu_set_model_map(model, model_size) == 0) {
            free(model); free_tensors(); return 1;
        }
    }

    int rc = 1;
    if (ds4_gpu_router_select_tensor(sel_t, w_t, p_t, model, model_size,
                                     bias_offset, 0, 0, token,
                                     n_expert, n_expert_used, scale,
                                     n_expert_groups, n_group_used,
                                     has_bias, hash_mode, l_t) == 0) {
        fprintf(stderr, "router_select[%s]: kernel returned error\n", label);
    } else {
        /* Read back the outputs. */
        std::vector<int32_t> got_sel(n_expert_used);
        std::vector<float>   got_w(n_expert_used);
        std::vector<float>   got_p(n_expert);
        bool ok =
            ds4_gpu_tensor_read(sel_t, 0, got_sel.data(),
                                (uint64_t)n_expert_used * sizeof(int32_t)) != 0 &&
            ds4_gpu_tensor_read(w_t,   0, got_w.data(),
                                (uint64_t)n_expert_used * sizeof(float))   != 0 &&
            ds4_gpu_tensor_read(p_t,   0, got_p.data(),
                                (uint64_t)n_expert * sizeof(float))        != 0;
        if (!ok) {
            fprintf(stderr, "router_select[%s]: tensor read failed\n", label);
        } else {
            /* Biased logit row for the reference. */
            std::vector<float> l(n_expert);
            for (uint32_t i = 0; i < n_expert; i++)
                l[i] = logits[i] + (bias ? bias[i] : 0.0f);

            /* Reference outputs. */
            std::vector<int32_t> ref_sel(n_expert_used);
            std::vector<double>  ref_p(n_expert);
            std::vector<float>   ref_w(n_expert_used);
            ref_topk(l.data(), n_expert, n_expert_used, ref_sel.data());
            ref_softmax_all(l.data(), n_expert, ref_p.data());
            ref_weights(l.data(), ref_sel.data(), n_expert_used,
                        (double)scale, ref_w.data());

            /* Compare (tolerance 1e-4). */
            ok = true;
            for (uint32_t k = 0; k < n_expert_used && ok; k++)
                if (got_sel[k] != ref_sel[k]) ok = false;
            for (uint32_t i = 0; i < n_expert && ok; i++)
                if (!(std::fabsf(got_p[i] - (float)ref_p[i]) <= 1e-4f)) ok = false;
            for (uint32_t k = 0; k < n_expert_used && ok; k++)
                if (!(std::fabsf(got_w[k] - ref_w[k]) <= 1e-4f)) ok = false;

            if (!ok) {
                fprintf(stderr, "--- router_select[%s] diagnostic (token=%u, bias=%d) ---\n",
                        label, token, has_bias ? 1 : 0);
                fprintf(stderr, "l: ");
                for (uint32_t i = 0; i < n_expert; i++) fprintf(stderr, "%.4f ", l[i]);
                fprintf(stderr, "\nselected got:");
                for (uint32_t k = 0; k < n_expert_used; k++) fprintf(stderr, " %d", got_sel[k]);
                fprintf(stderr, " want:");
                for (uint32_t k = 0; k < n_expert_used; k++) fprintf(stderr, " %d", ref_sel[k]);
                fprintf(stderr, "\nweights got:");
                for (uint32_t k = 0; k < n_expert_used; k++) fprintf(stderr, " %.6f", got_w[k]);
                fprintf(stderr, " want:");
                for (uint32_t k = 0; k < n_expert_used; k++) fprintf(stderr, " %.6f", ref_w[k]);
                fprintf(stderr, "\nprobs got:");
                for (uint32_t i = 0; i < n_expert; i++) fprintf(stderr, " %.6f", got_p[i]);
                fprintf(stderr, " want:");
                for (uint32_t i = 0; i < n_expert; i++) fprintf(stderr, " %.6f", (float)ref_p[i]);
                fprintf(stderr, "\n");
            }
            rc = ok ? 0 : 1;
        }
    }

    if (model) free(model);
    free_tensors();
    return rc;
}

static int test_router_select(void) {
    const uint32_t n_expert = 8, n_expert_used = 2;
    const uint32_t token = 12345;
    const float scale = 1.5f;
    float logits[n_expert] = {
        0.1f, 0.5f, -0.2f, 2.0f, 1.0f, -1.0f, 0.0f, 3.0f,
    };
    int rc = 0;

    /* Case 1: no bias.  n_expert_groups=2 exercises the documented group
     * fallback (still a global top-k in this first verified version). */
    rc |= run_router_case(logits, token, nullptr, scale,
                          n_expert, n_expert_used, 2, 1, false, false,
                          "no-bias groups-fallback");

    /* Case 2: hash_mode=true must not crash; output is the global top-k
     * (hash path is a documented no-op, the engine overrides it later). */
    rc |= run_router_case(logits, token, nullptr, scale,
                          n_expert, n_expert_used, 0, 0, false, true,
                          "hash-mode-ignored");

    /* Case 3: has_bias with a synthetic model buffer registered via
     * ds4_gpu_set_model_map. */
    float bias[n_expert] = { -0.5f, 0.1f, 0.2f, -0.3f, 0.05f, -0.05f, 0.0f, 0.15f };
    rc |= run_router_case(logits, token, bias, scale,
                          n_expert, n_expert_used, 0, 0, true, false,
                          "has-bias");

    /* Case 4: all -inf logits -> weights fallback 1/n_expert_used and
     * probs 1/n_expert. */
    float ninf_logits[n_expert];
    for (uint32_t i = 0; i < n_expert; i++) ninf_logits[i] = -INFINITY;
    rc |= run_router_case(ninf_logits, token, nullptr, scale,
                          n_expert, n_expert_used, 0, 0, false, false,
                          "all-ninf");

    /* Case 5: error path - null pointers must return 0. */
    if (ds4_gpu_router_select_tensor(nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0,
                                     0, n_expert, n_expert_used, scale,
                                     0, 0, false, false, nullptr) != 0) {
        fprintf(stderr, "router_select: null-pointer call should return 0\n");
        rc = 1;
    }

    return rc;
}
REGISTER_TEST(router_select, test_router_select);
