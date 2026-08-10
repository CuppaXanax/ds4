/* Kernel test: ds4_gpu_router_select_tensor (canonical Vulkan MoE router).
 *
 * DeepSeek V4 routing uses sqrt(softplus(logit)) probabilities, optional
 * bias only for top-k selection, and normalized unbiased selected weights.
 * The reference below is computed inline in double precision.
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>

static float ref_router_prob(float logit) {
    float softplus;
    if (logit > 20.0f) softplus = logit;
    else if (logit < -20.0f) softplus = std::exp(logit);
    else softplus = std::log1p(std::exp(logit));
    return std::sqrt(softplus);
}

static void ref_topk(const float *prob, const float *bias,
                     uint32_t n, uint32_t k, int32_t *sel) {
    std::vector<int32_t> idx(n);
    for (uint32_t i = 0; i < n; i++) idx[i] = (int32_t)i;
    std::stable_sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) {
        const float av = prob[a] + (bias ? bias[a] : 0.0f);
        const float bv = prob[b] + (bias ? bias[b] : 0.0f);
        if (av != bv) return av > bv;
        return a < b;
    });
    for (uint32_t i = 0; i < k; i++) sel[i] = idx[i];
}

/* Reference weights: normalize selected router probabilities, then scale. */
static void ref_weights(const float *prob, const int32_t *sel, uint32_t n_expert,
                        uint32_t k, double scale, float *w) {
    double sum = 0.0;
    for (uint32_t i = 0; i < k; i++)
        if (sel[i] >= 0 && (uint32_t)sel[i] < n_expert) sum += prob[sel[i]];
    if (sum < 6.103515625e-5) sum = 6.103515625e-5;
    for (uint32_t i = 0; i < k; i++)
        w[i] = sel[i] >= 0 && (uint32_t)sel[i] < n_expert
            ? (float)(prob[sel[i]] / sum * scale) : 0.0f;
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
    uint64_t model_size = 16, bias_offset = 0;
    if (has_bias && bias) {
        bias_offset = 16;
        model_size = bias_offset + (uint64_t)n_expert * sizeof(float);
    }
    unsigned char *model = (unsigned char*)malloc(model_size);
    if (!model) { free_tensors(); return 1; }
    std::memset(model, 0xAA, model_size);
    if (has_bias && bias) {
        std::memcpy(model + bias_offset, bias, (uint64_t)n_expert * sizeof(float));
    }
    if (ds4_gpu_set_model_map(model, model_size) == 0) {
        free(model); free_tensors(); return 1;
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
            std::vector<float> ref_prob(n_expert);
            for (uint32_t i = 0; i < n_expert; i++)
                ref_prob[i] = ref_router_prob(logits[i]);

            /* Reference outputs. */
            std::vector<int32_t> ref_sel(n_expert_used);
            std::vector<float>   ref_w(n_expert_used);
            ref_topk(ref_prob.data(), bias, n_expert, n_expert_used, ref_sel.data());
            ref_weights(ref_prob.data(), ref_sel.data(), n_expert, n_expert_used,
                        (double)scale, ref_w.data());

            /* Compare (tolerance 1e-4). */
            ok = true;
            for (uint32_t k = 0; k < n_expert_used && ok; k++)
                if (got_sel[k] != ref_sel[k]) ok = false;
            for (uint32_t i = 0; i < n_expert && ok; i++)
                if (!(std::fabsf(got_p[i] - ref_prob[i]) <= 1e-4f)) ok = false;
            for (uint32_t k = 0; k < n_expert_used && ok; k++)
                if (!(std::fabsf(got_w[k] - ref_w[k]) <= 1e-4f)) ok = false;

            if (!ok) {
                fprintf(stderr, "--- router_select[%s] diagnostic (token=%u, bias=%d) ---\n",
                        label, token, has_bias ? 1 : 0);
                fprintf(stderr, "prob: ");
                for (uint32_t i = 0; i < n_expert; i++) fprintf(stderr, "%.4f ", ref_prob[i]);
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
                for (uint32_t i = 0; i < n_expert; i++) fprintf(stderr, " %.6f", ref_prob[i]);
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
    const uint32_t n_expert = 256, n_expert_used = 6;
    const uint32_t token = 12345;
    const float scale = 1.5f;
    float logits[n_expert] = {
        0.1f, 0.5f, -0.2f, 2.0f, 1.0f, -1.0f, 0.0f, 3.0f,
    };
    int rc = 0;

    /* Case 1: no bias. */
    rc |= run_router_case(logits, token, nullptr, scale,
                          n_expert, n_expert_used, 0, 0, false, false,
                          "no-bias");

    /* Case 2: has_bias with a synthetic model buffer registered via
     * ds4_gpu_set_model_map. */
    float bias[n_expert] = { -0.5f, 0.1f, 0.2f, -0.3f, 0.05f, -0.05f, 0.0f, 0.15f };
    rc |= run_router_case(logits, token, bias, scale,
                          n_expert, n_expert_used, 0, 0, true, false,
                          "has-bias");

    /* Case 3: all -inf logits -> zero probabilities and zero weights. */
    float ninf_logits[n_expert];
    for (uint32_t i = 0; i < n_expert; i++) ninf_logits[i] = -INFINITY;
    rc |= run_router_case(ninf_logits, token, nullptr, scale,
                          n_expert, n_expert_used, 0, 0, false, false,
                          "all-ninf");

    /* Case 4: error path - null pointers must return 0. */
    if (ds4_gpu_router_select_tensor(nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0,
                                     0, n_expert, n_expert_used, scale,
                                     0, 0, false, false, nullptr) != 0) {
        fprintf(stderr, "router_select: null-pointer call should return 0\n");
        rc = 1;
    }

    return rc;
}
REGISTER_TEST(router_select, test_router_select);

static int test_router_select_batch(void) {
    const uint32_t n_expert = 256, n_expert_used = 6, n_tokens = 2;
    const float scale = 1.5f;
    const float logits[n_tokens * n_expert] = {
        0.1f, 0.5f, -0.2f, 2.0f, 1.0f, -1.0f, 0.0f, 3.0f,
        1.2f, -0.4f, 2.5f, 0.3f, 1.8f, 0.1f, -2.0f, 0.7f,
    };
    const int32_t tokens[n_tokens] = {3, 99};
    const float bias[n_expert] = {
        -0.5f, 0.1f, 0.2f, -0.3f, 0.05f, -0.05f, 0.0f, 0.15f,
    };
    const uint32_t hash_rows = 8;
    const uint64_t bias_offset = 16;
    const uint64_t hash_offset = bias_offset + sizeof(bias);
    const uint64_t model_size = hash_offset +
        (uint64_t)hash_rows * n_expert_used * sizeof(int32_t);
    std::vector<unsigned char> model(model_size, 0);
    std::memcpy(model.data() + bias_offset, bias, sizeof(bias));
    int32_t *hash = (int32_t *)(model.data() + hash_offset);
    for (uint32_t row = 0; row < hash_rows; row++) {
        hash[(uint64_t)row * n_expert_used] = (int32_t)((row + 1u) % n_expert);
        hash[(uint64_t)row * n_expert_used + 1u] = (int32_t)((row + 4u) % n_expert);
    }
    hash[1] = -1;

    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(
        (uint64_t)n_tokens * n_expert_used * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(
        (uint64_t)n_tokens * n_expert_used * sizeof(float));
    ds4_gpu_tensor *probs = ds4_gpu_tensor_alloc(
        (uint64_t)n_tokens * n_expert * sizeof(float));
    ds4_gpu_tensor *logits_t = ds4_gpu_tensor_alloc(sizeof(logits));
    ds4_gpu_tensor *tokens_t = ds4_gpu_tensor_alloc(sizeof(tokens));
    auto cleanup = [&]() {
        if (tokens_t) ds4_gpu_tensor_free(tokens_t);
        if (logits_t) ds4_gpu_tensor_free(logits_t);
        if (probs) ds4_gpu_tensor_free(probs);
        if (weights) ds4_gpu_tensor_free(weights);
        if (selected) ds4_gpu_tensor_free(selected);
    };
    if (!selected || !weights || !probs || !logits_t || !tokens_t ||
        !ds4_gpu_tensor_write(logits_t, 0, logits, sizeof(logits)) ||
        !ds4_gpu_tensor_write(tokens_t, 0, tokens, sizeof(tokens))) {
        cleanup();
        return 1;
    }

    auto run_case = [&](bool hash_mode, const char *label) -> int {
        if (!ds4_gpu_router_select_batch_tensor(
                selected, weights, probs, model.data(), model.size(),
                bias_offset, hash_offset, hash_rows, 0, 0,
                !hash_mode, hash_mode, logits_t, tokens_t,
                n_expert, n_expert_used, scale, n_tokens)) {
            fprintf(stderr, "router_select_batch[%s]: kernel returned error\n", label);
            return 1;
        }
        std::vector<int32_t> got_selected((uint64_t)n_tokens * n_expert_used);
        std::vector<float> got_weights((uint64_t)n_tokens * n_expert_used);
        std::vector<float> got_probs((uint64_t)n_tokens * n_expert);
        float got_logits[n_tokens * n_expert];
        if (!ds4_gpu_tensor_read(selected, 0, got_selected.data(),
                                 got_selected.size() * sizeof(int32_t)) ||
            !ds4_gpu_tensor_read(weights, 0, got_weights.data(),
                                 got_weights.size() * sizeof(float)) ||
            !ds4_gpu_tensor_read(probs, 0, got_probs.data(),
                                 got_probs.size() * sizeof(float)) ||
            !ds4_gpu_tensor_read(logits_t, 0, got_logits, sizeof(got_logits)) ||
            std::memcmp(got_logits, logits, sizeof(logits)) != 0) {
            fprintf(stderr, "router_select_batch[%s]: read or immutability failure\n", label);
            return 1;
        }
        for (uint32_t t = 0; t < n_tokens; t++) {
            float ref_probs[n_expert];
            int32_t ref_selected[n_expert_used];
            float expected_weights[n_expert_used];
            for (uint32_t i = 0; i < n_expert; i++)
                ref_probs[i] = ref_router_prob(logits[(uint64_t)t * n_expert + i]);
            if (hash_mode) {
                const uint32_t hash_token = tokens[t] >= 0 &&
                    (uint32_t)tokens[t] < hash_rows ? (uint32_t)tokens[t] : 0u;
                std::memcpy(ref_selected,
                            hash + (uint64_t)hash_token * n_expert_used,
                            sizeof(ref_selected));
            } else {
                ref_topk(ref_probs, bias, n_expert, n_expert_used, ref_selected);
            }
            ref_weights(ref_probs, ref_selected, n_expert, n_expert_used,
                        scale, expected_weights);
            for (uint32_t i = 0; i < n_expert; i++) {
                if (std::fabs(got_probs[(uint64_t)t * n_expert + i] - ref_probs[i]) > 1e-4f)
                    return 1;
            }
            for (uint32_t i = 0; i < n_expert_used; i++) {
                if (got_selected[(uint64_t)t * n_expert_used + i] != ref_selected[i] ||
                    std::fabs(got_weights[(uint64_t)t * n_expert_used + i] - expected_weights[i]) > 1e-4f)
                    return 1;
            }
        }
        return 0;
    };

    const int rc = run_case(false, "biased") | run_case(true, "hash");
    cleanup();
    return rc;
}

REGISTER_TEST(router_select_batch, test_router_select_batch);
