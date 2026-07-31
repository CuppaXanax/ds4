/* Kernel tests: DS4 compressed-attention indexer (host-side CPU fallbacks).
 *
 * The three kernels under test are CPU-hosted over the host-mapped tensor
 * memory (same pattern as ds4_gpu_add_tensor / ds4_gpu_router_select_tensor),
 * so NO begin_commands/end_commands are needed to read back the outputs.
 *
 * Reference math, replicated from the engine CPU path (ds4.c
 * indexer_allowed_decode_one* score loop) and the Metal/CUDA indexer score
 * kernels:
 *
 *   score[c] = sum_h max(0, dot(q[h], index_comp[c])) * weights[h] * scale
 *
 * with q [n_head][head_dim] (single-token) or [n_tokens][n_head][head_dim]
 * (batched), weights [n_head] / [n_tokens][n_head], index_comp
 * [n_comp][head_dim], scores [n_comp] / [n_tokens][n_comp].
 *
 * The batched variant additionally applies the causal visibility mask of the
 * Metal tiled kernel: at position p = pos0 + t only the first (p + 1) / ratio
 * compressed rows exist; rows beyond are written as -INFINITY.
 *
 * top-k writes per-token indices in descending score order, lower index wins
 * ties (matches ds4.c and CUDA topk_score_better); masked -INF rows sort last.
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>

/* Per-row score used by both host fallbacks and the test reference. */
static float ref_row_score(const float *qh, const float *w,
                           const float *kv, uint32_t n_head,
                           uint32_t head_dim, float scale) {
    float acc = 0.0f;
    for (uint32_t h = 0; h < n_head; h++) {
        const float *q = qh + (uint64_t)h * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; d++) dot += kv[d] * q[d];
        if (dot < 0.0f) dot = 0.0f;
        acc += dot * (w[h] * scale);
    }
    return acc;
}

/* ------------------------------------------------------------------ */
/* indexer_score_one: one token vs. all compressed rows               */
/* ------------------------------------------------------------------ */
static int test_indexer_score_one(void) {
    const uint32_t n_head = 4, head_dim = 8, n_comp = 6;
    const float scale = 1.0f / std::sqrtf((float)(n_head * head_dim));

    /* q[h][d]: deterministic pattern with mixed signs so ReLU matters. */
    std::vector<float> q(n_head * head_dim);
    for (uint32_t h = 0; h < n_head; h++)
        for (uint32_t d = 0; d < head_dim; d++)
            q[h * head_dim + d] = 0.5f * (float)(h + 1) * (float)(d + 1) - 1.25f;

    /* weights[h]: mixed signs. */
    std::vector<float> w(n_head);
    for (uint32_t h = 0; h < n_head; h++)
        w[h] = 0.25f * (float)(h + 1) - 0.5f;

    /* index_comp[c][d]: distinct rows. */
    std::vector<float> k(n_comp * head_dim);
    for (uint32_t c = 0; c < n_comp; c++)
        for (uint32_t d = 0; d < head_dim; d++)
            k[c * head_dim + d] = 0.1f * (float)(c * 3 + d) - 0.3f * (float)c;

    std::vector<float> ref(n_comp);
    for (uint32_t c = 0; c < n_comp; c++)
        ref[c] = ref_row_score(q.data(), w.data(), &k[c * head_dim],
                               n_head, head_dim, scale);

    ds4_gpu_tensor *st = ds4_gpu_tensor_alloc((uint64_t)n_comp * sizeof(float));
    ds4_gpu_tensor *qt = ds4_gpu_tensor_alloc(q.size() * sizeof(float));
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc(w.size() * sizeof(float));
    ds4_gpu_tensor *kt = ds4_gpu_tensor_alloc(k.size() * sizeof(float));
    ds4_gpu_tensor *tiny = nullptr;
    std::vector<float> got(n_comp);
    int rc = 1;
    if (!st || !qt || !wt || !kt) goto done;
    if (ds4_gpu_tensor_write(qt, 0, q.data(), q.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(wt, 0, w.data(), w.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(kt, 0, k.data(), k.size() * sizeof(float)) == 0) {
        fprintf(stderr, "indexer_score_one: input write failed\n");
        goto done;
    }
    if (ds4_gpu_indexer_score_one_tensor(st, qt, wt, kt, n_comp, n_head,
                                         head_dim, scale) == 0) {
        fprintf(stderr, "indexer_score_one: kernel returned error\n");
        goto done;
    }
    if (ds4_gpu_tensor_read(st, 0, got.data(), n_comp * sizeof(float)) == 0) {
        fprintf(stderr, "indexer_score_one: read failed\n");
        goto done;
    }
    rc = 0;
    for (uint32_t c = 0; c < n_comp; c++) {
        if (!(std::fabsf(got[c] - ref[c]) <= 1e-3f)) {
            fprintf(stderr, "indexer_score_one: c=%u got %.6f want %.6f\n",
                    c, got[c], ref[c]);
            rc = 1;
        }
    }
    /* Error paths: null tensors / undersized buffers must return 0. */
    if (ds4_gpu_indexer_score_one_tensor(st, nullptr, wt, kt, n_comp, n_head,
                                         head_dim, scale) != 0) rc = 1;
    tiny = ds4_gpu_tensor_alloc(sizeof(float));
    if (!tiny) goto done;
    if (ds4_gpu_indexer_score_one_tensor(tiny, qt, wt, kt, n_comp, n_head,
                                         head_dim, scale) != 0) rc = 1;
done:
    if (tiny) ds4_gpu_tensor_free(tiny);
    if (kt) ds4_gpu_tensor_free(kt);
    if (wt) ds4_gpu_tensor_free(wt);
    if (qt) ds4_gpu_tensor_free(qt);
    if (st) ds4_gpu_tensor_free(st);
    return rc;
}
REGISTER_TEST(indexer_score_one, test_indexer_score_one);

/* ------------------------------------------------------------------ */
/* indexer_scores_decode_batch: n_tokens vs. all rows + visibility     */
/* ------------------------------------------------------------------ */
static int test_indexer_scores_decode_batch(void) {
    const uint32_t n_head = 2, head_dim = 4, n_comp = 8, n_tokens = 3;
    const uint32_t pos0 = 10, ratio = 4;
    const float scale = 0.35f;

    std::vector<float> q(n_tokens * n_head * head_dim);
    std::vector<float> w(n_tokens * n_head);
    std::vector<float> k(n_comp * head_dim);
    for (uint32_t t = 0; t < n_tokens; t++)
        for (uint32_t h = 0; h < n_head; h++)
            for (uint32_t d = 0; d < head_dim; d++)
                q[(t * n_head + h) * head_dim + d] =
                    0.4f * (float)(t + 1) * (float)(h + 1) + 0.05f * (float)d;
    for (uint32_t t = 0; t < n_tokens; t++)
        for (uint32_t h = 0; h < n_head; h++)
            w[t * n_head + h] = 0.5f - 0.2f * (float)(t + h);
    for (uint32_t c = 0; c < n_comp; c++)
        for (uint32_t d = 0; d < head_dim; d++)
            k[c * head_dim + d] = 0.2f * (float)(c + d) - 0.6f;

    /* Reference with the causal visibility mask: row c visible for token t
     * iff c < (pos0 + t + 1) / ratio (clamped to n_comp). */
    std::vector<float> ref(n_tokens * n_comp);
    for (uint32_t t = 0; t < n_tokens; t++) {
        uint32_t visible = (pos0 + t + 1u) / ratio;
        if (visible > n_comp) visible = n_comp;
        for (uint32_t c = 0; c < n_comp; c++) {
            if (c >= visible) {
                ref[t * n_comp + c] = -INFINITY;
            } else {
                ref[t * n_comp + c] =
                    ref_row_score(&q[(uint64_t)t * n_head * head_dim],
                                  &w[(uint64_t)t * n_head],
                                  &k[(uint64_t)c * head_dim],
                                  n_head, head_dim, scale);
            }
        }
    }

    ds4_gpu_tensor *st = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_comp * sizeof(float));
    ds4_gpu_tensor *qt = ds4_gpu_tensor_alloc(q.size() * sizeof(float));
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc(w.size() * sizeof(float));
    ds4_gpu_tensor *kt = ds4_gpu_tensor_alloc(k.size() * sizeof(float));
    ds4_gpu_tensor *tiny = nullptr;
    std::vector<float> got(n_tokens * n_comp);
    int rc = 1;
    if (!st || !qt || !wt || !kt) goto done;
    if (ds4_gpu_tensor_write(qt, 0, q.data(), q.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(wt, 0, w.data(), w.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(kt, 0, k.data(), k.size() * sizeof(float)) == 0) {
        fprintf(stderr, "indexer_scores_decode_batch: input write failed\n");
        goto done;
    }
    if (ds4_gpu_indexer_scores_decode_batch_tensor(
            st, qt, wt, kt, n_comp, n_tokens, pos0, n_head, head_dim,
            ratio, scale) == 0) {
        fprintf(stderr, "indexer_scores_decode_batch: kernel returned error\n");
        goto done;
    }
    if (ds4_gpu_tensor_read(st, 0, got.data(), got.size() * sizeof(float)) == 0) {
        fprintf(stderr, "indexer_scores_decode_batch: read failed\n");
        goto done;
    }
    rc = 0;
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t c = 0; c < n_comp; c++) {
            const float g = got[t * n_comp + c], r = ref[t * n_comp + c];
            bool ok;
            if (std::isinf(r) && r < 0.0f)
                ok = std::isinf(g) && g < 0.0f;   /* masked row: -INFINITY */
            else
                ok = std::fabsf(g - r) <= 1e-3f;
            if (!ok) {
                fprintf(stderr,
                        "indexer_scores_decode_batch: t=%u c=%u got %.6f want %.6f\n",
                        t, c, g, r);
                rc = 1;
            }
        }
    }
    /* Error paths. */
    if (ds4_gpu_indexer_scores_decode_batch_tensor(
            st, qt, wt, kt, n_comp, n_tokens, pos0, n_head, head_dim,
            0 /* ratio */, scale) != 0) rc = 1;
    tiny = ds4_gpu_tensor_alloc(sizeof(float));
    if (!tiny) goto done;
    if (ds4_gpu_indexer_scores_decode_batch_tensor(
            tiny, qt, wt, kt, n_comp, n_tokens, pos0, n_head, head_dim,
            ratio, scale) != 0) rc = 1;
done:
    if (tiny) ds4_gpu_tensor_free(tiny);
    if (kt) ds4_gpu_tensor_free(kt);
    if (wt) ds4_gpu_tensor_free(wt);
    if (qt) ds4_gpu_tensor_free(qt);
    if (st) ds4_gpu_tensor_free(st);
    return rc;
}
REGISTER_TEST(indexer_scores_decode_batch, test_indexer_scores_decode_batch);

/* ------------------------------------------------------------------ */
/* indexer_topk: per-token descending indices, tie -> lower index     */
/* ------------------------------------------------------------------ */
static void ref_topk_row(const float *row, uint32_t n_comp, uint32_t top_k,
                         uint32_t *out) {
    std::vector<uint32_t> idx(n_comp);
    for (uint32_t c = 0; c < n_comp; c++) idx[c] = c;
    std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) {
        if (std::isnan(row[a])) return false;
        if (std::isnan(row[b])) return true;
        return row[a] > row[b] || (row[a] == row[b] && a < b);
    });
    for (uint32_t k = 0; k < top_k; k++) out[k] = idx[k];
}

static int test_indexer_topk(void) {
    const uint32_t n_comp = 8, n_tokens = 2, top_k = 3;
    /* Row 0: tie at 2.5 (idx 1 and 3 -> idx 1 first), -INF masked row. */
    const float scores[n_tokens * n_comp] = {
        1.0f,  2.5f, -0.5f, 2.5f, 0.0f, -INFINITY, 0.75f, 1.25f,
        0.1f,  0.1f,  0.1f, 0.1f, 0.1f,  0.1f,    0.1f,  0.1f,
    };

    uint32_t ref[n_tokens * top_k];
    for (uint32_t t = 0; t < n_tokens; t++)
        ref_topk_row(scores + (uint64_t)t * n_comp, n_comp, top_k,
                     ref + (uint64_t)t * top_k);

    ds4_gpu_tensor *st = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_comp * sizeof(float));
    ds4_gpu_tensor *ot = ds4_gpu_tensor_alloc((uint64_t)n_tokens * top_k * sizeof(uint32_t));
    std::vector<uint32_t> got(n_tokens * top_k);
    int rc = 1;
    if (!st || !ot) goto done;
    if (ds4_gpu_tensor_write(st, 0, scores, sizeof(scores)) == 0) {
        fprintf(stderr, "indexer_topk: input write failed\n");
        goto done;
    }
    if (ds4_gpu_indexer_topk_tensor(ot, st, n_comp, n_tokens, top_k) == 0) {
        fprintf(stderr, "indexer_topk: kernel returned error\n");
        goto done;
    }
    if (ds4_gpu_tensor_read(ot, 0, got.data(),
                            got.size() * sizeof(uint32_t)) == 0) {
        fprintf(stderr, "indexer_topk: read failed\n");
        goto done;
    }
    rc = 0;
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t k = 0; k < top_k; k++) {
            if (got[t * top_k + k] != ref[t * top_k + k]) {
                fprintf(stderr, "indexer_topk: t=%u k=%u got %u want %u\n",
                        t, k, got[t * top_k + k], ref[t * top_k + k]);
                rc = 1;
            }
        }
    }
    /* Error paths: top_k > n_comp, null tensor. */
    if (ds4_gpu_indexer_topk_tensor(ot, st, n_comp, n_tokens, n_comp + 1) != 0)
        rc = 1;
    if (ds4_gpu_indexer_topk_tensor(ot, nullptr, n_comp, n_tokens, top_k) != 0)
        rc = 1;
done:
    if (ot) ds4_gpu_tensor_free(ot);
    if (st) ds4_gpu_tensor_free(st);
    return rc;
}
REGISTER_TEST(indexer_topk, test_indexer_topk);

/* ------------------------------------------------------------------ */
/* end-to-end decode path: score_one -> topk                          */
/* ------------------------------------------------------------------ */
static int test_indexer_end_to_end(void) {
    const uint32_t n_head = 2, head_dim = 4, n_comp = 10, top_k = 3;
    const float scale = 0.25f;

    std::vector<float> q(n_head * head_dim);
    std::vector<float> w(n_head);
    std::vector<float> k(n_comp * head_dim);
    for (uint32_t h = 0; h < n_head; h++)
        for (uint32_t d = 0; d < head_dim; d++)
            q[h * head_dim + d] = 0.5f + 0.1f * (float)(h * head_dim + d);
    for (uint32_t h = 0; h < n_head; h++) w[h] = 1.0f / (float)(h + 2);
    for (uint32_t c = 0; c < n_comp; c++)
        for (uint32_t d = 0; d < head_dim; d++)
            k[c * head_dim + d] = 0.25f * (float)(c + 1) * (float)(d + 1) - 0.8f;

    std::vector<float> ref_score(n_comp);
    for (uint32_t c = 0; c < n_comp; c++)
        ref_score[c] = ref_row_score(q.data(), w.data(), &k[c * head_dim],
                                     n_head, head_dim, scale);
    uint32_t ref_sel[top_k];
    ref_topk_row(ref_score.data(), n_comp, top_k, ref_sel);

    ds4_gpu_tensor *st = ds4_gpu_tensor_alloc((uint64_t)n_comp * sizeof(float));
    ds4_gpu_tensor *qt = ds4_gpu_tensor_alloc(q.size() * sizeof(float));
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc(w.size() * sizeof(float));
    ds4_gpu_tensor *kt = ds4_gpu_tensor_alloc(k.size() * sizeof(float));
    ds4_gpu_tensor *ot = ds4_gpu_tensor_alloc((uint64_t)top_k * sizeof(uint32_t));
    std::vector<uint32_t> got(top_k);
    int rc = 1;
    if (!st || !qt || !wt || !kt || !ot) goto done;
    if (ds4_gpu_tensor_write(qt, 0, q.data(), q.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(wt, 0, w.data(), w.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(kt, 0, k.data(), k.size() * sizeof(float)) == 0) {
        fprintf(stderr, "indexer_end_to_end: input write failed\n");
        goto done;
    }
    if (ds4_gpu_indexer_score_one_tensor(st, qt, wt, kt, n_comp, n_head,
                                         head_dim, scale) == 0 ||
        ds4_gpu_indexer_topk_tensor(ot, st, n_comp, 1, top_k) == 0) {
        fprintf(stderr, "indexer_end_to_end: kernel returned error\n");
        goto done;
    }
    if (ds4_gpu_tensor_read(ot, 0, got.data(), top_k * sizeof(uint32_t)) == 0) {
        fprintf(stderr, "indexer_end_to_end: read failed\n");
        goto done;
    }
    rc = 0;
    for (uint32_t k = 0; k < top_k; k++) {
        if (got[k] != ref_sel[k]) {
            fprintf(stderr, "indexer_end_to_end: k=%u got %u want %u\n",
                    k, got[k], ref_sel[k]);
            rc = 1;
        }
    }
done:
    if (ot) ds4_gpu_tensor_free(ot);
    if (kt) ds4_gpu_tensor_free(kt);
    if (wt) ds4_gpu_tensor_free(wt);
    if (qt) ds4_gpu_tensor_free(qt);
    if (st) ds4_gpu_tensor_free(st);
    return rc;
}
REGISTER_TEST(indexer_end_to_end, test_indexer_end_to_end);
