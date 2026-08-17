/* Kernel tests: DS4 compressed-attention indexer (real Vulkan dispatches).
 *
 * The indexer score, top-k, and mask paths are GPU dispatches. Tensor reads
 * below synchronize the recorded work before checking the results.
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
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>
#include <chrono>

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
    const float scale = 1.0f / std::sqrt((float)(n_head * head_dim));

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
        if (!(std::fabs(got[c] - ref[c]) <= 1e-3f)) {
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
                ok = std::fabs(g - r) <= 1e-3f;
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

static void set_indexer_select_wave64_env(const char *value) {
#ifdef _WIN32
    _putenv_s("DS4_VULKAN_INDEXER_SELECT_WAVE64", value ? value : "");
#else
    if (value) setenv("DS4_VULKAN_INDEXER_SELECT_WAVE64", value, 1);
    else unsetenv("DS4_VULKAN_INDEXER_SELECT_WAVE64");
#endif
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

/* The canonical indexer keeps up to 512 compressed rows.  This exercises the
 * full output width without making the regular small ordering test expensive. */
static int test_indexer_topk_512_masked(void) {
    const uint32_t n_comp = 4096, n_tokens = 1, top_k = 512;
    std::vector<float> scores(n_comp);
    const uint32_t finite_rows = 37;
    for (uint32_t c = 0; c < finite_rows; c++) scores[c] = (float)(finite_rows - c);
    for (uint32_t c = finite_rows; c < 3500; c++) scores[c] = -INFINITY;
    for (uint32_t c = 3500; c < n_comp; c++) scores[c] = NAN;

    std::vector<uint32_t> ref(top_k);
    ref_topk_row(scores.data(), n_comp, top_k, ref.data());
    ds4_gpu_tensor *st = ds4_gpu_tensor_alloc(scores.size() * sizeof(float));
    ds4_gpu_tensor *ot = ds4_gpu_tensor_alloc((uint64_t)top_k * sizeof(uint32_t));
    ds4_gpu_tensor *mt = ds4_gpu_tensor_alloc((uint64_t)n_comp * sizeof(float));
    std::vector<uint32_t> got(top_k);
    std::vector<float> mask(n_comp);
    int rc = 1;
    if (!st || !ot || !mt) goto done;
    if (ds4_gpu_tensor_write(st, 0, scores.data(), scores.size() * sizeof(float)) == 0 ||
        ds4_gpu_indexer_topk_tensor(ot, st, n_comp, n_tokens, top_k) == 0 ||
        ds4_gpu_dsv4_topk_mask_tensor(mt, ot, n_comp, n_tokens, top_k) == 0 ||
        ds4_gpu_tensor_read(ot, 0, got.data(), got.size() * sizeof(uint32_t)) == 0 ||
        ds4_gpu_tensor_read(mt, 0, mask.data(), mask.size() * sizeof(float)) == 0)
        goto done;

    rc = 0;
    for (uint32_t k = 0; k < top_k; k++) {
        if (got[k] != ref[k]) {
            fprintf(stderr, "indexer_topk_512: k=%u got %u want %u\n",
                    k, got[k], ref[k]);
            rc = 1;
        }
    }
    for (uint32_t c = 0; c < n_comp; c++) {
        const bool selected_row = std::find(ref.begin(), ref.end(), c) != ref.end();
        const float want = selected_row ? 0.0f : -INFINITY;
        if ((std::isinf(want) && !(std::isinf(mask[c]) && mask[c] < 0.0f)) ||
            (!std::isinf(want) && mask[c] != want)) {
            fprintf(stderr, "indexer_topk_512: mask c=%u got %.6f\n", c, mask[c]);
            rc = 1;
        }
    }
done:
    if (mt) ds4_gpu_tensor_free(mt);
    if (ot) ds4_gpu_tensor_free(ot);
    if (st) ds4_gpu_tensor_free(st);
    return rc;
}
REGISTER_TEST(indexer_topk_512_masked, test_indexer_topk_512_masked);

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

/* Production-only fused selector gate.  The inputs make the score of row c
 * an exactly representable positive integer, while the odd permutation keeps
 * every visible row unique.  This isolates selector ordering from floating
 * tolerance and also lets the 32768-row gate avoid a 268M-op CPU reference. */
static int run_indexer_select_wave64_case(uint32_t n_comp, uint32_t pos0,
                                          bool compare_legacy,
                                          bool benchmark,
                                          bool disabled_fallback = false) {
    const uint32_t n_head = 64u, head_dim = 128u, ratio = 4u, top_k = 512u;
    const uint32_t visible = std::min(n_comp, (pos0 + 1u) / ratio);
    std::vector<float> q((size_t)n_head * head_dim, 0.0f);
    std::vector<float> weights(n_head, 0.0f);
    std::vector<float> index_comp((size_t)n_comp * head_dim, 0.0f);
    std::vector<float> reference_scores(n_comp, -INFINITY);
    std::vector<uint32_t> reference(top_k), candidate(top_k), legacy(top_k);
    q[0] = 1.0f;
    weights[0] = 1.0f;
    for (uint32_t c = 0; c < n_comp; c++) {
        const float score = (float)(((uint64_t)c * 4051u) % n_comp + 1u);
        index_comp[(size_t)c * head_dim] = score;
        if (c < visible) reference_scores[c] = score;
    }
    ref_topk_row(reference_scores.data(), n_comp, top_k, reference.data());

    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc((uint64_t)top_k * sizeof(uint32_t));
    ds4_gpu_tensor *scores = ds4_gpu_tensor_alloc((uint64_t)n_comp * sizeof(float));
    ds4_gpu_tensor *scratch = ds4_gpu_tensor_alloc((uint64_t)n_comp * sizeof(float));
    ds4_gpu_tensor *qt = ds4_gpu_tensor_alloc(q.size() * sizeof(float));
    ds4_gpu_tensor *wt = ds4_gpu_tensor_alloc(weights.size() * sizeof(float));
    ds4_gpu_tensor *kt = ds4_gpu_tensor_alloc(index_comp.size() * sizeof(float));
    ds4_gpu_tensor *legacy_selected = compare_legacy
        ? ds4_gpu_tensor_alloc((uint64_t)top_k * sizeof(uint32_t)) : nullptr;
    ds4_gpu_tensor *legacy_scores = compare_legacy
        ? ds4_gpu_tensor_alloc((uint64_t)n_comp * sizeof(float)) : nullptr;
    const char *saved = std::getenv("DS4_VULKAN_INDEXER_SELECT_WAVE64");
    const bool had_saved = saved != nullptr;
    const std::string saved_value = saved ? saved : "";
    int rc = 1;
    if (!selected || !scores || !scratch || !qt || !wt || !kt ||
        (compare_legacy && (!legacy_selected || !legacy_scores))) goto done;
    if (!ds4_gpu_tensor_write(qt, 0, q.data(), q.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(wt, 0, weights.data(), weights.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(kt, 0, index_comp.data(),
                              index_comp.size() * sizeof(float))) goto done;

    if (disabled_fallback) {
        set_indexer_select_wave64_env("0");
        if (ds4_gpu_indexer_select_decode_wave64_tensor(
                selected, scores, scratch, qt, wt, kt, n_comp, 1u, pos0,
                n_head, head_dim, ratio, top_k, 1.0f) != 0 ||
            ds4_gpu_indexer_select_decode_wave64_used() != 0 ||
            !ds4_gpu_indexer_scores_decode_batch_tensor(
                legacy_scores, qt, wt, kt, n_comp, 1u, pos0, n_head,
                head_dim, ratio, 1.0f) ||
            !ds4_gpu_indexer_topk_tensor(legacy_selected, legacy_scores,
                                         n_comp, 1u, top_k) ||
            !ds4_gpu_tensor_read(legacy_selected, 0, legacy.data(),
                                 legacy.size() * sizeof(uint32_t)) ||
            legacy != reference) {
            fprintf(stderr,
                    "indexer_select_wave64: forced-disable fallback failed\n");
            goto done;
        }
        rc = 0;
        goto done;
    }

    set_indexer_select_wave64_env("1");
    if (!ds4_gpu_indexer_select_decode_wave64_tensor(
            selected, scores, scratch, qt, wt, kt, n_comp, 1u, pos0,
            n_head, head_dim, ratio, top_k, 1.0f) ||
        ds4_gpu_indexer_select_decode_wave64_used() != 1) {
        rc = std::getenv("DS4_TEST_REQUIRE_INDEXER_SELECT_WAVE64") ? 1 : 0;
        goto done;
    }
    if (!ds4_gpu_tensor_read(selected, 0, candidate.data(),
                             candidate.size() * sizeof(uint32_t))) goto done;
    if (candidate != reference) {
        for (uint32_t k = 0; k < top_k; k++) {
            if (candidate[k] != reference[k]) {
                fprintf(stderr,
                        "indexer_select_wave64: n_comp=%u k=%u got=%u want=%u\n",
                        n_comp, k, candidate[k], reference[k]);
                break;
            }
        }
        goto done;
    }

    if (compare_legacy) {
        if (!ds4_gpu_indexer_scores_decode_batch_tensor(
                legacy_scores, qt, wt, kt, n_comp, 1u, pos0, n_head,
                head_dim, ratio, 1.0f) ||
            !ds4_gpu_indexer_topk_tensor(legacy_selected, legacy_scores,
                                         n_comp, 1u, top_k) ||
            !ds4_gpu_tensor_read(legacy_selected, 0, legacy.data(),
                                 legacy.size() * sizeof(uint32_t)) ||
            legacy != candidate) {
            fprintf(stderr, "indexer_select_wave64: legacy A/B mismatch n_comp=%u\n",
                    n_comp);
            goto done;
        }
    }

    if (benchmark) {
        const uint32_t warmups = 3u, rounds = 12u;
        for (uint32_t i = 0; i < warmups; i++) {
            if (compare_legacy &&
                (!ds4_gpu_indexer_scores_decode_batch_tensor(
                    legacy_scores, qt, wt, kt, n_comp, 1u, pos0, n_head,
                    head_dim, ratio, 1.0f) ||
                 !ds4_gpu_indexer_topk_tensor(legacy_selected, legacy_scores,
                                              n_comp, 1u, top_k))) goto done;
            if (!ds4_gpu_indexer_select_decode_wave64_tensor(
                    selected, scores, scratch, qt, wt, kt, n_comp, 1u, pos0,
                    n_head, head_dim, ratio, top_k, 1.0f)) goto done;
        }
        const auto old_begin = std::chrono::steady_clock::now();
        if (compare_legacy) {
            for (uint32_t i = 0; i < rounds; i++) {
                if (!ds4_gpu_indexer_scores_decode_batch_tensor(
                        legacy_scores, qt, wt, kt, n_comp, 1u, pos0, n_head,
                        head_dim, ratio, 1.0f) ||
                    !ds4_gpu_indexer_topk_tensor(legacy_selected, legacy_scores,
                                                 n_comp, 1u, top_k)) goto done;
            }
        }
        const auto old_end = std::chrono::steady_clock::now();
        const auto new_begin = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < rounds; i++) {
            if (!ds4_gpu_indexer_select_decode_wave64_tensor(
                    selected, scores, scratch, qt, wt, kt, n_comp, 1u, pos0,
                    n_head, head_dim, ratio, top_k, 1.0f)) goto done;
        }
        const auto new_end = std::chrono::steady_clock::now();
        const double old_ms = std::chrono::duration<double, std::milli>(
            old_end - old_begin).count() / rounds;
        const double new_ms = std::chrono::duration<double, std::milli>(
            new_end - new_begin).count() / rounds;
        fprintf(stderr,
                "indexer_select_wave64_bench: n_comp=%u visible=%u "
                "legacy_ms=%.3f candidate_ms=%.3f speedup=%.3fx\n",
                n_comp, visible, old_ms, new_ms,
                new_ms != 0.0 ? old_ms / new_ms : 0.0);
    }
    rc = 0;
done:
    if (had_saved) set_indexer_select_wave64_env(saved_value.c_str());
    else set_indexer_select_wave64_env(nullptr);
    if (legacy_scores) ds4_gpu_tensor_free(legacy_scores);
    if (legacy_selected) ds4_gpu_tensor_free(legacy_selected);
    if (kt) ds4_gpu_tensor_free(kt);
    if (wt) ds4_gpu_tensor_free(wt);
    if (qt) ds4_gpu_tensor_free(qt);
    if (scratch) ds4_gpu_tensor_free(scratch);
    if (scores) ds4_gpu_tensor_free(scores);
    if (selected) ds4_gpu_tensor_free(selected);
    return rc;
}

static int test_indexer_select_wave64_exact_ab(void) {
    return run_indexer_select_wave64_case(1024u, 4095u, true, false);
}
REGISTER_TEST(indexer_select_wave64_exact_ab,
              test_indexer_select_wave64_exact_ab);

static int test_indexer_select_wave64_disabled_fallback(void) {
    return run_indexer_select_wave64_case(1024u, 4095u, true, false, true);
}
REGISTER_TEST(indexer_select_wave64_disabled_fallback,
              test_indexer_select_wave64_disabled_fallback);

static int test_indexer_select_wave64_causal(void) {
    return run_indexer_select_wave64_case(1024u, 1599u, true, false);
}
REGISTER_TEST(indexer_select_wave64_causal,
              test_indexer_select_wave64_causal);

static int test_indexer_select_wave64_32768(void) {
    if (!std::getenv("DS4_TEST_INDEXER_SELECT_32768")) return 0;
    return run_indexer_select_wave64_case(32768u, 131071u, false, false);
}
REGISTER_TEST(indexer_select_wave64_32768,
              test_indexer_select_wave64_32768);

static int test_indexer_select_wave64_bench(void) {
    if (!std::getenv("DS4_TEST_BENCH_INDEXER_SELECT")) return 0;
    return run_indexer_select_wave64_case(1088u, 4351u, true, true);
}
REGISTER_TEST(indexer_select_wave64_bench,
              test_indexer_select_wave64_bench);

static int test_indexed_attention_causal_filler(void) {
    const uint32_t tokens = 2, heads = 1, dim = 4, raw_cap = 4, n_raw = 2;
    const uint32_t n_comp = 3, top_k = 3, pos0 = 1, ratio = 2;
    const float sinks[1] = {0.0f};
    const float q[] = {1.0f, 0.0f, 0.0f, 0.0f,
                       0.5f, 0.5f, 0.0f, 0.0f};
    const float raw[] = {0.0f, 0.0f, 0.0f, 0.0f,
                         0.0f, 0.0f, 0.0f, 0.0f,
                         2.0f, 0.0f, 0.0f, 0.0f,
                         0.0f, 2.0f, 0.0f, 0.0f};
    const float comp[] = {3.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 3.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 3.0f, 0.0f};
    const uint32_t topk[] = {0u, 99u, 1u, 0u, 99u, 1u};
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(float) * tokens * heads * dim);
    ds4_gpu_tensor *qt = ds4_gpu_tensor_alloc(sizeof(q));
    ds4_gpu_tensor *rt = ds4_gpu_tensor_alloc(sizeof(raw));
    ds4_gpu_tensor *ct = ds4_gpu_tensor_alloc(sizeof(comp));
    ds4_gpu_tensor *tt = ds4_gpu_tensor_alloc(sizeof(topk));
    unsigned char model[sizeof(sinks) + 16] = {};
    std::memcpy(model + 16, sinks, sizeof(sinks));
    int rc = 1;
    if (!out || !qt || !rt || !ct || !tt ||
        !ds4_gpu_set_model_map(model, sizeof(model)) ||
        !ds4_gpu_tensor_write(qt, 0, q, sizeof(q)) ||
        !ds4_gpu_tensor_write(rt, 0, raw, sizeof(raw)) ||
        !ds4_gpu_tensor_write(ct, 0, comp, sizeof(comp)) ||
        !ds4_gpu_tensor_write(tt, 0, topk, sizeof(topk)) ||
        !ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
            out, model, sizeof(model), 16, qt, rt, ct, 0, tt, tokens, pos0,
            n_raw, raw_cap, 2, n_comp, top_k, 2, ratio, heads, dim))
        goto done;
    {
        std::vector<float> got(tokens * heads * dim);
        if (!ds4_gpu_tensor_read(out, 0, got.data(), sizeof(float) * got.size())) goto done;
        const float scale = 0.5f;
        for (uint32_t t = 0; t < tokens; t++) {
            const uint32_t qpos = pos0 + t;
            const uint32_t visible = (qpos + 1u) / ratio;
            std::vector<const float *> rows;
            const uint32_t first_raw_pos = pos0 + tokens - n_raw;
            uint32_t raw_first = first_raw_pos;
            if (qpos + 1u > 2u) raw_first = std::max(raw_first, qpos + 1u - 2u);
            for (uint32_t raw_pos = raw_first; raw_pos <= qpos; raw_pos++)
                rows.push_back(raw + (2u + raw_pos - first_raw_pos) * dim);
            for (uint32_t k = 0; k < top_k; k++)
                if (topk[t * top_k + k] < visible)
                    rows.push_back(comp + topk[t * top_k + k] * dim);
            std::vector<float> score(rows.size());
            float max_score = sinks[0];
            for (size_t i = 0; i < rows.size(); i++) {
                for (uint32_t d = 0; d < dim; d++) score[i] += q[t * dim + d] * rows[i][d];
                score[i] *= scale;
                max_score = std::max(max_score, score[i]);
            }
            float denom = std::exp(sinks[0] - max_score);
            for (float &s : score) { s = std::exp(s - max_score); denom += s; }
            for (uint32_t d = 0; d < dim; d++) {
                float want = 0.0f;
                for (size_t i = 0; i < rows.size(); i++) want += score[i] * rows[i][d];
                want /= denom;
                if (std::fabs(got[t * dim + d] - want) > 1e-3f) goto done;
            }
        }
        rc = 0;
    }
done:
    if (tt) ds4_gpu_tensor_free(tt);
    if (ct) ds4_gpu_tensor_free(ct);
    if (rt) ds4_gpu_tensor_free(rt);
    if (qt) ds4_gpu_tensor_free(qt);
    if (out) ds4_gpu_tensor_free(out);
    return rc;
}
REGISTER_TEST(indexed_attention_causal_filler, test_indexed_attention_causal_filler);

/* Same-binary gate for the opt-in BC-250 indexed wave64 accumulator.  The
 * shape mirrors the long-context production path: 128-wide heads, all 512
 * selected compressed rows visible, and a non-empty raw ring suffix. */
extern "C" int ds4_gpu_attention_indexed_wave64_used(void);
extern "C" int ds4_gpu_attention_indexed_wave64_inv_rope_used(void);
extern "C" int ds4_gpu_attention_indexed_wave64_inv_rope_available(void);
extern "C" void ds4_gpu_set_decode_attn_rope_fuse(
    uint32_t head_dim, uint32_t n_rot, uint32_t pos0, uint32_t n_ctx_orig,
    bool inverse, float freq_base, float freq_scale, float ext_factor,
    float attn_factor, float beta_fast, float beta_slow);

static void set_indexed_wave64_env(const char *value) {
#if defined(_WIN32)
    _putenv_s("DS4_VULKAN_ATTN_INDEXED_WAVE64", value ? value : "");
#else
    if (value) setenv("DS4_VULKAN_ATTN_INDEXED_WAVE64", value, 1);
    else unsetenv("DS4_VULKAN_ATTN_INDEXED_WAVE64");
#endif
}

static void set_indexed_wave64_inv_rope_env(const char *value) {
#if defined(_WIN32)
    _putenv_s("DS4_VULKAN_ATTN_INDEXED_WAVE64_INV_ROPE", value ? value : "");
#else
    if (value) setenv("DS4_VULKAN_ATTN_INDEXED_WAVE64_INV_ROPE", value, 1);
    else unsetenv("DS4_VULKAN_ATTN_INDEXED_WAVE64_INV_ROPE");
#endif
}

static int test_indexed_attention_wave64_exact_ab(void) {
    const uint32_t n_tokens = 1, n_head = 2, head_dim = 128;
    const uint32_t n_comp = 512, top_k = 512, raw_cap = 8, n_raw = 4;
    const uint32_t pos0 = 4096, raw_start = 3, ratio = 4;
    const uint32_t q_bytes = n_tokens * n_head * head_dim * sizeof(float);
    const uint32_t raw_bytes = raw_cap * head_dim * sizeof(float);
    const uint32_t comp_bytes = n_comp * head_dim * sizeof(float);
    const uint32_t out_bytes = n_tokens * n_head * head_dim * sizeof(float);

    std::vector<float> q(n_tokens * n_head * head_dim);
    std::vector<float> raw(raw_cap * head_dim);
    std::vector<float> comp(n_comp * head_dim);
    std::vector<uint32_t> topk(top_k);
    std::vector<float> plain(n_tokens * n_head * head_dim);
    std::vector<float> base(n_tokens * n_head * head_dim);
    std::vector<float> opt(n_tokens * n_head * head_dim);
    std::vector<float> ordinary(n_tokens * n_head * head_dim);
    std::vector<float> ref(n_tokens * n_head * head_dim);
    for (uint32_t i = 0; i < q.size(); i++)
        q[i] = 0.03125f * (float)((i * 17u) % 29u) - 0.4f;
    for (uint32_t i = 0; i < raw.size(); i++)
        raw[i] = 0.015625f * (float)((i * 11u) % 23u) - 0.2f;
    for (uint32_t i = 0; i < comp.size(); i++)
        comp[i] = 0.0078125f * (float)((i * 7u) % 41u) - 0.15f;
    for (uint32_t c = 0; c < top_k; c++) topk[c] = c;

    const float sinks[n_head] = {0.125f, -0.25f};
    unsigned char model[16 + sizeof(sinks)] = {};
    std::memcpy(model + 16, sinks, sizeof(sinks));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *qt = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *rt = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *ct = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *tt = ds4_gpu_tensor_alloc(topk.size() * sizeof(uint32_t));
    ds4_gpu_tensor *ordinary_out = ds4_gpu_tensor_alloc(out_bytes);
    const char *saved = std::getenv("DS4_VULKAN_ATTN_INDEXED_WAVE64");
    const char *saved_rope = std::getenv("DS4_VULKAN_ATTN_INDEXED_WAVE64_INV_ROPE");
    const bool had_saved = saved != nullptr;
    const bool had_saved_rope = saved_rope != nullptr;
    const std::string saved_value = saved ? saved : "";
    const std::string saved_rope_value = saved_rope ? saved_rope : "";
    const float rope_base = 10000.0f;
    bool batch_active = false;
    bool fused_used = false;
    int rc = 1;
    if (!out || !qt || !rt || !ct || !tt || !ordinary_out ||
        !ds4_gpu_set_model_map(model, sizeof(model)) ||
        !ds4_gpu_tensor_write(qt, 0, q.data(), q_bytes) ||
        !ds4_gpu_tensor_write(rt, 0, raw.data(), raw_bytes) ||
        !ds4_gpu_tensor_write(ct, 0, comp.data(), comp_bytes) ||
        !ds4_gpu_tensor_write(tt, 0, topk.data(), topk.size() * sizeof(uint32_t)))
        goto done;

    /* Explicit kill switches establish the canonical fallback regardless of
     * whether this test is running on the BC-250 default device. */
    set_indexed_wave64_env("0");
    set_indexed_wave64_inv_rope_env("0");
    if (!ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
            out, model, sizeof(model), 16, qt, rt, ct, 0, tt, n_tokens, pos0,
            n_raw, raw_cap, raw_start, n_comp, top_k, 0, ratio, n_head, head_dim) ||
        ds4_gpu_attention_indexed_wave64_used() != 0 ||
        !ds4_gpu_tensor_read(out, 0, plain.data(), out_bytes) ||
        !ds4_gpu_rope_tail_tensor(out, n_tokens, n_head, head_dim, 64u, pos0,
                                  0u, true, rope_base, 1.0f, 0.0f, 1.0f,
                                  0.0f, 0.0f) ||
        !ds4_gpu_tensor_read(out, 0, base.data(), out_bytes))
        goto done;

    /* Empty overrides exercise the promoted BC-250 default. */
    set_indexed_wave64_env(nullptr);
    set_indexed_wave64_inv_rope_env(nullptr);
    if (ds4_gpu_attention_indexed_wave64_inv_rope_available() == 0) {
        fprintf(stderr, "indexed_attention_wave64_exact_ab: skipped (BC-250 Wave64 unavailable)\n");
        rc = (std::getenv("DS4_TEST_REQUIRE_INDEXED_WAVE64") != nullptr) ? 1 : 0;
        goto done;
    }
    ds4_gpu_set_decode_attn_rope_fuse(head_dim, 64u, pos0, 0u, true,
                                      rope_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    /* Keep fused and ordinary indexed dispatches in one command-buffer
     * lifetime. The second dispatch deliberately disables fusion; it must
     * not inherit the fused RoPE fields from the first dispatch. */
    if (!ds4_gpu_batch_layer_begin(0))
        goto done;
    batch_active = true;
    if (!ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
            out, model, sizeof(model), 16, qt, rt, ct, 0, tt, n_tokens, pos0,
            n_raw, raw_cap, raw_start, n_comp, top_k, 0, ratio, n_head, head_dim) ||
        ds4_gpu_attention_indexed_wave64_used() != 1)
        goto done;
    fused_used = ds4_gpu_attention_indexed_wave64_inv_rope_used() == 1;
    set_indexed_wave64_inv_rope_env("0");
    if (!ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
            ordinary_out, model, sizeof(model), 16, qt, rt, ct, 0, tt, n_tokens, pos0,
            n_raw, raw_cap, raw_start, n_comp, top_k, 0, ratio, n_head, head_dim) ||
        !ds4_gpu_batch_layer_end(0))
        goto done;
    batch_active = false;
    /* Command-ring layer retirement is intentionally nonblocking; tensor
     * reads expose mapped memory and do not fence the submitted work. */
    if (!ds4_gpu_synchronize())
        goto done;
    if (!ds4_gpu_tensor_read(out, 0, opt.data(), out_bytes) ||
        !ds4_gpu_tensor_read(ordinary_out, 0, ordinary.data(), out_bytes))
        goto done;
    /* Each mode-1 API call resets the per-call activation markers, so retain
     * the fused result before the deliberate ordinary follow-up dispatch. */
    if (ds4_gpu_attention_indexed_wave64_used() != 1 || !fused_used) {
        fprintf(stderr, "indexed_attention_wave64_exact_ab: skipped (wave64 unavailable)\n");
        rc = (std::getenv("DS4_TEST_REQUIRE_INDEXED_WAVE64") != nullptr) ? 1 : 0;
        goto done;
    }

    if (std::memcmp(plain.data(), ordinary.data(), out_bytes) != 0) {
        for (uint32_t i = 0; i < plain.size(); i++) {
            if (plain[i] != ordinary[i]) {
                fprintf(stderr, "indexed_attention_wave64_exact_ab: ordinary push mismatch i=%u "
                        "plain=%a ordinary=%a\n", i, plain[i], ordinary[i]);
                break;
            }
        }
        goto done;
    }

    if (std::memcmp(base.data(), opt.data(), out_bytes) != 0) {
        for (uint32_t i = 0; i < base.size(); i++) {
            if (base[i] != opt[i]) {
                fprintf(stderr, "indexed_attention_wave64_exact_ab: byte mismatch i=%u "
                        "base=%a opt=%a\n", i, base[i], opt[i]);
                break;
            }
        }
        goto done;
    }

    /* Reference the same online order, retaining the canonical raw-then-topk
     * sequence and float accumulator operations. */
    for (uint32_t h = 0; h < n_head; h++) {
        const float *qh = q.data() + h * head_dim;
        float *oh = ref.data() + h * head_dim;
        float max_score = sinks[h], denom = 1.0f;
        for (uint32_t r = 0; r < n_raw; r++) {
            const float *v = raw.data() + ((raw_start + r) % raw_cap) * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * v[d];
            const float score = dot * (1.0f / std::sqrt((float)head_dim));
            const float factor = score > max_score ? std::exp(max_score - score) : 1.0f;
            if (score > max_score) max_score = score;
            const float weight = std::exp(score - max_score);
            denom = denom * factor + weight;
            for (uint32_t d = 0; d < head_dim; d++) oh[d] = oh[d] * factor + weight * v[d];
        }
        for (uint32_t c = 0; c < top_k; c++) {
            const float *v = comp.data() + topk[c] * head_dim;
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * v[d];
            const float score = dot * (1.0f / std::sqrt((float)head_dim));
            const float factor = score > max_score ? std::exp(max_score - score) : 1.0f;
            if (score > max_score) max_score = score;
            const float weight = std::exp(score - max_score);
            denom = denom * factor + weight;
            for (uint32_t d = 0; d < head_dim; d++) oh[d] = oh[d] * factor + weight * v[d];
        }
        for (uint32_t d = 0; d < head_dim; d++) oh[d] /= denom;
        for (uint32_t pair = 0; pair < 32u; pair++) {
            const uint32_t i = pair * 2u;
            const float theta = float(pos0) *
                std::pow(rope_base, -float(i) / 64.0f);
            const float c = std::cos(theta);
            const float s = -std::sin(theta);
            const uint32_t tail = head_dim - 64u;
            const float v0 = oh[tail + i];
            const float v1 = oh[tail + i + 1u];
            oh[tail + i] = v0 * c - v1 * s;
            oh[tail + i + 1u] = v0 * s + v1 * c;
        }
    }
    for (uint32_t i = 0; i < ref.size(); i++) {
        if (std::fabs(base[i] - ref[i]) > 1e-3f ||
            std::fabs(opt[i] - ref[i]) > 1e-3f) {
            fprintf(stderr, "indexed_attention_wave64_exact_ab: reference mismatch i=%u "
                    "base=%a opt=%a ref=%a\n", i, base[i], opt[i], ref[i]);
            goto done;
        }
    }
    rc = 0;
done:
    if (batch_active) (void)ds4_gpu_batch_layer_end(0);
    if (had_saved) set_indexed_wave64_env(saved_value.c_str());
    else set_indexed_wave64_env(nullptr);
    if (had_saved_rope) set_indexed_wave64_inv_rope_env(saved_rope_value.c_str());
    else set_indexed_wave64_inv_rope_env(nullptr);
    if (tt) ds4_gpu_tensor_free(tt);
    if (ordinary_out) ds4_gpu_tensor_free(ordinary_out);
    if (ct) ds4_gpu_tensor_free(ct);
    if (rt) ds4_gpu_tensor_free(rt);
    if (qt) ds4_gpu_tensor_free(qt);
    if (out) ds4_gpu_tensor_free(out);
    return rc;
}
REGISTER_TEST(indexed_attention_wave64_exact_ab, test_indexed_attention_wave64_exact_ab);

/* Optional wall-time probe.  It is deliberately a separate test so normal
 * kernel-test runs do not pay for 64 synchronous dispatches.  Set
 * DS4_TEST_BENCH_INDEXED_WAVE64=1 on BC-250; set
 * DS4_TEST_REQUIRE_INDEXED_WAVE64=1 as well to turn an unavailable candidate
 * into a hard failure instead of a skip. */
static int test_indexed_attention_wave64_bench(void) {
    if (!std::getenv("DS4_TEST_BENCH_INDEXED_WAVE64")) return 0;
    const uint32_t n_tokens = 1, n_head = 2, head_dim = 128;
    const uint32_t n_comp = 512, top_k = 512, raw_cap = 128, n_raw = 128;
    const uint32_t pos0 = 4096, raw_start = 0, ratio = 4, rounds = 32;
    const uint32_t q_bytes = n_tokens * n_head * head_dim * sizeof(float);
    const uint32_t raw_bytes = raw_cap * head_dim * sizeof(float);
    const uint32_t comp_bytes = n_comp * head_dim * sizeof(float);
    const uint32_t out_bytes = n_tokens * n_head * head_dim * sizeof(float);
    std::vector<float> q(n_tokens * n_head * head_dim);
    std::vector<float> raw(raw_cap * head_dim);
    std::vector<float> comp(n_comp * head_dim);
    std::vector<uint32_t> topk(top_k);
    std::vector<float> sink_out(n_tokens * n_head * head_dim);
    for (uint32_t i = 0; i < q.size(); i++)
        q[i] = 0.03125f * (float)((i * 17u) % 29u) - 0.4f;
    for (uint32_t i = 0; i < raw.size(); i++)
        raw[i] = 0.015625f * (float)((i * 11u) % 23u) - 0.2f;
    for (uint32_t i = 0; i < comp.size(); i++)
        comp[i] = 0.0078125f * (float)((i * 7u) % 41u) - 0.15f;
    for (uint32_t c = 0; c < top_k; c++) topk[c] = c;

    const float sinks[n_head] = {0.125f, -0.25f};
    unsigned char model[16 + sizeof(sinks)] = {};
    std::memcpy(model + 16, sinks, sizeof(sinks));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *qt = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *rt = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *ct = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *tt = ds4_gpu_tensor_alloc(topk.size() * sizeof(uint32_t));
    const char *saved = std::getenv("DS4_VULKAN_ATTN_INDEXED_WAVE64");
    const bool had_saved = saved != nullptr;
    const std::string saved_value = saved ? saved : "";
    std::chrono::steady_clock::time_point fallback_begin, fallback_end;
    std::chrono::steady_clock::time_point candidate_begin, candidate_end;
    auto dispatch_and_read = [&]() -> bool {
        if (!ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                out, model, sizeof(model), 16, qt, rt, ct, 0, tt, n_tokens, pos0,
                n_raw, raw_cap, raw_start, n_comp, top_k, 0, ratio,
                n_head, head_dim)) return false;
        return ds4_gpu_tensor_read(out, 0, sink_out.data(), out_bytes) != 0;
    };
    int rc = 1;
    if (!out || !qt || !rt || !ct || !tt ||
        !ds4_gpu_set_model_map(model, sizeof(model)) ||
        !ds4_gpu_tensor_write(qt, 0, q.data(), q_bytes) ||
        !ds4_gpu_tensor_write(rt, 0, raw.data(), raw_bytes) ||
        !ds4_gpu_tensor_write(ct, 0, comp.data(), comp_bytes) ||
        !ds4_gpu_tensor_write(tt, 0, topk.data(), topk.size() * sizeof(uint32_t)))
        goto done;

    set_indexed_wave64_env("0");
    if (!dispatch_and_read()) goto done;
    fallback_begin = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < rounds; i++)
        if (!dispatch_and_read()) goto done;
    fallback_end = std::chrono::steady_clock::now();

    set_indexed_wave64_env("1");
    if (!dispatch_and_read()) goto done;
    if (ds4_gpu_attention_indexed_wave64_used() != 1) {
        fprintf(stderr, "indexed_attention_wave64_bench: skipped (wave64 unavailable)\n");
        rc = (std::getenv("DS4_TEST_REQUIRE_INDEXED_WAVE64") != nullptr) ? 1 : 0;
        goto done;
    }
    candidate_begin = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < rounds; i++)
        if (!dispatch_and_read()) goto done;
    candidate_end = std::chrono::steady_clock::now();

    {
        const double fallback_ms = std::chrono::duration<double, std::milli>(
            fallback_end - fallback_begin).count();
        const double candidate_ms = std::chrono::duration<double, std::milli>(
            candidate_end - candidate_begin).count();
        fprintf(stderr,
                "indexed_attention_wave64_bench: rounds=%u "
                "fallback_total_ms=%.3f fallback_per_dispatch_ms=%.3f "
                "candidate_total_ms=%.3f candidate_per_dispatch_ms=%.3f\n",
                rounds, fallback_ms, fallback_ms / rounds,
                candidate_ms, candidate_ms / rounds);
    }
    rc = 0;
done:
    if (had_saved) set_indexed_wave64_env(saved_value.c_str());
    else set_indexed_wave64_env(nullptr);
    if (tt) ds4_gpu_tensor_free(tt);
    if (ct) ds4_gpu_tensor_free(ct);
    if (rt) ds4_gpu_tensor_free(rt);
    if (qt) ds4_gpu_tensor_free(qt);
    if (out) ds4_gpu_tensor_free(out);
    return rc;
}
REGISTER_TEST(indexed_attention_wave64_bench, test_indexed_attention_wave64_bench);

/* Focused production-shaped comparison for the opt-in indexed inverse-RoPE
 * fusion.  The fallback performs indexed attention followed by standalone
 * rope_tail; the candidate performs the same indexed attention with the
 * inverse tail fused into its wave64 accumulator.  This is intentionally
 * separate from the older Wave64-only probe so its timing covers the exact
 * dispatch pair being considered for promotion.
 *
 * Set DS4_TEST_BENCH_INDEXED_WAVE64_INV_ROPE=1 on BC-250.  Set
 * DS4_TEST_REQUIRE_INDEXED_WAVE64=1 to turn an unavailable Wave64 path into a
 * hard failure. */
static int test_indexed_attention_wave64_inv_rope_bench(void) {
    if (!std::getenv("DS4_TEST_BENCH_INDEXED_WAVE64_INV_ROPE")) return 0;
    const uint32_t n_tokens = 1, n_head = 32, head_dim = 128;
    const uint32_t n_comp = 4096, top_k = 512, raw_cap = 1024, n_raw = 256;
    const uint32_t pos0 = 16384, raw_start = 0, ratio = 4;
    const uint32_t warmups = 2, rounds = 8;
    const uint32_t q_bytes = n_tokens * n_head * head_dim * sizeof(float);
    const uint32_t raw_bytes = raw_cap * head_dim * sizeof(float);
    const uint32_t comp_bytes = n_comp * head_dim * sizeof(float);
    const uint32_t out_bytes = n_tokens * n_head * head_dim * sizeof(float);
    std::vector<float> q(n_tokens * n_head * head_dim);
    std::vector<float> raw(raw_cap * head_dim);
    std::vector<float> comp(n_comp * head_dim);
    std::vector<uint32_t> topk(top_k);
    std::vector<float> reference(n_tokens * n_head * head_dim);
    std::vector<float> candidate(n_tokens * n_head * head_dim);
    std::vector<float> sink_out(n_tokens * n_head * head_dim);
    for (uint32_t i = 0; i < q.size(); i++)
        q[i] = 0.03125f * (float)((i * 17u) % 29u) - 0.4f;
    for (uint32_t i = 0; i < raw.size(); i++)
        raw[i] = 0.015625f * (float)((i * 11u) % 23u) - 0.2f;
    for (uint32_t i = 0; i < comp.size(); i++)
        comp[i] = 0.0078125f * (float)((i * 7u) % 41u) - 0.15f;
    for (uint32_t c = 0; c < top_k; c++) topk[c] = c;

    std::vector<float> sinks(n_head);
    for (uint32_t h = 0; h < n_head; h++)
        sinks[h] = 0.125f - 0.03125f * (float)(h % 11u);
    std::vector<unsigned char> model(16 + n_head * sizeof(float));
    std::memcpy(model.data() + 16, sinks.data(), n_head * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *qt = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *rt = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *ct = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *tt = ds4_gpu_tensor_alloc(topk.size() * sizeof(uint32_t));
    const char *saved_wave64 = std::getenv("DS4_VULKAN_ATTN_INDEXED_WAVE64");
    const char *saved_inv = std::getenv("DS4_VULKAN_ATTN_INDEXED_WAVE64_INV_ROPE");
    const bool had_saved_wave64 = saved_wave64 != nullptr;
    const bool had_saved_inv = saved_inv != nullptr;
    const std::string saved_wave64_value = saved_wave64 ? saved_wave64 : "";
    const std::string saved_inv_value = saved_inv ? saved_inv : "";
    const float rope_base = 10000.0f;
    int rc = 1;

    auto run_fallback = [&](std::vector<float> *capture) -> bool {
        set_indexed_wave64_inv_rope_env("0");
        if (!ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                out, model.data(), model.size(), 16, qt, rt, ct, 0, tt,
                n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
                0, ratio, n_head, head_dim) ||
            !ds4_gpu_rope_tail_tensor(out, n_tokens, n_head, head_dim, 64u,
                                      pos0, 0u, true, rope_base, 1.0f, 0.0f,
                                      1.0f, 0.0f, 0.0f) ||
            !ds4_gpu_tensor_read(out, 0, sink_out.data(), out_bytes))
            return false;
        if (capture) *capture = sink_out;
        return true;
    };
    auto run_candidate = [&](std::vector<float> *capture) -> bool {
        set_indexed_wave64_inv_rope_env("1");
        ds4_gpu_set_decode_attn_rope_fuse(
            head_dim, 64u, pos0, 0u, true, rope_base, 1.0f, 0.0f, 1.0f,
            0.0f, 0.0f);
        if (!ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                out, model.data(), model.size(), 16, qt, rt, ct, 0, tt,
                n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
                0, ratio, n_head, head_dim) ||
            ds4_gpu_attention_indexed_wave64_used() != 1 ||
            ds4_gpu_attention_indexed_wave64_inv_rope_used() != 1 ||
            !ds4_gpu_tensor_read(out, 0, sink_out.data(), out_bytes))
            return false;
        if (capture) *capture = sink_out;
        return true;
    };
    std::chrono::steady_clock::time_point fallback_begin, fallback_end;
    std::chrono::steady_clock::time_point candidate_begin, candidate_end;

    if (!out || !qt || !rt || !ct || !tt ||
        !ds4_gpu_set_model_map(model.data(), model.size()) ||
        !ds4_gpu_tensor_write(qt, 0, q.data(), q_bytes) ||
        !ds4_gpu_tensor_write(rt, 0, raw.data(), raw_bytes) ||
        !ds4_gpu_tensor_write(ct, 0, comp.data(), comp_bytes) ||
        !ds4_gpu_tensor_write(tt, 0, topk.data(), topk.size() * sizeof(uint32_t)))
        goto done;

    set_indexed_wave64_env("1");
    for (uint32_t i = 0; i < warmups; i++) {
        if (!run_fallback(nullptr) || !run_candidate(nullptr)) goto done;
    }
    if (!run_fallback(&reference)) goto done;
    if (ds4_gpu_attention_indexed_wave64_used() != 1) goto done;
    fallback_begin = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < rounds; i++)
        if (!run_fallback(nullptr)) goto done;
    fallback_end = std::chrono::steady_clock::now();

    if (!run_candidate(&candidate)) goto done;
    candidate_begin = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < rounds; i++)
        if (!run_candidate(nullptr)) goto done;
    candidate_end = std::chrono::steady_clock::now();

    if (std::memcmp(reference.data(), candidate.data(), out_bytes) != 0) {
        for (uint32_t i = 0; i < reference.size(); i++) {
            if (reference[i] != candidate[i]) {
                fprintf(stderr, "indexed_attention_wave64_inv_rope_bench: byte mismatch "
                        "i=%u reference=%a candidate=%a\n",
                        i, reference[i], candidate[i]);
                break;
            }
        }
        goto done;
    }
    {
        const double fallback_ms = std::chrono::duration<double, std::milli>(
            fallback_end - fallback_begin).count();
        const double candidate_ms = std::chrono::duration<double, std::milli>(
            candidate_end - candidate_begin).count();
        fprintf(stderr,
                "indexed_attention_wave64_inv_rope_bench: heads=%u raw=%u "
                "comp=%u topk=%u warmups=%u rounds=%u "
                "fallback_total_ms=%.3f fallback_per_iter_ms=%.3f "
                "candidate_total_ms=%.3f candidate_per_iter_ms=%.3f\n",
                n_head, n_raw, n_comp, top_k, warmups, rounds,
                fallback_ms, fallback_ms / rounds,
                candidate_ms, candidate_ms / rounds);
    }
    rc = 0;
done:
    if (had_saved_wave64) set_indexed_wave64_env(saved_wave64_value.c_str());
    else set_indexed_wave64_env(nullptr);
    if (had_saved_inv) set_indexed_wave64_inv_rope_env(saved_inv_value.c_str());
    else set_indexed_wave64_inv_rope_env(nullptr);
    if (tt) ds4_gpu_tensor_free(tt);
    if (ct) ds4_gpu_tensor_free(ct);
    if (rt) ds4_gpu_tensor_free(rt);
    if (qt) ds4_gpu_tensor_free(qt);
    if (out) ds4_gpu_tensor_free(out);
    return rc;
}
REGISTER_TEST(indexed_attention_wave64_inv_rope_bench,
              test_indexed_attention_wave64_inv_rope_bench);

/* Dedicated production-shape gate for the 512-wide indexed candidate.  The
 * legacy 128-wide gates above remain unchanged. */
static int test_indexed_attention_wave64_512_impl(uint32_t n_comp,
                                                  bool benchmark) {
    if (benchmark && !std::getenv("DS4_TEST_BENCH_INDEXED_WAVE64_512")) return 0;
    if (!benchmark && n_comp == 32768u &&
        !std::getenv("DS4_TEST_INDEXED_WAVE64_32768")) return 0;
    const uint32_t n_tokens = 1, n_head = 64, head_dim = 512;
    const uint32_t top_k = 512, raw_cap = 128, n_raw = 128;
    const uint32_t pos0 = n_comp == 32768u ? 131072u : 4096u;
    const uint32_t raw_start = 3, window = 128, ratio = 4;
    const size_t q_values = (size_t)n_tokens * n_head * head_dim;
    const size_t raw_values = (size_t)raw_cap * head_dim;
    const size_t comp_values = (size_t)n_comp * head_dim;
    const size_t out_values = q_values;
    std::vector<float> q(q_values), raw(raw_values), comp(comp_values, 0.0f);
    std::vector<uint32_t> topk(top_k);
    std::vector<float> fallback(out_values), candidate(out_values);
    for (size_t i = 0; i < q.size(); i++)
        q[i] = 0.03125f * (float)((i * 17u) % 29u) - 0.4f;
    for (size_t i = 0; i < raw.size(); i++)
        raw[i] = 0.015625f * (float)((i * 11u) % 23u) - 0.2f;
    for (uint32_t c = 0; c < top_k; c++) {
        const uint32_t selected_row = n_comp == 32768u
            ? n_comp - top_k + c : c;
        topk[c] = selected_row;
        for (uint32_t d = 0; d < head_dim; d++)
            comp[(size_t)selected_row * head_dim + d] =
                0.0078125f * (float)(((selected_row + d) * 7u) % 41u) - 0.15f;
    }
    float sinks[n_head] = {};
    for (uint32_t h = 0; h < n_head; h++) sinks[h] = 0.125f - 0.003f * h;
    std::vector<unsigned char> model(16u + sizeof(sinks), 0u);
    std::memcpy(model.data() + 16u, sinks, sizeof(sinks));
    const uint64_t q_bytes = q.size() * sizeof(float);
    const uint64_t raw_bytes = raw.size() * sizeof(float);
    const uint64_t comp_bytes = comp.size() * sizeof(float);
    const uint64_t out_bytes = out_values * sizeof(float);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    ds4_gpu_tensor *qt = ds4_gpu_tensor_alloc(q_bytes);
    ds4_gpu_tensor *rt = ds4_gpu_tensor_alloc(raw_bytes);
    ds4_gpu_tensor *ct = ds4_gpu_tensor_alloc(comp_bytes);
    ds4_gpu_tensor *tt = ds4_gpu_tensor_alloc(topk.size() * sizeof(uint32_t));
    const char *saved = std::getenv("DS4_VULKAN_ATTN_INDEXED_WAVE64");
    const char *saved_inv = std::getenv("DS4_VULKAN_ATTN_INDEXED_WAVE64_INV_ROPE");
    const bool had_saved = saved != nullptr, had_saved_inv = saved_inv != nullptr;
    const std::string saved_value = saved ? saved : "";
    const std::string saved_inv_value = saved_inv ? saved_inv : "";
    int rc = 1;
    do {
        if (!out || !qt || !rt || !ct || !tt ||
            !ds4_gpu_set_model_map(model.data(), model.size()) ||
            !ds4_gpu_tensor_write(qt, 0, q.data(), q_bytes) ||
            !ds4_gpu_tensor_write(rt, 0, raw.data(), raw_bytes) ||
            !ds4_gpu_tensor_write(ct, 0, comp.data(), comp_bytes) ||
            !ds4_gpu_tensor_write(tt, 0, topk.data(), topk.size() * sizeof(uint32_t)))
            break;
        set_indexed_wave64_inv_rope_env("0");
        auto dispatch_and_read = [&](ds4_gpu_tensor *dst, std::vector<float> &data) -> bool {
            if (!ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
                    dst, model.data(), model.size(), 16u, qt, rt, ct, 0u, tt,
                    n_tokens, pos0, n_raw, raw_cap, raw_start, n_comp, top_k,
                    window, ratio, n_head, head_dim)) return false;
            return ds4_gpu_tensor_read(dst, 0, data.data(), out_bytes) != 0;
        };
        set_indexed_wave64_env("0");
        if (!dispatch_and_read(out, fallback)) break;
        if (benchmark) {
            const uint32_t rounds = 8;
            const auto begin = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < rounds; i++)
                if (!dispatch_and_read(out, fallback)) break;
            const auto end = std::chrono::steady_clock::now();
            const double fallback_ms = std::chrono::duration<double, std::milli>(end - begin).count();
            set_indexed_wave64_env("1");
            if (!dispatch_and_read(out, candidate) ||
                ds4_gpu_attention_indexed_wave64_used() != 1) break;
            const auto candidate_begin = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < rounds; i++)
                if (!dispatch_and_read(out, candidate)) break;
            const auto candidate_end = std::chrono::steady_clock::now();
            const double candidate_ms = std::chrono::duration<double, std::milli>(candidate_end - candidate_begin).count();
            fprintf(stderr, "indexed_attention_wave64_512_bench: n_comp=%u rounds=%u fallback_per_dispatch_ms=%.3f candidate_per_dispatch_ms=%.3f\n",
                    n_comp, rounds, fallback_ms / rounds, candidate_ms / rounds);
        } else {
            set_indexed_wave64_env("1");
            if (!dispatch_and_read(out, candidate) ||
                ds4_gpu_attention_indexed_wave64_used() != 1) {
                rc = std::getenv("DS4_TEST_REQUIRE_INDEXED_WAVE64") ? 1 : 0;
                break;
            }
        }
        if (std::memcmp(fallback.data(), candidate.data(), out_bytes) != 0) break;
        /* The production decode also fuses inverse RoPE into this shader.
         * Compare it against the ordinary indexed path plus standalone
         * rope_tail while retaining the non-fused A/B above. */
        const float rope_base = 10000.0f;
        set_indexed_wave64_env("0");
        set_indexed_wave64_inv_rope_env("0");
        ds4_gpu_set_decode_attn_rope_fuse(head_dim, 64u, pos0, 0u, true,
                                          rope_base, 1.0f, 0.0f, 1.0f,
                                          0.0f, 0.0f);
        if (!dispatch_and_read(out, fallback) ||
            !ds4_gpu_rope_tail_tensor(out, n_tokens, n_head, head_dim, 64u,
                                      pos0, 0u, true, rope_base, 1.0f, 0.0f,
                                      1.0f, 0.0f, 0.0f) ||
            !ds4_gpu_tensor_read(out, 0, fallback.data(), out_bytes)) break;
        set_indexed_wave64_env("1");
        set_indexed_wave64_inv_rope_env("1");
        ds4_gpu_set_decode_attn_rope_fuse(head_dim, 64u, pos0, 0u, true,
                                          rope_base, 1.0f, 0.0f, 1.0f,
                                          0.0f, 0.0f);
        if (ds4_gpu_attention_indexed_wave64_inv_rope_available() == 0) {
            rc = std::getenv("DS4_TEST_REQUIRE_INDEXED_WAVE64") ? 1 : 0;
            break;
        }
        if (!dispatch_and_read(out, candidate) ||
            ds4_gpu_attention_indexed_wave64_used() != 1 ||
            ds4_gpu_attention_indexed_wave64_inv_rope_used() != 1 ||
            std::memcmp(fallback.data(), candidate.data(), out_bytes) != 0)
            break;
        rc = 0;
    } while (false);
    if (had_saved) set_indexed_wave64_env(saved_value.c_str());
    else set_indexed_wave64_env(nullptr);
    if (had_saved_inv) set_indexed_wave64_inv_rope_env(saved_inv_value.c_str());
    else set_indexed_wave64_inv_rope_env(nullptr);
    if (tt) ds4_gpu_tensor_free(tt);
    if (ct) ds4_gpu_tensor_free(ct);
    if (rt) ds4_gpu_tensor_free(rt);
    if (qt) ds4_gpu_tensor_free(qt);
    if (out) ds4_gpu_tensor_free(out);
    return rc;
}

static int test_indexed_attention_wave64_512_exact_ab(void) {
    return test_indexed_attention_wave64_512_impl(1024u, false);
}
REGISTER_TEST(indexed_attention_wave64_512_exact_ab,
              test_indexed_attention_wave64_512_exact_ab);

static int test_indexed_attention_wave64_512_bench(void) {
    return test_indexed_attention_wave64_512_impl(1024u, true);
}
REGISTER_TEST(indexed_attention_wave64_512_bench,
              test_indexed_attention_wave64_512_bench);

static int test_indexed_attention_wave64_512_32768_exact(void) {
    return test_indexed_attention_wave64_512_impl(32768u, false);
}
REGISTER_TEST(indexed_attention_wave64_512_32768_exact,
              test_indexed_attention_wave64_512_32768_exact);
