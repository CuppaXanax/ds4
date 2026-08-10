/* Kernel tests: ds4_gpu_embed_token_q8_0_tensor,
 * ds4_gpu_embed_tokens_q8_0_tensor, ds4_gpu_embed_token_quant_tensor,
 * ds4_gpu_embed_tokens_quant_tensor.
 *
 * The embedders are host-side: they dequantize the token embedding row(s)
 * of the synthetic model buffer and write f32 activations directly into
 * the output tensor (no command buffer, mirroring the *_hc_tensor
 * embedders).  Tests build a GGUF-style Q8_0 table (blocks of {scale f16,
 * 32 x int8}, 34 bytes/block) and an F16 table, register them with
 * ds4_gpu_set_model_map, and compare against the ds4.c reference math
 * (embed_token_q8_0 / embed_token_f16), including partial-block padding
 * (n_embd % 32 != 0) and out-of-range token clamping to row 0.
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>

/* f32 -> IEEE half (round toward zero; test values are normal-range). */
static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t e    = (x >> 23) & 0xffu;
    uint32_t m    = x & 0x7fffffu;
    if (e == 0xffu) return (uint16_t)(sign | 0x7c00u | (m ? 0x200u : 0u));
    if (e == 0u)    return (uint16_t)sign;              /* ±0 / flush subnormal */
    int32_t e16 = (int32_t)e - 127 + 15;
    if (e16 >= 31)  return (uint16_t)(sign | 0x7c00u);  /* ±inf */
    if (e16 <= 0) {                                     /* subnormal f16 */
        uint32_t shift = 126u - e;
        uint32_t m16 = (0x800000u | m) >> shift;
        return (uint16_t)(sign | m16);
    }
    return (uint16_t)(sign | ((uint32_t)e16 << 10) | (m >> 13));
}

/* IEEE half -> f32 (exact decode, same semantics as the backend). */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e    = (h >> 10) & 0x1fu;
    uint32_t m    = h & 0x3ffu;
    uint32_t bits;
    if (e == 0u) {
        if (m == 0u) bits = sign;                       /* ±0 */
        else {                                          /* subnormal */
            uint32_t mm = m, p = 0u;
            while (!(mm & 0x400u) && p < 10u) { mm <<= 1; p++; }
            bits = sign | ((113u - p) << 23) | ((mm & 0x3ffu) << 13);
        }
    } else if (e == 31u) {
        bits = sign | 0x7f800000u | (m << 13);          /* ±inf/nan */
    } else {
        bits = sign | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

/* Build a synthetic Q8_0 embedding table:
 *   row(token) = blocks x [scale f16 | 32 x int8]   (34 bytes/block)
 * Values are deterministic per token/block/element so the CPU reference and
 * the kernel see identical bytes. */
static unsigned char *make_q8_model(uint32_t n_vocab, uint32_t n_embd,
                                    uint64_t *out_model_size,
                                    uint64_t *out_weight_offset) {
    const uint64_t header = 16;
    const uint64_t blocks = ((uint64_t)n_embd + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t model_size = header + (uint64_t)n_vocab * row_bytes;
    unsigned char *model = (unsigned char *)malloc(model_size);
    if (!model) return NULL;
    std::memset(model, 0xAA, header);
    for (uint32_t tok = 0; tok < n_vocab; tok++) {
        uint8_t *row = model + header + (uint64_t)tok * row_bytes;
        for (uint64_t b = 0; b < blocks; b++) {
            const float scale = 0.5f + 0.25f * (float)(tok + 1) + 0.125f * (float)(b + 1);
            const uint16_t sb = f32_to_f16(scale);
            std::memcpy(row + b * 34u, &sb, 2);
            int8_t *qs = (int8_t *)(row + b * 34u + 2u);
            for (uint32_t i = 0; i < 32; i++) {
                int v = (int)((tok + 1) * (b + 1) * (i + 1)) % 96 - 48;  /* [-48,47] */
                qs[i] = (int8_t)v;
            }
        }
    }
    *out_model_size = model_size;
    *out_weight_offset = header;
    return model;
}

/* Build a synthetic F16 embedding table: row(token) = n_embd x IEEE half. */
static unsigned char *make_f16_model(uint32_t n_vocab, uint32_t n_embd,
                                     uint64_t *out_model_size,
                                     uint64_t *out_weight_offset) {
    const uint64_t header = 16;
    const uint64_t row_bytes = (uint64_t)n_embd * 2u;
    const uint64_t model_size = header + (uint64_t)n_vocab * row_bytes;
    unsigned char *model = (unsigned char *)malloc(model_size);
    if (!model) return NULL;
    std::memset(model, 0xAA, header);
    for (uint32_t tok = 0; tok < n_vocab; tok++) {
        uint16_t *row = (uint16_t *)(model + header + (uint64_t)tok * row_bytes);
        for (uint32_t i = 0; i < n_embd; i++) {
            float v = (float)((uint64_t)tok * n_embd + i + 1) * 0.125f;
            if (((tok + i) & 1u) != 0) v = -v;
            row[i] = f32_to_f16(v);
        }
    }
    *out_model_size = model_size;
    *out_weight_offset = header;
    return model;
}

/* CPU reference, exactly ds4.c embed_token_q8_0. */
static void ref_embed_q8(const uint8_t *row, uint32_t n_embd, float *out) {
    const uint64_t blocks = ((uint64_t)n_embd + 31u) / 32u;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t sb;
        std::memcpy(&sb, row + b * 34u, 2);
        const float scale = f16_to_f32(sb);
        const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
        const uint64_t i0 = b * 32u;
        const uint64_t bn = (uint64_t)n_embd - i0 < 32u ? (uint64_t)n_embd - i0 : 32u;
        for (uint64_t i = 0; i < bn; i++) out[i0 + i] = scale * (float)qs[i];
    }
}

/* CPU reference, exactly ds4.c embed_token_f16. */
static void ref_embed_f16(const uint16_t *row, uint32_t n_embd, float *out) {
    for (uint32_t i = 0; i < n_embd; i++) out[i] = f16_to_f32(row[i]);
}

static int check_f32(const char *what, const float *got, const float *want, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        if (!(std::fabsf(got[i] - want[i]) <= 1e-3f)) {
            fprintf(stderr, "--- %s mismatch[%u]: got %.6f want %.6f\n",
                    what, i, got[i], want[i]);
            return 1;
        }
    }
    return 0;
}

/* Run one single-token Q8_0 embed and compare to the reference. */
static int check_embed_token_q8_one(const unsigned char *model, uint64_t model_size,
                                    uint64_t weight_offset, uint32_t n_vocab,
                                    uint32_t n_embd, uint32_t token, const char *what) {
    const uint64_t row_bytes = ((uint64_t)n_embd + 31u) / 32u * 34u;
    const uint32_t id = token >= n_vocab ? 0 : token;
    float want[64];
    ref_embed_q8(model + weight_offset + (uint64_t)id * row_bytes, n_embd, want);

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(n_embd * sizeof(float));
    if (!out) return 1;
    int ok = 1;
    if (ds4_gpu_embed_token_q8_0_tensor(out, model, model_size, weight_offset,
                                        n_vocab, token, n_embd) != 0) {
        float got[64];
        if (ds4_gpu_tensor_read(out, 0, got, n_embd * sizeof(float)) != 0) {
            char label[96];
            std::snprintf(label, sizeof(label), "%s token=%u", what, token);
            ok = check_f32(label, got, want, n_embd);
        }
    }
    ds4_gpu_tensor_free(out);
    return ok;
}

static int test_embed_token_q8_0(void) {
    /* Aligned case: n_embd=64 -> 2 full 32-element blocks per row. */
    uint64_t ms = 0, wo = 0;
    unsigned char *model = make_q8_model(4, 64, &ms, &wo);
    if (!model) return 1;
    if (ds4_gpu_set_model_map(model, ms) == 0) { free(model); return 1; }
    const uint32_t aligned_tokens[] = {0, 2, 4, 100};  /* 4,100 out of range -> row 0 */
    for (uint32_t k = 0; k < 4; k++) {
        if (check_embed_token_q8_one(model, ms, wo, 4, 64, aligned_tokens[k],
                                     "embed_token_q8_0/64") != 0) {
            free(model);
            return 1;
        }
    }
    free(model);

    /* Padded case: n_embd=8 -> a single partial block (8 of 32 int8 used). */
    uint64_t ms2 = 0, wo2 = 0;
    unsigned char *model2 = make_q8_model(4, 8, &ms2, &wo2);
    if (!model2) return 1;
    if (ds4_gpu_set_model_map(model2, ms2) == 0) { free(model2); return 1; }
    const uint32_t padded_tokens[] = {1, 3, 9};        /* 9 out of range -> row 0 */
    for (uint32_t k = 0; k < 3; k++) {
        if (check_embed_token_q8_one(model2, ms2, wo2, 4, 8, padded_tokens[k],
                                     "embed_token_q8_0/8") != 0) {
            free(model2);
            return 1;
        }
    }
    free(model2);
    return 0;
}
REGISTER_TEST(embed_token_q8_0, test_embed_token_q8_0);

static int test_embed_tokens_q8_0(void) {
    int rc = 1;

    /* Aligned batch: n_tokens=4 over a 64-wide table, with negative and
     * out-of-range ids clamped to row 0. */
    uint64_t ms = 0, wo = 0;
    unsigned char *model = make_q8_model(4, 64, &ms, &wo);
    if (!model) return 1;
    if (ds4_gpu_set_model_map(model, ms) == 0) { free(model); return 1; }

    const int32_t toks[4] = {0, 2, -1, 7};
    ds4_gpu_tensor *tok_t = ds4_gpu_tensor_alloc(sizeof(toks));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(4 * 64 * sizeof(float));
    if (!tok_t || !out) {
        if (tok_t) ds4_gpu_tensor_free(tok_t);
        if (out) ds4_gpu_tensor_free(out);
        free(model);
        return 1;
    }
    if (ds4_gpu_tensor_write(tok_t, 0, toks, sizeof(toks)) == 0) {
        ds4_gpu_tensor_free(tok_t); ds4_gpu_tensor_free(out); free(model);
        return 1;
    }

    const uint64_t row_bytes = 2u * 34u;  /* n_embd=64 -> 2 blocks */
    float want[4 * 64];
    for (uint32_t t = 0; t < 4; t++) {
        int32_t id = toks[t];
        if (id < 0 || (uint32_t)id >= 4) id = 0;
        ref_embed_q8(model + wo + (uint64_t)id * row_bytes, 64, want + t * 64);
    }

    if (ds4_gpu_embed_tokens_q8_0_tensor(out, tok_t, model, ms, wo, 4, 4, 64) != 0) {
        float got[4 * 64];
        if (ds4_gpu_tensor_read(out, 0, got, sizeof(got)) != 0)
            rc = check_f32("embed_tokens_q8_0/64", got, want, 4 * 64);
    }
    ds4_gpu_tensor_free(tok_t);
    ds4_gpu_tensor_free(out);
    free(model);
    if (rc != 0) return rc;

    /* Padded batch: n_tokens=3 over an 8-wide table (partial blocks). */
    uint64_t ms2 = 0, wo2 = 0;
    unsigned char *model2 = make_q8_model(4, 8, &ms2, &wo2);
    if (!model2) return 1;
    if (ds4_gpu_set_model_map(model2, ms2) == 0) { free(model2); return 1; }

    const int32_t toks2[3] = {3, 1, 12};
    ds4_gpu_tensor *tok2 = ds4_gpu_tensor_alloc(sizeof(toks2));
    ds4_gpu_tensor *out2 = ds4_gpu_tensor_alloc(3 * 8 * sizeof(float));
    if (!tok2 || !out2) {
        if (tok2) ds4_gpu_tensor_free(tok2);
        if (out2) ds4_gpu_tensor_free(out2);
        free(model2);
        return 1;
    }
    if (ds4_gpu_tensor_write(tok2, 0, toks2, sizeof(toks2)) == 0) {
        ds4_gpu_tensor_free(tok2); ds4_gpu_tensor_free(out2); free(model2);
        return 1;
    }

    float want2[3 * 8];
    for (uint32_t t = 0; t < 3; t++) {
        int32_t id = toks2[t];
        if (id < 0 || (uint32_t)id >= 4) id = 0;
        ref_embed_q8(model2 + wo2 + (uint64_t)id * 34u, 8, want2 + t * 8);
    }

    if (ds4_gpu_embed_tokens_q8_0_tensor(out2, tok2, model2, ms2, wo2, 4, 3, 8) != 0) {
        float got2[3 * 8];
        if (ds4_gpu_tensor_read(out2, 0, got2, sizeof(got2)) != 0)
            rc = check_f32("embed_tokens_q8_0/8", got2, want2, 3 * 8);
    }
    ds4_gpu_tensor_free(tok2);
    ds4_gpu_tensor_free(out2);
    free(model2);
    return rc;
}
REGISTER_TEST(embed_tokens_q8_0, test_embed_tokens_q8_0);

static int test_embed_hc_f16(void) {
    constexpr uint32_t n_vocab = 4;
    constexpr uint32_t n_embd = 8;
    constexpr uint32_t n_hc = 3;
    uint64_t model_size = 0, weight_offset = 0;
    unsigned char *model = make_f16_model(n_vocab, n_embd,
                                           &model_size, &weight_offset);
    if (!model) return 1;
    if (ds4_gpu_set_model_map(model, model_size) == 0) {
        free(model);
        return 1;
    }

    float row[n_embd];
    float want[n_hc * n_embd];
    ref_embed_f16((const uint16_t *)(model + weight_offset + 2u * n_embd * 2u),
                  n_embd, row);
    for (uint32_t h = 0; h < n_hc; h++)
        std::memcpy(want + h * n_embd, row, sizeof(row));

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(want));
    if (!out) {
        free(model);
        return 1;
    }
    int rc = 1;
    if (ds4_gpu_embed_token_hc_tensor(out, model, model_size, weight_offset,
                                      n_vocab, 2, n_embd, n_hc) != 0) {
        float got[n_hc * n_embd];
        if (ds4_gpu_tensor_read(out, 0, got, sizeof(got)) != 0)
            rc = check_f32("embed_token_hc/f16", got, want, n_hc * n_embd);
    }
    ds4_gpu_tensor_free(out);
    if (rc != 0) {
        free(model);
        return rc;
    }

    const int32_t tokens[2] = {3, -1};
    ds4_gpu_tensor *token_tensor = ds4_gpu_tensor_alloc(sizeof(tokens));
    ds4_gpu_tensor *batch_out = ds4_gpu_tensor_alloc(2u * sizeof(want));
    if (!token_tensor || !batch_out) {
        if (token_tensor) ds4_gpu_tensor_free(token_tensor);
        if (batch_out) ds4_gpu_tensor_free(batch_out);
        free(model);
        return 1;
    }
    if (ds4_gpu_tensor_write(token_tensor, 0, tokens, sizeof(tokens)) == 0) {
        ds4_gpu_tensor_free(token_tensor);
        ds4_gpu_tensor_free(batch_out);
        free(model);
        return 1;
    }

    float batch_want[2 * n_hc * n_embd];
    for (uint32_t t = 0; t < 2; t++) {
        const uint32_t id = tokens[t] < 0 ? 0u : (uint32_t)tokens[t];
        ref_embed_f16((const uint16_t *)(model + weight_offset +
                                        (uint64_t)id * n_embd * 2u),
                      n_embd, row);
        for (uint32_t h = 0; h < n_hc; h++)
            std::memcpy(batch_want + ((uint64_t)t * n_hc + h) * n_embd,
                        row, sizeof(row));
    }

    rc = 1;
    if (ds4_gpu_embed_tokens_hc_tensor(batch_out, token_tensor, model, model_size,
                                       weight_offset, n_vocab, 2, n_embd, n_hc) != 0) {
        float got[2 * n_hc * n_embd];
        if (ds4_gpu_tensor_read(batch_out, 0, got, sizeof(got)) != 0)
            rc = check_f32("embed_tokens_hc/f16", got, batch_want,
                           2 * n_hc * n_embd);
    }
    ds4_gpu_tensor_free(token_tensor);
    ds4_gpu_tensor_free(batch_out);
    free(model);
    return rc;
}
REGISTER_TEST(embed_hc_f16, test_embed_hc_f16);

/* Single-token quant dispatch: weight_type 8 -> Q8_0, 1 -> F16,
 * anything else -> 0. */
static int test_embed_token_quant(void) {
    int rc = 1;

    /* weight_type 8 (Q8_0): n_embd=64, one in-range and one out-of-range id. */
    uint64_t ms = 0, wo = 0;
    unsigned char *model = make_q8_model(4, 64, &ms, &wo);
    if (!model) return 1;
    if (ds4_gpu_set_model_map(model, ms) == 0) { free(model); return 1; }
    for (uint32_t k = 0; k < 2; k++) {
        if (check_embed_token_q8_one(model, ms, wo, 4, 64, k == 0 ? 2u : 99u,
                                     "embed_token_quant/q8") != 0) {
            free(model);
            return rc;
        }
    }
    free(model);

    /* weight_type 1 (F16): n_embd=64. */
    uint64_t mf = 0, wf = 0;
    unsigned char *model_f = make_f16_model(4, 64, &mf, &wf);
    if (!model_f) return 1;
    if (ds4_gpu_set_model_map(model_f, mf) == 0) { free(model_f); return 1; }

    const uint32_t tok_f16 = 3;
    float want_f16[64];
    ref_embed_f16((const uint16_t *)(model_f + wf + (uint64_t)tok_f16 * 64 * 2u),
                  64, want_f16);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(64 * sizeof(float));
    if (!out) { free(model_f); return 1; }
    if (ds4_gpu_embed_token_quant_tensor(out, model_f, mf, wf, 1, 4, tok_f16, 64) != 0) {
        float got[64];
        if (ds4_gpu_tensor_read(out, 0, got, sizeof(got)) != 0)
            rc = check_f32("embed_token_quant/f16", got, want_f16, 64);
    }
    ds4_gpu_tensor_free(out);
    free(model_f);
    if (rc != 0) return rc;

    /* Unsupported weight type -> 0. */
    ds4_gpu_tensor *out_u = ds4_gpu_tensor_alloc(64 * sizeof(float));
    if (!out_u) return 1;
    const int unsupported_ok =
        ds4_gpu_embed_token_quant_tensor(out_u, NULL, 0, 0, 0, 4, 0, 64) == 0;
    ds4_gpu_tensor_free(out_u);
    return unsupported_ok ? 0 : 1;
}
REGISTER_TEST(embed_token_quant, test_embed_token_quant);

/* Batch quant dispatch: weight_type 1 (F16) and 8 (Q8_0). */
static int test_embed_tokens_quant(void) {
    int rc = 1;

    /* weight_type 1 (F16): n_tokens=3, with -1 clamped to row 0. */
    uint64_t mf = 0, wf = 0;
    unsigned char *model_f = make_f16_model(4, 64, &mf, &wf);
    if (!model_f) return 1;
    if (ds4_gpu_set_model_map(model_f, mf) == 0) { free(model_f); return 1; }

    const int32_t toks[3] = {1, 3, -1};
    ds4_gpu_tensor *tok_t = ds4_gpu_tensor_alloc(sizeof(toks));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(3 * 64 * sizeof(float));
    if (!tok_t || !out) {
        if (tok_t) ds4_gpu_tensor_free(tok_t);
        if (out) ds4_gpu_tensor_free(out);
        free(model_f);
        return 1;
    }
    if (ds4_gpu_tensor_write(tok_t, 0, toks, sizeof(toks)) == 0) {
        ds4_gpu_tensor_free(tok_t); ds4_gpu_tensor_free(out); free(model_f);
        return 1;
    }

    float want[3 * 64];
    for (uint32_t t = 0; t < 3; t++) {
        int32_t id = toks[t];
        if (id < 0 || (uint32_t)id >= 4) id = 0;
        ref_embed_f16((const uint16_t *)(model_f + wf + (uint64_t)id * 64 * 2u),
                      64, want + t * 64);
    }

    if (ds4_gpu_embed_tokens_quant_tensor(out, tok_t, model_f, mf, wf, 1, 4, 3, 64) != 0) {
        float got[3 * 64];
        if (ds4_gpu_tensor_read(out, 0, got, sizeof(got)) != 0)
            rc = check_f32("embed_tokens_quant/f16", got, want, 3 * 64);
    }
    ds4_gpu_tensor_free(tok_t);
    ds4_gpu_tensor_free(out);
    free(model_f);
    if (rc != 0) return rc;

    /* weight_type 8 (Q8_0): n_tokens=2 over a padded 8-wide table. */
    uint64_t ms = 0, wo = 0;
    unsigned char *model = make_q8_model(4, 8, &ms, &wo);
    if (!model) return 1;
    if (ds4_gpu_set_model_map(model, ms) == 0) { free(model); return 1; }

    const int32_t toks2[2] = {2, 7};                   /* 7 out of range -> row 0 */
    ds4_gpu_tensor *tok2 = ds4_gpu_tensor_alloc(sizeof(toks2));
    ds4_gpu_tensor *out2 = ds4_gpu_tensor_alloc(2 * 8 * sizeof(float));
    if (!tok2 || !out2) {
        if (tok2) ds4_gpu_tensor_free(tok2);
        if (out2) ds4_gpu_tensor_free(out2);
        free(model);
        return 1;
    }
    if (ds4_gpu_tensor_write(tok2, 0, toks2, sizeof(toks2)) == 0) {
        ds4_gpu_tensor_free(tok2); ds4_gpu_tensor_free(out2); free(model);
        return 1;
    }

    float want2[2 * 8];
    for (uint32_t t = 0; t < 2; t++) {
        int32_t id = toks2[t];
        if (id < 0 || (uint32_t)id >= 4) id = 0;
        ref_embed_q8(model + wo + (uint64_t)id * 34u, 8, want2 + t * 8);
    }

    if (ds4_gpu_embed_tokens_quant_tensor(out2, tok2, model, ms, wo, 8, 4, 2, 8) != 0) {
        float got2[2 * 8];
        if (ds4_gpu_tensor_read(out2, 0, got2, sizeof(got2)) != 0)
            rc = check_f32("embed_tokens_quant/q8", got2, want2, 2 * 8);
    }
    ds4_gpu_tensor_free(tok2);
    ds4_gpu_tensor_free(out2);
    free(model);
    return rc;
}
REGISTER_TEST(embed_tokens_quant, test_embed_tokens_quant);
