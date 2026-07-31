/* Kernel test: ds4_gpu_matmul_quant_tensor (host-side dense quantized matmul).
 *
 * out[t][o] = sum_i x[t][i] * W[o][i]
 *
 * The kernel dispatches by weight_type:
 *   F32(0)/F16(1)/Q8_0(8) -> existing matmul_*_tensor entries,
 *   Q2_K(10)/IQ2_XXS(16)  -> host-side port of the ds4.c matvec helpers
 *   (ds4_vec_dot_q2_K_q8_K / ds4_vec_dot_iq2_xxs_q8_K): the f32 activation
 *   is quantized to Q8_K, then each weight row is dotted against it.  The
 *   GGUF block layouts and dot math are the same ds4gk_* helpers used by
 *   ds4_gpu_routed_moe_*_tensor, so the CPU reference below is a verbatim
 *   transcription of those helpers (like t_routed_moe_one.cpp).
 *
 * The Q2_K / IQ2_XXS paths are CPU-hosted over the model map and host-mapped
 * tensor memory (like ds4_gpu_add_tensor), so no begin_commands/end_commands
 * are needed to read back the outputs.
 *
 * Model layout: junk header, then the row-major quantized W matrix at
 * weight_offset.  Q2_K rows are 84 B/256-elem block (scales[16] + qs[64] +
 * f16 d + f16 dmin); IQ2_XXS rows are 66 B/256-elem block (f16 d + 32x u16).
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <vector>

/* ---------------- f16 helpers (from ds4.c / t_matmul_f16.cpp) ------------- */
static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t e    = (x >> 23) & 0xffu;
    uint32_t m    = x & 0x7fffffu;
    if (e == 0xffu) return (uint16_t)(sign | 0x7c00u | (m ? 0x200u : 0u));
    if (e == 0u)    return (uint16_t)sign;
    int32_t e16 = (int32_t)e - 127 + 15;
    if (e16 >= 31)  return (uint16_t)(sign | 0x7c00u);
    if (e16 <= 0) {
        uint32_t shift = 126u - e;
        uint32_t m16 = (0x800000u | m) >> shift;
        return (uint16_t)(sign | m16);
    }
    return (uint16_t)(sign | ((uint32_t)e16 << 10) | (m >> 13));
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e    = (h >> 10) & 0x1fu;
    uint32_t m    = h & 0x3ffu;
    uint32_t bits;
    if (e == 0u) {
        if (m == 0u) bits = sign;
        else {
            uint32_t mm = m, p = 0u;
            while (!(mm & 0x400u) && p < 10u) { mm <<= 1; p++; }
            bits = sign | ((113u - p) << 23) | ((mm & 0x3ffu) << 13);
        }
    } else if (e == 31u) {
        bits = sign | 0x7f800000u | (m << 13);
    } else {
        bits = sign | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

/* ---------------- Q8_K / Q2_K / IQ2_XXS blocks (ds4.c) -------------------- */
static const uint32_t QK_K = 256;

struct q8k_block { float d; int8_t qs[QK_K]; int16_t bsums[QK_K / 16]; };
struct q2k_block { uint8_t scales[QK_K / 16]; uint8_t qs[QK_K / 4]; uint16_t d; uint16_t dmin; };
struct iq2_block { uint16_t d; uint16_t qs[QK_K / 8]; };

static void quantize_row_q8_K(const float *x, q8k_block *y, int64_t k) {
    const int64_t nb = k / (int64_t)QK_K;
    for (int64_t b = 0; b < nb; b++) {
        float max = 0.0f, amax = 0.0f;
        for (int j = 0; j < (int)QK_K; j++) {
            const float ax = std::fabsf(x[j]);
            if (ax > amax) { amax = ax; max = x[j]; }
        }
        if (amax == 0.0f) {
            y[b].d = 0.0f;
            std::memset(y[b].qs, 0, sizeof(y[b].qs));
            std::memset(y[b].bsums, 0, sizeof(y[b].bsums));
            x += QK_K;
            continue;
        }
        const float iscale = -127.0f / max;
        for (int j = 0; j < (int)QK_K; j++) {
            int v = (int)lrintf(iscale * x[j]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            y[b].qs[j] = (int8_t)v;
        }
        for (int j = 0; j < (int)(QK_K / 16); j++) {
            int sum = 0;
            for (int i = 0; i < 16; i++) sum += y[b].qs[j * 16 + i];
            y[b].bsums[j] = (int16_t)sum;
        }
        y[b].d = 1.0f / iscale;
        x += QK_K;
    }
}

static int32_t dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    for (uint32_t i = 0; i < 16; i++) sum += (int32_t)q8[i] * (int32_t)((q2[i] >> shift) & 3);
    return sum;
}

static float vec_dot_q2_K_q8_K(int n, const q2k_block *x, const q8k_block *y) {
    const int nb = n / (int)QK_K;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;
        int summs = 0;
        for (int j = 0; j < 16; j++) summs += y[i].bsums[j] * (sc[j] >> 4);
        const float dall = y[i].d * f16_to_f32(x[i].d);
        const float dmin = y[i].d * f16_to_f32(x[i].dmin);
        int isum = 0, is = 0;
        for (int k = 0; k < (int)(QK_K / 128); k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                int isuml = dot_q2_16(q2, q8, shift);
                isum += d * isuml;
                d = sc[is++] & 0x0f;
                isuml = dot_q2_16(q2 + 16, q8 + 16, shift);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * (float)isum - dmin * (float)summs;
    }
    return sumf;
}

/* IQ2_XXS tables (verbatim from ds4.c, same as t_routed_moe_one.cpp). */
static const uint8_t iq2_kmask[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };
static const uint8_t iq2_ksigns[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};
static const uint64_t iq2_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

static int8_t iq2_signed_grid[256][128][8];
static int iq2_grid_ready = 0;

static void iq2_grid_init(void) {
    if (iq2_grid_ready) return;
    for (uint32_t g = 0; g < 256; g++) {
        const uint8_t *grid = (const uint8_t *)(iq2_grid + g);
        for (uint32_t s = 0; s < 128; s++) {
            const uint8_t signs = iq2_ksigns[s];
            for (uint32_t j = 0; j < 8; j++) {
                const int v = (int)grid[j];
                iq2_signed_grid[g][s][j] = (int8_t)((signs & iq2_kmask[j]) ? -v : v);
            }
        }
    }
    iq2_grid_ready = 1;
}

static int32_t dot_iq2_pair_16(const int8_t *grid0, const int8_t *grid1, const int8_t *q8) {
    int32_t sum = 0;
    for (uint32_t i = 0; i < 8; i++) sum += (int32_t)grid0[i] * (int32_t)q8[i];
    for (uint32_t i = 0; i < 8; i++) sum += (int32_t)grid1[i] * (int32_t)q8[8 + i];
    return sum;
}

static float vec_dot_iq2_xxs_q8_K(int n, const iq2_block *x, const q8k_block *y) {
    iq2_grid_init();
    const int nb = n / (int)QK_K;
    uint32_t aux32[2];
    const uint8_t *aux8 = (const uint8_t *)aux32;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const float d = f16_to_f32(x[i].d) * y[i].d;
        const uint16_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < (int)(QK_K / 32); ib32++) {
            std::memcpy(aux32, q2, 2 * sizeof(uint32_t));
            q2 += 4;
            const uint32_t ls = 2 * (aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; l += 2) {
                const uint32_t sign_idx0 = (aux32[1] >> (7 * l)) & 127;
                const uint32_t sign_idx1 = (aux32[1] >> (7 * (l + 1))) & 127;
                sumi += dot_iq2_pair_16(iq2_signed_grid[aux8[l]][sign_idx0],
                                        iq2_signed_grid[aux8[l + 1]][sign_idx1],
                                        q8);
                q8 += 16;
            }
            bsum += sumi * (int32_t)ls;
        }
        sumf += d * (float)bsum;
    }
    return 0.125f * sumf;
}

/* ---------------- generic row dot dispatch -------------------------------- */
static float ref_dot(uint32_t type, const uint8_t *row, uint64_t in_dim,
                     const std::vector<q8k_block> &xqk) {
    if (type == 10) {
        return vec_dot_q2_K_q8_K((int)in_dim, (const q2k_block *)row, xqk.data());
    }
    return vec_dot_iq2_xxs_q8_K((int)in_dim, (const iq2_block *)row, xqk.data());
}

/* ---------------- one end-to-end case ------------------------------------- */
static int run_case(const char *label, uint32_t type,
                    uint64_t in_dim, uint64_t out_dim, uint64_t n_tok,
                    uint64_t weight_offset, uint64_t row_bytes,
                    const std::vector<uint8_t> &model,
                    const std::vector<float> &xv)
{
    ds4_gpu_tensor *x   = ds4_gpu_tensor_alloc(n_tok * in_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(n_tok * out_dim * sizeof(float));
    auto freet = [&]() {
        if (x)   ds4_gpu_tensor_free(x);
        if (out) ds4_gpu_tensor_free(out);
    };
    if (!x || !out) { freet(); return 1; }
    if (ds4_gpu_set_model_map(model.data(), model.size()) == 0 ||
        ds4_gpu_tensor_write(x, 0, xv.data(), (uint64_t)xv.size() * sizeof(float)) == 0) {
        freet();
        return 1;
    }

    const int ok = ds4_gpu_matmul_quant_tensor(
            out, model.data(), model.size(), weight_offset, type,
            in_dim, out_dim, x, n_tok);
    if (!ok) {
        fprintf(stderr, "matmul_quant[%s]: kernel returned 0\n", label);
        freet();
        return 1;
    }

    std::vector<float> got(n_tok * out_dim);
    if (ds4_gpu_tensor_read(out, 0, got.data(), got.size() * sizeof(float)) == 0) {
        fprintf(stderr, "matmul_quant[%s]: tensor read failed\n", label);
        freet();
        return 1;
    }

    /* CPU reference: quantize each token to Q8_K, dot every weight row. */
    std::vector<float> ref(n_tok * out_dim);
    std::vector<q8k_block> xqk(in_dim / QK_K);
    for (uint64_t t = 0; t < n_tok; t++) {
        quantize_row_q8_K(xv.data() + t * in_dim, xqk.data(), (int64_t)in_dim);
        for (uint64_t o = 0; o < out_dim; o++) {
            const uint8_t *row = model.data() + weight_offset + o * row_bytes;
            ref[t * out_dim + o] = ref_dot(type, row, in_dim, xqk);
        }
    }

    const float tol = 1e-3f;
    bool good = true;
    for (uint64_t i = 0; i < ref.size(); i++) {
        if (!(std::fabsf(got[i] - ref[i]) <= tol)) {
            if (good) fprintf(stderr, "--- matmul_quant[%s] mismatch (type %u) ---\n",
                              label, type);
            fprintf(stderr, "  [%llu] got=%.6f want=%.6f\n",
                    (unsigned long long)i, got[i], ref[i]);
            good = false;
        }
    }
    freet();
    return good ? 0 : 1;
}

static int test_matmul_quant(void) {
    int rc = 0;
    const uint64_t header = 4096;   /* distinct from other tests' cached ranges */

    /* ============ Case A: Q2_K, in=256 out=8, n_tok=2 ======================
     * One 84-byte Q2_K block per row; dmin and per-group scales are nonzero
     * so the dmin / min term of the dot is exercised. */
    {
        const uint64_t in_dim = 256, out_dim = 8, n_tok = 2;
        const uint64_t row_bytes = 84;                 /* sizeof(q2k_block) */
        const uint64_t w_bytes = out_dim * row_bytes;
        const uint64_t weight_offset = header;
        std::vector<uint8_t> model(header + w_bytes, 0xAA);

        for (uint64_t o = 0; o < out_dim; o++) {
            q2k_block blk;
            blk.d    = f32_to_f16(0.25f * (float)(o + 1));
            blk.dmin = f32_to_f16(0.0625f * (float)((o % 3) + 1));
            for (uint32_t g = 0; g < 16; g++) {
                const uint8_t sc = (uint8_t)((g + (uint32_t)o) % 6 + 1);       /* 1..6 */
                const uint8_t mn = (uint8_t)(((uint32_t)o * 2 + g) % 5 + 1);   /* 1..5 */
                blk.scales[g] = (uint8_t)(sc | (mn << 4));
            }
            for (uint32_t k = 0; k < 64; k++) {
                uint8_t byte = 0;
                for (uint32_t t = 0; t < 4; t++) {
                    const uint32_t i = k * 4 + t;
                    const uint32_t q = (i * 7 + (uint32_t)o * 13 + t * 3) % 4;
                    byte |= (uint8_t)(q << (2 * t));
                }
                blk.qs[k] = byte;
            }
            std::memcpy(model.data() + weight_offset + o * row_bytes, &blk, sizeof(blk));
        }

        std::vector<float> xv(n_tok * in_dim);
        for (uint64_t t = 0; t < n_tok; t++)
            for (uint64_t i = 0; i < in_dim; i++)
                xv[t * in_dim + i] = (float)((int)(((t + 1) * i * 13 + 5) % 17) - 8) * 0.125f;

        rc |= run_case("q2k", 10, in_dim, out_dim, n_tok,
                       weight_offset, row_bytes, model, xv);
    }

    /* ============ Case B: IQ2_XXS, in=256 out=8, n_tok=2 ===================
     * One 66-byte IQ2_XXS block per row; grid/sign/ls fields varied per row. */
    {
        const uint64_t in_dim = 256, out_dim = 8, n_tok = 2;
        const uint64_t row_bytes = 66;                 /* sizeof(iq2_block) */
        const uint64_t w_bytes = out_dim * row_bytes;
        const uint64_t weight_offset = header;
        std::vector<uint8_t> model(header + w_bytes, 0xAA);

        for (uint64_t o = 0; o < out_dim; o++) {
            iq2_block blk;
            blk.d = f32_to_f16(2.0f * (float)(o + 1));  /* scale = 0.5*(2*ls+1)*d/4 */
            for (uint32_t g = 0; g < 32; g += 4) {
                const uint32_t l = g / 4;              /* 8-element group */
                const uint32_t ls = (uint32_t)((o + l) % 3);
                uint32_t aux0 = 0, aux1 = ls << 28;
                for (uint32_t t = 0; t < 4; t++) {
                    const uint32_t grid_idx = (uint32_t)((o * 7 + l * 11 + t * 17) % 256);
                    ((uint8_t *)&aux0)[t] = (uint8_t)grid_idx;
                    aux1 |= (((o * 5 + l * 13 + t) & 127) << (7 * t));
                }
                blk.qs[g + 0] = (uint16_t)(aux0 & 0xffffu);
                blk.qs[g + 1] = (uint16_t)(aux0 >> 16);
                blk.qs[g + 2] = (uint16_t)(aux1 & 0xffffu);
                blk.qs[g + 3] = (uint16_t)(aux1 >> 16);
            }
            std::memcpy(model.data() + weight_offset + o * row_bytes, &blk, sizeof(blk));
        }

        std::vector<float> xv(n_tok * in_dim);
        for (uint64_t t = 0; t < n_tok; t++)
            for (uint64_t i = 0; i < in_dim; i++)
                xv[t * in_dim + i] = (float)((int)(((t + 1) * i * 7 + 3) % 19) - 9) * 0.125f;

        rc |= run_case("iq2xxs", 16, in_dim, out_dim, n_tok,
                       weight_offset, row_bytes, model, xv);
    }

    /* ============ Error paths ============================================= */
    {
        ds4_gpu_tensor *x   = ds4_gpu_tensor_alloc(256 * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(8 * sizeof(float));
        if (!x || !out) {
            if (x)   ds4_gpu_tensor_free(x);
            if (out) ds4_gpu_tensor_free(out);
            return 1;
        }
        /* Null arguments must return 0. */
        if (ds4_gpu_matmul_quant_tensor(nullptr, nullptr, 0, 0, 10,
                                        256, 8, nullptr, 1) != 0) {
            fprintf(stderr, "matmul_quant: null-pointer call should return 0\n");
            rc = 1;
        }
        /* Unsupported weight_type must return 0. */
        unsigned char model[64] = {0};
        if (ds4_gpu_set_model_map(model, sizeof(model)) == 0) { rc = 1; }
        else if (ds4_gpu_matmul_quant_tensor(out, model, sizeof(model), 0, 99,
                                             256, 8, x, 1) != 0) {
            fprintf(stderr, "matmul_quant: unsupported weight_type should return 0\n");
            rc = 1;
        }
        /* Weight range past the end of the model map must return 0. */
        if (ds4_gpu_matmul_quant_tensor(out, model, sizeof(model), 1, 10,
                                        256, 8, x, 1) != 0) {
            fprintf(stderr, "matmul_quant: out-of-range weights should return 0\n");
            rc = 1;
        }
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
    }

    return rc;
}
REGISTER_TEST(matmul_quant, test_matmul_quant);
