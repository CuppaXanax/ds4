/* Kernel test: ds4_gpu_routed_moe_one_tensor (host-side single-token routed MoE).
 *
 * The kernel is CPU-hosted over the model map and the host-mapped tensor
 * memory (like ds4_gpu_add_tensor), so no begin_commands/end_commands are
 * needed to read back the outputs.
 *
 * For every selected expert the kernel replicates the ds4.c reference
 * (layer_routed_moe_one_prealloc + matvec_*_prequant):
 *   gate = dequant(gate_row[e]) . x
 *   up   = dequant(up_row[e]) . x
 *   clamp (if clamp > 1e-6: gate=min(gate,clamp), up=clamp)
 *   mid  = silu(gate) * up * weight[e]              (weighted SwiGLU)
 *   out += dequant(down_row[e]) . requant_q(mid)    (down sees the weighted mid)
 * and writes gate/up/mid (clamped/weighted) plus the per-expert down output
 * into the tensors consumed by the downstream shared-down / hc kernels.
 *
 * Quant types covered:
 *   - Q8_0   (gate/up/down): f16 block scale + 32 int8, 34 B/block
 *   - Q2_K   (gate/up/down): 256-elem blocks, 84 B/block
 *   - IQ2_XXS (gate/up):     256-elem blocks, 66 B/block (grid/sign tables)
 * The CPU reference below is a verbatim transcription of the ds4.c helpers
 * (quantize_q8_0_activation, ds4_quantize_row_q8_K, dot_q8_0_row,
 * ds4_vec_dot_q2_K_q8_K, ds4_vec_dot_iq2_xxs_q8_K).
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void set_routed_iq2_words_env(const char *value) {
#ifdef _WIN32
    _putenv_s("DS4_VULKAN_ROUTED_IQ2_WORDS", value ? value : "");
#else
    if (value) setenv("DS4_VULKAN_ROUTED_IQ2_WORDS", value, 1);
    else unsetenv("DS4_VULKAN_ROUTED_IQ2_WORDS");
#endif
}

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

/* ---------------- activation / swiglu helpers (ds4.c) -------------------- */
static float sigmoid_stable(float x) {
    if (x >= 0.0f) {
        const float e = std::exp(-x);
        return 1.0f / (1.0f + e);
    } else {
        const float e = std::exp(x);
        return e / (1.0f + e);
    }
}
static float silu(float x) { return x * sigmoid_stable(x); }

/* ---------------- Q8_0 (ds4.c quantize_q8_0_activation + dot_q8_0_row) --- */
static void quantize_q8_0_activation(const float *x, int8_t *xq, float *scale, uint64_t n) {
    const uint64_t blocks = (n + 31) / 32;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = n - i0 < 32 ? n - i0 : 32;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) {
            const float ax = std::fabsf(x[i0 + i]);
            if (ax > amax) amax = ax;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        scale[b] = d;
        for (uint64_t i = 0; i < bn; i++) {
            int v = (int)lrintf(x[i0 + i] * id);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            xq[i0 + i] = (int8_t)v;
        }
        for (uint64_t i = bn; i < 32 && i0 + i < blocks * 32; i++) xq[i0 + i] = 0;
    }
}

static int32_t dot_i8_32(const int8_t *a, const int8_t *b, uint64_t n) {
    int32_t sum = 0;
    for (uint64_t i = 0; i < n; i++) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

static float dot_q8_0_row(const uint8_t *row, const int8_t *xq, const float *xscale,
                          uint64_t in_dim, uint64_t blocks) {
    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t scale_bits;
        std::memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
        const int8_t *qs = (const int8_t *)(row + b * 34 + 2);
        const uint64_t i0 = b * 32;
        const uint64_t n = in_dim - i0 < 32 ? in_dim - i0 : 32;
        acc += f16_to_f32(scale_bits) * xscale[b] * (float)dot_i8_32(qs, xq + i0, n);
    }
    return acc;
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

/* IQ2_XXS tables (verbatim from ds4.c). */
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
static float ref_dot_row(uint32_t type, const uint8_t *row, uint32_t in_dim,
                         const std::vector<int8_t> &xq8,
                         const std::vector<float> &xscale8,
                         const std::vector<q8k_block> &xqk) {
    if (type == 8) {
        const uint64_t blocks = xq8.size() / 32;
        return dot_q8_0_row(row, xq8.data(), xscale8.data(), in_dim, blocks);
    }
    if (type == 10) {
        return vec_dot_q2_K_q8_K((int)in_dim, (const q2k_block *)row, xqk.data());
    }
    return vec_dot_iq2_xxs_q8_K((int)in_dim, (const iq2_block *)row, xqk.data());
}

/* ---------------- CPU reference (mirror of the kernel) -------------------- */
static void ref_moe_one(
        uint32_t gate_type, uint32_t down_type,
        uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const std::vector<float> &x,
        const std::vector<int32_t> &sel,
        const std::vector<float> &wgt,
        const std::vector<uint8_t> &model,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        const std::vector<float> &add_in,
        std::vector<float> &ref_gate, std::vector<float> &ref_up,
        std::vector<float> &ref_mid, std::vector<float> &ref_experts,
        std::vector<float> &ref_out)
{
    (void)n_total_expert;
    std::vector<int8_t> xq8;
    std::vector<float> xscale8;
    std::vector<q8k_block> xqk;
    if (gate_type == 8) {
        const uint64_t blocks = (in_dim + 31) / 32;
        xq8.resize(blocks * 32, 0);
        xscale8.resize(blocks);
        quantize_q8_0_activation(x.data(), xq8.data(), xscale8.data(), in_dim);
    } else {
        xqk.resize(in_dim / QK_K);
        quantize_row_q8_K(x.data(), xqk.data(), in_dim);
    }

    ref_gate.assign((uint64_t)n_expert * mid_dim, 0.0f);
    ref_up.assign((uint64_t)n_expert * mid_dim, 0.0f);
    ref_mid.assign((uint64_t)n_expert * mid_dim, 0.0f);
    ref_experts.assign((uint64_t)n_expert * out_dim, 0.0f);
    ref_out.assign(out_dim, 0.0f);

    std::vector<float> midv(mid_dim);
    for (uint32_t e = 0; e < n_expert; e++) {
        const uint32_t expert = (uint32_t)sel[e];
        const float weight = wgt[e];
        const uint8_t *gb = model.data() + gate_offset + (uint64_t)expert * gate_expert_bytes;
        const uint8_t *ub = model.data() + up_offset + (uint64_t)expert * gate_expert_bytes;
        const uint8_t *db = model.data() + down_offset + (uint64_t)expert * down_expert_bytes;

        for (uint32_t r = 0; r < mid_dim; r++) {
            float g = ref_dot_row(gate_type, gb + (uint64_t)r * gate_row_bytes,
                                  in_dim, xq8, xscale8, xqk);
            float u = ref_dot_row(gate_type, ub + (uint64_t)r * gate_row_bytes,
                                  in_dim, xq8, xscale8, xqk);
            if (clamp > 1.0e-6f) {
                if (g > clamp) g = clamp;
                if (u > clamp) u = clamp;
                if (u < -clamp) u = -clamp;
            }
            const float m = silu(g) * u * weight;
            ref_gate[(uint64_t)e * mid_dim + r] = g;
            ref_up[(uint64_t)e * mid_dim + r] = u;
            ref_mid[(uint64_t)e * mid_dim + r] = m;
            midv[r] = m;
        }

        std::vector<int8_t> midq8;
        std::vector<float> midscale8;
        std::vector<q8k_block> midqk;
        if (down_type == 8) {
            const uint64_t blocks = (mid_dim + 31) / 32;
            midq8.resize(blocks * 32, 0);
            midscale8.resize(blocks);
            quantize_q8_0_activation(midv.data(), midq8.data(), midscale8.data(), mid_dim);
        } else {
            midqk.resize(mid_dim / QK_K);
            quantize_row_q8_K(midv.data(), midqk.data(), mid_dim);
        }
        for (uint32_t r2 = 0; r2 < out_dim; r2++) {
            const float dv = ref_dot_row(down_type, db + (uint64_t)r2 * down_row_bytes,
                                         mid_dim, midq8, midscale8, midqk);
            ref_experts[(uint64_t)e * out_dim + r2] = dv;
            ref_out[r2] += dv;
        }
    }
    if (!add_in.empty()) {
        for (uint32_t r = 0; r < out_dim; r++) ref_out[r] += add_in[r];
    }
}

/* ---------------- one end-to-end case ------------------------------------- */
static int run_moe_case(
        const char *label,
        uint32_t gate_type, uint32_t down_type,
        uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const std::vector<float> &x,
        const std::vector<int32_t> &sel,
        const std::vector<float> &wgt,
        const std::vector<uint8_t> &model,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        const std::vector<float> &add_in,
        bool compare_cpu_reference = true)
{
    const bool exact_iq2_ab = gate_type == 16;
    const char *saved_iq2_words_env = getenv("DS4_VULKAN_ROUTED_IQ2_WORDS");
    const bool had_iq2_words_env = saved_iq2_words_env != nullptr;
    const std::string saved_iq2_words = saved_iq2_words_env
        ? saved_iq2_words_env : "";
    ds4_gpu_tensor *out_t  = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    ds4_gpu_tensor *gate_t = ds4_gpu_tensor_alloc((uint64_t)n_expert * mid_dim * sizeof(float));
    ds4_gpu_tensor *up_t   = ds4_gpu_tensor_alloc((uint64_t)n_expert * mid_dim * sizeof(float));
    ds4_gpu_tensor *mid_t  = ds4_gpu_tensor_alloc((uint64_t)n_expert * mid_dim * sizeof(float));
    ds4_gpu_tensor *exp_t  = ds4_gpu_tensor_alloc((uint64_t)n_expert * out_dim * sizeof(float));
    ds4_gpu_tensor *sel_t  = ds4_gpu_tensor_alloc((uint64_t)n_expert * sizeof(int32_t));
    ds4_gpu_tensor *w_t    = ds4_gpu_tensor_alloc((uint64_t)n_expert * sizeof(float));
    ds4_gpu_tensor *x_t    = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *add_t  = add_in.empty() ? nullptr :
                             ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    auto freet = [&]() {
        if (exact_iq2_ab) {
            set_routed_iq2_words_env(had_iq2_words_env
                ? saved_iq2_words.c_str() : nullptr);
        }
        if (out_t)  ds4_gpu_tensor_free(out_t);
        if (gate_t) ds4_gpu_tensor_free(gate_t);
        if (up_t)   ds4_gpu_tensor_free(up_t);
        if (mid_t)  ds4_gpu_tensor_free(mid_t);
        if (exp_t)  ds4_gpu_tensor_free(exp_t);
        if (sel_t)  ds4_gpu_tensor_free(sel_t);
        if (w_t)    ds4_gpu_tensor_free(w_t);
        if (x_t)    ds4_gpu_tensor_free(x_t);
        if (add_t)  ds4_gpu_tensor_free(add_t);
    };
    if (!out_t || !gate_t || !up_t || !mid_t || !exp_t ||
        !sel_t || !w_t || !x_t || (!add_in.empty() && !add_t)) {
        freet();
        return 1;
    }
    if (ds4_gpu_set_model_map(model.data(), model.size()) == 0) { freet(); return 1; }
    if (ds4_gpu_tensor_write(sel_t, 0, sel.data(), (uint64_t)sel.size() * sizeof(int32_t)) == 0 ||
        ds4_gpu_tensor_write(w_t,   0, wgt.data(), (uint64_t)wgt.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(x_t,   0, x.data(),   (uint64_t)x.size() * sizeof(float)) == 0 ||
        (!add_in.empty() && ds4_gpu_tensor_write(add_t, 0, add_in.data(),
                                                 (uint64_t)add_in.size() * sizeof(float)) == 0)) {
        freet();
        return 1;
    }

    auto run_kernel = [&]() -> int {
        return ds4_gpu_routed_moe_one_tensor(
            out_t, gate_t, up_t, mid_t, exp_t,
            model.data(), model.size(),
            gate_offset, up_offset, down_offset,
            gate_type, down_type,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim,
            sel_t, w_t, n_total_expert, n_expert,
            clamp, x_t, add_t, 0, false);
    };
    if (exact_iq2_ab) {
        /* First call warms the model mappings and shader path.  The following
         * baseline/candidate pair is therefore a cache-hot same-binary A/B. */
        set_routed_iq2_words_env("0");
        if (!run_kernel()) {
            fprintf(stderr, "routed_moe_one[%s]: baseline warm-up returned 0\n", label);
            freet();
            return 1;
        }
    }
    int ok = run_kernel();
    if (!ok) {
        fprintf(stderr, "routed_moe_one[%s]: kernel returned 0\n", label);
        freet();
        return 1;
    }

    std::vector<float> ref_gate, ref_up, ref_mid, ref_experts, ref_out;
    /* Multi-block GPU projection rows reduce per-lane block partials as a
     * tree, while the scalar CPU oracle accumulates every block serially.
     * Their association is only comparable for the one-block fixture; the
     * production fixture uses the strict full-pipeline A/B above instead. */
    if (compare_cpu_reference) {
        ref_moe_one(gate_type, down_type, in_dim, mid_dim, out_dim,
                    n_total_expert, n_expert, clamp, x, sel, wgt, model,
                    gate_offset, up_offset, down_offset,
                    gate_expert_bytes, gate_row_bytes,
                    down_expert_bytes, down_row_bytes,
                    add_in, ref_gate, ref_up, ref_mid, ref_experts, ref_out);
    }

    std::vector<float> got_gate((uint64_t)n_expert * mid_dim);
    std::vector<float> got_up((uint64_t)n_expert * mid_dim);
    std::vector<float> got_mid((uint64_t)n_expert * mid_dim);
    std::vector<float> got_exp((uint64_t)n_expert * out_dim);
    std::vector<float> got_out(out_dim);
    auto read_outputs = [&](std::vector<float> &read_gate,
                            std::vector<float> &read_up,
                            std::vector<float> &read_mid,
                            std::vector<float> &read_exp,
                            std::vector<float> &read_out) -> bool {
        return ds4_gpu_tensor_read(gate_t, 0, read_gate.data(), read_gate.size() * sizeof(float)) != 0 &&
               ds4_gpu_tensor_read(up_t,   0, read_up.data(),   read_up.size() * sizeof(float)) != 0 &&
               ds4_gpu_tensor_read(mid_t,  0, read_mid.data(),  read_mid.size() * sizeof(float)) != 0 &&
               ds4_gpu_tensor_read(exp_t,  0, read_exp.data(),  read_exp.size() * sizeof(float)) != 0 &&
               ds4_gpu_tensor_read(out_t,  0, read_out.data(),  read_out.size() * sizeof(float)) != 0;
    };
    bool read_ok = read_outputs(got_gate, got_up, got_mid, got_exp, got_out);
    if (!read_ok) {
        fprintf(stderr, "routed_moe_one[%s]: tensor read failed\n", label);
        freet();
        return 1;
    }

    bool exact_ab_ok = true;
    if (exact_iq2_ab) {
        std::vector<float> ab_gate(got_gate.size());
        std::vector<float> ab_up(got_up.size());
        std::vector<float> ab_mid(got_mid.size());
        std::vector<float> ab_exp(got_exp.size());
        std::vector<float> ab_out(got_out.size());
        set_routed_iq2_words_env("1");
        if (!run_kernel() ||
            !read_outputs(ab_gate, ab_up, ab_mid, ab_exp, ab_out)) {
            fprintf(stderr, "routed_moe_one[%s]: IQ2 word-load A/B run failed\n", label);
            freet();
            return 1;
        }
        auto exact_vec = [&](const char *what, const std::vector<float> &baseline,
                             const std::vector<float> &candidate) -> bool {
            if (std::memcmp(baseline.data(), candidate.data(),
                            baseline.size() * sizeof(float)) == 0) return true;
            for (size_t i = 0; i < baseline.size(); ++i) {
                uint32_t baseline_bits = 0, candidate_bits = 0;
                std::memcpy(&baseline_bits, &baseline[i], sizeof(baseline_bits));
                std::memcpy(&candidate_bits, &candidate[i], sizeof(candidate_bits));
                if (baseline_bits != candidate_bits) {
                    fprintf(stderr,
                        "routed_moe_one[%s]: IQ2 word-load %s bit mismatch "
                        "[%zu] baseline=%08x candidate=%08x\n",
                        label, what, i, baseline_bits, candidate_bits);
                    break;
                }
            }
            return false;
        };
        exact_ab_ok &= exact_vec("gate", got_gate, ab_gate);
        exact_ab_ok &= exact_vec("up", got_up, ab_up);
        exact_ab_ok &= exact_vec("mid", got_mid, ab_mid);
        exact_ab_ok &= exact_vec("experts", got_exp, ab_exp);
        exact_ab_ok &= exact_vec("out", got_out, ab_out);
        got_gate.swap(ab_gate);
        got_up.swap(ab_up);
        got_mid.swap(ab_mid);
        got_exp.swap(ab_exp);
        got_out.swap(ab_out);
    }

    auto cmp_vec = [&](const char *what, const std::vector<float> &got,
                       const std::vector<float> &ref, size_t n) -> bool {
        bool good = true;
        for (size_t i = 0; i < n; i++) {
            const float tol = 1e-4f + 2e-7f * std::fabsf(ref[i]);
            if (!(std::fabsf(got[i] - ref[i]) <= tol)) {
                if (good) fprintf(stderr, "--- routed_moe_one[%s] %s mismatch ---\n", label, what);
                fprintf(stderr, "  [%zu] got=%.6f want=%.6f\n", i, got[i], ref[i]);
                good = false;
            }
        }
        return good;
    };
    bool ok_all = exact_ab_ok;
    if (compare_cpu_reference) {
        ok_all &= cmp_vec("gate", got_gate, ref_gate, got_gate.size());
        ok_all &= cmp_vec("up",   got_up,   ref_up,   got_up.size());
        ok_all &= cmp_vec("mid",  got_mid,  ref_mid,  got_mid.size());
        ok_all &= cmp_vec("experts", got_exp, ref_experts, got_exp.size());
        ok_all &= cmp_vec("out",  got_out,  ref_out,  got_out.size());
    } else {
        auto finite_vec = [&](const char *what,
                              const std::vector<float> &values) -> bool {
            for (size_t i = 0; i < values.size(); ++i) {
                if (!std::isfinite(values[i])) {
                    fprintf(stderr,
                        "routed_moe_one[%s]: production %s[%zu] is not finite\n",
                        label, what, i);
                    return false;
                }
            }
            return true;
        };
        ok_all &= finite_vec("gate", got_gate);
        ok_all &= finite_vec("up", got_up);
        ok_all &= finite_vec("mid", got_mid);
        ok_all &= finite_vec("experts", got_exp);
        ok_all &= finite_vec("out", got_out);
    }
    if (!ok_all) {
        fprintf(stderr, "--- routed_moe_one[%s] inputs ---\n", label);
        fprintf(stderr, "x: ");
        for (uint32_t i = 0; i < in_dim && i < 32; i++) fprintf(stderr, "%.3f ", x[i]);
        fprintf(stderr, "\nsel:");
        for (uint32_t i = 0; i < n_expert; i++) fprintf(stderr, " %d", sel[i]);
        fprintf(stderr, "\nwgt:");
        for (uint32_t i = 0; i < n_expert; i++) fprintf(stderr, " %.3f", wgt[i]);
        fprintf(stderr, "\n");
    }

    freet();
    return ok_all ? 0 : 1;
}

/* ---------------- synthetic Q8_0 model (34 B/32-elem block) --------------- */
static void fill_q8_row(uint8_t *dst, uint32_t n_elem, float scale,
                        uint32_t row, uint32_t expert, uint32_t kind) {
    uint16_t d16 = f32_to_f16(scale);
    std::memcpy(dst, &d16, 2);
    int8_t *qs = (int8_t *)(dst + 2);
    for (uint32_t i = 0; i < n_elem; i++) {
        const int v = (int)(((i * 7 + row * 13 + expert * 5 + kind * 11) % 9) - 4);
        qs[i] = (int8_t)v;
    }
}

static int test_routed_moe_one(void) {
    int rc = 0;

    /* ============ Case A: Q8_0 gate/up + Q8_0 down, small dims ============
     * in=8, mid=8, out=8, 2 experts, both selected. */
    {
        const uint32_t in_dim = 8, mid_dim = 8, out_dim = 8;
        const uint32_t n_total = 2, n_expert = 2;
        const float clamp = 0.25f;
        const uint64_t row_bytes = 34;               /* 1 Q8_0 block (8 < 32) */
        const uint64_t gate_expert_bytes = (uint64_t)mid_dim * row_bytes;
        const uint64_t down_expert_bytes = (uint64_t)out_dim * row_bytes;
        const uint64_t header = 16;
        const uint64_t gate_offset = header;
        const uint64_t up_offset = gate_offset + (uint64_t)n_total * gate_expert_bytes;
        const uint64_t down_offset = up_offset + (uint64_t)n_total * gate_expert_bytes;
        const uint64_t model_size = down_offset + (uint64_t)n_total * down_expert_bytes;

        std::vector<uint8_t> model(model_size, 0xAA);
        for (uint32_t e = 0; e < n_total; e++) {
            for (uint32_t r = 0; r < mid_dim; r++) {
                fill_q8_row(model.data() + gate_offset + (uint64_t)e * gate_expert_bytes + (uint64_t)r * row_bytes,
                            in_dim, 0.25f * (float)((r % 3) + 1), r, e, 0);
                fill_q8_row(model.data() + up_offset + (uint64_t)e * gate_expert_bytes + (uint64_t)r * row_bytes,
                            in_dim, 0.5f * (float)((r % 2) + 1), r, e, 1);
            }
            for (uint32_t r2 = 0; r2 < out_dim; r2++) {
                fill_q8_row(model.data() + down_offset + (uint64_t)e * down_expert_bytes + (uint64_t)r2 * row_bytes,
                            mid_dim, 0.125f * (float)((r2 % 4) + 1), r2, e, 2);
            }
        }

        std::vector<float> x = { 0.5f, -1.0f, 0.75f, 0.25f, -0.5f, 1.25f, -0.75f, 0.375f };
        std::vector<int32_t> sel = { 0, 1 };
        std::vector<float> wgt = { 0.6f, 0.4f };
        rc |= run_moe_case("q8_0-small", 8, 8, in_dim, mid_dim, out_dim,
                           n_total, n_expert, clamp, x, sel, wgt, model,
                           gate_offset, up_offset, down_offset,
                           gate_expert_bytes, row_bytes,
                           down_expert_bytes, row_bytes, {});

        /* Same case with add_in (folded shared expert path). */
        std::vector<float> add_in = { 0.1f, -0.2f, 0.05f, 0.3f, -0.1f, 0.02f, -0.3f, 0.15f };
        rc |= run_moe_case("q8_0-small-addin", 8, 8, in_dim, mid_dim, out_dim,
                           n_total, n_expert, clamp, x, sel, wgt, model,
                           gate_offset, up_offset, down_offset,
                           gate_expert_bytes, row_bytes,
                           down_expert_bytes, row_bytes, add_in);

        /* Clamp disabled must produce different (larger) mid values; just
         * verify it still runs and matches its own reference. */
        rc |= run_moe_case("q8_0-small-noclamp", 8, 8, in_dim, mid_dim, out_dim,
                           n_total, n_expert, 0.0f, x, sel, wgt, model,
                           gate_offset, up_offset, down_offset,
                           gate_expert_bytes, row_bytes,
                           down_expert_bytes, row_bytes, {});
    }

    /* ============ Case B: Q2_K gate/up + Q2_K down =========================
     * in=256, mid=256, out=8 (QK_K-aligned blocks, 84 B each). */
    {
        const uint32_t in_dim = 256, mid_dim = 256, out_dim = 8;
        const uint32_t n_total = 2, n_expert = 2;
        const float clamp = 0.25f;
        const uint64_t row_bytes = 84;               /* 1 Q2_K block (256 elems) */
        const uint64_t gate_expert_bytes = (uint64_t)mid_dim * row_bytes;
        const uint64_t down_expert_bytes = (uint64_t)out_dim * row_bytes;
        const uint64_t header = 16;
        const uint64_t gate_offset = header;
        const uint64_t up_offset = gate_offset + (uint64_t)n_total * gate_expert_bytes;
        const uint64_t down_offset = up_offset + (uint64_t)n_total * gate_expert_bytes;
        const uint64_t model_size = down_offset + (uint64_t)n_total * down_expert_bytes;

        std::vector<uint8_t> model(model_size, 0xAA);
        for (uint32_t e = 0; e < n_total; e++) {
            for (uint32_t r = 0; r < mid_dim; r++) {
                q2k_block blk;
                blk.d = f32_to_f16(0.5f);
                blk.dmin = 0;
                for (uint32_t g = 0; g < 16; g++) blk.scales[g] = 0x01;
                for (uint32_t k = 0; k < 64; k++) {
                    uint8_t byte = 0;
                    for (uint32_t t = 0; t < 4; t++) {
                        const uint32_t i = k * 4 + t;
                        const uint32_t q = (i * 7 + r * 13 + e * 5) % 4;
                        byte |= (uint8_t)(q << (2 * t));
                    }
                    blk.qs[k] = byte;
                }
                std::memcpy(model.data() + gate_offset + (uint64_t)e * gate_expert_bytes + (uint64_t)r * row_bytes,
                            &blk, sizeof(blk));
                std::memcpy(model.data() + up_offset + (uint64_t)e * gate_expert_bytes + (uint64_t)r * row_bytes,
                            &blk, sizeof(blk));
            }
            for (uint32_t r2 = 0; r2 < out_dim; r2++) {
                q2k_block blk;
                blk.d = f32_to_f16(0.125f);
                blk.dmin = 0;
                for (uint32_t g = 0; g < 16; g++) blk.scales[g] = 0x02;   /* scale 2 */
                for (uint32_t k = 0; k < 64; k++) {
                    uint8_t byte = 0;
                    for (uint32_t t = 0; t < 4; t++) {
                        const uint32_t i = k * 4 + t;
                        const uint32_t q = (i * 3 + r2 * 7 + e * 11) % 4;
                        byte |= (uint8_t)(q << (2 * t));
                    }
                    blk.qs[k] = byte;
                }
                std::memcpy(model.data() + down_offset + (uint64_t)e * down_expert_bytes + (uint64_t)r2 * row_bytes,
                            &blk, sizeof(blk));
            }
        }

        std::vector<float> x(in_dim);
        for (uint32_t i = 0; i < in_dim; i++) x[i] = (float)((int)((i * 13 + 5) % 17) - 8) * 0.125f;
        std::vector<int32_t> sel = { 1, 0 };          /* reversed order on purpose */
        std::vector<float> wgt = { 0.6f, 0.4f };
        rc |= run_moe_case("q2k-all", 10, 10, in_dim, mid_dim, out_dim,
                           n_total, n_expert, clamp, x, sel, wgt, model,
                           gate_offset, up_offset, down_offset,
                           gate_expert_bytes, row_bytes,
                           down_expert_bytes, row_bytes, {});
    }

    /* ============ Case C: IQ2_XXS gate/up + Q2_K down ======================
     * The production gate/up layout (66 B/block). */
    {
        const bool production_shape = getenv("DS4_TEST_PRODUCTION_SHAPE") != nullptr;
        const uint32_t in_dim = production_shape ? 7168 : 256;
        const uint32_t mid_dim = 2048, out_dim = 1024;
        const uint32_t n_total = 8, n_expert = 6;
        const float clamp = 0.25f;
        const uint32_t gate_blocks = in_dim / 256;
        const uint64_t gate_row_bytes = (uint64_t)gate_blocks * 66;
        const uint64_t down_row_bytes = 8 * 84;      /* 8 Q2_K blocks */
        const uint64_t gate_expert_bytes = (uint64_t)mid_dim * gate_row_bytes;
        const uint64_t down_expert_bytes = (uint64_t)out_dim * down_row_bytes;
        const uint64_t header = 16;
        const uint64_t gate_offset = header;
        const uint64_t up_offset = gate_offset + (uint64_t)n_total * gate_expert_bytes;
        const uint64_t down_offset = up_offset + (uint64_t)n_total * gate_expert_bytes;
        const uint64_t model_size = down_offset + (uint64_t)n_total * down_expert_bytes;

        std::vector<uint8_t> model(model_size, 0xAA);
        for (uint32_t e = 0; e < n_total; e++) {
            for (uint32_t r = 0; r < mid_dim; r++) {
                for (uint32_t b = 0; b < gate_blocks; ++b) {
                    iq2_block blk;
                    blk.d = f32_to_f16(4.0f);        /* scale = 0.5 * (2*ls+1) */
                    for (uint32_t g = 0; g < 32; g += 4) {
                        const uint32_t l = g / 4;    /* 8-element group */
                        const uint32_t ls = (r + e + b) % 3;
                        uint32_t aux0 = 0, aux1 = ls << 28;
                        for (uint32_t t = 0; t < 4; t++) {
                            const uint32_t grid_idx =
                                (r * 7 + l * 11 + e * 3 + b * 19 + t * 17) % 256;
                            ((uint8_t *)&aux0)[t] = (uint8_t)grid_idx;
                            aux1 |= (((r * 5 + l * 13 + e * 7 + b * 3 + t) & 127) << (7 * t));
                        }
                        blk.qs[g + 0] = (uint16_t)(aux0 & 0xffffu);
                        blk.qs[g + 1] = (uint16_t)(aux0 >> 16);
                        blk.qs[g + 2] = (uint16_t)(aux1 & 0xffffu);
                        blk.qs[g + 3] = (uint16_t)(aux1 >> 16);
                    }
                    const uint64_t row_base = (uint64_t)e * gate_expert_bytes +
                                              (uint64_t)r * gate_row_bytes +
                                              (uint64_t)b * sizeof(blk);
                    std::memcpy(model.data() + gate_offset + row_base,
                                &blk, sizeof(blk));
                    std::memcpy(model.data() + up_offset + row_base,
                                &blk, sizeof(blk));
                }
            }
            for (uint32_t r2 = 0; r2 < out_dim; r2++) {
                for (uint32_t b = 0; b < 8; b++) {
                    q2k_block blk;
                    blk.d = f32_to_f16(0.125f);
                    blk.dmin = 0;
                    for (uint32_t g = 0; g < 16; g++) blk.scales[g] = 0x02;
                    for (uint32_t k = 0; k < 64; k++) {
                        uint8_t byte = 0;
                        for (uint32_t t = 0; t < 4; t++) {
                            const uint32_t i = b * 256 + k * 4 + t;
                            const uint32_t q = (i * 3 + r2 * 7 + e * 11) % 4;
                            byte |= (uint8_t)(q << (2 * t));
                        }
                        blk.qs[k] = byte;
                    }
                    std::memcpy(model.data() + down_offset + (uint64_t)e * down_expert_bytes +
                                (uint64_t)r2 * down_row_bytes + (uint64_t)b * sizeof(blk),
                                &blk, sizeof(blk));
                }
            }
        }

        std::vector<float> x(in_dim);
        for (uint32_t i = 0; i < in_dim; i++) x[i] = (float)((int)((i * 7 + 3) % 19) - 9) * 0.125f;
        std::vector<int32_t> sel = { 0, 1, 2, 3, 4, 5 };
        std::vector<float> wgt = { 0.20f, 0.18f, 0.17f, 0.16f, 0.15f, 0.14f };
        rc |= run_moe_case(production_shape
                               ? "iq2xxs-gate-q2k-down-production"
                               : "iq2xxs-gate-q2k-down",
                           16, 10, in_dim, mid_dim, out_dim,
                           n_total, n_expert, clamp, x, sel, wgt, model,
                           gate_offset, up_offset, down_offset,
                           gate_expert_bytes, gate_row_bytes,
                           down_expert_bytes, down_row_bytes, {},
                           !production_shape);

        if (!production_shape) {
            const uint64_t iq2_down_row_bytes = 8 * sizeof(iq2_block);
            const uint64_t iq2_down_expert_bytes =
                (uint64_t)out_dim * iq2_down_row_bytes;
            for (uint32_t e = 0; e < n_total; e++) {
                for (uint32_t row = 0; row < out_dim; row++) {
                    for (uint32_t block = 0; block < 8; block++) {
                        iq2_block value{};
                        value.d = f32_to_f16(0.125f);
                        std::memcpy(model.data() + down_offset +
                                        (uint64_t)e * iq2_down_expert_bytes +
                                        (uint64_t)row * iq2_down_row_bytes +
                                        (uint64_t)block * sizeof(value),
                                    &value, sizeof(value));
                    }
                }
            }
            rc |= run_moe_case("iq2xxs-all", 16, 16,
                               in_dim, mid_dim, out_dim,
                               n_total, n_expert, clamp, x, sel, wgt, model,
                               gate_offset, up_offset, down_offset,
                               gate_expert_bytes, gate_row_bytes,
                               iq2_down_expert_bytes, iq2_down_row_bytes, {});
        }
    }

    /* ============ Error paths ============================================= */
    {
        ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc(8 * sizeof(float));
        ds4_gpu_tensor *sel_t = ds4_gpu_tensor_alloc(2 * sizeof(int32_t));
        ds4_gpu_tensor *w_t   = ds4_gpu_tensor_alloc(2 * sizeof(float));
        ds4_gpu_tensor *x_t   = ds4_gpu_tensor_alloc(8 * sizeof(float));
        if (!out_t || !sel_t || !w_t || !x_t) {
            if (out_t) ds4_gpu_tensor_free(out_t);
            if (sel_t) ds4_gpu_tensor_free(sel_t);
            if (w_t)   ds4_gpu_tensor_free(w_t);
            if (x_t)   ds4_gpu_tensor_free(x_t);
            return 1;
        }
        /* Null tensors must return 0. */
        if (ds4_gpu_routed_moe_one_tensor(nullptr, nullptr, nullptr, nullptr, nullptr,
                                          nullptr, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0,
                                          8, 8, 8, nullptr, nullptr, 2, 2, 0.25f,
                                          nullptr, nullptr, 0, false) != 0) {
            fprintf(stderr, "routed_moe_one: null-pointer call should return 0\n");
            rc = 1;
        }
        /* Out-of-range selected expert must return 0. */
        unsigned char model[64] = {0};
        int32_t bad_sel[2] = { 0, 99 };
        float bad_w[2] = { 0.5f, 0.5f };
        float xv[8] = { 0 };
        if (ds4_gpu_set_model_map(model, sizeof(model)) == 0) { rc = 1; }
        else if (ds4_gpu_tensor_write(sel_t, 0, bad_sel, sizeof(bad_sel)) == 0 ||
                 ds4_gpu_tensor_write(w_t, 0, bad_w, sizeof(bad_w)) == 0 ||
                 ds4_gpu_tensor_write(x_t, 0, xv, sizeof(xv)) == 0) {
            rc = 1;
        } else if (ds4_gpu_routed_moe_one_tensor(out_t, out_t, out_t, out_t, out_t,
                                                 model, sizeof(model), 0, 0, 0,
                                                 8, 8, 272, 34, 272, 34,
                                                 8, 8, 8, sel_t, w_t, 2, 2, 0.25f,
                                                 x_t, nullptr, 0, false) != 0) {
            fprintf(stderr, "routed_moe_one: out-of-range expert should return 0\n");
            rc = 1;
        }
        ds4_gpu_tensor_free(out_t);
        ds4_gpu_tensor_free(sel_t);
        ds4_gpu_tensor_free(w_t);
        ds4_gpu_tensor_free(x_t);
    }

    return rc;
}
REGISTER_TEST(routed_moe_one, test_routed_moe_one);
