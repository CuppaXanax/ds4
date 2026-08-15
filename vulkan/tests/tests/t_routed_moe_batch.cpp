/* Kernel test: ds4_gpu_routed_moe_batch_tensor (host-side prefill routed MoE).
 *
 * The kernel is CPU-hosted over the model map and the host-mapped tensor
 * memory (like ds4_gpu_add_tensor), so no begin_commands/end_commands are
 * needed to read back the outputs.
 *
 * The batch API runs the exact single-token math
 * (ds4_gpu_routed_moe_one_tensor) once per token with batch-slot tensor
 * layout:
 *   selected[t*n_expert+e], weights[t*n_expert+e]
 *   gate/up/mid[t*n_expert*expert_mid_dim + e*expert_mid_dim + r]
 *   experts[t*n_expert*out_dim + e*out_dim + r2]
 *   out[t*out_dim + r2]
 * For every selected expert pair:
 *   gate = dequant(gate_row[e]) . x
 *   up   = dequant(up_row[e]) . x
 *   clamp (if clamp > 1e-6: gate=min(gate,clamp), up=clamp)
 *   mid  = silu(gate) * up * weight[e]              (weighted SwiGLU)
 *   out += dequant(down_row[e]) . requant_q(mid)    (down sees the weighted mid)
 *
 * Quant types covered:
 *   - Q8_0 (gate/up/down): f16 block scale + 32 int8, 34 B/block
 *   - Q2_K (gate/up/down): 256-elem blocks, 84 B/block (Q8_K activation)
 * The CPU reference below is a verbatim transcription of the ds4.c helpers
 * (quantize_q8_0_activation, ds4_quantize_row_q8_K, dot_q8_0_row,
 * ds4_vec_dot_q2_K_q8_K).
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

/* ---------------- Q8_K / Q2_K blocks (ds4.c) ------------------------------ */
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

static float vec_dot_iq2_fixture_q8_K(int n, const iq2_block *x, const q8k_block *y) {
    float total = 0.0f;
    for (int block = 0; block < n / (int)QK_K; block++) {
        const float d = f16_to_f32(x[block].d) * y[block].d;
        int sum = 0;
        for (int i = 0; i < (int)QK_K; i++)
            sum += 8 * y[block].qs[i];
        total += 0.125f * d * (float)sum;
    }
    return total;
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
    if (type == 10)
        return vec_dot_q2_K_q8_K((int)in_dim, (const q2k_block *)row, xqk.data());
    return vec_dot_iq2_fixture_q8_K((int)in_dim, (const iq2_block *)row, xqk.data());
}

/* ---------------- CPU reference (batch: mirror of the kernel) ------------- */
static void ref_moe_batch(
        uint32_t gate_type, uint32_t down_type,
        uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        uint32_t n_tokens,
        const std::vector<float> &x,
        const std::vector<int32_t> &sel,
        const std::vector<float> &wgt,
        const std::vector<uint8_t> &model,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        std::vector<float> &ref_gate, std::vector<float> &ref_up,
        std::vector<float> &ref_mid, std::vector<float> &ref_experts,
        std::vector<float> &ref_out)
{
    (void)n_total_expert;
    const uint64_t pair_stride = (uint64_t)n_expert * mid_dim;
    const uint64_t exp_stride  = (uint64_t)n_expert * out_dim;

    ref_gate.assign((uint64_t)n_tokens * pair_stride, 0.0f);
    ref_up.assign((uint64_t)n_tokens * pair_stride, 0.0f);
    ref_mid.assign((uint64_t)n_tokens * pair_stride, 0.0f);
    ref_experts.assign((uint64_t)n_tokens * exp_stride, 0.0f);
    ref_out.assign((uint64_t)n_tokens * out_dim, 0.0f);

    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *xt = x.data() + (uint64_t)t * in_dim;
        const int32_t *selt = sel.data() + (uint64_t)t * n_expert;
        const float *wgt_t = wgt.data() + (uint64_t)t * n_expert;
        float *gt = ref_gate.data() + (uint64_t)t * pair_stride;
        float *ut = ref_up.data() + (uint64_t)t * pair_stride;
        float *mt = ref_mid.data() + (uint64_t)t * pair_stride;
        float *et = ref_experts.data() + (uint64_t)t * exp_stride;
        float *ot = ref_out.data() + (uint64_t)t * out_dim;

        std::vector<int8_t> xq8;
        std::vector<float> xscale8;
        std::vector<q8k_block> xqk;
        if (gate_type == 8) {
            const uint64_t blocks = (in_dim + 31) / 32;
            xq8.resize(blocks * 32, 0);
            xscale8.resize(blocks);
            quantize_q8_0_activation(xt, xq8.data(), xscale8.data(), in_dim);
        } else {
            xqk.resize(in_dim / QK_K);
            quantize_row_q8_K(xt, xqk.data(), in_dim);
        }

        std::vector<float> midv(mid_dim);
        for (uint32_t e = 0; e < n_expert; e++) {
            const uint32_t expert = (uint32_t)selt[e];
            const float weight = wgt_t[e];
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
                gt[(uint64_t)e * mid_dim + r] = g;
                ut[(uint64_t)e * mid_dim + r] = u;
                mt[(uint64_t)e * mid_dim + r] = m;
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
                et[(uint64_t)e * out_dim + r2] = dv;
                ot[r2] += dv;
            }
        }
    }
}

/* ---------------- one end-to-end case ------------------------------------- */
static int run_moe_batch_case(
        const char *label,
        uint32_t gate_type, uint32_t down_type,
        uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        uint32_t n_tokens,
        const std::vector<float> &x,
        const std::vector<int32_t> &sel,
        const std::vector<float> &wgt,
        const std::vector<uint8_t> &model,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes)
{
    ds4_gpu_tensor *out_t  = ds4_gpu_tensor_alloc((uint64_t)n_tokens * out_dim * sizeof(float));
    ds4_gpu_tensor *gate_t = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_expert * mid_dim * sizeof(float));
    ds4_gpu_tensor *up_t   = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_expert * mid_dim * sizeof(float));
    ds4_gpu_tensor *mid_t  = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_expert * mid_dim * sizeof(float));
    ds4_gpu_tensor *exp_t  = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_expert * out_dim * sizeof(float));
    ds4_gpu_tensor *sel_t  = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_expert * sizeof(int32_t));
    ds4_gpu_tensor *w_t    = ds4_gpu_tensor_alloc((uint64_t)n_tokens * n_expert * sizeof(float));
    ds4_gpu_tensor *x_t    = ds4_gpu_tensor_alloc((uint64_t)n_tokens * in_dim * sizeof(float));
    auto freet = [&]() {
        if (out_t)  ds4_gpu_tensor_free(out_t);
        if (gate_t) ds4_gpu_tensor_free(gate_t);
        if (up_t)   ds4_gpu_tensor_free(up_t);
        if (mid_t)  ds4_gpu_tensor_free(mid_t);
        if (exp_t)  ds4_gpu_tensor_free(exp_t);
        if (sel_t)  ds4_gpu_tensor_free(sel_t);
        if (w_t)    ds4_gpu_tensor_free(w_t);
        if (x_t)    ds4_gpu_tensor_free(x_t);
    };
    if (!out_t || !gate_t || !up_t || !mid_t || !exp_t ||
        !sel_t || !w_t || !x_t) {
        freet();
        return 1;
    }
    if (ds4_gpu_set_model_map(model.data(), model.size()) == 0) { freet(); return 1; }
    if (ds4_gpu_tensor_write(sel_t, 0, sel.data(), (uint64_t)sel.size() * sizeof(int32_t)) == 0 ||
        ds4_gpu_tensor_write(w_t,   0, wgt.data(), (uint64_t)wgt.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(x_t,   0, x.data(),   (uint64_t)x.size() * sizeof(float)) == 0) {
        freet();
        return 1;
    }

    bool mid_f16 = true;
    int ok = ds4_gpu_routed_moe_batch_tensor(
            out_t, gate_t, up_t, mid_t, exp_t,
            model.data(), model.size(),
            gate_offset, up_offset, down_offset,
            gate_type, down_type,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim,
            sel_t, w_t, n_total_expert, n_expert,
            clamp, x_t, 0, n_tokens, &mid_f16, false);
    if (!ok) {
        fprintf(stderr, "routed_moe_batch[%s]: kernel returned 0\n", label);
        freet();
        return 1;
    }
    /* Host path always writes f32 mid. */
    if (mid_f16) {
        fprintf(stderr, "routed_moe_batch[%s]: mid_is_f16 should be false\n", label);
        freet();
        return 1;
    }

    std::vector<float> ref_gate, ref_up, ref_mid, ref_experts, ref_out;
    ref_moe_batch(gate_type, down_type, in_dim, mid_dim, out_dim,
                  n_total_expert, n_expert, clamp, n_tokens,
                  x, sel, wgt, model,
                  gate_offset, up_offset, down_offset,
                  gate_expert_bytes, gate_row_bytes,
                  down_expert_bytes, down_row_bytes,
                  ref_gate, ref_up, ref_mid, ref_experts, ref_out);

    std::vector<float> got_gate((uint64_t)n_tokens * n_expert * mid_dim);
    std::vector<float> got_up((uint64_t)n_tokens * n_expert * mid_dim);
    std::vector<float> got_mid((uint64_t)n_tokens * n_expert * mid_dim);
    std::vector<float> got_exp((uint64_t)n_tokens * n_expert * out_dim);
    std::vector<float> got_out((uint64_t)n_tokens * out_dim);
    bool read_ok =
        ds4_gpu_tensor_read(gate_t, 0, got_gate.data(), got_gate.size() * sizeof(float)) != 0 &&
        ds4_gpu_tensor_read(up_t,   0, got_up.data(),   got_up.size() * sizeof(float)) != 0 &&
        ds4_gpu_tensor_read(mid_t,  0, got_mid.data(),  got_mid.size() * sizeof(float)) != 0 &&
        ds4_gpu_tensor_read(exp_t,  0, got_exp.data(),  got_exp.size() * sizeof(float)) != 0 &&
        ds4_gpu_tensor_read(out_t,  0, got_out.data(),  got_out.size() * sizeof(float)) != 0;
    if (!read_ok) {
        fprintf(stderr, "routed_moe_batch[%s]: tensor read failed\n", label);
        freet();
        return 1;
    }

    const float tol = 1e-3f;
    auto cmp_vec = [&](const char *what, const std::vector<float> &got,
                       const std::vector<float> &ref, size_t n) -> bool {
        bool good = true;
        for (size_t i = 0; i < n; i++) {
            if (!(std::fabsf(got[i] - ref[i]) <= tol)) {
                if (good) fprintf(stderr, "--- routed_moe_batch[%s] %s mismatch ---\n", label, what);
                fprintf(stderr, "  [%zu] got=%.6f want=%.6f\n", i, got[i], ref[i]);
                good = false;
            }
        }
        return good;
    };
    bool ok_all = true;
    ok_all &= cmp_vec("gate", got_gate, ref_gate, got_gate.size());
    ok_all &= cmp_vec("up",   got_up,   ref_up,   got_up.size());
    ok_all &= cmp_vec("mid",  got_mid,  ref_mid,  got_mid.size());
    ok_all &= cmp_vec("experts", got_exp, ref_experts, got_exp.size());
    ok_all &= cmp_vec("out",  got_out,  ref_out,  got_out.size());
    if (!ok_all) {
        fprintf(stderr, "--- routed_moe_batch[%s] inputs ---\n", label);
        for (uint32_t t = 0; t < n_tokens; t++) {
            fprintf(stderr, "t%u x:", t);
            for (uint32_t i = 0; i < in_dim && i < 32; i++) fprintf(stderr, " %.3f", x[(uint64_t)t * in_dim + i]);
            fprintf(stderr, "\nt%u sel:", t);
            for (uint32_t i = 0; i < n_expert; i++) fprintf(stderr, " %d", sel[(uint64_t)t * n_expert + i]);
            fprintf(stderr, "\nt%u wgt:", t);
            for (uint32_t i = 0; i < n_expert; i++) fprintf(stderr, " %.3f", wgt[(uint64_t)t * n_expert + i]);
            fprintf(stderr, "\n");
        }
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

static int test_routed_moe_batch(void) {
    int rc = 0;

    /* ============ Case A: Q8_0 gate/up + Q8_0 down, small dims ============
     * Cross the backend's 256-token dispatch boundary with alternating routes. */
    {
        const uint32_t in_dim = 8, mid_dim = 8, out_dim = 8;
        const uint32_t n_total = 2, n_expert = 2;
        const uint32_t n_tokens = 257;
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

        std::vector<float> x((uint64_t)n_tokens * in_dim);
        std::vector<int32_t> sel((uint64_t)n_tokens * n_expert);
        std::vector<float> wgt((uint64_t)n_tokens * n_expert);
        for (uint32_t t = 0; t < n_tokens; t++) {
            for (uint32_t i = 0; i < in_dim; i++)
                x[(uint64_t)t * in_dim + i] =
                    0.125f * (float)((int32_t)((t * 11u + i * 7u) % 17u) - 8);
            sel[(uint64_t)t * n_expert] = (int32_t)(t & 1u);
            sel[(uint64_t)t * n_expert + 1u] = (int32_t)(1u - (t & 1u));
            wgt[(uint64_t)t * n_expert] = 0.6f;
            wgt[(uint64_t)t * n_expert + 1u] = 0.4f;
        }
        rc |= run_moe_batch_case("q8_0-small", 8, 8, in_dim, mid_dim, out_dim,
                                 n_total, n_expert, clamp, n_tokens,
                                 x, sel, wgt, model,
                                 gate_offset, up_offset, down_offset,
                                 gate_expert_bytes, row_bytes,
                                 down_expert_bytes, row_bytes);

        /* Clamp disabled must still match its own reference. */
        rc |= run_moe_batch_case("q8_0-small-noclamp", 8, 8, in_dim, mid_dim, out_dim,
                                 n_total, n_expert, 0.0f, n_tokens,
                                 x, sel, wgt, model,
                                 gate_offset, up_offset, down_offset,
                                 gate_expert_bytes, row_bytes,
                                 down_expert_bytes, row_bytes);
    }

    /* ============ Case B: Q2_K gate/up + Q2_K down =========================
     * in=256, mid=256, out=8 (QK_K-aligned blocks, 84 B each), 2 tokens. */
    {
        const uint32_t in_dim = 256, mid_dim = 2048, out_dim = 1024;
        const uint32_t n_total = 2, n_expert = 2;
        const uint32_t n_tokens = 2;
        const float clamp = 0.25f;
        const uint64_t gate_row_bytes = 66;          /* 1 IQ2_XXS block */
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
                iq2_block blk{};
                blk.d = f32_to_f16(4.0f);
                for (uint32_t k = 0; k < 32; k += 4) {
                    blk.qs[k + 0] = 0;
                    blk.qs[k + 1] = 0;
                    blk.qs[k + 2] = 0;
                    blk.qs[k + 3] = 0;
                }
                std::memcpy(model.data() + gate_offset + (uint64_t)e * gate_expert_bytes + (uint64_t)r * gate_row_bytes,
                            &blk, sizeof(blk));
                std::memcpy(model.data() + up_offset + (uint64_t)e * gate_expert_bytes + (uint64_t)r * gate_row_bytes,
                            &blk, sizeof(blk));
            }
            for (uint32_t r2 = 0; r2 < out_dim; r2++) {
                for (uint32_t b = 0; b < 8; b++) {
                    q2k_block blk;
                    blk.d = f32_to_f16(0.125f);
                    blk.dmin = 0;
                    for (uint32_t g = 0; g < 16; g++) blk.scales[g] = 0x02;   /* scale 2 */
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

        std::vector<float> x(2 * in_dim);
        for (uint32_t t = 0; t < 2; t++)
            for (uint32_t i = 0; i < in_dim; i++)
                x[(uint64_t)t * in_dim + i] = (float)((int)((i * 13 + 5 + t * 7) % 17) - 8) * 0.125f;
        std::vector<int32_t> sel = { 1, 0,   0, 1 };
        std::vector<float> wgt = { 0.6f, 0.4f,   0.5f, 0.5f };
        rc |= run_moe_batch_case("iq2xxs-gate-q2k-down", 16, 10, in_dim, mid_dim, out_dim,
                                 n_total, n_expert, clamp, n_tokens,
                                 x, sel, wgt, model,
                                 gate_offset, up_offset, down_offset,
                                 gate_expert_bytes, gate_row_bytes,
                                 down_expert_bytes, down_row_bytes);

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
        rc |= run_moe_batch_case("iq2xxs-all", 16, 16,
                                 in_dim, mid_dim, out_dim,
                                 n_total, n_expert, clamp, n_tokens,
                                 x, sel, wgt, model,
                                 gate_offset, up_offset, down_offset,
                                 gate_expert_bytes, gate_row_bytes,
                                 iq2_down_expert_bytes, iq2_down_row_bytes);
    }

    /* ============ Error paths ============================================= */
    {
        ds4_gpu_tensor *out_t = ds4_gpu_tensor_alloc(8 * sizeof(float));
        ds4_gpu_tensor *sel_t = ds4_gpu_tensor_alloc(4 * sizeof(int32_t)); /* 2 tok x 2 exp */
        ds4_gpu_tensor *w_t   = ds4_gpu_tensor_alloc(4 * sizeof(float));
        ds4_gpu_tensor *x_t   = ds4_gpu_tensor_alloc(8 * sizeof(float));
        if (!out_t || !sel_t || !w_t || !x_t) {
            if (out_t) ds4_gpu_tensor_free(out_t);
            if (sel_t) ds4_gpu_tensor_free(sel_t);
            if (w_t)   ds4_gpu_tensor_free(w_t);
            if (x_t)   ds4_gpu_tensor_free(x_t);
            return 1;
        }
        /* Null tensors must return 0. */
        if (ds4_gpu_routed_moe_batch_tensor(nullptr, nullptr, nullptr, nullptr, nullptr,
                                            nullptr, 0, 0, 0, 0, 8, 8, 0, 0, 0, 0,
                                            8, 8, 8, nullptr, nullptr, 2, 2, 0.25f,
                                            nullptr, 0, 1, nullptr, false) != 0) {
            fprintf(stderr, "routed_moe_batch: null-pointer call should return 0\n");
            rc = 1;
        }
        /* Out-of-range selected expert must return 0. */
        unsigned char model[64] = {0};
        int32_t bad_sel[4] = { 0, 99, 1, 0 };        /* token0 slot1 -> expert 99 */
        float bad_w[4] = { 0.5f, 0.5f, 0.5f, 0.5f };
        float xv[8] = { 0 };
        if (ds4_gpu_set_model_map(model, sizeof(model)) == 0) { rc = 1; }
        else if (ds4_gpu_tensor_write(sel_t, 0, bad_sel, sizeof(bad_sel)) == 0 ||
                 ds4_gpu_tensor_write(w_t, 0, bad_w, sizeof(bad_w)) == 0 ||
                 ds4_gpu_tensor_write(x_t, 0, xv, sizeof(xv)) == 0) {
            rc = 1;
        } else if (ds4_gpu_routed_moe_batch_tensor(out_t, out_t, out_t, out_t, out_t,
                                                   model, sizeof(model), 0, 0, 0,
                                                   8, 8, 272, 34, 272, 34,
                                                   8, 8, 8, sel_t, w_t, 2, 2, 0.25f,
                                                   x_t, 0, 2, nullptr, false) != 0) {
            fprintf(stderr, "routed_moe_batch: out-of-range expert should return 0\n");
            rc = 1;
        }
        ds4_gpu_tensor_free(out_t);
        ds4_gpu_tensor_free(sel_t);
        ds4_gpu_tensor_free(w_t);
        ds4_gpu_tensor_free(x_t);
    }

    return rc;
}
REGISTER_TEST(routed_moe_batch, test_routed_moe_batch);
