/* Kernel tests: shared expert (DS4 layer-FFN half that runs for every token).
 *
 * These entry points are host-side (CPU over the model map and host-mapped
 * tensor memory, like ds4_gpu_add_tensor / ds4_gpu_routed_moe_one_tensor),
 * so no begin_commands/end_commands are needed to read back the outputs.
 *
 * Covered functions:
 *   ds4_gpu_shared_gate_up_swiglu_q8_0_tensor  (gate + up + mid)
 *   ds4_gpu_shared_mid_swiglu_q8_0_tensor      (mid only)
 *   ds4_gpu_shared_down_hc_expand_q8_0_tensor  (shared down + HC expand)
 *
 * The CPU reference is a verbatim transcription of the ds4.c helpers
 * (quantize_q8_0_activation, dot_q8_0_row, sigmoid_stable/silu, swiglu,
 * hc_post_one).  Weights are GGUF Q8_0 (f16 block scale + 32 int8 quants,
 * 34 B/block); the model buffer mimics a GGUF-style file with a junk header
 * and the raw weight ranges at known offsets.
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

/* ds4.c swiglu(): clamp before silu.  gate is clamped positive-only, up is
 * clamped symmetrically (the fused Metal kernel does the same). */
static void swiglu_ref(float *out, const float *gate, const float *up,
                       uint64_t n, float clamp) {
    for (uint64_t i = 0; i < n; i++) {
        float g = gate[i];
        float u = up[i];
        if (clamp > 1.0e-6f) {
            if (g > clamp) g = clamp;
            if (u > clamp) u = clamp;
            if (u < -clamp) u = -clamp;
        }
        out[i] = silu(g) * u;
    }
}

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

/* ---------------- synthetic Q8_0 weight rows ----------------------------- */
static void fill_q8_row(uint8_t *dst, uint32_t n_elem, float scale,
                        uint32_t row, uint32_t kind) {
    uint16_t d16 = f32_to_f16(scale);
    for (uint32_t block = 0; block < (n_elem + 31u) / 32u; block++) {
        std::memcpy(dst + block * 34u, &d16, 2);
        int8_t *qs = (int8_t *)(dst + block * 34u + 2u);
        for (uint32_t i = 0; i < 32u; i++) {
            const uint32_t index = block * 32u + i;
            const int v = (int)(((index * 7 + row * 13 + kind * 11) % 65) - 32);
            qs[i] = (int8_t)v;
        }
    }
}

/* =========================================================================
 * ds4_gpu_shared_gate_up_swiglu_q8_0_tensor
 * ========================================================================= */
static int run_gate_up_case(const char *label,
                            uint64_t in_dim, uint64_t out_dim, float clamp,
                            bool store_mid_only) {
    const uint64_t n_blocks  = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = n_blocks * 34u;      /* GGUF Q8_0: 34 B/block */
    const uint64_t w_bytes   = out_dim * row_bytes;
    const uint64_t header    = 4096;                /* avoid weight-cache aliasing */
    const uint64_t gate_offset = header;
    const uint64_t up_offset   = gate_offset + w_bytes;
    const uint64_t model_size  = up_offset + w_bytes;

    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    ds4_gpu_tensor *up   = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    ds4_gpu_tensor *mid  = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    ds4_gpu_tensor *x    = ds4_gpu_tensor_alloc(in_dim * sizeof(float));
    if (!gate || !up || !mid || !x) {
        if (gate) ds4_gpu_tensor_free(gate);
        if (up)   ds4_gpu_tensor_free(up);
        if (mid)  ds4_gpu_tensor_free(mid);
        if (x)    ds4_gpu_tensor_free(x);
        return 1;
    }

    /* Synthetic "model file": junk header + gate/up Q8_0 matrices. */
    std::vector<unsigned char> model(model_size, 0xAA);
    for (uint64_t o = 0; o < out_dim; o++) {
        fill_q8_row(model.data() + gate_offset + o * row_bytes,
                    (uint32_t)in_dim, (float)((o % 5) + 1) * 0.25f, (uint32_t)o, 0);
        fill_q8_row(model.data() + up_offset + o * row_bytes,
                    (uint32_t)in_dim, (float)((o % 3) + 1) * 0.5f, (uint32_t)o, 1);
    }
    if (ds4_gpu_set_model_map(model.data(), model.size()) == 0) {
        ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up);
        ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(x);
        return 1;
    }

    /* Known input activation (f32). */
    std::vector<float> xv(in_dim);
    for (uint64_t i = 0; i < in_dim; i++)
        xv[i] = (float)((int)((i * 5 + 1) % 21) - 10) * 0.125f;
    if (ds4_gpu_tensor_write(x, 0, xv.data(), xv.size() * sizeof(float)) == 0) {
        ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up);
        ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(x);
        return 1;
    }

    /* CPU reference: quantize x to Q8_0, dequant gate/up rows, clamp, swiglu
     * (verbatim ds4.c swiglu + quantize + dot helpers). */
    std::vector<int8_t>  xq(n_blocks * 32);
    std::vector<float>   xscale(n_blocks);
    quantize_q8_0_activation(xv.data(), xq.data(), xscale.data(), in_dim);

    std::vector<float> ref_gate(out_dim), ref_up(out_dim), ref_mid(out_dim);
    for (uint64_t o = 0; o < out_dim; o++) {
        ref_gate[o] = dot_q8_0_row(model.data() + gate_offset + o * row_bytes,
                                   xq.data(), xscale.data(), in_dim, n_blocks);
        ref_up[o]   = dot_q8_0_row(model.data() + up_offset + o * row_bytes,
                                   xq.data(), xscale.data(), in_dim, n_blocks);
    }
    swiglu_ref(ref_mid.data(), ref_gate.data(), ref_up.data(), out_dim, clamp);

    int ok = 0;
    if (store_mid_only) {
        ok = ds4_gpu_shared_mid_swiglu_q8_0_tensor(
                mid, model.data(), model.size(), gate_offset, up_offset,
                in_dim, out_dim, x, clamp);
    } else {
        ok = ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                gate, up, mid, model.data(), model.size(), gate_offset, up_offset,
                in_dim, out_dim, x, clamp);
    }
    if (!ok) {
        fprintf(stderr, "shared_expert[%s]: kernel returned 0\n", label);
        ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up);
        ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(x);
        return 1;
    }

    std::vector<float> got_gate(out_dim), got_up(out_dim), got_mid(out_dim);
    bool read_ok = true;
    if (store_mid_only) {
        read_ok = ds4_gpu_tensor_read(mid, 0, got_mid.data(), got_mid.size() * sizeof(float)) != 0;
    } else {
        read_ok = ds4_gpu_tensor_read(gate, 0, got_gate.data(), got_gate.size() * sizeof(float)) != 0 &&
                  ds4_gpu_tensor_read(up,   0, got_up.data(),   got_up.size() * sizeof(float)) != 0 &&
                  ds4_gpu_tensor_read(mid,  0, got_mid.data(),  got_mid.size() * sizeof(float)) != 0;
    }
    if (!read_ok) {
        fprintf(stderr, "shared_expert[%s]: tensor read failed\n", label);
        ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up);
        ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(x);
        return 1;
    }

    const float tol = 1.0e-3f;
    auto cmp = [&](const char *what, const std::vector<float> &got,
                   const std::vector<float> &ref, size_t n) -> bool {
        bool good = true;
        for (size_t i = 0; i < n; i++) {
            const float g = got[i], r = ref[i];
            bool close;
            if (std::isnan(g) && std::isnan(r)) close = true;
            else if (std::isinf(g) || std::isinf(r)) close = (g == r);
            else close = std::fabsf(g - r) <= tol;
            if (!close) {
                if (good) fprintf(stderr, "--- shared_expert[%s] %s mismatch ---\n", label, what);
                fprintf(stderr, "  [%zu] got=%.6f want=%.6f\n", i, g, r);
                good = false;
            }
        }
        return good;
    };

    bool ok_all = true;
    ok_all &= cmp("mid", got_mid, ref_mid, out_dim);
    if (!store_mid_only) {
        ok_all &= cmp("gate", got_gate, ref_gate, out_dim);
        ok_all &= cmp("up",   got_up,   ref_up,   out_dim);
    }
    if (!ok_all) {
        fprintf(stderr, "--- shared_expert[%s] inputs ---\n", label);
        fprintf(stderr, "x: ");
        for (uint64_t i = 0; i < in_dim && i < 16; i++) fprintf(stderr, "%.3f ", xv[i]);
        fprintf(stderr, "\nclamp=%.3f in_dim=%llu out_dim=%llu\n",
                clamp, (unsigned long long)in_dim, (unsigned long long)out_dim);
    }

    ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(x);
    return ok_all ? 0 : 1;
}

static int test_shared_gate_up_swiglu_q8_0(void) {
    int rc = 0;
    rc |= run_gate_up_case("gate-up-64x8-clamp",  64, 8,  0.5f, false);
    rc |= run_gate_up_case("gate-up-64x8-noclamp", 64, 8,  0.0f, false);
    rc |= run_gate_up_case("gate-up-128x16",       128, 16, 0.25f, false);

    /* Error paths: null tensors and out-of-range weight offsets must fail. */
    {
        ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(8 * sizeof(float));
        ds4_gpu_tensor *x   = ds4_gpu_tensor_alloc(64 * sizeof(float));
        if (!mid || !x) {
            if (mid) ds4_gpu_tensor_free(mid);
            if (x)   ds4_gpu_tensor_free(x);
            return 1;
        }
        if (ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 64, 8,
                nullptr, 0.5f) != 0) {
            fprintf(stderr, "shared_gate_up: null-pointer call should return 0\n");
            rc = 1;
        }
        unsigned char model[4096 + 34] = {0};
        if (ds4_gpu_set_model_map(model, sizeof(model)) == 0) rc = 1;
        /* weight range (4096+34) extends past model_size -> must fail. */
        if (ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
                x, x, mid, model, sizeof(model), 4096, 4096, 64, 8,
                x, 0.5f) != 0) {
            fprintf(stderr, "shared_gate_up: out-of-range weights should return 0\n");
            rc = 1;
        }
        ds4_gpu_tensor_free(mid);
        ds4_gpu_tensor_free(x);
    }
    return rc;
}
REGISTER_TEST(shared_gate_up_swiglu_q8_0, test_shared_gate_up_swiglu_q8_0);

static int test_shared_mid_swiglu_q8_0(void) {
    return run_gate_up_case("mid-only-64x8", 64, 8, 0.5f, true);
}
REGISTER_TEST(shared_mid_swiglu_q8_0, test_shared_mid_swiglu_q8_0);

/* =========================================================================
 * ds4_gpu_shared_down_hc_expand_q8_0_tensor
 * ========================================================================= */
static int run_down_hc_case(const char *label,
                            uint64_t mid_dim, uint32_t n_embd, uint32_t n_hc,
                            float clamp_unused) {
    (void)clamp_unused;
    const uint64_t n_blocks  = (mid_dim + 31u) / 32u;
    const uint64_t row_bytes = n_blocks * 34u;
    const uint64_t w_bytes   = (uint64_t)n_embd * row_bytes;
    const uint64_t header    = 4096;
    const uint64_t weight_offset = header;
    const uint64_t model_size    = header + w_bytes;

    ds4_gpu_tensor *out_hc     = ds4_gpu_tensor_alloc((uint64_t)n_hc * n_embd * sizeof(float));
    ds4_gpu_tensor *shared_out = ds4_gpu_tensor_alloc((uint64_t)n_embd * sizeof(float));
    ds4_gpu_tensor *shared_mid = ds4_gpu_tensor_alloc(mid_dim * sizeof(float));
    ds4_gpu_tensor *routed_out = ds4_gpu_tensor_alloc((uint64_t)n_embd * sizeof(float));
    ds4_gpu_tensor *residual_hc = ds4_gpu_tensor_alloc((uint64_t)n_hc * n_embd * sizeof(float));
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    ds4_gpu_tensor *split  = ds4_gpu_tensor_alloc(mix_hc * sizeof(float));
    if (!out_hc || !shared_out || !shared_mid || !routed_out || !residual_hc || !split) {
        if (out_hc) ds4_gpu_tensor_free(out_hc);
        if (shared_out) ds4_gpu_tensor_free(shared_out);
        if (shared_mid) ds4_gpu_tensor_free(shared_mid);
        if (routed_out) ds4_gpu_tensor_free(routed_out);
        if (residual_hc) ds4_gpu_tensor_free(residual_hc);
        if (split) ds4_gpu_tensor_free(split);
        return 1;
    }

    /* Synthetic model: junk header + Q8_0 down matrix (n_embd x mid_dim). */
    std::vector<unsigned char> model(model_size, 0xAA);
    for (uint32_t d = 0; d < n_embd; d++) {
        fill_q8_row(model.data() + weight_offset + (uint64_t)d * row_bytes,
                    (uint32_t)mid_dim, (float)((d % 4) + 1) * 0.125f, d, 2);
    }
    if (ds4_gpu_set_model_map(model.data(), model.size()) == 0) {
        ds4_gpu_tensor_free(out_hc); ds4_gpu_tensor_free(shared_out);
        ds4_gpu_tensor_free(shared_mid); ds4_gpu_tensor_free(routed_out);
        ds4_gpu_tensor_free(residual_hc); ds4_gpu_tensor_free(split);
        return 1;
    }

    std::vector<float> midv(mid_dim), routed(n_embd), resid((uint64_t)n_hc * n_embd);
    std::vector<float> splitv(mix_hc);
    for (uint64_t i = 0; i < mid_dim; i++)
        midv[i] = (float)((int)((i * 3 + 2) % 19) - 9) * 0.125f;
    for (uint32_t d = 0; d < n_embd; d++)
        routed[d] = (float)((int)(d * 7 % 11) - 5) * 0.25f;
    for (uint32_t h = 0; h < n_hc; h++)
        for (uint32_t d = 0; d < n_embd; d++)
            resid[(uint64_t)h * n_embd + d] = (float)((int)((h * 13 + d * 5) % 17) - 8) * 0.125f;
    for (uint64_t i = 0; i < mix_hc; i++)
        splitv[i] = (float)((int)(i * 11 % 23) - 11) * 0.0625f;

    if (ds4_gpu_tensor_write(shared_mid, 0, midv.data(), midv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(routed_out, 0, routed.data(), routed.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(residual_hc, 0, resid.data(), resid.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(split, 0, splitv.data(), splitv.size() * sizeof(float)) == 0) {
        ds4_gpu_tensor_free(out_hc); ds4_gpu_tensor_free(shared_out);
        ds4_gpu_tensor_free(shared_mid); ds4_gpu_tensor_free(routed_out);
        ds4_gpu_tensor_free(residual_hc); ds4_gpu_tensor_free(split);
        return 1;
    }

    /* CPU reference: quantize mid to Q8_0, dequant down rows, then hc_post
     * (hc_post_one from ds4.c, comb addressed [dst_hc, src_hc]). */
    std::vector<int8_t> midq(n_blocks * 32);
    std::vector<float> midscale(n_blocks);
    quantize_q8_0_activation(midv.data(), midq.data(), midscale.data(), mid_dim);

    std::vector<float> ref_shared(n_embd), ref_out((uint64_t)n_hc * n_embd);
    const float *postp = splitv.data() + n_hc;
    const float *combp = splitv.data() + 2 * (uint64_t)n_hc;
    for (uint32_t d = 0; d < n_embd; d++) {
        const float shared_v = dot_q8_0_row(model.data() + weight_offset + (uint64_t)d * row_bytes,
                                            midq.data(), midscale.data(), mid_dim, n_blocks);
        ref_shared[d] = shared_v;
        const float block_v = routed[d] + shared_v;
        for (uint32_t dst = 0; dst < n_hc; dst++) {
            float acc = block_v * postp[dst];
            for (uint32_t src = 0; src < n_hc; src++) {
                acc += combp[dst + (uint64_t)src * n_hc] * resid[(uint64_t)src * n_embd + d];
            }
            ref_out[(uint64_t)dst * n_embd + d] = acc;
        }
    }

    int ok = ds4_gpu_shared_down_hc_expand_q8_0_tensor(
            out_hc, shared_out, model.data(), model.size(), weight_offset,
            mid_dim, n_embd, shared_mid, routed_out, residual_hc, split,
            n_embd, n_hc);
    if (!ok) {
        fprintf(stderr, "shared_down_hc[%s]: kernel returned 0\n", label);
        ds4_gpu_tensor_free(out_hc); ds4_gpu_tensor_free(shared_out);
        ds4_gpu_tensor_free(shared_mid); ds4_gpu_tensor_free(routed_out);
        ds4_gpu_tensor_free(residual_hc); ds4_gpu_tensor_free(split);
        return 1;
    }

    std::vector<float> got_shared(n_embd), got_out((uint64_t)n_hc * n_embd);
    if (ds4_gpu_tensor_read(shared_out, 0, got_shared.data(), got_shared.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_read(out_hc, 0, got_out.data(), got_out.size() * sizeof(float)) == 0) {
        fprintf(stderr, "shared_down_hc[%s]: tensor read failed\n", label);
        ds4_gpu_tensor_free(out_hc); ds4_gpu_tensor_free(shared_out);
        ds4_gpu_tensor_free(shared_mid); ds4_gpu_tensor_free(routed_out);
        ds4_gpu_tensor_free(residual_hc); ds4_gpu_tensor_free(split);
        return 1;
    }

    const float tol = 1.0e-3f;
    auto cmp = [&](const char *what, const std::vector<float> &got,
                   const std::vector<float> &ref, size_t n) -> bool {
        bool good = true;
        for (size_t i = 0; i < n; i++) {
            const float g = got[i], r = ref[i];
            bool close;
            if (std::isnan(g) && std::isnan(r)) close = true;
            else if (std::isinf(g) || std::isinf(r)) close = (g == r);
            else close = std::fabsf(g - r) <= tol;
            if (!close) {
                if (good) fprintf(stderr, "--- shared_down_hc[%s] %s mismatch ---\n", label, what);
                fprintf(stderr, "  [%zu] got=%.6f want=%.6f\n", i, g, r);
                good = false;
            }
        }
        return good;
    };

    bool ok_all = true;
    ok_all &= cmp("shared", got_shared, ref_shared, n_embd);
    ok_all &= cmp("out_hc", got_out, ref_out, got_out.size());
    if (!ok_all) {
        fprintf(stderr, "--- shared_down_hc[%s] inputs ---\n", label);
        fprintf(stderr, "mid_dim=%llu n_embd=%u n_hc=%u\n",
                (unsigned long long)mid_dim, n_embd, n_hc);
    }

    ds4_gpu_tensor_free(out_hc); ds4_gpu_tensor_free(shared_out);
    ds4_gpu_tensor_free(shared_mid); ds4_gpu_tensor_free(routed_out);
    ds4_gpu_tensor_free(residual_hc); ds4_gpu_tensor_free(split);
    return ok_all ? 0 : 1;
}

static int test_shared_down_hc_expand_q8_0(void) {
    int rc = 0;
    rc |= run_down_hc_case("down-hc-8x8x4",  8, 8, 4, 0.0f);
    rc |= run_down_hc_case("down-hc-32x16x4", 32, 16, 4, 0.0f);

    /* Error paths: null tensors and out-of-range weights must fail. */
    {
        ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(32 * sizeof(float));
        ds4_gpu_tensor *o   = ds4_gpu_tensor_alloc(4 * 8 * sizeof(float));
        if (!mid || !o) {
            if (mid) ds4_gpu_tensor_free(mid);
            if (o)   ds4_gpu_tensor_free(o);
            return 1;
        }
        if (ds4_gpu_shared_down_hc_expand_q8_0_tensor(
                nullptr, nullptr, nullptr, 0, 0, 32, 8,
                nullptr, nullptr, nullptr, nullptr, 8, 4) != 0) {
            fprintf(stderr, "shared_down_hc: null-pointer call should return 0\n");
            rc = 1;
        }
        unsigned char model[4096 + 34] = {0};
        if (ds4_gpu_set_model_map(model, sizeof(model)) == 0) rc = 1;
        if (ds4_gpu_shared_down_hc_expand_q8_0_tensor(
                o, o, model, sizeof(model), 4096, 32, 8,
                mid, o, o, o, 8, 4) != 0) {
            fprintf(stderr, "shared_down_hc: out-of-range weights should return 0\n");
            rc = 1;
        }
        ds4_gpu_tensor_free(mid);
        ds4_gpu_tensor_free(o);
    }
    return rc;
}
REGISTER_TEST(shared_down_hc_expand_q8_0, test_shared_down_hc_expand_q8_0);
