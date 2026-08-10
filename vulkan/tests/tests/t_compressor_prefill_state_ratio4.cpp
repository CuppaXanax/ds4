/* Kernel test: ds4_gpu_compressor_prefill_state_ratio4_tensor (host-side in
 * the backend).
 *
 * The kernel re-initializes the rolling compressor state of a ratio-4 layer
 * from the 4 projected tail rows (the last complete prefill window):
 *
 *   state_kv[0..7]        = 0
 *   state_score[0..7]     = -INFINITY
 *   for r in [0,4): dst = r (attention lane), src = r, phase = (pos0 + r) % 4
 *     state_kv[dst][j]    = kv_tail[src][j]
 *     state_score[dst][j] = sc_tail[src][j] + ape[phase * width + j]
 *
 * with width = 2 * head_dim, state_rows = 8.  The indexer lane (rows 4..7)
 * stays empty until the decoder stores into it.  This matches the CUDA
 * compressor_set_rows_kernel call (src0=0, dst0=0, rows=4) and the Metal
 * compressor_set_rows_projected path (dst rows {0,1,2,3}).
 *
 * The kernel is host-side, so NO begin_commands/end_commands are needed.
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include "../../ds4_gpu_mgpu.h"   /* complete ds4_gpu_tensor for stack views */
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int check_f32(const char *what, const float *got, const float *want, uint32_t n) {
    int bad = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float g = got[i], w = want[i];
        bool ok;
        if (std::isnan(g) && std::isnan(w)) ok = true;
        else if (std::isinf(g) || std::isinf(w)) ok = (g == w);
        else ok = std::fabsf(g - w) <= 1e-6f;
        if (!ok) {
            if (bad < 8)
                fprintf(stderr, "--- %s mismatch[%u]: got %.9g want %.9g\n",
                        what, i, g, w);
            bad++;
        }
    }
    if (bad) fprintf(stderr, "--- %s: %d mismatches\n", what, bad);
    return bad ? 1 : 0;
}

/* Deterministic synthetic values: alternating signs, in-range. */
static void synth_vals(float *v, uint64_t n) {
    for (uint64_t i = 0; i < n; i++) {
        float f = (float)(i + 1) * 0.375f + 0.125f * (float)(i % 7);
        if ((i & 1u) != 0) f = -f;
        v[i] = f;
    }
}

/* f16 round trip helpers (ds4.c f32_to_f16 / f16_to_f32). */
static uint16_t ref_f32_to_f16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) {
        if (((bits >> 23) & 0xffu) == 0xffu && mant != 0) return (uint16_t)(sign | 0x7e00u);
        return (uint16_t)(sign | 0x7c00u);
    }
    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half & 1u))) half++;
    return (uint16_t)half;
}

static float ref_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t expo = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (expo == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
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
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* ---------- model map builder (APE only) ---------- */

struct Ratio4Model {
    std::vector<unsigned char> data;
    uint64_t ape_offset;
};

/* APE is laid out as [ratio][width] f32/f16 (flat index = phase * width + j),
 * exactly the layout the CUDA kernels and the GGUF column-major tensor use. */
static Ratio4Model build_ratio4_model(uint32_t head_dim, uint32_t ape_type) {
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint64_t elem = ape_type == 1u ? 2u : 4u;
    const uint64_t header = 16;
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem;
    Ratio4Model m;
    m.data.assign(header + ape_bytes, 0x00);
    m.ape_offset = header;
    for (uint32_t ph = 0; ph < ratio; ph++) {
        for (uint32_t j = 0; j < width; j++) {
            const float v = 0.5f * (float)(ph + 1) + 0.125f * (float)(j % 5);
            if (ape_type == 1u) {
                const uint16_t h = ref_f32_to_f16(v);
                memcpy(m.data.data() + m.ape_offset + ((uint64_t)ph * width + j) * 2, &h, 2);
            } else {
                memcpy(m.data.data() + m.ape_offset + ((uint64_t)ph * width + j) * 4, &v, 4);
            }
        }
    }
    return m;
}

static float ref_ape_val(const unsigned char *model, uint64_t ape_offset,
                         uint32_t ape_type, uint64_t idx) {
    if (ape_type == 1u) {
        uint16_t h;
        memcpy(&h, model + ape_offset + idx * 2, 2);
        return ref_f16_to_f32(h);
    }
    float v;
    memcpy(&v, model + ape_offset + idx * 4, 4);
    return v;
}

/* ---------- CPU reference ---------- */

/* compressor_set_rows_kernel with src0=0, dst0=0, rows=ratio, on top of a
 * zeroed / -INFINITY state (the exact prefill_state_ratio4 semantics). */
static void ref_prefill_state_ratio4(std::vector<float> &skv, std::vector<float> &ssc,
                                     const std::vector<float> &kv_tail,
                                     const std::vector<float> &sc_tail,
                                     const unsigned char *model, uint64_t ape_offset,
                                     uint32_t ape_type, uint32_t head_dim,
                                     uint32_t pos0) {
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint64_t state_n = (uint64_t)state_rows * width;
    skv.assign(state_n, 0.0f);
    ssc.assign(state_n, -INFINITY);
    for (uint32_t r = 0; r < ratio; r++) {
        const uint32_t src = r;
        const uint32_t dst = r;
        const uint32_t phase = (pos0 + src) % ratio;
        for (uint32_t j = 0; j < width; j++) {
            skv[(uint64_t)dst * width + j] = kv_tail[(uint64_t)src * width + j];
            ssc[(uint64_t)dst * width + j] = sc_tail[(uint64_t)src * width + j] +
                ref_ape_val(model, ape_offset, ape_type, (uint64_t)phase * width + j);
        }
    }
}

static int run_ratio4_case(const char *what, uint32_t head_dim, uint32_t pos0,
                           uint32_t ape_type) {
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    Ratio4Model m = build_ratio4_model(head_dim, ape_type);
    if (!ds4_gpu_set_model_map(m.data.data(), m.data.size())) return 1;

    ds4_gpu_tensor *kv_tail = ds4_gpu_tensor_alloc((uint64_t)ratio * width * sizeof(float));
    ds4_gpu_tensor *sc_tail = ds4_gpu_tensor_alloc((uint64_t)ratio * width * sizeof(float));
    ds4_gpu_tensor *skv = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *ssc = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    if (!kv_tail || !sc_tail || !skv || !ssc) {
        if (kv_tail) ds4_gpu_tensor_free(kv_tail);
        if (sc_tail) ds4_gpu_tensor_free(sc_tail);
        if (skv) ds4_gpu_tensor_free(skv);
        if (ssc) ds4_gpu_tensor_free(ssc);
        return 1;
    }
    std::vector<float> kvv((uint64_t)ratio * width), scv((uint64_t)ratio * width);
    synth_vals(kvv.data(), kvv.size());
    synth_vals(scv.data(), scv.size());
    /* Garbage in the state: the kernel must fully overwrite rows 0..3 and
     * zero/-INF rows 4..7. */
    if (ds4_gpu_tensor_write(kv_tail, 0, kvv.data(), kvv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(sc_tail, 0, scv.data(), scv.size() * sizeof(float)) == 0 ||
        ds4_gpu_tensor_fill_f32(skv, 123.0f, (uint64_t)state_rows * width) == 0 ||
        ds4_gpu_tensor_fill_f32(ssc, 123.0f, (uint64_t)state_rows * width) == 0) {
        ds4_gpu_tensor_free(kv_tail); ds4_gpu_tensor_free(sc_tail);
        ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc);
        return 1;
    }

    std::vector<float> wskv, wssc;
    ref_prefill_state_ratio4(wskv, wssc, kvv, scv, m.data.data(), m.ape_offset,
                             ape_type, head_dim, pos0);

    int rc = 1;
    if (ds4_gpu_compressor_prefill_state_ratio4_tensor(
            skv, ssc, kv_tail, sc_tail, m.data.data(), m.data.size(),
            m.ape_offset, ape_type, head_dim, pos0) != 0) {
        std::vector<float> gkv((uint64_t)state_rows * width);
        std::vector<float> gsc((uint64_t)state_rows * width);
        if (ds4_gpu_tensor_read(skv, 0, gkv.data(), gkv.size() * sizeof(float)) != 0 &&
            ds4_gpu_tensor_read(ssc, 0, gsc.data(), gsc.size() * sizeof(float)) != 0) {
            if (check_f32(what, gkv.data(), wskv.data(), (uint32_t)gkv.size()) != 0) rc = 1;
            else if (check_f32("ratio4/state_score", gsc.data(), wssc.data(),
                               (uint32_t)gsc.size()) != 0) rc = 1;
            else rc = 0;
        }
    }
    ds4_gpu_tensor_free(kv_tail); ds4_gpu_tensor_free(sc_tail);
    ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc);
    return rc;
}

static int test_compressor_prefill_state_ratio4(void) {
    return run_ratio4_case("ratio4/state_kv", 8, 3, 0);
}
REGISTER_TEST(compressor_prefill_state_ratio4, test_compressor_prefill_state_ratio4);

/* f16 APE path with a different phase alignment (pos0 not multiple of 4). */
static int test_compressor_prefill_state_ratio4_f16(void) {
    return run_ratio4_case("ratio4_f16/state_kv", 16, 5, 1);
}
REGISTER_TEST(compressor_prefill_state_ratio4_f16, test_compressor_prefill_state_ratio4_f16);

/* Validation: undersized state/tail, out-of-range APE, NULL args. */
static int test_compressor_prefill_state_ratio4_bounds(void) {
    const uint32_t head_dim = 8;
    const uint32_t width = 16, state_rows = 8;
    Ratio4Model m = build_ratio4_model(head_dim, 0);

    ds4_gpu_tensor *kv_tail = ds4_gpu_tensor_alloc(4ull * width * sizeof(float));
    ds4_gpu_tensor *sc_tail = ds4_gpu_tensor_alloc(4ull * width * sizeof(float));
    ds4_gpu_tensor *skv = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *ssc = ds4_gpu_tensor_alloc((uint64_t)state_rows * width * sizeof(float));
    ds4_gpu_tensor *skv_small = ds4_gpu_tensor_alloc((uint64_t)(state_rows - 1) * width * sizeof(float));
    ds4_gpu_tensor *tail_small = ds4_gpu_tensor_alloc(4ull * width * sizeof(float) - 4);
    if (!kv_tail || !sc_tail || !skv || !ssc || !skv_small || !tail_small) {
        if (kv_tail) ds4_gpu_tensor_free(kv_tail);
        if (sc_tail) ds4_gpu_tensor_free(sc_tail);
        if (skv) ds4_gpu_tensor_free(skv);
        if (ssc) ds4_gpu_tensor_free(ssc);
        if (skv_small) ds4_gpu_tensor_free(skv_small);
        if (tail_small) ds4_gpu_tensor_free(tail_small);
        return 1;
    }
    int rc = 0;

    /* Undersized state -> 0. */
    if (ds4_gpu_compressor_prefill_state_ratio4_tensor(
            skv_small, ssc, kv_tail, sc_tail, m.data.data(), m.data.size(),
            m.ape_offset, 0, head_dim, 0) != 0) {
        fprintf(stderr, "--- bounds: accepted undersized state\n");
        rc = 1;
    }
    /* Undersized tail -> 0. */
    if (ds4_gpu_compressor_prefill_state_ratio4_tensor(
            skv, ssc, tail_small, sc_tail, m.data.data(), m.data.size(),
            m.ape_offset, 0, head_dim, 0) != 0) {
        fprintf(stderr, "--- bounds: accepted undersized kv tail\n");
        rc = 1;
    }
    /* APE range outside the mapped model -> 0. */
    if (ds4_gpu_compressor_prefill_state_ratio4_tensor(
            skv, ssc, kv_tail, sc_tail, m.data.data(), m.data.size(),
            m.data.size() + 8, 0, head_dim, 0) != 0) {
        fprintf(stderr, "--- bounds: accepted out-of-range APE offset\n");
        rc = 1;
    }
    /* Bad ape_type -> 0. */
    if (ds4_gpu_compressor_prefill_state_ratio4_tensor(
            skv, ssc, kv_tail, sc_tail, m.data.data(), m.data.size(),
            m.ape_offset, 7, head_dim, 0) != 0) {
        fprintf(stderr, "--- bounds: accepted bad ape_type\n");
        rc = 1;
    }
    /* NULL tail -> 0. */
    if (ds4_gpu_compressor_prefill_state_ratio4_tensor(
            skv, ssc, nullptr, sc_tail, m.data.data(), m.data.size(),
            m.ape_offset, 0, head_dim, 0) != 0) {
        fprintf(stderr, "--- bounds: accepted NULL kv tail\n");
        rc = 1;
    }
    /* head_dim == 0 -> 0. */
    if (ds4_gpu_compressor_prefill_state_ratio4_tensor(
            skv, ssc, kv_tail, sc_tail, m.data.data(), m.data.size(),
            m.ape_offset, 0, 0, 0) != 0) {
        fprintf(stderr, "--- bounds: accepted head_dim == 0\n");
        rc = 1;
    }

    ds4_gpu_tensor_free(kv_tail); ds4_gpu_tensor_free(sc_tail);
    ds4_gpu_tensor_free(skv); ds4_gpu_tensor_free(ssc);
    ds4_gpu_tensor_free(skv_small); ds4_gpu_tensor_free(tail_small);
    return rc;
}
REGISTER_TEST(compressor_prefill_state_ratio4_bounds, test_compressor_prefill_state_ratio4_bounds);
