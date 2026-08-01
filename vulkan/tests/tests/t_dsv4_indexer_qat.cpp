/* Kernel test: ds4_gpu_dsv4_indexer_qat_tensor (host-side in the backend).
 *
 * The kernel applies, in-place to every row of x, the official DS4 indexer
 * QAT chain: a 128-wide Hadamard rotation followed by an FP4 activation-
 * simulation round trip.  It is host-side (tensor->ptr is host-mapped), so
 * NO begin_commands/end_commands are needed to read back the output.
 *
 * Reference math is replicated from ds4.c dsv4_indexer_qat_row_inplace_cpu
 * (dsv4_hadamard128_inplace_cpu + dsv4_fp4_act_quantize_row_inplace_cpu +
 * dsv4_e2m1fn_value_cpu / dsv4_e2m1fn_dequant_cpu), which is also what the
 * CUDA indexer_hadamard_fp4_kernel and the Metal
 * kernel_dsv4_indexer_hadamard_fp4_f32 implement.
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include "../../ds4_gpu_mgpu.h"   /* complete ds4_gpu_tensor for stack views */
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
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

/* ds4.c dsv4_hadamard128_inplace_cpu. */
static void ref_hadamard128(float *x) {
    for (uint32_t stride = 1; stride < 128; stride <<= 1) {
        for (uint32_t base = 0; base < 128; base += 2u * stride) {
            for (uint32_t i = 0; i < stride; i++) {
                const float a = x[base + i];
                const float b = x[base + stride + i];
                x[base + i] = a + b;
                x[base + stride + i] = a - b;
            }
        }
    }
    const float scale = 0.08838834764831845f;
    for (uint32_t i = 0; i < 128; i++) x[i] *= scale;
}

/* ds4.c dsv4_e2m1fn_value_cpu / dsv4_e2m1fn_dequant_cpu. */
static float ref_e2m1fn_value(int i) {
    static const float values[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    return values[i & 7];
}

static float ref_e2m1fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - ref_e2m1fn_value(0));
    for (int i = 1; i < 8; i++) {
        const float diff = fabsf(ax - ref_e2m1fn_value(i));
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * ref_e2m1fn_value(best);
}

/* ds4.c dsv4_fp4_act_quantize_row_inplace_cpu (32-aligned rows assumed). */
static void ref_fp4_act_quantize_row(float *x, uint32_t n) {
    for (uint32_t off = 0; off < n; off += 32) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < 32; i++) {
            const float av = fabsf(x[off + i]);
            if (av > amax) amax = av;
        }
        if (amax < 7.052966104933725e-38f) amax = 7.052966104933725e-38f;
        const float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 6.0f)));
        for (uint32_t i = 0; i < 32; i++) {
            float v = x[off + i] / scale;
            if (v > 6.0f) v = 6.0f;
            if (v < -6.0f) v = -6.0f;
            x[off + i] = ref_e2m1fn_dequant(v) * scale;
        }
    }
}

static int run_qat_case(const char *what, uint32_t n_rows, uint32_t head_dim,
                        uint32_t seed) {
    std::vector<float> xv((uint64_t)n_rows * head_dim);
    for (uint32_t r = 0; r < n_rows; r++) {
        for (uint32_t i = 0; i < head_dim; i++) {
            /* Moderate magnitudes with mixed signs so the Hadamard sums stay
             * in normal float range and the FP4 clamp/round path is hit. */
            float f = 0.25f * (float)((i + r * 31u + seed) % 97u + 1) + 0.125f * (float)(i % 7);
            if ((i + r) & 1u) f = -f;
            xv[(uint64_t)r * head_dim + i] = f;
        }
    }

    std::vector<float> ref = xv;
    for (uint32_t r = 0; r < n_rows; r++) {
        ref_hadamard128(ref.data() + (uint64_t)r * head_dim);
        ref_fp4_act_quantize_row(ref.data() + (uint64_t)r * head_dim, head_dim);
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(xv.size() * sizeof(float));
    if (!x) return 1;
    if (ds4_gpu_tensor_write(x, 0, xv.data(), xv.size() * sizeof(float)) == 0) {
        ds4_gpu_tensor_free(x);
        return 1;
    }

    int rc = 1;
    if (ds4_gpu_dsv4_indexer_qat_tensor(x, n_rows, head_dim) != 0) {
        std::vector<float> got(xv.size());
        if (ds4_gpu_tensor_read(x, 0, got.data(), got.size() * sizeof(float)) != 0) {
            if (check_f32(what, got.data(), ref.data(), (uint32_t)got.size()) == 0)
                rc = 0;
        }
    }
    ds4_gpu_tensor_free(x);
    return rc;
}

/* Main case: n_rows=2, head_dim=128 (the model shape). */
static int test_dsv4_indexer_qat(void) {
    return run_qat_case("dsv4_indexer_qat", 2, 128, 3);
}
REGISTER_TEST(dsv4_indexer_qat, test_dsv4_indexer_qat);

/* Single-row case, different synthetic seed. */
static int test_dsv4_indexer_qat_one_row(void) {
    return run_qat_case("dsv4_indexer_qat/one_row", 1, 128, 17);
}
REGISTER_TEST(dsv4_indexer_qat_one_row, test_dsv4_indexer_qat_one_row);

/* Validation: wrong head_dim / zero rows / undersized buffer must fail. */
static int test_dsv4_indexer_qat_bounds(void) {
    const uint32_t head_dim = 128;
    int rc = 0;

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)2 * head_dim * sizeof(float));
    if (!x) return 1;
    if (ds4_gpu_tensor_fill_f32(x, 1.0f, (uint64_t)2 * head_dim) == 0) {
        ds4_gpu_tensor_free(x);
        return 1;
    }

    /* head_dim != 128 -> 0 (the reference dies for non-128 indexer rows). */
    if (ds4_gpu_dsv4_indexer_qat_tensor(x, 2, 64) != 0) {
        fprintf(stderr, "--- bounds: accepted head_dim != 128\n");
        rc = 1;
    }
    /* n_rows == 0 -> 0. */
    if (ds4_gpu_dsv4_indexer_qat_tensor(x, 0, head_dim) != 0) {
        fprintf(stderr, "--- bounds: accepted n_rows == 0\n");
        rc = 1;
    }
    /* NULL tensor -> 0. */
    if (ds4_gpu_dsv4_indexer_qat_tensor(nullptr, 2, head_dim) != 0) {
        fprintf(stderr, "--- bounds: accepted NULL tensor\n");
        rc = 1;
    }

    /* Undersized buffer: only 1 row allocated, ask for 2 -> 0. */
    ds4_gpu_tensor small;
    small.ptr = x->ptr;
    small.bytes = (uint64_t)head_dim * sizeof(float);
    small.owner = 0;
    small.device_id = x->device_id;
    if (ds4_gpu_dsv4_indexer_qat_tensor(&small, 2, head_dim) != 0) {
        fprintf(stderr, "--- bounds: accepted undersized buffer\n");
        rc = 1;
    }

    ds4_gpu_tensor_free(x);
    return rc;
}
REGISTER_TEST(dsv4_indexer_qat_bounds, test_dsv4_indexer_qat_bounds);
