/* Example kernel test: ds4_gpu_add_tensor (real GPU dispatch of add_f32).
 * out = a + b elementwise; validates the harness end-to-end on the GPU. */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

static int test_add(void) {
    const uint32_t n = 4096;
    const uint64_t view_offset = 256;
    ds4_gpu_tensor *abase = ds4_gpu_tensor_alloc(n * sizeof(float) + view_offset);
    ds4_gpu_tensor *bbase = ds4_gpu_tensor_alloc(n * sizeof(float) + view_offset);
    ds4_gpu_tensor *obase = ds4_gpu_tensor_alloc(n * sizeof(float) + view_offset);
    ds4_gpu_tensor *a = ds4_gpu_tensor_view(abase, view_offset, n * sizeof(float));
    ds4_gpu_tensor *b = ds4_gpu_tensor_view(bbase, view_offset, n * sizeof(float));
    ds4_gpu_tensor *o = ds4_gpu_tensor_view(obase, view_offset, n * sizeof(float));
    if (!abase || !bbase || !obase || !a || !b || !o) return 1;
    ds4_gpu_tensor_fill_f32(a, 1.5f, n);
    ds4_gpu_tensor_fill_f32(b, 2.25f, n);
    int rc = 1;
    if (ds4_gpu_add_tensor(o, a, b, n) != 0) {
        float v = 0.0f;
        if (ds4_gpu_tensor_read(o, 0, &v, sizeof(v)) != 0) {
            rc = std::fabsf(v - 3.75f) < 1e-4f ? 0 : 1;
        }
    }
    ds4_gpu_tensor_free(o);
    ds4_gpu_tensor_free(b);
    ds4_gpu_tensor_free(a);
    ds4_gpu_tensor_free(obase);
    ds4_gpu_tensor_free(bbase);
    ds4_gpu_tensor_free(abase);
    return rc;
}
REGISTER_TEST(add_f32, test_add);

static void set_layer_span(const char *value) {
#if defined(_WIN32)
    _putenv_s("DS4_VULKAN_BATCH_LAYER_SPAN", value);
#else
    setenv("DS4_VULKAN_BATCH_LAYER_SPAN", value, 1);
#endif
}

static int run_cross_layer_span(uint32_t span) {
    const uint32_t n = 4096;
    const uint64_t bytes = (uint64_t)n * sizeof(float);
    std::vector<float> av(n), bv(n), dv(n), zv(n, 0.0f), got(n);
    std::vector<ds4_gpu_tensor *> snapshots(span, nullptr);
    ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *b = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *delta = ds4_gpu_tensor_alloc(bytes);
    ds4_gpu_tensor *zero = ds4_gpu_tensor_alloc(bytes);
    int rc = 1;
    for (uint32_t layer = 0; layer < span; layer++)
        snapshots[layer] = ds4_gpu_tensor_alloc(bytes);
    if (!a || !b || !delta || !zero) goto done;
    for (ds4_gpu_tensor *snapshot : snapshots)
        if (!snapshot) goto done;
    for (uint32_t i = 0; i < n; i++) {
        av[i] = (float)((int)(i % 37) - 18) * 0.0625f;
        bv[i] = -99.0f;
        dv[i] = (float)((int)(i % 11) - 5) * 0.03125f;
    }
    if (!ds4_gpu_tensor_write(a, 0, av.data(), bytes) ||
        !ds4_gpu_tensor_write(b, 0, bv.data(), bytes) ||
        !ds4_gpu_tensor_write(delta, 0, dv.data(), bytes) ||
        !ds4_gpu_tensor_write(zero, 0, zv.data(), bytes)) goto done;
    set_layer_span(span == 2 ? "2" : "4");
    if (!ds4_gpu_begin_commands()) goto done;
    for (uint32_t layer = 0; layer < span; layer++) {
        ds4_gpu_tensor *input = (layer & 1u) ? b : a;
        ds4_gpu_tensor *output = (layer & 1u) ? a : b;
        if (!ds4_gpu_batch_layer_begin(layer, input, output) ||
            !ds4_gpu_add_tensor(output, input, delta, n) ||
            !ds4_gpu_add_tensor(snapshots[layer], output, zero, n) ||
            !ds4_gpu_batch_layer_end(layer, input, output)) goto done;
    }
    if (!ds4_gpu_end_commands()) goto done;
    {
        ds4_gpu_tensor *result = (span & 1u) ? b : a;
        if (!ds4_gpu_tensor_read(result, 0, got.data(), bytes)) goto done;
    }
    for (uint32_t i = 0; i < n; i++) {
        const float expected = av[i] + (float)span * dv[i];
        if (std::fabsf(got[i] - expected) > 1.0e-6f) {
            fprintf(stderr,
                    "cross_layer_span[%u]: mismatch at %u got=%g want=%g\n",
                    span, i, (double)got[i], (double)expected);
            goto done;
        }
    }
    for (uint32_t layer = 0; layer < span; layer++) {
        if (!ds4_gpu_tensor_read(snapshots[layer], 0, got.data(), bytes))
            goto done;
        for (uint32_t i = 0; i < n; i++) {
            const float expected = av[i] + (float)(layer + 1u) * dv[i];
            if (std::fabsf(got[i] - expected) > 1.0e-6f) {
                fprintf(stderr,
                        "cross_layer_span[%u]: layer %u mismatch at %u got=%g want=%g\n",
                        span, layer, i, (double)got[i], (double)expected);
                goto done;
            }
        }
    }
    rc = 0;
done:
    set_layer_span("1");
    (void)ds4_gpu_synchronize();
    ds4_gpu_tensor_free(delta);
    ds4_gpu_tensor_free(zero);
    ds4_gpu_tensor_free(b);
    ds4_gpu_tensor_free(a);
    for (ds4_gpu_tensor *snapshot : snapshots) ds4_gpu_tensor_free(snapshot);
    return rc;
}

static int test_cross_layer_span_2(void) { return run_cross_layer_span(2); }
static int test_cross_layer_span_4(void) { return run_cross_layer_span(4); }
REGISTER_TEST(cross_layer_span_2, test_cross_layer_span_2);
REGISTER_TEST(cross_layer_span_4, test_cross_layer_span_4);
