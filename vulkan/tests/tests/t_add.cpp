/* Example kernel test: ds4_gpu_add_tensor (real GPU dispatch of add_f32).
 * out = a + b elementwise; validates the harness end-to-end on the GPU. */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cmath>

static int test_add(void) {
    const uint32_t n = 4096;
    ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(n * sizeof(float));
    ds4_gpu_tensor *b = ds4_gpu_tensor_alloc(n * sizeof(float));
    ds4_gpu_tensor *o = ds4_gpu_tensor_alloc(n * sizeof(float));
    if (!a || !b || !o) return 1;
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
    return rc;
}
REGISTER_TEST(add_f32, test_add);
