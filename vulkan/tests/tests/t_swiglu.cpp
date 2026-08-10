/* Example kernel test: ds4_gpu_swiglu_tensor (real GPU dispatch). */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cmath>

static int test_swiglu(void) {
    const uint32_t n = 513;
    const float clamp = 1.25f;
    const float weight = 0.75f;
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(n * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(n * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(n * sizeof(float));
    if (!gate || !up || !out) return 1;

    float gate_values[n], up_values[n];
    for (uint32_t i = 0; i < n; i++) {
        gate_values[i] = (float)((int)(i % 17) - 8) * 0.5f;
        up_values[i] = (float)((int)(i % 13) - 6) * 0.5f;
    }
    ds4_gpu_tensor_write(gate, 0, gate_values, sizeof(gate_values));
    ds4_gpu_tensor_write(up, 0, up_values, sizeof(up_values));

    int rc = 1;
    if (ds4_gpu_swiglu_tensor(out, gate, up, n, clamp, weight) != 0) {
        float got[n];
        if (ds4_gpu_tensor_read(out, 0, got, sizeof(got)) != 0) {
            rc = 0;
            for (uint32_t i = 0; i < n; i++) {
                float gate_value = std::fmin(gate_values[i], clamp);
                float up_value = std::fmin(clamp, std::fmax(-clamp, up_values[i]));
                float silu = gate_value / (1.0f + std::exp(-gate_value));
                float want = silu * up_value * weight;
                if (std::fabs(got[i] - want) > 2e-4f) { rc = 1; break; }
            }
        }
    }
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(gate);
    return rc;
}
REGISTER_TEST(swiglu, test_swiglu);