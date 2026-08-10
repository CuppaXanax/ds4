#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>

static int test_argmax(void) {
    const float values[] = {-3.0f, 7.0f, 2.0f, 7.0f, 1.0f};
    ds4_gpu_tensor *logits = ds4_gpu_tensor_alloc(sizeof(values));
    ds4_gpu_tensor *index = ds4_gpu_tensor_alloc(sizeof(int32_t));
    if (!logits || !index ||
        !ds4_gpu_tensor_write(logits, 0, values, sizeof(values)) ||
        !ds4_gpu_argmax_tensor(index, logits, 5)) {
        if (index) ds4_gpu_tensor_free(index);
        if (logits) ds4_gpu_tensor_free(logits);
        return 1;
    }
    int32_t got = -1;
    const int ok = ds4_gpu_tensor_read(index, 0, &got, sizeof(got)) && got == 1;
    ds4_gpu_tensor_free(index);
    ds4_gpu_tensor_free(logits);
    return ok ? 0 : 1;
}

REGISTER_TEST(argmax, test_argmax);