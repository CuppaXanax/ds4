#include "../tests.h"
#include "../../../ds4_gpu.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

static int test_routed_moe_iq2_execution_artifact(void) {
    const uint32_t in_dim = 256;
    const uint32_t mid_dim = 256;
    const uint32_t out_dim = 256;
    const uint64_t gate_row_bytes = 66;
    const uint64_t down_row_bytes = 84;
    const uint64_t gate_bytes = (uint64_t)mid_dim * gate_row_bytes;
    const uint64_t down_bytes = (uint64_t)out_dim * down_row_bytes;
    const uint64_t gate_offset = 4096;
    const uint64_t up_offset = gate_offset + gate_bytes + 4096;
    const uint64_t down_offset = up_offset + gate_bytes + 4096;
    std::vector<uint8_t> model((size_t)(down_offset + down_bytes + 4096), 0);
    for (uint64_t row = 0; row < mid_dim; ++row) {
        uint8_t *gate = model.data() + gate_offset + row * gate_row_bytes;
        uint8_t *up = model.data() + up_offset + row * gate_row_bytes;
        gate[0] = 0x00; gate[1] = 0x3c; /* f16 1.0 */
        up[0] = 0x00; up[1] = 0x3c;
        for (uint32_t i = 2; i < gate_row_bytes; ++i) {
            gate[i] = (uint8_t)(0x11u + (row + i) % 17u);
            up[i] = (uint8_t)(0x13u + (row + i) % 19u);
        }
    }
    for (uint64_t row = 0; row < out_dim; ++row) {
        uint8_t *down = model.data() + down_offset + row * down_row_bytes;
        for (uint32_t i = 0; i < down_row_bytes; ++i)
            down[i] = (uint8_t)(0x21u + (row + i) % 29u);
    }
    std::vector<float> input(in_dim, 0.25f);
    std::vector<int32_t> selected(1, 0);
    std::vector<float> route_weight(1, 1.0f);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(in_dim * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(mid_dim * sizeof(float) + 1024u);
    ds4_gpu_tensor *experts = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc(sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(sizeof(float));
    int result = 1;
    auto cleanup = [&]() {
        ds4_gpu_set_model_map(model.data(), model.size());
        ds4_gpu_tensor_free(weights); ds4_gpu_tensor_free(sel);
        ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(experts);
        ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(x);
        unsetenv("DS4_VULKAN_EXECUTION_ARTIFACT_IQ2");
        unsetenv("DS4_VULKAN_REQUIRE_EXECUTION_ARTIFACT_IQ2");
        unsetenv("DS4_VULKAN_EXECUTION_ARTIFACT_IQ2_TEST");
        unsetenv("DS4_VULKAN_TRACE_KERNELS");
        return result;
    };
    if (!x || !mid || !experts || !out || !sel || !weights ||
        !ds4_gpu_tensor_write(x, 0, input.data(), input.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(sel, 0, selected.data(), sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(weights, 0, route_weight.data(), sizeof(float)))
        return cleanup();
    setenv("DS4_VULKAN_EXECUTION_ARTIFACT_IQ2_TEST", "1", 1);
    setenv("DS4_VULKAN_TRACE_KERNELS", "1", 1);
    /* First establish the exact raw-GGUF result with the artifact consumer
     * disabled.  The second run then has an independent reference rather
     * than comparing two executions of the candidate itself. */
    setenv("DS4_VULKAN_EXECUTION_ARTIFACT_IQ2", "0", 1);
    unsetenv("DS4_VULKAN_REQUIRE_EXECUTION_ARTIFACT_IQ2");
    if (!ds4_gpu_set_model_map(model.data(), model.size()))
        return cleanup();
    auto run = [&](std::vector<float> &values) -> bool {
        std::vector<float> zeros(out_dim, 0.0f);
        if (!ds4_gpu_tensor_write(out, 0, zeros.data(), zeros.size() * sizeof(float)) ||
            !ds4_gpu_batch_layer_begin(0)) return false;
        const int ok = ds4_gpu_routed_moe_one_tensor(
            out, nullptr, nullptr, mid, experts, model.data(), model.size(),
            gate_offset, up_offset, down_offset, 16, 10,
            gate_bytes, gate_row_bytes, down_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim, sel, weights, 1, 1, 0.0f, x, nullptr,
            0, false);
        const int ended = ds4_gpu_batch_layer_end(0);
        if (!ok || !ended || values.size() != out_dim) return false;
        return ds4_gpu_tensor_read(out, 0, values.data(), values.size() * sizeof(float)) != 0;
    };
    std::vector<float> reference(out_dim), artifact(out_dim);
    if (!run(reference)) return cleanup();

    setenv("DS4_VULKAN_EXECUTION_ARTIFACT_IQ2", "1", 1);
    setenv("DS4_VULKAN_REQUIRE_EXECUTION_ARTIFACT_IQ2", "1", 1);
    if (!ds4_gpu_set_model_map(model.data(), model.size()) ||
        !ds4_gpu_cache_iq2_expert_range(model.data(), model.size(), gate_offset,
                                         gate_bytes, in_dim, mid_dim, "iq2-exec-gate") ||
        !ds4_gpu_cache_iq2_expert_range(model.data(), model.size(), up_offset,
                                         gate_bytes, in_dim, mid_dim, "iq2-exec-up"))
        return cleanup();
    std::memset(model.data() + gate_offset, 0, (size_t)gate_bytes);
    std::memset(model.data() + up_offset, 0, (size_t)gate_bytes);
    if (!run(artifact)) return cleanup();
    if (std::memcmp(reference.data(), artifact.data(),
                    reference.size() * sizeof(float)) != 0)
        return cleanup();
    result = 0;
    return cleanup();
}

REGISTER_TEST(routed_moe_iq2_execution_artifact,
              test_routed_moe_iq2_execution_artifact);
