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
    constexpr uint32_t max_tokens = 4096;
    std::vector<float> input((uint64_t)max_tokens * in_dim, 0.25f);
    std::vector<int32_t> selected(max_tokens, 0);
    std::vector<float> route_weight(max_tokens, 1.0f);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)max_tokens * in_dim * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc((uint64_t)max_tokens * mid_dim * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc((uint64_t)max_tokens * mid_dim * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc((uint64_t)max_tokens * mid_dim * sizeof(float) + 1024u);
    ds4_gpu_tensor *experts = ds4_gpu_tensor_alloc((uint64_t)max_tokens * out_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)max_tokens * out_dim * sizeof(float));
    ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc((uint64_t)max_tokens * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc((uint64_t)max_tokens * sizeof(float));
    int result = 1;
    auto cleanup = [&]() {
        ds4_gpu_set_model_map(model.data(), model.size());
        ds4_gpu_tensor_free(weights); ds4_gpu_tensor_free(sel);
        ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(experts);
        ds4_gpu_tensor_free(mid); ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(x);
        unsetenv("DS4_VULKAN_EXECUTION_ARTIFACT_IQ2");
        unsetenv("DS4_VULKAN_REQUIRE_EXECUTION_ARTIFACT_IQ2");
        unsetenv("DS4_VULKAN_EXECUTION_ARTIFACT_IQ2_TEST");
        unsetenv("DS4_VULKAN_ROUTED_MID_ONLY");
        unsetenv("DS4_VULKAN_TRACE_KERNELS");
        return result;
    };
    if (!x || !gate || !up || !mid || !experts || !out || !sel || !weights ||
        !ds4_gpu_tensor_write(x, 0, input.data(), input.size() * sizeof(float)) ||
        !ds4_gpu_tensor_write(sel, 0, selected.data(), selected.size() * sizeof(int32_t)) ||
        !ds4_gpu_tensor_write(weights, 0, route_weight.data(), route_weight.size() * sizeof(float)))
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
    auto run_decode = [&](std::vector<float> &values) -> bool {
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
        return ds4_gpu_tensor_read(out, 0, values.data(),
                                   values.size() * sizeof(float)) != 0;
    };
    auto run_batch = [&](uint32_t n_tokens, std::vector<float> &values) -> bool {
        std::vector<float> zeros((uint64_t)n_tokens * out_dim, 0.0f);
        if (!ds4_gpu_tensor_write(out, 0, zeros.data(), zeros.size() * sizeof(float)) ||
            !ds4_gpu_batch_layer_begin(0)) return false;
        bool mid_is_f16 = false;
        const int ok = ds4_gpu_routed_moe_batch_tensor(
            out, gate, up, mid, experts, model.data(), model.size(),
            gate_offset, up_offset, down_offset, 16, 10,
            gate_bytes, gate_row_bytes, down_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim, sel, weights, 1, 1, 0.0f, x, 0,
            n_tokens, &mid_is_f16, false);
        const int ended = ds4_gpu_batch_layer_end(0);
        if (!ok || !ended || mid_is_f16 || values.size() != (uint64_t)n_tokens * out_dim) return false;
        return ds4_gpu_tensor_read(out, 0, values.data(), values.size() * sizeof(float)) != 0;
    };
    auto decode_is_rejected = [&]() -> bool {
        if (!ds4_gpu_batch_layer_begin(0)) return false;
        const int ok = ds4_gpu_routed_moe_one_tensor(
            out, nullptr, nullptr, mid, experts, model.data(), model.size(),
            gate_offset, up_offset, down_offset, 16, 10,
            gate_bytes, gate_row_bytes, down_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim, sel, weights, 1, 1, 0.0f, x, nullptr,
            0, false);
        const int ended = ds4_gpu_batch_layer_end(0);
        return !ok && ended;
    };
    auto batch_is_rejected = [&](uint32_t n_tokens) -> bool {
        if (!ds4_gpu_batch_layer_begin(0)) return false;
        bool mid_is_f16 = false;
        const int ok = ds4_gpu_routed_moe_batch_tensor(
            out, gate, up, mid, experts, model.data(), model.size(),
            gate_offset, up_offset, down_offset, 16, 10,
            gate_bytes, gate_row_bytes, down_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim, sel, weights, 1, 1, 0.0f, x, 0,
            n_tokens, &mid_is_f16, false);
        const int ended = ds4_gpu_batch_layer_end(0);
        return !ok && ended;
    };
    std::vector<float> reference_one(out_dim), artifact_one(out_dim);
    std::vector<float> reference((uint64_t)2 * out_dim), artifact((uint64_t)2 * out_dim);
    if (!run_decode(reference_one) || !run_batch(2, reference)) return cleanup();

    setenv("DS4_VULKAN_EXECUTION_ARTIFACT_IQ2", "1", 1);
    setenv("DS4_VULKAN_REQUIRE_EXECUTION_ARTIFACT_IQ2", "1", 1);
    /* Strict mode must reject a missing artifact before any raw upload. */
    if (!ds4_gpu_set_model_map(model.data(), model.size()) ||
        !decode_is_rejected() ||
        !ds4_gpu_cache_iq2_expert_range(model.data(), model.size(), gate_offset,
                                         gate_bytes, in_dim, mid_dim, "iq2-exec-gate") ||
        !ds4_gpu_cache_iq2_expert_range(model.data(), model.size(), up_offset,
                                         gate_bytes, in_dim, mid_dim, "iq2-exec-up"))
        return cleanup();
    std::memset(model.data() + gate_offset, 0, (size_t)gate_bytes);
    std::memset(model.data() + up_offset, 0, (size_t)gate_bytes);
    if (!run_decode(artifact_one) || !run_batch(2, artifact)) return cleanup();
    if (std::memcmp(reference_one.data(), artifact_one.data(),
                    reference_one.size() * sizeof(float)) != 0)
        return cleanup();
    if (std::memcmp(reference.data(), artifact.data(),
                    reference.size() * sizeof(float)) != 0)
        return cleanup();
    /* Required mode wins over both an explicit artifact disable and an
     * unsupported generic routed path; neither may rematerialize raw bytes. */
    setenv("DS4_VULKAN_EXECUTION_ARTIFACT_IQ2", "0", 1);
    if (!decode_is_rejected()) return cleanup();
    setenv("DS4_VULKAN_EXECUTION_ARTIFACT_IQ2", "1", 1);
    setenv("DS4_VULKAN_ROUTED_MID_ONLY", "0", 1);
    if (!batch_is_rejected(2)) return cleanup();
    unsetenv("DS4_VULKAN_ROUTED_MID_ONLY");
    /* Make the 4096-token reachability proof cheap: validation is disabled
     * inside the layer batch and the routed shader must bounds-check IDs.
     * A raw fallback still reaches the artifact ownership guard and fails. */
    std::vector<int32_t> invalid(max_tokens, 99);
    if (!ds4_gpu_tensor_write(sel, 0, invalid.data(),
                              invalid.size() * sizeof(int32_t)))
        return cleanup();
    std::vector<float> large((uint64_t)max_tokens * out_dim);
    if (!run_batch(max_tokens, large)) return cleanup();
    for (float value : large)
        if (value != 0.0f) return cleanup();
    result = 0;
    return cleanup();
}

REGISTER_TEST(routed_moe_iq2_execution_artifact,
              test_routed_moe_iq2_execution_artifact);
