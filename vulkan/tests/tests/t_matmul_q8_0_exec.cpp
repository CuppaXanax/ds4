#include "../tests.h"
#include "../../../ds4_gpu.h"
#include <vulkan/vulkan.h>
#include "../../include/vk_mem_alloc.h"
#include "../../../ds4_vulkan.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint16_t exec_f32_to_f16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) return (uint16_t)(sign | 0x7c00u);
    if (exponent == 0u) return (uint16_t)sign;
    const int32_t half_exponent = (int32_t)exponent - 127 + 15;
    if (half_exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    if (half_exponent <= 0) return (uint16_t)sign;
    return (uint16_t)(sign | ((uint32_t)half_exponent << 10) | (mantissa >> 13));
}

static float exec_f16_to_f32(uint16_t half) {
    const uint32_t sign = (uint32_t)(half & 0x8000u) << 16;
    const uint32_t exponent = (half >> 10) & 0x1fu;
    const uint32_t mantissa = half & 0x3ffu;
    uint32_t bits;
    if (exponent == 0u) bits = sign;
    else if (exponent == 31u) bits = sign | 0x7f800000u | (mantissa << 13);
    else bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static int test_matmul_q8_0_execution_artifact(void) {
    const uint64_t in_dim = 512;
    const uint64_t out_dim = 8;
    const uint64_t n_tok = 1;
    const uint64_t blocks = in_dim / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t source_offset = 4096;
    const uint64_t weight_bytes = out_dim * row_bytes;
    std::vector<uint8_t> model(source_offset + weight_bytes + 4096u, 0xa5);
    for (uint64_t row = 0; row < out_dim; ++row) {
        for (uint64_t block = 0; block < blocks; ++block) {
            uint8_t *dst = model.data() + source_offset + row * row_bytes + block * 34u;
            const uint16_t scale = exec_f32_to_f16(0.125f * (float)(row + block + 1u));
            std::memcpy(dst, &scale, sizeof(scale));
            for (uint32_t i = 0; i < 32u; ++i)
                dst[2u + i] = (uint8_t)((int)((row * 17u + block * 11u + i * 5u) % 61u) - 30);
        }
    }
    const std::vector<uint8_t> raw_model = model;
    std::vector<float> input(in_dim);
    for (uint64_t i = 0; i < in_dim; ++i)
        input[i] = (float)((int)((i * 19u + 7u) % 101u) - 50) * 0.03125f;

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(input.size() * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(blocks * 36u);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    int result = 1;
    auto cleanup = [&]() {
        ds4_gpu_set_model_map(model.data(), model.size());
        ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(q);
        ds4_gpu_tensor_free(x);
        unsetenv("DS4_VULKAN_EXECUTION_ARTIFACT_Q8");
        unsetenv("DS4_VULKAN_REQUIRE_EXECUTION_ARTIFACT_Q8");
        unsetenv("DS4_VULKAN_TRACE_KERNELS");
        return result;
    };
    if (!x || !q || !out || !ds4_gpu_tensor_write(x, 0, input.data(),
                                                   input.size() * sizeof(float)))
        return cleanup();

    setenv("DS4_VULKAN_EXECUTION_ARTIFACT_Q8", "1", 1);
    setenv("DS4_VULKAN_REQUIRE_EXECUTION_ARTIFACT_Q8", "1", 1);
    setenv("DS4_VULKAN_TRACE_KERNELS", "1", 1);
    if (!ds4_gpu_set_model_map(model.data(), model.size()) ||
        !ds4_gpu_cache_q8_f16_range(model.data(), model.size(), source_offset,
                                    weight_bytes, in_dim, out_dim, "exec-test"))
        return cleanup();

    /* The dispatch below must consume the immutable artifact.  Zeroing the
     * source range makes a fallback to raw GGUF or a failed prebuild fail
     * loudly instead of producing a false green result. */
    std::memset(model.data() + source_offset, 0, (size_t)weight_bytes);
    if (!ds4_gpu_begin_commands() ||
        !ds4_gpu_quantize_q8_0_tensor(q, x, in_dim, n_tok) ||
        !ds4_gpu_end_commands())
        return cleanup();
    std::vector<uint8_t> packed(ds4_gpu_tensor_bytes(q));
    if (!ds4_gpu_tensor_read(q, 0, packed.data(), packed.size())) return cleanup();

    if (!ds4_gpu_begin_commands() ||
        !ds4_gpu_matmul_q8_0_prequant_tensor(out, model.data(), model.size(),
                                             source_offset, in_dim, out_dim, q, n_tok) ||
        !ds4_gpu_end_commands())
        return cleanup();
    std::vector<float> got(out_dim);
    if (!ds4_gpu_tensor_read(out, 0, got.data(), got.size() * sizeof(float)))
        return cleanup();

    for (uint64_t row = 0; row < out_dim; ++row) {
        float expected = 0.0f;
        for (uint64_t block = 0; block < blocks; ++block) {
            uint16_t scale_bits;
            std::memcpy(&scale_bits,
                        raw_model.data() + source_offset + row * row_bytes + block * 34u,
                        sizeof(scale_bits));
            const uint8_t *q_weight = raw_model.data() + source_offset +
                                      row * row_bytes + block * 34u + 2u;
            const uint8_t *q_input = packed.data() + block * 36u + 4u;
            uint32_t activation_bits;
            std::memcpy(&activation_bits, packed.data() + block * 36u, sizeof(activation_bits));
            float activation_scale;
            std::memcpy(&activation_scale, &activation_bits, sizeof(activation_scale));
            int dot = 0;
            for (uint32_t i = 0; i < 32u; ++i) {
                const int8_t w = (int8_t)q_weight[i];
                const int8_t a = (int8_t)q_input[i];
                dot += (int)w * (int)a;
            }
            expected += exec_f16_to_f32(scale_bits) *
                        activation_scale *
                        (float)dot;
        }
        if (!std::isfinite(got[row]) || std::fabs(got[row] - expected) > 1e-5f)
            return cleanup();
    }
    result = 0;
    return cleanup();
}

REGISTER_TEST(matmul_q8_0_execution_artifact, test_matmul_q8_0_execution_artifact);
