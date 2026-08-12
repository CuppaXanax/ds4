#include "../tests.h"
#include "../../../ds4_gpu.h"
#include "../../q8_aligned_artifact.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static uint16_t q8_aligned_f32_to_f16(float value) {
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

static int test_q8_aligned_artifact() {
    const uint64_t in_dim = 37;
    const uint64_t out_dim = 5;
    const uint64_t artifact_out_dim = 7;
    const uint64_t n_tok = 3;
    const uint64_t source_offset = 4096;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t raw_bytes = artifact_out_dim * blocks * 34u;
    std::vector<uint8_t> model(source_offset + raw_bytes, 0xa5);
    for (uint64_t row = 0; row < artifact_out_dim; row++) {
        for (uint64_t block = 0; block < blocks; block++) {
            uint8_t *dst = model.data() + source_offset + (row * blocks + block) * 34u;
            const uint16_t scale = q8_aligned_f32_to_f16(0.125f * (float)(row + block + 1u));
            std::memcpy(dst, &scale, sizeof(scale));
            for (uint64_t i = 0; i < 32u; i++)
                dst[2u + i] = (uint8_t)((int)((row * 29u + block * 17u + i * 7u) % 63u) - 31);
        }
    }
    const std::vector<uint8_t> original = model;
    std::vector<float> input(n_tok * in_dim);
    for (uint64_t token = 0; token < n_tok; token++)
        for (uint64_t i = 0; i < in_dim; i++)
            input[token * in_dim + i] =
                (float)((int)((i * 13u + token * 19u) % 29u) - 14) * 0.03125f;

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(input.size() * sizeof(float));
    ds4_gpu_tensor *raw_out = ds4_gpu_tensor_alloc(n_tok * artifact_out_dim * sizeof(float));
    ds4_gpu_tensor *aligned_out = ds4_gpu_tensor_alloc(n_tok * out_dim * sizeof(float));
    int result = 1;
    ds4_vulkan_q8_aligned_artifact artifact{};
    auto cleanup = [&]() {
        ds4_gpu_set_model_map(model.data(), model.size());
        ds4_vulkan_q8_aligned_free(&artifact);
        ds4_gpu_tensor_free(aligned_out);
        ds4_gpu_tensor_free(raw_out);
        ds4_gpu_tensor_free(x);
        return result;
    };
    if (!x || !raw_out || !aligned_out ||
        !ds4_gpu_tensor_write(x, 0, input.data(), input.size() * sizeof(float)))
        return cleanup();

    if (!ds4_gpu_set_model_map(model.data(), model.size()) ||
        !ds4_gpu_begin_commands() ||
        !ds4_gpu_matmul_q8_0_tensor(raw_out, model.data(), model.size(), source_offset,
                                    in_dim, artifact_out_dim, x, n_tok) ||
        !ds4_gpu_end_commands())
        return cleanup();
    std::vector<float> raw(n_tok * artifact_out_dim);
    if (!ds4_gpu_tensor_read(raw_out, 0, raw.data(), raw.size() * sizeof(float)))
        return cleanup();

    if (!ds4_gpu_set_model_map(model.data(), model.size())) return cleanup();
    if (!ds4_vulkan_q8_aligned_build(&artifact, model.data(), model.size(), source_offset,
                                     in_dim, artifact_out_dim, 64u) ||
        artifact.payload_offset % 256u != 0 ||
        artifact.payload_bytes != artifact_out_dim * blocks * 32u)
        return cleanup();
    for (uint64_t record = 0; record < artifact_out_dim * blocks; record++) {
        const uint8_t *source = original.data() + source_offset + record * 34u;
        if (std::memcmp(artifact.data + record * 2u, source, 2u) != 0 ||
            std::memcmp(artifact.data + artifact.payload_offset + record * 32u,
                        source + 2u, 32u) != 0)
            return cleanup();
    }
    for (uint64_t i = artifact_out_dim * blocks * 2u; i < artifact.payload_offset; i++)
        if (artifact.data[i] != 0) return cleanup();
    if (!ds4_gpu_cache_q8_f16_range(model.data(), model.size(), source_offset,
                                    raw_bytes, in_dim, artifact_out_dim, "q8-aligned-test"))
        return cleanup();

    std::memset(model.data() + source_offset, 0, (size_t)raw_bytes);
    if (!ds4_gpu_begin_commands() ||
        !ds4_gpu_matmul_q8_0_tensor(aligned_out, model.data(), model.size(), source_offset,
                                    in_dim, out_dim, x, n_tok) ||
        !ds4_gpu_end_commands())
        return cleanup();
    std::vector<float> aligned(n_tok * out_dim);
    if (!ds4_gpu_tensor_read(aligned_out, 0, aligned.data(), aligned.size() * sizeof(float)))
        return cleanup();
    bool tokens_differ = false;
    for (uint64_t token = 0; token < n_tok; token++) {
        for (uint64_t row = 0; row < out_dim; row++) {
            const uint64_t aligned_index = token * out_dim + row;
            const uint64_t raw_index = token * artifact_out_dim + row;
            if (std::fabs(aligned[aligned_index] - raw[raw_index]) > 1e-5f)
                return cleanup();
            if (token > 0 && std::fabs(raw[raw_index] - raw[row]) > 1e-5f)
                tokens_differ = true;
        }
    }
    if (!tokens_differ) return cleanup();
    result = 0;
    return cleanup();
}

REGISTER_TEST(q8_aligned_artifact, test_q8_aligned_artifact);