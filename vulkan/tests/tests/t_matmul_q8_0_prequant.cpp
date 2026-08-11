/* Kernel test for reusable Q8_0 activation quantization and consumption. */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <vulkan/vulkan.h>
#include "../../include/vk_mem_alloc.h"
#include "../../../ds4_vulkan.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <vector>

static uint16_t f32_to_f16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t exp = (bits >> 23) & 0xffu;
    uint32_t mant = bits & 0x7fffffu;
    if (exp == 0xffu) return (uint16_t)(sign | 0x7c00u);
    if (exp == 0u) return (uint16_t)sign;
    int32_t e16 = (int32_t)exp - 127 + 15;
    if (e16 >= 31) return (uint16_t)(sign | 0x7c00u);
    if (e16 <= 0) return (uint16_t)sign;
    return (uint16_t)(sign | ((uint32_t)e16 << 10) | (mant >> 13));
}

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0u) bits = sign;
    else if (exp == 31u) bits = sign | 0x7f800000u | (mant << 13);
    else bits = sign | ((exp + 112u) << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

static void make_q8_weights(std::vector<uint8_t> &model, uint64_t offset,
                            uint64_t in_dim, uint64_t out_dim, int variant) {
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    for (uint64_t row = 0; row < out_dim; row++) {
        uint8_t *dst = model.data() + offset + row * row_bytes;
        for (uint64_t block = 0; block < blocks; block++) {
            uint16_t scale = f32_to_f16(0.25f * (float)(row + 1u));
            std::memcpy(dst + block * 34u, &scale, sizeof(scale));
            int8_t *q = (int8_t *)(dst + block * 34u + 2u);
            for (uint32_t i = 0; i < 32u; i++) {
                uint64_t k = block * 32u + i;
                q[i] = (k < in_dim) ? (int8_t)(((int)(k * 7u + row * 11u +
                    (uint64_t)variant * 13u) % 63) - 31) : 0;
            }
        }
    }
}

static int test_matmul_q8_0_prequant() {
    const uint64_t in_dim = 47;
    const uint64_t out_dim = 4;
    const uint64_t n_tok = 3;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t weight0 = 4096;
    const uint64_t weight1 = weight0 + out_dim * row_bytes + 4096;
    std::vector<uint8_t> model(weight1 + out_dim * row_bytes, 0xA5);
    make_q8_weights(model, weight0, in_dim, out_dim, 0);
    make_q8_weights(model, weight1, in_dim, out_dim, 1);

    std::vector<float> input(n_tok * in_dim, 0.0f);
    for (uint64_t t = 0; t < n_tok; t++)
        for (uint64_t i = 0; i < in_dim; i++)
            input[t * in_dim + i] = (t == 1 && i < 32u) ? 0.0f :
                (float)((int)((i * 19u + t * 23u) % 101u) - 50) * 0.03125f;

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(input.size() * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(n_tok * blocks * 36u);
    ds4_gpu_tensor *out0 = ds4_gpu_tensor_alloc(n_tok * out_dim * sizeof(float));
    ds4_gpu_tensor *out1 = ds4_gpu_tensor_alloc(n_tok * out_dim * sizeof(float));
    ds4_gpu_tensor *out_ref = ds4_gpu_tensor_alloc(n_tok * out_dim * sizeof(float));
    if (!x || !q || !out0 || !out1 || !out_ref) {
        ds4_gpu_tensor_free(out_ref);
        ds4_gpu_tensor_free(out1); ds4_gpu_tensor_free(out0);
        ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(x);
        return 1;
    }
    int rc = 1;
    auto cleanup = [&]() {
        ds4_gpu_tensor_free(out_ref);
        ds4_gpu_tensor_free(out1); ds4_gpu_tensor_free(out0);
        ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(x);
        return rc;
    };
    if (!ds4_gpu_tensor_write(x, 0, input.data(), input.size() * sizeof(float)) ||
        !ds4_gpu_begin_commands() ||
        !ds4_gpu_quantize_q8_0_tensor(q, x, in_dim, n_tok) ||
        !ds4_gpu_end_commands()) return cleanup();

    std::vector<uint8_t> packed(q->bytes);
    if (!ds4_gpu_tensor_read(q, 0, packed.data(), packed.size())) return cleanup();
    const uint8_t *zero_block = packed.data() + (1u * blocks) * 36u;
    for (uint32_t i = 0; i < 36u; i++)
        if (zero_block[i] != 0) return cleanup();
    const uint8_t *tail_block = packed.data() + 36u;
    for (uint32_t i = 4u + 15u; i < 36u; i++)
        if (tail_block[i] != 0) return cleanup();

    if (!ds4_gpu_begin_commands() ||
        !ds4_gpu_matmul_q8_0_prequant_tensor(out0, model.data(), model.size(),
            weight0, in_dim, out_dim, q, n_tok) ||
        !ds4_gpu_matmul_q8_0_prequant_tensor(out1, model.data(), model.size(),
            weight1, in_dim, out_dim, q, n_tok) ||
        !ds4_gpu_end_commands()) return cleanup();

    std::vector<float> established[2] = {
        std::vector<float>(n_tok * out_dim),
        std::vector<float>(n_tok * out_dim),
    };
    const uint64_t established_weights[2] = {weight0, weight1};
    for (int which = 0; which < 2; which++) {
        if (!ds4_gpu_begin_commands() ||
            !ds4_gpu_matmul_q8_0_tensor(out_ref, model.data(), model.size(),
                established_weights[which], in_dim, out_dim, x, n_tok) ||
            !ds4_gpu_end_commands() ||
            !ds4_gpu_tensor_read(out_ref, 0, established[which].data(),
                                 established[which].size() * sizeof(float)))
            return cleanup();
    }

    for (int which = 0; which < 2; which++) {
        std::vector<float> got(n_tok * out_dim);
        ds4_gpu_tensor *out = which == 0 ? out0 : out1;
        if (!ds4_gpu_tensor_read(out, 0, got.data(), got.size() * sizeof(float))) return cleanup();
        const uint64_t weight = which == 0 ? weight0 : weight1;
        for (uint64_t t = 0; t < n_tok; t++) {
            int8_t xq[64] = {};
            float xscale[2] = {};
            for (uint64_t b = 0; b < blocks; b++) {
                float amax = 0.0f;
                for (uint32_t i = 0; i < 32u && b * 32u + i < in_dim; i++)
                    amax = std::fmaxf(amax, std::fabsf(input[t * in_dim + b * 32u + i]));
                xscale[b] = amax / 127.0f;
                for (uint32_t i = 0; i < 32u && b * 32u + i < in_dim; i++) {
                    int qv = (int)std::lrintf(input[t * in_dim + b * 32u + i] /
                                             (xscale[b] == 0.0f ? 1.0f : xscale[b]));
                    xq[b * 32u + i] = (int8_t)std::fmax(-128, std::fmin(127, qv));
                }
            }
            for (uint64_t row = 0; row < out_dim; row++) {
                const uint8_t *w = model.data() + weight + row * row_bytes;
                float sum = 0.0f;
                for (uint64_t b = 0; b < blocks; b++) {
                    uint16_t sb; std::memcpy(&sb, w + b * 34u, sizeof(sb));
                    int dot = 0;
                    for (uint32_t i = 0; i < 32u; i++)
                        dot += (int)((int8_t)w[b * 34u + 2u + i]) * xq[b * 32u + i];
                    sum += f16_to_f32(sb) * xscale[b] * (float)dot;
                }
                if (std::fabsf(got[t * out_dim + row] - sum) > 1e-5f) return cleanup();
                if (std::fabsf(got[t * out_dim + row] -
                               established[which][t * out_dim + row]) > 1e-5f)
                    return cleanup();
            }
        }
    }
    rc = 0;
    return cleanup();
}

REGISTER_TEST(matmul_q8_0_prequant, test_matmul_q8_0_prequant);