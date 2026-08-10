#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static uint16_t dispatch_order_f16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) return (uint16_t)(sign | 0x7c00u | (mantissa ? 0x200u : 0u));
    if (exponent == 0u) return (uint16_t)sign;
    const int32_t half_exponent = (int32_t)exponent - 127 + 15;
    if (half_exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    if (half_exponent <= 0) return (uint16_t)sign;
    return (uint16_t)(sign | ((uint32_t)half_exponent << 10) | (mantissa >> 13));
}

static void dispatch_order_reference(float *out, const float *input,
                                     const float *weights,
                                     uint32_t in_dim, uint32_t out_dim) {
    for (uint32_t row = 0; row < out_dim; row++) {
        double sum = 0.0;
        for (uint32_t col = 0; col < in_dim; col++)
            sum += (double)input[col] * weights[(uint64_t)row * in_dim + col];
        out[row] = (float)sum;
    }
}

static int test_dispatch_order(void) {
    const uint32_t in_dim = 4;
    const uint32_t out_dim = 3;
    const uint64_t weight_bytes = (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const uint64_t weight0_offset = 16ull * 1024ull * 1024ull;
    const uint64_t weight1_offset = weight0_offset + weight_bytes;
    const uint64_t model_size = weight1_offset + weight_bytes;

    unsigned char *model = (unsigned char *)calloc(1, (size_t)model_size);
    ds4_gpu_tensor *input0 = ds4_gpu_tensor_alloc(in_dim * sizeof(float));
    ds4_gpu_tensor *input1 = ds4_gpu_tensor_alloc(in_dim * sizeof(float));
    ds4_gpu_tensor *out0 = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    ds4_gpu_tensor *out1 = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    ds4_gpu_tensor *normalized = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    auto cleanup = [&]() {
        if (normalized) ds4_gpu_tensor_free(normalized);
        if (out1) ds4_gpu_tensor_free(out1);
        if (out0) ds4_gpu_tensor_free(out0);
        if (input1) ds4_gpu_tensor_free(input1);
        if (input0) ds4_gpu_tensor_free(input0);
        free(model);
    };
    if (!model || !input0 || !input1 || !out0 || !out1 || !normalized) {
        cleanup();
        return 1;
    }

    const float weights0[in_dim * out_dim] = {
         1.0f,  2.0f,  3.0f,  4.0f,
        -1.0f,  0.5f,  2.0f, -0.5f,
         0.25f, 1.5f, -2.0f,  3.0f,
    };
    const float weights1[in_dim * out_dim] = {
        -2.0f,  1.0f,  0.5f,  3.0f,
         4.0f, -1.0f,  2.0f,  0.25f,
         1.0f,  1.0f,  1.0f,  1.0f,
    };
    uint16_t *stored0 = (uint16_t *)(model + weight0_offset);
    uint16_t *stored1 = (uint16_t *)(model + weight1_offset);
    for (uint32_t i = 0; i < in_dim * out_dim; i++) {
        stored0[i] = dispatch_order_f16(weights0[i]);
        stored1[i] = dispatch_order_f16(weights1[i]);
    }

    const float values0[in_dim] = {1.0f, -2.0f, 0.5f, 3.0f};
    const float values1[in_dim] = {-1.0f, 0.25f, 2.0f, -0.5f};
    float reference0[out_dim], reference1[out_dim], reference_norm[out_dim];
    dispatch_order_reference(reference0, values0, weights0, in_dim, out_dim);
    dispatch_order_reference(reference1, values1, weights1, in_dim, out_dim);
    double squares = 0.0;
    for (uint32_t i = 0; i < out_dim; i++) squares += (double)reference0[i] * reference0[i];
    const float inverse_rms = 1.0f / std::sqrt((float)(squares / out_dim) + 1e-6f);
    for (uint32_t i = 0; i < out_dim; i++) reference_norm[i] = reference0[i] * inverse_rms;

    if (ds4_gpu_commands_active() ||
        !ds4_gpu_set_model_map(model, model_size) ||
        !ds4_gpu_tensor_write(input0, 0, values0, sizeof(values0)) ||
        !ds4_gpu_tensor_write(input1, 0, values1, sizeof(values1)) ||
        !ds4_gpu_begin_commands() || !ds4_gpu_commands_active() ||
        !ds4_gpu_matmul_f16_tensor(out0, model, model_size, weight0_offset,
                                   in_dim, out_dim, input0, 1) ||
        !ds4_gpu_matmul_f16_tensor(out1, model, model_size, weight1_offset,
                                   in_dim, out_dim, input1, 1) ||
        !ds4_gpu_rms_norm_plain_tensor(normalized, out0, out_dim, 1e-6f) ||
        !ds4_gpu_end_commands() || ds4_gpu_commands_active()) {
        cleanup();
        return 1;
    }

    {
        float got0[out_dim], got1[out_dim], got_norm[out_dim];
        if (!ds4_gpu_tensor_read(out0, 0, got0, sizeof(got0)) ||
            !ds4_gpu_tensor_read(out1, 0, got1, sizeof(got1)) ||
            !ds4_gpu_tensor_read(normalized, 0, got_norm, sizeof(got_norm))) {
            cleanup();
            return 1;
        }
        for (uint32_t i = 0; i < out_dim; i++) {
            if (std::fabs(got0[i] - reference0[i]) > 1e-3f ||
                std::fabs(got1[i] - reference1[i]) > 1e-3f ||
                std::fabs(got_norm[i] - reference_norm[i]) > 1e-3f) {
                std::fprintf(stderr,
                             "dispatch_order[%u]: out0=%g/%g out1=%g/%g norm=%g/%g\n",
                             i, got0[i], reference0[i], got1[i], reference1[i],
                             got_norm[i], reference_norm[i]);
                cleanup();
                return 1;
            }
        }
    }

    cleanup();
    return 0;
}

REGISTER_TEST(dispatch_order, test_dispatch_order);