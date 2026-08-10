/* End-to-end output-head composition using real Vulkan kernels. */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint16_t f32_to_f16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t e = (bits >> 23) & 0xffu;
    const uint32_t m = bits & 0x7fffffu;
    if (e == 0xffu) return (uint16_t)(sign | 0x7c00u | (m ? 0x200u : 0u));
    if (e == 0u) return (uint16_t)sign;
    const int32_t e16 = (int32_t)e - 127 + 15;
    if (e16 >= 31) return (uint16_t)(sign | 0x7c00u);
    if (e16 <= 0) return (uint16_t)(sign | ((0x800000u | m) >> (126u - e)));
    return (uint16_t)(sign | ((uint32_t)e16 << 10) | (m >> 13));
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1fu;
    const uint32_t m = h & 0x3ffu;
    uint32_t bits;
    if (e == 0u) {
        if (m == 0u) bits = sign;
        else {
            uint32_t mm = m, p = 0u;
            while (!(mm & 0x400u) && p < 10u) { mm <<= 1; p++; }
            bits = sign | ((113u - p) << 23) | ((mm & 0x3ffu) << 13);
        }
    } else if (e == 31u) bits = sign | 0x7f800000u | (m << 13);
    else bits = sign | ((e + 112u) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static void quant_q8_row(uint8_t *row, const float *x, uint32_t n) {
    for (uint32_t b = 0; b < (n + 31u) / 32u; b++) {
        const uint32_t first = b * 32u;
        const uint32_t count = n - first < 32u ? n - first : 32u;
        float amax = 0.0f;
        for (uint32_t i = 0; i < count; i++) amax = std::fmax(amax, std::fabs(x[first + i]));
        const float scale = amax == 0.0f ? 0.0f : amax / 127.0f;
        const uint16_t hs = f32_to_f16(scale);
        row[b * 34u] = (uint8_t)hs;
        row[b * 34u + 1u] = (uint8_t)(hs >> 8);
        for (uint32_t i = 0; i < 32u; i++) {
            const float value = i < count && scale != 0.0f ? x[first + i] / scale : 0.0f;
            int q = (int)std::lrint(value);
            if (q < -127) q = -127;
            if (q > 127) q = 127;
            row[b * 34u + 2u + i] = (uint8_t)(int8_t)q;
        }
    }
}

static void ref_q8_dot(float *out, const float *x, const uint8_t *weights,
                       uint32_t in_dim, uint32_t vocab) {
    const uint32_t blocks = (in_dim + 31u) / 32u;
    for (uint32_t o = 0; o < vocab; o++) {
        double acc = 0.0;
        const uint8_t *row = weights + (uint64_t)o * blocks * 34u;
        for (uint32_t b = 0; b < blocks; b++) {
            uint16_t hs = (uint16_t)row[b * 34u] | ((uint16_t)row[b * 34u + 1u] << 8);
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32u && b * 32u + i < in_dim; i++)
                amax = std::fmax(amax, std::fabs(x[b * 32u + i]));
            const float xd = amax / 127.0f;
            const float wd = f16_to_f32(hs);
            int dot = 0;
            for (uint32_t i = 0; i < 32u && b * 32u + i < in_dim; i++) {
                int xq = xd == 0.0f ? 0 : (int)std::nearbyint(x[b * 32u + i] / xd);
                if (xq < -128) xq = -128;
                if (xq > 127) xq = 127;
                dot += (int)(int8_t)row[b * 34u + 2u + i] * xq;
            }
            acc += (double)wd * (double)xd * (double)dot;
        }
        out[o] = (float)acc;
    }
}

static int check_output_head(const std::vector<float> &got,
                             const std::vector<float> &want) {
    if (got.size() != want.size()) return 1;
    for (size_t i = 0; i < got.size(); i++) {
        if (!std::isfinite(got[i]) || std::fabs(got[i] - want[i]) > 2e-3f) {
            std::fprintf(stderr, "output_head[%zu]: got=%g want=%g\n",
                         i, got[i], want[i]);
            return 1;
        }
    }
    return 0;
}

static int test_output_head(void) {
    const uint32_t n_hc = 4, n_embd = 33, vocab = 65537;
    const uint32_t hc_dim = n_hc * n_embd;
    const uint32_t blocks = (n_embd + 31u) / 32u;
    const uint64_t fn_offset = 64;
    const uint64_t scale_offset = fn_offset + (uint64_t)n_hc * hc_dim * 2u;
    const uint64_t base_offset = scale_offset + sizeof(float);
    const uint64_t norm_offset = base_offset + n_hc * sizeof(float);
    const uint64_t output_offset = norm_offset + n_embd * sizeof(float);
    const uint64_t output_bytes = (uint64_t)vocab * blocks * 34u;
    const uint64_t model_size = output_offset + output_bytes;

    std::vector<uint8_t> model(model_size, 0);
    uint16_t *fn = reinterpret_cast<uint16_t *>(model.data() + fn_offset);
    std::vector<float> fn_f32((size_t)n_hc * hc_dim), scale(1, 1.15f);
    std::vector<float> base(n_hc), norm_w(n_embd);
    for (uint32_t h = 0; h < n_hc; h++) {
        base[h] = -0.2f + 0.1f * h;
        for (uint32_t i = 0; i < hc_dim; i++) {
            const float value = ((int)((h + 3u) * (i + 5u) % 17u) - 8) * 0.035f;
            fn_f32[(size_t)h * hc_dim + i] = value;
            fn[(size_t)h * hc_dim + i] = f32_to_f16(value);
        }
    }
    for (uint32_t i = 0; i < n_embd; i++) norm_w[i] = 0.8f + 0.01f * (i % 9u);
    std::memcpy(model.data() + scale_offset, scale.data(), sizeof(float));
    std::memcpy(model.data() + base_offset, base.data(), base.size() * sizeof(float));
    std::memcpy(model.data() + norm_offset, norm_w.data(), norm_w.size() * sizeof(float));

    std::vector<float> q8_weights((size_t)vocab * n_embd);
    std::vector<uint8_t> q8(model.begin() + output_offset, model.end());
    for (uint32_t o = 0; o < vocab; o++)
        for (uint32_t i = 0; i < n_embd; i++)
            q8_weights[(size_t)o * n_embd + i] =
                ((int)((o * 7u + i * 3u) % 23u) - 11) * 0.021f;
    for (uint32_t o = 0; o < vocab; o++)
        quant_q8_row(q8.data() + (uint64_t)o * blocks * 34u,
                     q8_weights.data() + (uint64_t)o * n_embd, n_embd);
    std::memcpy(model.data() + output_offset, q8.data(), q8.size());

    std::vector<float> input(hc_dim);
    for (uint32_t i = 0; i < hc_dim; i++) input[i] = ((int)(i * 11u % 29u) - 14) * 0.07f;
    std::vector<float> flat(hc_dim), pre(n_hc), weights(n_hc), embd(n_embd), norm(n_embd), want(vocab);
    double ss = 0.0;
    for (float v : input) ss += (double)v * v;
    const float inv = 1.0f / std::sqrt((float)(ss / hc_dim) + 1e-6f);
    for (uint32_t i = 0; i < hc_dim; i++) flat[i] = input[i] * inv;
    for (uint32_t h = 0; h < n_hc; h++) {
        double acc = 0.0;
        for (uint32_t i = 0; i < hc_dim; i++)
            acc += (double)f16_to_f32(fn[(size_t)h * hc_dim + i]) * flat[i];
        pre[h] = (float)acc;
        weights[h] = 1.0f / (1.0f + std::exp(-(pre[h] * scale[0] + base[h]))) + 1e-6f;
    }
    for (uint32_t i = 0; i < n_embd; i++) {
        double acc = 0.0;
        for (uint32_t h = 0; h < n_hc; h++) acc += (double)input[h * n_embd + i] * weights[h];
        embd[i] = (float)acc;
    }
    ss = 0.0;
    for (float v : embd) ss += (double)v * v;
    const float norm_inv = 1.0f / std::sqrt((float)(ss / n_embd) + 1e-6f);
    for (uint32_t i = 0; i < n_embd; i++) norm[i] = embd[i] * norm_inv * norm_w[i];
    ref_q8_dot(want.data(), norm.data(), q8.data(), n_embd, vocab);

    ds4_gpu_tensor *input_t = ds4_gpu_tensor_alloc((uint64_t)hc_dim * sizeof(float));
    ds4_gpu_tensor *flat_t = ds4_gpu_tensor_alloc((uint64_t)hc_dim * sizeof(float));
    ds4_gpu_tensor *pre_t = ds4_gpu_tensor_alloc((uint64_t)n_hc * sizeof(float));
    ds4_gpu_tensor *weights_t = ds4_gpu_tensor_alloc((uint64_t)n_hc * sizeof(float));
    ds4_gpu_tensor *embd_t = ds4_gpu_tensor_alloc((uint64_t)n_embd * sizeof(float));
    ds4_gpu_tensor *norm_t = ds4_gpu_tensor_alloc((uint64_t)n_embd * sizeof(float));
    ds4_gpu_tensor *logits_t = ds4_gpu_tensor_alloc((uint64_t)vocab * sizeof(float));
    auto cleanup = [&]() {
        ds4_gpu_tensor_free(input_t); ds4_gpu_tensor_free(flat_t);
        ds4_gpu_tensor_free(pre_t); ds4_gpu_tensor_free(weights_t);
        ds4_gpu_tensor_free(embd_t); ds4_gpu_tensor_free(norm_t);
        ds4_gpu_tensor_free(logits_t);
    };
    if (!input_t || !flat_t || !pre_t || !weights_t || !embd_t || !norm_t || !logits_t ||
        !ds4_gpu_set_model_map(model.data(), model.size()) ||
        !ds4_gpu_tensor_write(input_t, 0, input.data(), input.size() * sizeof(float))) {
        cleanup();
        return 1;
    }

    int rc = 1;
    if (ds4_gpu_rms_norm_plain_tensor(flat_t, input_t, hc_dim, 1e-6f) != 0 &&
        ds4_gpu_matmul_f16_tensor(pre_t, model.data(), model.size(), fn_offset,
                                  hc_dim, n_hc, flat_t, 1) != 0 &&
        ds4_gpu_output_hc_weights_tensor(weights_t, pre_t, model.data(), model.size(),
                                         scale_offset, base_offset, n_hc, 1e-6f) != 0 &&
        ds4_gpu_hc_weighted_sum_tensor(embd_t, input_t, weights_t, n_embd, n_hc) != 0 &&
        ds4_gpu_rms_norm_weight_tensor(norm_t, embd_t, model.data(), model.size(),
                                       norm_offset, n_embd, 1e-6f) != 0 &&
        ds4_gpu_matmul_q8_0_tensor(logits_t, model.data(), model.size(), output_offset,
                                   n_embd, vocab, norm_t, 1) != 0) {
        std::vector<float> got(vocab);
        if (ds4_gpu_tensor_read(logits_t, 0, got.data(), got.size() * sizeof(float)) != 0)
            rc = check_output_head(got, want);
    }
    cleanup();
    return rc;
}

REGISTER_TEST(output_head, test_output_head);