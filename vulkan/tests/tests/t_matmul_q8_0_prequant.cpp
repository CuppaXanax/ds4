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
#include <cstdlib>
#include <chrono>
#include <string>
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

    std::vector<uint8_t> packed(ds4_gpu_tensor_bytes(q));
    if (!ds4_gpu_tensor_read(q, 0, packed.data(), packed.size())) return cleanup();
    const uint8_t *zero_block = packed.data() + (1u * blocks) * 36u;
    for (uint32_t i = 0; i < 36u; i++)
        if (zero_block[i] != 0) return cleanup();
    const uint8_t *tail_block = packed.data() + 36u;
    for (uint32_t i = 4u + 16u; i < 36u; i++)
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

    const bool production_ok =
        ds4_gpu_matmul_q8_0_tensor(out_ref, model.data(), model.size(),
                                    weight0, in_dim, out_dim, x, n_tok) != 0;
    std::vector<float> production(n_tok * out_dim);
    const bool production_read = production_ok &&
        ds4_gpu_tensor_read(out_ref, 0, production.data(),
                             production.size() * sizeof(float)) != 0;
    if (!production_read) return cleanup();
    for (size_t i = 0; i < production.size(); i++)
        if (std::fabsf(production[i] - established[0][i]) > 1e-5f)
            return cleanup();

    rc = 0;
    return cleanup();
}

REGISTER_TEST(matmul_q8_0_prequant, test_matmul_q8_0_prequant);

static void set_q8_env(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

static int run_q8_wave64_unpack(bool benchmark) {
    if (benchmark && !std::getenv("DS4_TEST_BENCH_Q8_WAVE64_UNPACK")) return 0;
    const uint64_t in_dim = 8192u, out_dim = 4096u, n_tok = 1u;
    const uint64_t blocks = 256u, row_bytes = blocks * 34u;
    const uint64_t weight_offset = 4096u;
    const uint64_t model_size = weight_offset + out_dim * row_bytes;
    std::vector<uint8_t> model(model_size, 0u);
    for (uint64_t record = 0; record < out_dim * blocks; ++record) {
        uint8_t *block = model.data() + weight_offset + record * 34u;
        uint16_t scale_bits;
        if (record % 19u == 0u) scale_bits = 0u;
        else if (record % 19u == 1u)
            scale_bits = (uint16_t)(1u + (record % 0x03ffu));
        else
            scale_bits = f32_to_f16((float)((record % 13u) + 1u) /
                                    1024.0f);
        std::memcpy(block, &scale_bits, sizeof(scale_bits));
        int8_t *payload = reinterpret_cast<int8_t *>(block + 2u);
        for (uint32_t i = 0; i < 32u; ++i)
            payload[i] = (int8_t)((int)((record * 17u + i * 29u) % 255u) -
                                  127);
    }
    std::vector<float> input(in_dim);
    for (uint64_t i = 0; i < in_dim; ++i)
        input[i] = (float)((int)((i * 37u) % 127u) - 63) * (1.0f / 64.0f);
    std::vector<float> reference(out_dim), candidate(out_dim);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(input.size() * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(blocks * 36u);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_dim * sizeof(float));

    const char *saved_wave64_ptr = std::getenv("DS4_VULKAN_Q8_WAVE64");
    const char *saved_unpack_ptr = std::getenv("DS4_VULKAN_Q8_WAVE64_UNPACK");
    const char *saved_mode_ptr = std::getenv("DS4_VULKAN_Q8_MODE");
    const bool had_wave64 = saved_wave64_ptr != nullptr;
    const bool had_unpack = saved_unpack_ptr != nullptr;
    const bool had_mode = saved_mode_ptr != nullptr;
    const std::string saved_wave64 = saved_wave64_ptr ? saved_wave64_ptr : "";
    const std::string saved_unpack = saved_unpack_ptr ? saved_unpack_ptr : "";
    const std::string saved_mode = saved_mode_ptr ? saved_mode_ptr : "";
    int rc = 1;
    if (!x || !q || !out ||
        !ds4_gpu_set_model_map(model.data(), model.size()) ||
        !ds4_gpu_tensor_write(x, 0, input.data(), input.size() * sizeof(float)) ||
        !ds4_gpu_begin_commands() ||
        !ds4_gpu_quantize_q8_0_tensor(q, x, in_dim, n_tok) ||
        !ds4_gpu_end_commands()) goto done;

    set_q8_env("DS4_VULKAN_Q8_MODE", nullptr);
    set_q8_env("DS4_VULKAN_Q8_WAVE64", "1");
    set_q8_env("DS4_VULKAN_Q8_WAVE64_UNPACK", "0");
    if (!ds4_gpu_matmul_q8_0_prequant_tensor(
            out, model.data(), model.size(), weight_offset,
            in_dim, out_dim, q, n_tok) ||
        ds4_gpu_q8_wave64_unpack_used() != 0 ||
        !ds4_gpu_tensor_read(out, 0, reference.data(),
                             reference.size() * sizeof(float))) goto done;

    set_q8_env("DS4_VULKAN_Q8_WAVE64_UNPACK", "1");
    if (!ds4_gpu_matmul_q8_0_prequant_tensor(
            out, model.data(), model.size(), weight_offset,
            in_dim, out_dim, q, n_tok) ||
        ds4_gpu_q8_wave64_unpack_used() != 1 ||
        !ds4_gpu_tensor_read(out, 0, candidate.data(),
                             candidate.size() * sizeof(float))) goto done;
    if (std::memcmp(reference.data(), candidate.data(),
                    reference.size() * sizeof(float)) != 0) {
        for (uint64_t i = 0; i < out_dim; ++i) {
            if (std::memcmp(&reference[i], &candidate[i], sizeof(float)) != 0) {
                fprintf(stderr,
                        "q8_wave64_unpack: i=%llu reference=%a candidate=%a\n",
                        (unsigned long long)i, reference[i], candidate[i]);
                break;
            }
        }
        goto done;
    }

    if (benchmark) {
        const uint32_t warmups = 4u, rounds = 16u;
        for (uint32_t i = 0; i < warmups; ++i) {
            set_q8_env("DS4_VULKAN_Q8_WAVE64_UNPACK", (i & 1u) ? "1" : "0");
            if (!ds4_gpu_matmul_q8_0_prequant_tensor(
                    out, model.data(), model.size(), weight_offset,
                    in_dim, out_dim, q, n_tok)) goto done;
        }
        double base_total_ms = 0.0, candidate_total_ms = 0.0;
        for (uint32_t i = 0; i < rounds; ++i) {
            const bool candidate_first = (i & 1u) != 0u;
            for (uint32_t pass = 0; pass < 2u; ++pass) {
                const bool use_candidate = candidate_first == (pass == 0u);
                set_q8_env("DS4_VULKAN_Q8_WAVE64_UNPACK",
                           use_candidate ? "1" : "0");
                const auto begin = std::chrono::steady_clock::now();
                if (!ds4_gpu_matmul_q8_0_prequant_tensor(
                        out, model.data(), model.size(), weight_offset,
                        in_dim, out_dim, q, n_tok)) goto done;
                const auto end = std::chrono::steady_clock::now();
                const double ms = std::chrono::duration<double, std::milli>(
                    end - begin).count();
                if (use_candidate) candidate_total_ms += ms;
                else base_total_ms += ms;
            }
        }
        const double base_ms = base_total_ms / rounds;
        const double candidate_ms = candidate_total_ms / rounds;
        fprintf(stderr,
                "q8_wave64_unpack_bench: shape=%llux%llu base_ms=%.3f "
                "candidate_ms=%.3f speedup=%.3fx\n",
                (unsigned long long)out_dim, (unsigned long long)in_dim,
                base_ms, candidate_ms,
                candidate_ms != 0.0 ? base_ms / candidate_ms : 0.0);
    }
    rc = 0;
done:
    if (had_wave64) set_q8_env("DS4_VULKAN_Q8_WAVE64", saved_wave64.c_str());
    else set_q8_env("DS4_VULKAN_Q8_WAVE64", nullptr);
    if (had_unpack) set_q8_env("DS4_VULKAN_Q8_WAVE64_UNPACK", saved_unpack.c_str());
    else set_q8_env("DS4_VULKAN_Q8_WAVE64_UNPACK", nullptr);
    if (had_mode) set_q8_env("DS4_VULKAN_Q8_MODE", saved_mode.c_str());
    else set_q8_env("DS4_VULKAN_Q8_MODE", nullptr);
    if (out) ds4_gpu_tensor_free(out);
    if (q) ds4_gpu_tensor_free(q);
    if (x) ds4_gpu_tensor_free(x);
    return rc;
}

static int test_q8_wave64_unpack_exact_ab(void) {
    return run_q8_wave64_unpack(false);
}
REGISTER_TEST(q8_wave64_unpack_exact_ab, test_q8_wave64_unpack_exact_ab);

static int test_q8_wave64_unpack_bench(void) {
    return run_q8_wave64_unpack(true);
}
REGISTER_TEST(q8_wave64_unpack_bench, test_q8_wave64_unpack_bench);

/* Exercise all 256 lanes and make a change to the final FP32 addition order
 * observable.  2^24 + 1 - 2^24 is zero when accumulated left-to-right, but
 * can be one under a parallel tree.  Every factor is exactly representable,
 * so this is an exact ordering gate rather than a tolerance-based check. */
static int test_matmul_q8_0_aligned_reduction() {
    const uint64_t in_dim = 8192;  /* 256 Q8 blocks: four Wave64s on BC-250. */
    const uint64_t out_dim = 1;
    const uint64_t n_tok = 1;
    const uint64_t blocks = in_dim / 32u;
    const uint64_t row_bytes = blocks * 34u;
    const uint64_t weight_offset = 12288;
    std::vector<uint8_t> model(weight_offset + out_dim * row_bytes, 0xA5);
    uint8_t *weights = model.data() + weight_offset;
    const uint16_t unit_scale = f32_to_f16(1.0f);
    const uint16_t large_scale = f32_to_f16(512.0f);
    for (uint64_t block = 0; block < blocks; block++) {
        uint8_t *w = weights + block * 34u;
        std::memcpy(w, &unit_scale, sizeof(unit_scale));
        std::memset(w + 2u, 0, 32u);
    }

    /* xq is {127, 127, 1, 1, 1, 1, 1, 0...}, with xscale exactly one.
     * Its dot with {127, 127, 127, 127, 127, 127, 2} is 32768. */
    std::vector<float> input(n_tok * in_dim, 0.0f);
    for (uint64_t block = 0; block < blocks; block++) {
        float *x_block = input.data() + block * 32u;
        x_block[0] = 127.0f;
        x_block[1] = 127.0f;
        for (uint32_t i = 2u; i < 7u; i++) x_block[i] = 1.0f;
    }
    const int8_t large_q[7] = {127, 127, 127, 127, 127, 127, 2};
    for (uint32_t i = 0; i < 7u; i++) {
        weights[2u + i] = (uint8_t)large_q[i];
        weights[2u + 34u + i] = (uint8_t)(i == 6u ? 1 : 0);
        weights[2u + 2u * 34u + i] = (uint8_t)-large_q[i];
    }
    std::memcpy(weights, &large_scale, sizeof(large_scale));
    std::memcpy(weights + 2u * 34u, &large_scale, sizeof(large_scale));

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(input.size() * sizeof(float));
    ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(n_tok * blocks * 36u);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(n_tok * out_dim * sizeof(float));
    if (!x || !q || !out) {
        ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(x);
        return 1;
    }
    int rc = 1;
    auto cleanup = [&]() {
        ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(q); ds4_gpu_tensor_free(x);
        return rc;
    };

    if (!ds4_gpu_tensor_write(x, 0, input.data(), input.size() * sizeof(float)) ||
        !ds4_gpu_begin_commands() ||
        !ds4_gpu_quantize_q8_0_tensor(q, x, in_dim, n_tok) ||
        !ds4_gpu_matmul_q8_0_prequant_tensor(out, model.data(), model.size(),
                                               weight_offset, in_dim, out_dim, q, n_tok) ||
        !ds4_gpu_end_commands()) return cleanup();

    std::vector<float> got(n_tok * out_dim);
    if (!ds4_gpu_tensor_read(out, 0, got.data(), got.size() * sizeof(float)))
        return cleanup();
    if (got[0] != 0.0f) {
        fprintf(stderr, "matmul_q8_0_aligned_reduction: got=%g want=0\n", got[0]);
        return cleanup();
    }

    rc = 0;
    return cleanup();
}

REGISTER_TEST(matmul_q8_0_aligned_reduction, test_matmul_q8_0_aligned_reduction);
