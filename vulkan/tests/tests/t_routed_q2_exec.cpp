#include "../tests.h"
#include "../../../ds4_gpu.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint16_t q2_exec_f16(float value) {
    uint32_t bits = 0; std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) return (uint16_t)(sign | 0x7c00u);
    const int32_t e = (int32_t)exponent - 127 + 15;
    if (e >= 31) return (uint16_t)(sign | 0x7c00u);
    if (e <= 0) return (uint16_t)sign;
    return (uint16_t)(sign | ((uint32_t)e << 10) | (mantissa >> 13));
}

static void fill_q2_block(uint8_t *dst, uint32_t row, uint32_t expert,
                          uint32_t block) {
    std::memset(dst, 0, 84);
    uint16_t d = q2_exec_f16(0.125f + 0.03125f * (float)(expert + 1));
    for (uint32_t i = 0; i < 16; ++i)
        dst[i] = (uint8_t)(0x11u + ((row + expert + block + i) & 7u));
    for (uint32_t i = 0; i < 64; ++i) {
        uint8_t q0 = (uint8_t)((i + row * 3u + expert * 5u + block) & 3u);
        uint8_t q1 = (uint8_t)((i + row * 7u + expert * 2u + block * 3u) & 3u);
        dst[16 + i] = (uint8_t)(q0 | (q1 << 2) | (q0 << 4) | (q1 << 6));
    }
    std::memcpy(dst + 80, &d, sizeof(d));
}

static int test_routed_q2_execution_artifact(void) {
    const uint32_t in_dim = 256, mid_dim = 256, out_dim = 8;
    const uint32_t n_total = 2, n_selected = 2;
    const uint64_t row_bytes = 84;
    const uint64_t gate_expert_bytes = (uint64_t)mid_dim * row_bytes;
    const uint64_t down_expert_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t gate_offset = 4096;
    const uint64_t up_offset = gate_offset + (uint64_t)n_total * gate_expert_bytes;
    const uint64_t down_offset = up_offset + (uint64_t)n_total * gate_expert_bytes;
    const uint64_t down_bytes = (uint64_t)n_total * down_expert_bytes;
    const uint64_t model_size = down_offset + down_bytes + 4096;
    std::vector<uint8_t> model((size_t)model_size, 0xa5);
    for (uint32_t e = 0; e < n_total; ++e) {
        for (uint32_t row = 0; row < mid_dim; ++row) {
            fill_q2_block(model.data() + gate_offset + (uint64_t)e * gate_expert_bytes +
                              (uint64_t)row * row_bytes, row, e, 0);
            fill_q2_block(model.data() + up_offset + (uint64_t)e * gate_expert_bytes +
                              (uint64_t)row * row_bytes, row + 11, e, 1);
        }
        for (uint32_t row = 0; row < out_dim; ++row)
            fill_q2_block(model.data() + down_offset + (uint64_t)e * down_expert_bytes +
                              (uint64_t)row * row_bytes, row + 23, e, 2);
    }
    std::vector<float> x(in_dim);
    for (uint32_t i = 0; i < in_dim; ++i)
        x[i] = (float)((int)((i * 13u + 5u) % 17u) - 8) * 0.125f;
    const int32_t selected[n_selected] = {1, 0};
    const float weights[n_selected] = {0.6f, 0.4f};
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(mid_dim * n_selected * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(mid_dim * n_selected * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(mid_dim * n_selected * sizeof(float));
    ds4_gpu_tensor *experts = ds4_gpu_tensor_alloc(out_dim * n_selected * sizeof(float));
    ds4_gpu_tensor *sel = ds4_gpu_tensor_alloc(n_selected * sizeof(int32_t));
    ds4_gpu_tensor *wgt = ds4_gpu_tensor_alloc(n_selected * sizeof(float));
    ds4_gpu_tensor *input = ds4_gpu_tensor_alloc(in_dim * sizeof(float));
    int result = 1;
    auto cleanup = [&]() {
        /* Retire the backend's model identity while the synthetic storage is
         * still alive.  The next test may reuse this allocation address; if
         * the dangling identity survives, cached execution/artifact state can
         * be mistaken for the next model and poison later router tests. */
        ds4_gpu_set_model_map(model.data(), model.size());
        unsetenv("DS4_VULKAN_ROUTED_Q2_EXECUTION");
        unsetenv("DS4_VULKAN_REQUIRE_ROUTED_Q2_EXECUTION");
        unsetenv("DS4_VULKAN_TEST_ROUTED_DOWN_REDUCE");
        ds4_gpu_tensor_free(experts); ds4_gpu_tensor_free(mid);
        ds4_gpu_tensor_free(up); ds4_gpu_tensor_free(gate); ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(wgt); ds4_gpu_tensor_free(sel); ds4_gpu_tensor_free(input);
        return result;
    };
    if (!out || !gate || !up || !mid || !experts || !sel || !wgt || !input ||
        !ds4_gpu_set_model_map(model.data(), model.size()) ||
        !ds4_gpu_tensor_write(sel, 0, selected, sizeof(selected)) ||
        !ds4_gpu_tensor_write(wgt, 0, weights, sizeof(weights)) ||
        !ds4_gpu_tensor_write(input, 0, x.data(), x.size() * sizeof(float)))
        return cleanup();
    auto run = [&]() {
        if (!ds4_gpu_batch_layer_begin(1)) return 0;
        const int ok = ds4_gpu_routed_moe_one_tensor(
            out, gate, up, mid, experts, model.data(), model.size(),
            gate_offset, up_offset, down_offset, 10, 10,
            gate_expert_bytes, row_bytes, down_expert_bytes, row_bytes,
            in_dim, mid_dim, out_dim, sel, wgt, n_total, n_selected,
            0.25f, input, nullptr, 0, false);
        return (ok && ds4_gpu_batch_layer_end(1)) ? 1 : 0;
    };
    setenv("DS4_VULKAN_TEST_ROUTED_DOWN_REDUCE", "1", 1);
    setenv("DS4_VULKAN_ROUTED_Q2_EXECUTION", "0", 1);
    if (!run()) { std::fprintf(stderr, "q2_exec: baseline run failed\n"); return cleanup(); }
    /* Retire the baseline before reading it or replacing its raw weight
     * range.  A command-ring read can otherwise observe the old mapped
     * contents and leave the overlap guard unable to remove the raw cache. */
    if (!ds4_gpu_synchronize()) {
        std::fprintf(stderr, "q2_exec: baseline synchronize failed\n");
        return cleanup();
    }
    std::vector<float> baseline(out_dim);
    if (!ds4_gpu_tensor_read(out, 0, baseline.data(), baseline.size() * sizeof(float))) {
        std::fprintf(stderr, "q2_exec: baseline read failed\n");
        return cleanup();
    }
    /* Build the execution artifact while the execution path is enabled.  The
     * cache API intentionally honors DS4_VULKAN_ROUTED_Q2_EXECUTION=0 by
     * taking the raw fallback, so calling it in the baseline mode would leave
     * the required artifact absent for the zeroed-raw verification below. */
    setenv("DS4_VULKAN_ROUTED_Q2_EXECUTION", "1", 1);
    setenv("DS4_VULKAN_REQUIRE_ROUTED_Q2_EXECUTION", "1", 1);
    if (!ds4_gpu_cache_q2_execution_range(
            model.data(), model.size(), down_offset, down_bytes,
            mid_dim, (uint64_t)n_total * out_dim, "routed-q2-exec-test")) {
        std::fprintf(stderr, "q2_exec: artifact cache failed\n");
        return cleanup();
    }
    std::memset(model.data() + down_offset, 0, (size_t)down_bytes);
    if (!run()) { std::fprintf(stderr, "q2_exec: artifact run failed\n"); return cleanup(); }
    std::vector<float> got(out_dim);
    if (!ds4_gpu_tensor_read(out, 0, got.data(), got.size() * sizeof(float))) {
        std::fprintf(stderr, "q2_exec: artifact read failed\n");
        return cleanup();
    }
    if (std::memcmp(got.data(), baseline.data(), got.size() * sizeof(float)) != 0) {
        std::fprintf(stderr, "q2_exec: mismatch\n");
        return cleanup();
    }
    result = 0;
    return cleanup();
}

REGISTER_TEST(routed_q2_execution_artifact, test_routed_q2_execution_artifact);
