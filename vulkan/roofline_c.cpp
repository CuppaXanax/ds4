/* Benchmark C: isolated production Flash math roofline.
 *
 * This is intentionally not a session or distributed benchmark.  It invokes
 * the production Q8 GEMV once per sample and the production routed appliance
 * shape with only its hard math dependencies: input Q8 quantization, fused
 * IQ2 gate/up/SwiGLU, and Q2 down/reduce.  Timeline summaries are emitted by
 * the backend, so GPU time is not inferred from host wall time.
 *
 * Build with:
 *   make vulkan-roofline-c
 * Run on one BC-250 blade only; do not start the coordinator for this tool.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../ds4_gpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {

constexpr uint32_t kLayer = 4;
constexpr uint32_t kQ8In = 4096;
constexpr uint32_t kQ8Out = 4096;
constexpr uint32_t kQ8Blocks = kQ8In / 32;
constexpr uint32_t kQ8RowBytes = kQ8Blocks * 34;
constexpr uint64_t kQ8WeightBytes = uint64_t(kQ8Out) * kQ8RowBytes;
/* Q-B is the historically dominant Flash projection: 64 heads x 512. */
constexpr uint32_t kQBIn = 1024;
constexpr uint32_t kQBOut = 32768;
constexpr uint32_t kQBBlocks = kQBIn / 32;
constexpr uint32_t kQBRowBytes = kQBBlocks * 34;
constexpr uint64_t kQBWeightBytes = uint64_t(kQBOut) * kQBRowBytes;

constexpr uint32_t kExperts = 256;
constexpr uint32_t kTopK = 6;
constexpr uint32_t kExpertIn = 4096;
constexpr uint32_t kExpertMid = 2048;
constexpr uint32_t kExpertOut = 4096;
constexpr uint32_t kIQ2Blocks = kExpertIn / 256;
constexpr uint32_t kQ2Blocks = kExpertMid / 256;
constexpr uint32_t kIQ2RowBytes = kIQ2Blocks * 66;
constexpr uint32_t kQ2RowBytes = kQ2Blocks * 84;
constexpr uint64_t kGateExpertBytes = uint64_t(kExpertMid) * kIQ2RowBytes;
constexpr uint64_t kDownExpertBytes = uint64_t(kExpertOut) * kQ2RowBytes;
constexpr uint64_t kGateTableBytes = uint64_t(kExperts) * kGateExpertBytes;
constexpr uint64_t kDownTableBytes = uint64_t(kExperts) * kDownExpertBytes;
constexpr uint64_t kRoutedModelBytes = 4096 + 2 * kGateTableBytes + kDownTableBytes;
constexpr uint64_t kSelectedGateBytes = uint64_t(kTopK) * kGateExpertBytes;
constexpr uint64_t kSelectedDownBytes = uint64_t(kTopK) * kDownExpertBytes;

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t e = (x >> 23) & 0xffu;
    const uint32_t m = x & 0x7fffffu;
    if (e == 0xffu) return uint16_t(sign | 0x7c00u | (m ? 0x200u : 0u));
    if (e == 0u) return uint16_t(sign);
    const int32_t e16 = int32_t(e) - 127 + 15;
    if (e16 >= 31) return uint16_t(sign | 0x7c00u);
    if (e16 <= 0) {
        const uint32_t shift = 126u - e;
        return uint16_t(sign | ((0x800000u | m) >> shift));
    }
    return uint16_t(sign | (uint32_t(e16) << 10) | (m >> 13));
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = uint32_t(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1fu;
    const uint32_t m = h & 0x3ffu;
    uint32_t bits;
    if (e == 0u) {
        if (m == 0u) bits = sign;
        else {
            uint32_t mm = m, p = 0;
            while (!(mm & 0x400u) && p < 10u) { mm <<= 1; ++p; }
            bits = sign | ((113u - p) << 23) | ((mm & 0x3ffu) << 13);
        }
    } else if (e == 31u) bits = sign | 0x7f800000u | (m << 13);
    else bits = sign | ((e + 112u) << 23) | (m << 13);
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

static uint64_t fnv1a(const void *data, size_t bytes) {
    const auto *p = static_cast<const uint8_t *>(data);
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < bytes; ++i) { h ^= p[i]; h *= UINT64_C(1099511628211); }
    return h;
}

static void *model_alloc(uint64_t bytes) {
#if defined(__linux__)
    void *p = mmap(nullptr, size_t(bytes), PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p != MAP_FAILED) return p;
#endif
    return std::malloc(size_t(bytes));
}

static void model_free(void *p, uint64_t bytes) {
    if (!p) return;
#if defined(__linux__)
    if (munmap(p, size_t(bytes)) == 0) return;
#endif
    std::free(p);
}

static void fill_q8_row(uint8_t *row, uint32_t row_id, uint32_t blocks) {
    const uint16_t scale = f32_to_f16(0.125f + 0.0078125f * float(row_id & 7u));
    for (uint32_t b = 0; b < blocks; ++b) {
        std::memcpy(row + b * 34u, &scale, sizeof(scale));
        auto *q = reinterpret_cast<int8_t *>(row + b * 34u + 2u);
        for (uint32_t i = 0; i < 32; ++i)
            q[i] = int8_t(int((row_id * 13u + b * 17u + i * 7u) % 63u) - 31);
    }
}

static void fill_iq2_block(uint8_t *dst, uint32_t row, uint32_t expert,
                           uint32_t block) {
    /* Match the production IQ2_XXS fixture convention: d=4 gives the
     * canonical 0.5*(2*ls+1) scale after the shader's 0.125 epilogue. */
    const uint16_t d = f32_to_f16(4.0f);
    std::memcpy(dst, &d, sizeof(d));
    auto *words = reinterpret_cast<uint16_t *>(dst + 2u);
    for (uint32_t g = 0; g < 32; g += 4) {
        const uint32_t l = g / 4u;
        uint32_t aux0 = 0, aux1 = ((row + expert + block) % 3u) << 28;
        for (uint32_t t = 0; t < 4; ++t) {
            const uint32_t grid = (row * 7u + l * 11u + expert * 3u +
                                   block * 19u + t * 17u) & 255u;
            reinterpret_cast<uint8_t *>(&aux0)[t] = uint8_t(grid);
            aux1 |= ((row * 5u + l * 13u + expert * 7u + block * 3u + t) & 127u)
                    << (7u * t);
        }
        words[g + 0] = uint16_t(aux0 & 0xffffu);
        words[g + 1] = uint16_t(aux0 >> 16);
        words[g + 2] = uint16_t(aux1 & 0xffffu);
        words[g + 3] = uint16_t(aux1 >> 16);
    }
}

static void fill_q2_block(uint8_t *dst, uint32_t row, uint32_t expert) {
    std::memset(dst, 0, 84);
    for (uint32_t i = 0; i < 16; ++i) dst[i] = 0x02;
    auto *q = dst + 16;
    for (uint32_t i = 0; i < 64; ++i) {
        uint8_t v = 0;
        for (uint32_t t = 0; t < 4; ++t) {
            const uint32_t elem = i * 4u + t;
            v |= uint8_t(((elem * 3u + row * 7u + expert * 11u) & 3u) << (2u * t));
        }
        q[i] = v;
    }
    const uint16_t d = f32_to_f16(0.125f);
    std::memcpy(dst + 80, &d, sizeof(d));
}

static void fill_routed_model(uint8_t *model, uint64_t gate_offset,
                              uint64_t up_offset, uint64_t down_offset) {
    for (uint32_t e = 0; e < kTopK; ++e) {
        for (uint32_t row = 0; row < kExpertMid; ++row) {
            for (uint32_t b = 0; b < kIQ2Blocks; ++b) {
                const uint64_t rel = uint64_t(e) * kGateExpertBytes +
                                     uint64_t(row) * kIQ2RowBytes + uint64_t(b) * 66u;
                fill_iq2_block(model + gate_offset + rel, row, e, b);
                fill_iq2_block(model + up_offset + rel, row, e, b);
            }
        }
        for (uint32_t row = 0; row < kExpertOut; ++row) {
            for (uint32_t b = 0; b < kQ2Blocks; ++b) {
                const uint64_t rel = uint64_t(e) * kDownExpertBytes +
                                     uint64_t(row) * kQ2RowBytes + uint64_t(b) * 84u;
                fill_q2_block(model + down_offset + rel, row, e);
            }
        }
    }
}

static void fill_q8_model(uint8_t *model, uint64_t offset, uint32_t rows,
                          uint32_t blocks, uint32_t row_bytes) {
    for (uint32_t row = 0; row < rows; ++row)
        fill_q8_row(model + offset + uint64_t(row) * row_bytes, row, blocks);
}

static void fill_activation(ds4_gpu_tensor *x, uint32_t n) {
    std::vector<float> values(n);
    for (uint32_t i = 0; i < n; ++i)
        values[i] = float(int((i * 37u + 11u) % 257u) - 128) * 0.00390625f;
    ds4_gpu_tensor_write(x, 0, values.data(), values.size() * sizeof(float));
}

static bool finite_hash(const ds4_gpu_tensor *tensor, uint32_t count,
                        uint64_t &hash_out) {
    std::vector<float> values(count);
    if (!ds4_gpu_tensor_read(tensor, 0, values.data(), values.size() * sizeof(float)))
        return false;
    for (float v : values) if (!std::isfinite(v)) return false;
    hash_out = fnv1a(values.data(), values.size() * sizeof(float));
    return true;
}

static int run_q8_case(const char *label, uint32_t in_dim, uint32_t out_dim,
                       uint32_t blocks, uint32_t row_bytes,
                       uint64_t weight_bytes) {
    const uint64_t model_size = 4096u + weight_bytes;
    auto *model = static_cast<uint8_t *>(model_alloc(model_size));
    if (!model) return 1;
    std::memset(model, 0, 4096);
    fill_q8_model(model, 4096, out_dim, blocks, row_bytes);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(uint64_t(in_dim) * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(uint64_t(out_dim) * sizeof(float));
    if (!x || !out || !ds4_gpu_set_model_map(model, model_size)) return 1;
    fill_activation(x, in_dim);
    std::printf("roofline_c family=%s in=%u out=%u blocks=%u "
                "weight_bytes=%llu useful_weight_bytes=%llu dispatches=1\n",
                label, in_dim, out_dim, blocks,
                (unsigned long long)weight_bytes,
                (unsigned long long)weight_bytes);
    uint64_t baseline = 0;
    for (int iter = 0; iter < 2; ++iter) {
        if (!ds4_gpu_begin_commands()) return 1;
        ds4_gpu_timeline_layer_begin(kLayer);
        const int ok = ds4_gpu_matmul_q8_0_tensor(
            out, model, model_size, 4096, in_dim, out_dim, x, 1);
        ds4_gpu_timeline_stage_end(label);
        if (!ok || !ds4_gpu_end_commands() || !ds4_gpu_synchronize()) return 1;
        ds4_gpu_timeline_layer_end(kLayer);
        uint64_t hash = 0;
        if (!finite_hash(out, out_dim, hash)) return 1;
        if (iter == 0) baseline = hash;
        if (hash != baseline) return 1;
        std::printf("roofline_c %s_iter=%d output_fnv1a=%016llx checksum=PASS\n",
                    label,
                    iter, (unsigned long long)hash);
    }
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    model_free(model, model_size);
    return 0;
}

static int run_q8(void) {
    const int shared = run_q8_case("q8_shared_4096x4096", kQ8In, kQ8Out,
                                   kQ8Blocks, kQ8RowBytes, kQ8WeightBytes);
    if (shared != 0) return shared;
    return run_q8_case("q8_qb_1024x32768", kQBIn, kQBOut,
                        kQBBlocks, kQBRowBytes, kQBWeightBytes);
}

static int run_routed(void) {
    const uint64_t header = 4096;
    const uint64_t gate_offset = header;
    const uint64_t up_offset = gate_offset + kGateTableBytes;
    const uint64_t down_offset = up_offset + kGateTableBytes;
    auto *model = static_cast<uint8_t *>(model_alloc(kRoutedModelBytes));
    if (!model) return 1;
    std::memset(model, 0, header);
    fill_routed_model(model, gate_offset, up_offset, down_offset);
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(kExpertIn * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(kExpertOut * sizeof(float));
    /* Mid is the production compact-Q8 scratch owner in mid-only mode. */
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(64u * 1024u);
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(kTopK * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(kTopK * sizeof(float));
    if (!x || !out || !mid || !selected || !weights ||
        !ds4_gpu_set_model_map(model, kRoutedModelBytes)) return 1;
    fill_activation(x, kExpertIn);
    const int32_t ids[kTopK] = {0, 1, 2, 3, 4, 5};
    const float route_weights[kTopK] = {0.20f, 0.18f, 0.17f, 0.16f, 0.15f, 0.14f};
    ds4_gpu_tensor_write(selected, 0, ids, sizeof(ids));
    ds4_gpu_tensor_write(weights, 0, route_weights, sizeof(route_weights));
    std::printf("roofline_c family=routed_flash in=%u mid=%u out=%u experts=%u topk=%u "
                "gate_type=IQ2_XXS down_type=Q2_K model_table_bytes=%llu "
                "useful_weight_bytes=%llu hard_dispatches=3\n",
                kExpertIn, kExpertMid, kExpertOut, kExperts, kTopK,
                (unsigned long long)(2 * kGateTableBytes + kDownTableBytes),
                (unsigned long long)(2 * kSelectedGateBytes + kSelectedDownBytes));
    uint64_t baseline = 0;
    for (int iter = 0; iter < 2; ++iter) {
        if (!ds4_gpu_batch_layer_begin(kLayer)) return 1;
        ds4_gpu_timeline_layer_begin(kLayer);
        const int ok = ds4_gpu_routed_moe_one_tensor(
            out, nullptr, nullptr, mid, nullptr, model, kRoutedModelBytes,
            gate_offset, up_offset, down_offset, 16, 10,
            kGateExpertBytes, kIQ2RowBytes, kDownExpertBytes, kQ2RowBytes,
            kExpertIn, kExpertMid, kExpertOut, selected, weights,
            kExperts, kTopK, 0.25f, x, nullptr, kLayer, true);
        ds4_gpu_timeline_stage_end("routed_flash_math");
        ds4_gpu_timeline_layer_end(kLayer);
        if (!ok || !ds4_gpu_batch_layer_end(kLayer) || !ds4_gpu_synchronize()) return 1;
        uint64_t hash = 0;
        if (!finite_hash(out, kExpertOut, hash)) return 1;
        if (iter == 0) baseline = hash;
        if (hash != baseline) return 1;
        std::printf("roofline_c routed_iter=%d output_fnv1a=%016llx checksum=PASS\n",
                    iter, (unsigned long long)hash);
    }
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    model_free(model, kRoutedModelBytes);
    return 0;
}

} // namespace

int main() {
    if (!std::getenv("DS4_VULKAN_TIMELINE_LAYER"))
        setenv("DS4_VULKAN_TIMELINE_LAYER", "4", 1);
    setenv("DS4_VULKAN_TRACE_KERNELS", "1", 1);
    setenv("DS4_VULKAN_TIMELINE", "1", 1);
    if (!ds4_gpu_init()) {
        std::fprintf(stderr, "roofline_c: ds4_gpu_init failed\n");
        return 2;
    }
    const int q8_rc = run_q8();
    const int routed_rc = q8_rc == 0 ? run_routed() : 1;
    ds4_gpu_cleanup();
    if (q8_rc != 0 || routed_rc != 0) {
        std::fprintf(stderr, "roofline_c: FAIL q8=%d routed=%d\n", q8_rc, routed_rc);
        return 1;
    }
    std::puts("roofline_c: PASS");
    return 0;
}
