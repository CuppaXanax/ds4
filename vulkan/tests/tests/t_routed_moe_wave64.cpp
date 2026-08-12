/* Exact same-binary A/B for the production routed-MoE reduction shapes.
 *
 * The IQ2 gate/up projection has 28 Q8_K blocks (32 lanes per row); the Q2
 * down projection has eight blocks (eight lanes per row).  The portable and
 * Wave64 paths must produce identical bits for every intermediate and output.
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr const char *kDisableWave64 = "DS4_VULKAN_DISABLE_ROUTED_WAVE64";
constexpr const char *kRequireWave64 = "DS4_VULKAN_REQUIRE_ROUTED_WAVE64";

struct Iq2Block {
    uint16_t d;
    uint16_t qs[32];
};

struct Q2Block {
    uint8_t scales[16];
    uint8_t qs[64];
    uint16_t d;
    uint16_t dmin;
};

static_assert(sizeof(Iq2Block) == 66, "IQ2_XXS block layout");
static_assert(sizeof(Q2Block) == 84, "Q2_K block layout");

uint16_t f32_to_f16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const uint32_t exponent = (bits >> 23) & 0xffu;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu)
        return (uint16_t)(sign | 0x7c00u | (mantissa ? 0x0200u : 0u));
    if (exponent == 0u) return (uint16_t)sign;
    const int32_t half_exponent = (int32_t)exponent - 127 + 15;
    if (half_exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    if (half_exponent <= 0) {
        const uint32_t shift = 126u - exponent;
        const uint32_t half_mantissa = (0x800000u | mantissa) >> shift;
        return (uint16_t)(sign | half_mantissa);
    }
    return (uint16_t)(sign | ((uint32_t)half_exponent << 10) |
                      (mantissa >> 13));
}

void fill_iq2_block(Iq2Block &block, uint32_t row, uint32_t column,
                    uint32_t kind) {
    block.d = f32_to_f16(0.25f);
    for (uint32_t group = 0; group < 8; ++group) {
        uint32_t grids = 0;
        uint32_t signs_and_scale = ((row + column + kind) % 3u) << 28u;
        for (uint32_t i = 0; i < 4; ++i) {
            const uint32_t grid =
                (row * 7u + column * 17u + group * 29u + kind * 43u + i * 11u) & 255u;
            const uint32_t signs =
                (row * 5u + column * 13u + group * 19u + kind * 3u + i) & 127u;
            grids |= grid << (8u * i);
            signs_and_scale |= signs << (7u * i);
        }
        memcpy(&block.qs[group * 4u], &grids, sizeof(grids));
        memcpy(&block.qs[group * 4u + 2u], &signs_and_scale,
               sizeof(signs_and_scale));
    }
}

void fill_q2_block(Q2Block &block, uint32_t row, uint32_t column) {
    block.d = f32_to_f16(0.125f);
    block.dmin = 0;
    for (uint32_t i = 0; i < 16; ++i)
        block.scales[i] = (uint8_t)(1u + ((row + column + i) % 7u));
    for (uint32_t i = 0; i < 64; ++i)
        block.qs[i] = (uint8_t)(row * 31u + column * 17u + i * 13u);
}

struct Snapshot {
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> mid;
    std::vector<float> experts;
    std::vector<float> out;
};

bool read_snapshot(Snapshot &snapshot,
                   ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
                   ds4_gpu_tensor *mid, ds4_gpu_tensor *experts,
                   ds4_gpu_tensor *out) {
    return ds4_gpu_tensor_read(gate, 0, snapshot.gate.data(),
                               snapshot.gate.size() * sizeof(float)) != 0 &&
           ds4_gpu_tensor_read(up, 0, snapshot.up.data(),
                               snapshot.up.size() * sizeof(float)) != 0 &&
           ds4_gpu_tensor_read(mid, 0, snapshot.mid.data(),
                               snapshot.mid.size() * sizeof(float)) != 0 &&
           ds4_gpu_tensor_read(experts, 0, snapshot.experts.data(),
                               snapshot.experts.size() * sizeof(float)) != 0 &&
           ds4_gpu_tensor_read(out, 0, snapshot.out.data(),
                               snapshot.out.size() * sizeof(float)) != 0;
}

bool exact_vector(const char *name, const std::vector<float> &portable,
                  const std::vector<float> &wave64) {
    if (portable.size() == wave64.size() &&
        memcmp(portable.data(), wave64.data(),
               portable.size() * sizeof(float)) == 0)
        return true;
    const size_t count = portable.size() < wave64.size() ?
                         portable.size() : wave64.size();
    for (size_t i = 0; i < count; ++i) {
        uint32_t a, b;
        memcpy(&a, &portable[i], sizeof(a));
        memcpy(&b, &wave64[i], sizeof(b));
        if (a != b) {
            fprintf(stderr,
                    "routed_moe_wave64: %s[%zu] differs: portable=%g (0x%08x), Wave64=%g (0x%08x)\n",
                    name, i, (double)portable[i], a, (double)wave64[i], b);
            return false;
        }
    }
    fprintf(stderr, "routed_moe_wave64: %s size mismatch\n", name);
    return false;
}

int test_routed_moe_wave64(void) {
    if (getenv("DS4_TEST_PRODUCTION_SHAPE") == nullptr) {
        fprintf(stderr,
                "routed_moe_wave64: set DS4_TEST_PRODUCTION_SHAPE=1 for the exact production-shape A/B\n");
        return 0;
    }

    constexpr uint32_t in_dim = 7168;
    constexpr uint32_t mid_dim = 2048;
    constexpr uint32_t out_dim = 256;
    constexpr uint32_t n_total_expert = 1;
    constexpr uint32_t n_expert = 1;
    constexpr uint32_t gate_blocks = in_dim / 256;
    constexpr uint32_t down_blocks = mid_dim / 256;
    constexpr uint64_t gate_row_bytes = gate_blocks * sizeof(Iq2Block);
    constexpr uint64_t down_row_bytes = down_blocks * sizeof(Q2Block);
    constexpr uint64_t gate_expert_bytes = mid_dim * gate_row_bytes;
    constexpr uint64_t down_expert_bytes = out_dim * down_row_bytes;
    constexpr uint64_t gate_offset = 16;
    constexpr uint64_t up_offset = gate_offset + gate_expert_bytes;
    constexpr uint64_t down_offset = up_offset + gate_expert_bytes;
    constexpr uint64_t model_size = down_offset + down_expert_bytes;
    static_assert(gate_blocks == 28, "production gate block count");
    static_assert(down_blocks == 8, "production down block count");

    std::vector<uint8_t> model(model_size, 0);
    for (uint32_t row = 0; row < mid_dim; ++row) {
        for (uint32_t block = 0; block < gate_blocks; ++block) {
            Iq2Block gate_block{}, up_block{};
            fill_iq2_block(gate_block, row, block, 0);
            fill_iq2_block(up_block, row, block, 1);
            memcpy(model.data() + gate_offset + (uint64_t)row * gate_row_bytes +
                       (uint64_t)block * sizeof(Iq2Block),
                   &gate_block, sizeof(gate_block));
            memcpy(model.data() + up_offset + (uint64_t)row * gate_row_bytes +
                       (uint64_t)block * sizeof(Iq2Block),
                   &up_block, sizeof(up_block));
        }
    }
    for (uint32_t row = 0; row < out_dim; ++row) {
        for (uint32_t block = 0; block < down_blocks; ++block) {
            Q2Block down_block{};
            fill_q2_block(down_block, row, block);
            memcpy(model.data() + down_offset + (uint64_t)row * down_row_bytes +
                       (uint64_t)block * sizeof(Q2Block),
                   &down_block, sizeof(down_block));
        }
    }

    std::vector<float> x(in_dim);
    for (uint32_t i = 0; i < in_dim; ++i)
        x[i] = (float)((int)((i * 17u + 5u) % 41u) - 20) * 0.03125f;
    const int32_t selected = 0;
    const float weight = 0.75f;

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(mid_dim * sizeof(float));
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(mid_dim * sizeof(float));
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(mid_dim * sizeof(float));
    ds4_gpu_tensor *experts = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    ds4_gpu_tensor *selected_tensor = ds4_gpu_tensor_alloc(sizeof(selected));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(sizeof(weight));
    ds4_gpu_tensor *input = ds4_gpu_tensor_alloc(in_dim * sizeof(float));
    auto cleanup = [&]() {
        if (out) ds4_gpu_tensor_free(out);
        if (gate) ds4_gpu_tensor_free(gate);
        if (up) ds4_gpu_tensor_free(up);
        if (mid) ds4_gpu_tensor_free(mid);
        if (experts) ds4_gpu_tensor_free(experts);
        if (selected_tensor) ds4_gpu_tensor_free(selected_tensor);
        if (weights) ds4_gpu_tensor_free(weights);
        if (input) ds4_gpu_tensor_free(input);
    };
    if (!out || !gate || !up || !mid || !experts || !selected_tensor ||
        !weights || !input) {
        cleanup();
        return 1;
    }
    if (ds4_gpu_set_model_map(model.data(), model.size()) == 0 ||
        ds4_gpu_tensor_write(selected_tensor, 0, &selected, sizeof(selected)) == 0 ||
        ds4_gpu_tensor_write(weights, 0, &weight, sizeof(weight)) == 0 ||
        ds4_gpu_tensor_write(input, 0, x.data(), x.size() * sizeof(float)) == 0) {
        cleanup();
        return 1;
    }

    Snapshot portable{}, wave64{};
    for (Snapshot *snapshot : {&portable, &wave64}) {
        snapshot->gate.resize(mid_dim);
        snapshot->up.resize(mid_dim);
        snapshot->mid.resize(mid_dim);
        snapshot->experts.resize(out_dim);
        snapshot->out.resize(out_dim);
    }

    const char *old_disable = getenv(kDisableWave64);
    const bool had_disable = old_disable != nullptr;
    const std::string saved_disable = old_disable ? old_disable : "";
    const char *old_require = getenv(kRequireWave64);
    const bool had_require = old_require != nullptr;
    const std::string saved_require = old_require ? old_require : "";
    auto restore_env = [&]() {
        if (had_disable) setenv(kDisableWave64, saved_disable.c_str(), 1);
        else unsetenv(kDisableWave64);
        if (had_require) setenv(kRequireWave64, saved_require.c_str(), 1);
        else unsetenv(kRequireWave64);
    };
    auto run = [&](bool portable_path, Snapshot &snapshot) -> bool {
        if (portable_path) {
            setenv(kDisableWave64, "1", 1);
            unsetenv(kRequireWave64);
        } else {
            unsetenv(kDisableWave64);
            setenv(kRequireWave64, "1", 1);
        }
        const int ok = ds4_gpu_routed_moe_one_tensor(
            out, gate, up, mid, experts, model.data(), model.size(),
            gate_offset, up_offset, down_offset, 16, 10,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            in_dim, mid_dim, out_dim, selected_tensor, weights,
            n_total_expert, n_expert, 0.25f, input, nullptr, 0, false);
        return ok != 0 && read_snapshot(snapshot, gate, up, mid, experts, out);
    };

    bool ok = run(true, portable) && run(false, wave64);
    restore_env();
    if (!ok) {
        fprintf(stderr, "routed_moe_wave64: dispatch or readback failed\n");
        cleanup();
        return 1;
    }
    ok = exact_vector("gate", portable.gate, wave64.gate) &&
         exact_vector("up", portable.up, wave64.up) &&
         exact_vector("mid", portable.mid, wave64.mid) &&
         exact_vector("experts", portable.experts, wave64.experts) &&
         exact_vector("out", portable.out, wave64.out);
    if (ok)
        fprintf(stderr,
                "routed_moe_wave64: exact portable/Wave64 match (gate/up lanes=32, down lanes=8)\n");
    cleanup();
    return ok ? 0 : 1;
}

} // namespace

REGISTER_TEST(routed_moe_wave64, test_routed_moe_wave64);
