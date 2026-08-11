#include "../vulkan/q8_aligned_artifact.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <vector>

int main() {
    _putenv_s("DS4_VULKAN_Q8_ALIGNED", "1");
    const uint64_t in_dim = 33, out_dim = 3, blocks = 2;
    std::vector<uint8_t> raw(out_dim * blocks * 34);
    for (uint64_t i = 0; i < raw.size(); ++i) raw[i] = static_cast<uint8_t>(i * 37u + 11u);
    ds4_vulkan_q8_aligned_artifact artifact{};
    assert(ds4_vulkan_q8_aligned_build(&artifact, raw.data(), raw.size(), 0,
                                       in_dim, out_dim, 256));
    for (uint64_t row = 0; row < out_dim; ++row) {
        for (uint64_t block = 0; block < blocks; ++block) {
            const uint64_t record = row * blocks + block;
            const uint64_t source = record * 34;
            assert(artifact.data[record * 2] == raw[source]);
            assert(artifact.data[record * 2 + 1] == raw[source + 1]);
            for (uint64_t i = 0; i < 32; ++i)
                assert(artifact.data[artifact.payload_offset + record * 32 + i] == raw[source + 2 + i]);
        }
    }
    assert(artifact.payload_offset % 256 == 0);
    ds4_vulkan_q8_aligned_free(&artifact);
    return 0;
}