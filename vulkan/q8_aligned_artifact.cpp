#include "q8_aligned_artifact.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    const uint64_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

int ds4_vulkan_q8_aligned_enabled(void) {
    const char *value = std::getenv("DS4_VULKAN_Q8_ALIGNED");
    return value && std::strcmp(value, "1") == 0;
}

int ds4_vulkan_q8_aligned_build(
    ds4_vulkan_q8_aligned_artifact *artifact, const void *model_map,
    uint64_t model_size, uint64_t source_offset, uint64_t in_dim,
    uint64_t out_dim, uint64_t min_storage_buffer_offset_alignment) {
    if (!artifact || !model_map || !ds4_vulkan_q8_aligned_enabled() ||
        in_dim == 0 || out_dim == 0 || min_storage_buffer_offset_alignment == 0)
        return 0;
    if (in_dim > 8192 || in_dim > UINT64_MAX - 31) return 0;
    const uint64_t blocks = (in_dim + 31) / 32;
    if (out_dim > UINT64_MAX / blocks || out_dim * blocks > UINT64_MAX / 34)
        return 0;
    const uint64_t raw_bytes = out_dim * blocks * 34;
    if (source_offset > model_size || raw_bytes > model_size - source_offset) return 0;
    const uint64_t records = out_dim * blocks;
    if (records > UINT64_MAX / 2 || records > UINT64_MAX / 32) return 0;
    const uint64_t alignment = std::max<uint64_t>(256, min_storage_buffer_offset_alignment);
    const uint64_t scale_bytes = align_up(records * 2, 4);
    const uint64_t payload_bytes = records * 32;
    const uint64_t payload_offset = align_up(scale_bytes, alignment);
    if (payload_offset < scale_bytes || payload_bytes > UINT64_MAX - payload_offset)
        return 0;
    const uint64_t total_bytes = payload_offset + payload_bytes;
    if (total_bytes == 0 || total_bytes > SIZE_MAX) return 0;
    uint8_t *data = static_cast<uint8_t *>(std::calloc(1, static_cast<size_t>(total_bytes)));
    if (!data) return 0;
    const uint8_t *raw = static_cast<const uint8_t *>(model_map) + source_offset;
    for (uint64_t row = 0; row < out_dim; row++) {
        for (uint64_t block = 0; block < blocks; block++) {
            const uint64_t raw_block = (row * blocks + block) * 34;
            const uint64_t index = row * blocks + block;
            std::memcpy(data + index * 2, raw + raw_block, 2);
            std::memcpy(data + payload_offset + index * 32, raw + raw_block + 2, 32);
        }
    }
    *artifact = {data, total_bytes, source_offset, in_dim, out_dim, blocks,
                 scale_bytes, payload_offset, payload_bytes, alignment};
    return 1;
}

void ds4_vulkan_q8_aligned_free(ds4_vulkan_q8_aligned_artifact *artifact) {
    if (!artifact) return;
    std::free(artifact->data);
    *artifact = {};
}