#include "iq2_repacked_artifact.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    const uint64_t rem = value % alignment;
    return rem == 0 ? value : value + alignment - rem;
}

int ds4_vulkan_iq2_repacked_build(
    ds4_vulkan_iq2_repacked_artifact *artifact, const void *model_map,
    uint64_t model_size, uint64_t source_offset, uint64_t in_dim,
    uint64_t out_dim, uint64_t experts, uint64_t raw_row_bytes,
    uint64_t min_storage_buffer_offset_alignment) {
    if (!artifact || !model_map || !in_dim || !out_dim || !experts ||
        !raw_row_bytes || !min_storage_buffer_offset_alignment ||
        in_dim % 256u != 0) return 0;
    const uint64_t blocks = in_dim / 256u;
    if (blocks > UINT64_MAX / 68u) return 0;
    constexpr uint64_t tile_rows = 16u;
    const uint64_t payload = blocks * tile_rows * 68u;
    const uint64_t alignment = std::max<uint64_t>(256u,
                                                   min_storage_buffer_offset_alignment);
    const uint64_t row_bytes = align_up(payload, alignment);
    const uint64_t tiles = (out_dim + tile_rows - 1u) / tile_rows;
    if (row_bytes < payload || tiles > UINT64_MAX / row_bytes) return 0;
    const uint64_t expert_bytes = tiles * row_bytes;
    if (experts > UINT64_MAX / expert_bytes) return 0;
    const uint64_t total = experts * expert_bytes;
    const uint64_t raw_expert_bytes = out_dim * raw_row_bytes;
    if (raw_expert_bytes / raw_row_bytes != out_dim ||
        experts > UINT64_MAX / raw_expert_bytes) return 0;
    const uint64_t raw_total = experts * raw_expert_bytes;
    if (source_offset > model_size || raw_total > model_size - source_offset ||
        total == 0 || total > SIZE_MAX) return 0;

    uint8_t *data = static_cast<uint8_t *>(std::calloc(1, (size_t)total));
    if (!data) return 0;
    const uint8_t *raw = static_cast<const uint8_t *>(model_map) + source_offset;
    for (uint64_t expert = 0; expert < experts; ++expert) {
        for (uint64_t tile = 0; tile < tiles; ++tile) {
            uint8_t *dst = data + expert * expert_bytes + tile * row_bytes;
            for (uint64_t local = 0; local < tile_rows; ++local) {
                const uint64_t row = tile * tile_rows + local;
                if (row >= out_dim) continue;
                const uint8_t *src = raw + expert * raw_expert_bytes + row * raw_row_bytes;
                for (uint64_t block = 0; block < blocks; ++block) {
                    const uint8_t *rb = src + block * 66u;
                    uint32_t scale = 0;
                    std::memcpy(&scale, rb, sizeof(uint16_t));
                    std::memcpy(dst + (local * blocks + block) * 4u,
                                &scale, sizeof(scale));
                    for (uint64_t ib = 0; ib < 8; ++ib) {
                        uint32_t q = 0, aux = 0;
                        std::memcpy(&q, rb + 2u + ib * 8u, sizeof(q));
                        std::memcpy(&aux, rb + 6u + ib * 8u, sizeof(aux));
                        const uint64_t lane_word = local * blocks + block;
                        std::memcpy(dst + (16u * blocks + ib * 16u * blocks + lane_word) * 4u,
                                    &q, sizeof(q));
                        std::memcpy(dst + (9u * 16u * blocks + ib * 16u * blocks + lane_word) * 4u,
                                    &aux, sizeof(aux));
                    }
                }
            }
        }
    }
    *artifact = {data, total, source_offset, in_dim, out_dim, experts,
                 blocks, raw_row_bytes, row_bytes, alignment};
    return 1;
}

void ds4_vulkan_iq2_repacked_free(ds4_vulkan_iq2_repacked_artifact *artifact) {
    if (!artifact) return;
    std::free(artifact->data);
    *artifact = {};
}
