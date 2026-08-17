#include "../execution_artifact.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

static uint32_t source_block_bytes(uint32_t format) {
    return format == DS4_VULKAN_EXEC_Q8_0 ? 34u :
           format == DS4_VULKAN_EXEC_IQ2_XXS ? 66u : 84u;
}

static uint32_t source_blocks_per_tile(uint32_t format) {
    return format == DS4_VULKAN_EXEC_Q8_0 ? 8u : 1u;
}

static void check_format(uint32_t format) {
    constexpr uint64_t in_dim = 512;
    constexpr uint64_t out_dim = 5;
    const uint64_t bytes = out_dim * (in_dim / 256u) *
                           source_blocks_per_tile(format) * source_block_bytes(format);
    std::vector<uint8_t> source(bytes);
    for (uint64_t i = 0; i < bytes; ++i)
        source[i] = static_cast<uint8_t>((i * 37u + format) & 255u);

    ds4_vulkan_execution_artifact artifact{};
    assert(ds4_vulkan_execution_artifact_build(
        &artifact, source.data(), source.size(), 0, in_dim, out_dim, format, 256));
    assert(artifact.block_elements == 256 && artifact.tile_rows == 4);
    assert(artifact.blocks_per_row == 2 && artifact.tile_count == 2);
    assert(artifact.source_blocks_per_tile == source_blocks_per_tile(format));
    assert(artifact.source_bytes == bytes);

    for (uint32_t row = 0; row < out_dim; ++row) {
        const uint32_t tile = row / artifact.tile_rows;
        const uint32_t row_in_tile = row % artifact.tile_rows;
        for (uint32_t block = 0; block < artifact.blocks_per_row; ++block) {
            const uint64_t src = (static_cast<uint64_t>(row) * artifact.blocks_per_row + block) *
                                 artifact.source_blocks_per_tile * artifact.source_block_bytes;
            const uint64_t scale = ds4_vulkan_execution_artifact_index(
                &artifact, DS4_VULKAN_EXEC_PLANE_SCALE, tile, block, row_in_tile);
            for (uint32_t source_block = 0;
                 source_block < artifact.source_blocks_per_tile; ++source_block) {
                const uint32_t scale_bytes = format == DS4_VULKAN_EXEC_Q8_0 ? 2u :
                    artifact.plane_element_bytes[DS4_VULKAN_EXEC_PLANE_SCALE];
                assert(std::memcmp(
                    artifact.data + scale + source_block * scale_bytes,
                    source.data() + src + source_block * artifact.source_block_bytes,
                    scale_bytes) == 0);
            }
            const uint64_t payload = ds4_vulkan_execution_artifact_index(
                &artifact, DS4_VULKAN_EXEC_PLANE_PAYLOAD, tile, block, row_in_tile);
            const uint32_t delta = format == DS4_VULKAN_EXEC_Q2_K ? 16u : 2u;
            for (uint32_t source_block = 0;
                 source_block < artifact.source_blocks_per_tile; ++source_block) {
                const uint32_t payload_bytes = format == DS4_VULKAN_EXEC_Q8_0 ? 32u :
                    artifact.plane_element_bytes[DS4_VULKAN_EXEC_PLANE_PAYLOAD];
                assert(std::memcmp(
                    artifact.data + payload + source_block * payload_bytes,
                    source.data() + src + source_block * artifact.source_block_bytes + delta,
                    payload_bytes) == 0);
            }
            if (format == DS4_VULKAN_EXEC_Q2_K) {
                const uint64_t d = ds4_vulkan_execution_artifact_index(
                    &artifact, DS4_VULKAN_EXEC_PLANE_Q2_D, tile, block, row_in_tile);
                assert(std::memcmp(artifact.data + d, source.data() + src + 80, 4) == 0);
            }
        }
    }

    /* The fourth row of the second tile is padding and must stay zero. */
    for (uint32_t p = 0; p < artifact.plane_count; ++p) {
        const uint64_t off = ds4_vulkan_execution_artifact_index(
            &artifact, p, 1, 1, 3);
        for (uint32_t i = 0; i < artifact.plane_element_bytes[p]; ++i)
            assert(artifact.data[off + i] == 0);
    }
    ds4_vulkan_execution_artifact_free(&artifact);
}

static void check_arena() {
    uint8_t source[5 * 2 * 84] = {};
    for (uint32_t i = 0; i < sizeof(source); ++i) source[i] = (uint8_t)i;
    ds4_vulkan_execution_artifact artifact{};
    assert(ds4_vulkan_execution_artifact_build(
        &artifact, source, sizeof(source), 0, 512, 5,
        DS4_VULKAN_EXEC_Q2_K, 256));

    ds4_vulkan_execution_arena arena{};
    assert(ds4_vulkan_execution_arena_init(&arena, 256));
    uint32_t index = UINT32_MAX;
    assert(ds4_vulkan_execution_arena_add(&arena, 7, 3, &artifact, &index));
    assert(index == 0 && arena.bytes == artifact.bytes);
    const ds4_vulkan_execution_arena_entry *entry =
        ds4_vulkan_execution_arena_find(&arena, 7, 3);
    assert(entry && entry->data_offset == 0 && entry->bytes == artifact.bytes);
    assert(!ds4_vulkan_execution_arena_add(&arena, 7, 3, &artifact, nullptr));
    assert(!ds4_vulkan_execution_arena_find(&arena, 8, 3));
    ds4_vulkan_execution_arena_free(&arena);
    ds4_vulkan_execution_artifact_free(&artifact);
}

int main() {
    check_format(DS4_VULKAN_EXEC_Q8_0);
    check_format(DS4_VULKAN_EXEC_IQ2_XXS);
    check_format(DS4_VULKAN_EXEC_Q2_K);
    check_arena();
    return 0;
}
