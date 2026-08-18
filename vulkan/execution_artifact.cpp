#include "execution_artifact.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace {

constexpr uint32_t kBlockElements = 256;
constexpr uint32_t kTileRows = 4;

static bool mul_u64(uint64_t a, uint64_t b, uint64_t &out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    out = a * b;
    return true;
}

static bool add_u64(uint64_t a, uint64_t b, uint64_t &out) {
    if (b > UINT64_MAX - a) return false;
    out = a + b;
    return true;
}

static uint64_t align_up(uint64_t value, uint64_t alignment) {
    if (alignment == 0 || value > UINT64_MAX - (alignment - 1u)) return UINT64_MAX;
    const uint64_t remainder = value % alignment;
    return remainder == 0 ? value : value + alignment - remainder;
}

struct FormatLayout {
    uint32_t source_block_bytes;
    uint32_t source_blocks_per_tile;
    uint32_t plane_count;
    uint32_t plane_element_bytes[3];
};

static bool format_layout(uint32_t format, FormatLayout &layout) {
    layout = {};
    switch (format) {
    case DS4_VULKAN_EXEC_Q8_0:
        layout = {34, 8, 2, {16, 256, 0}};
        return true;
    case DS4_VULKAN_EXEC_IQ2_XXS:
        layout = {66, 1, 2, {2, 64, 0}};
        return true;
    case DS4_VULKAN_EXEC_Q2_K:
        layout = {84, 1, 3, {16, 64, 4}};
        return true;
    default:
        return false;
    }
}

static uint32_t plane_source_delta(uint32_t format, uint32_t plane) {
    if (format == DS4_VULKAN_EXEC_Q2_K) {
        return plane == DS4_VULKAN_EXEC_PLANE_PAYLOAD ? 16u :
               plane == DS4_VULKAN_EXEC_PLANE_Q2_D ? 80u : 0u;
    }
    return plane == DS4_VULKAN_EXEC_PLANE_SCALE ? 0u : 2u;
}

} // namespace

extern "C" int ds4_vulkan_execution_artifact_build(
    ds4_vulkan_execution_artifact *artifact,
    const void *model_map,
    uint64_t model_size,
    uint64_t source_offset,
    uint64_t in_dim,
    uint64_t out_dim,
    uint32_t format,
    uint64_t min_storage_buffer_offset_alignment) {
    if (!artifact || !model_map || model_size == 0 || in_dim == 0 ||
        out_dim == 0 || min_storage_buffer_offset_alignment == 0 ||
        in_dim % kBlockElements != 0)
        return 0;

    FormatLayout layout;
    if (!format_layout(format, layout)) return 0;
    const uint64_t blocks = in_dim / kBlockElements;
    const uint64_t tiles = (out_dim + kTileRows - 1u) / kTileRows;
    if (blocks > UINT32_MAX || tiles > UINT32_MAX) return 0;

    uint64_t source_row_bytes = 0, source_bytes = 0, entries = 0;
    if (!mul_u64(blocks, layout.source_blocks_per_tile, source_row_bytes) ||
        !mul_u64(source_row_bytes, layout.source_block_bytes, source_row_bytes) ||
        !mul_u64(out_dim, source_row_bytes, source_bytes) ||
        source_offset > model_size || source_bytes > model_size - source_offset ||
        !mul_u64(tiles, blocks, entries) || !mul_u64(entries, kTileRows, entries))
        return 0;

    const uint64_t alignment = std::max<uint64_t>(256, min_storage_buffer_offset_alignment);
    uint64_t plane_bytes[3] = {};
    uint64_t plane_offset[3] = {};
    for (uint32_t p = 0; p < layout.plane_count; ++p)
        if (!mul_u64(entries, layout.plane_element_bytes[p], plane_bytes[p])) return 0;

    uint64_t total_bytes = 0;
    for (uint32_t p = 0; p < layout.plane_count; ++p) {
        total_bytes = align_up(total_bytes, alignment);
        if (total_bytes == UINT64_MAX) return 0;
        plane_offset[p] = total_bytes;
        if (!add_u64(total_bytes, plane_bytes[p], total_bytes)) return 0;
    }
    if (total_bytes == 0 || total_bytes > SIZE_MAX) return 0;
    uint8_t *data = static_cast<uint8_t *>(std::calloc(1, static_cast<size_t>(total_bytes)));
    if (!data) return 0;

    const uint8_t *source = static_cast<const uint8_t *>(model_map) + source_offset;
    for (uint64_t tile = 0; tile < tiles; ++tile) {
        for (uint64_t block = 0; block < blocks; ++block) {
            for (uint32_t row_in_tile = 0; row_in_tile < kTileRows; ++row_in_tile) {
                const uint64_t row = tile * kTileRows + row_in_tile;
                if (row >= out_dim) continue;
                const uint8_t *src = source + row * source_row_bytes +
                                     block * layout.source_blocks_per_tile *
                                         layout.source_block_bytes;
                const uint64_t entry = (tile * blocks + block) * kTileRows + row_in_tile;
                for (uint32_t p = 0; p < layout.plane_count; ++p) {
                    const uint32_t bytes = layout.plane_element_bytes[p];
                    const uint64_t dst = plane_offset[p] +
                        entry * (uint64_t)bytes;
                    uint8_t *dst_ptr = data + dst;
                    const uint32_t delta = plane_source_delta(format, p);
                    for (uint32_t source_block = 0;
                         source_block < layout.source_blocks_per_tile;
                         ++source_block) {
                        const uint32_t source_bytes =
                            format == DS4_VULKAN_EXEC_Q8_0
                                ? (p == DS4_VULKAN_EXEC_PLANE_SCALE ? 2u : 32u)
                                : bytes;
                        std::memcpy(
                            dst_ptr,
                            src + source_block * layout.source_block_bytes + delta,
                            source_bytes);
                        dst_ptr += source_bytes;
                    }
                }
            }
        }
    }

    *artifact = {};
    artifact->data = data;
    artifact->bytes = total_bytes;
    artifact->source_offset = source_offset;
    artifact->source_bytes = source_bytes;
    artifact->in_dim = in_dim;
    artifact->out_dim = out_dim;
    artifact->format = format;
    artifact->block_elements = kBlockElements;
    artifact->source_block_bytes = layout.source_block_bytes;
    artifact->source_blocks_per_tile = layout.source_blocks_per_tile;
    artifact->blocks_per_row = static_cast<uint32_t>(blocks);
    artifact->tile_rows = kTileRows;
    artifact->tile_count = static_cast<uint32_t>(tiles);
    artifact->alignment = static_cast<uint32_t>(alignment);
    artifact->plane_count = layout.plane_count;
    for (uint32_t p = 0; p < layout.plane_count; ++p) {
        artifact->plane_offset[p] = plane_offset[p];
        artifact->plane_bytes[p] = plane_bytes[p];
        artifact->plane_element_bytes[p] = layout.plane_element_bytes[p];
    }
    return 1;
}

extern "C" void ds4_vulkan_execution_artifact_free(
    ds4_vulkan_execution_artifact *artifact) {
    if (!artifact) return;
    std::free(artifact->data);
    *artifact = {};
}

extern "C" uint64_t ds4_vulkan_execution_artifact_index(
    const ds4_vulkan_execution_artifact *artifact,
    uint32_t plane, uint32_t tile, uint32_t block, uint32_t row_in_tile) {
    if (!artifact || plane >= artifact->plane_count ||
        tile >= artifact->tile_count || block >= artifact->blocks_per_row ||
        row_in_tile >= artifact->tile_rows)
        return UINT64_MAX;
    const uint64_t entry =
        ((uint64_t)tile * artifact->blocks_per_row + block) * artifact->tile_rows +
        row_in_tile;
    return artifact->plane_offset[plane] +
           entry * artifact->plane_element_bytes[plane];
}

extern "C" int ds4_vulkan_execution_artifact_release_window(
    uint64_t model_size,
    uint64_t source_offset,
    uint64_t source_bytes,
    uint64_t page_size,
    uint64_t *release_offset,
    uint64_t *release_bytes) {
    if (!release_offset || !release_bytes || page_size == 0 ||
        (page_size & (page_size - 1u)) != 0 || source_bytes == 0 ||
        source_offset > model_size || source_bytes > model_size - source_offset ||
        source_offset > UINT64_MAX - (page_size - 1u))
        return 0;
    const uint64_t end = source_offset + source_bytes;
    const uint64_t begin = (source_offset + page_size - 1u) &
                           ~(page_size - 1u);
    const uint64_t aligned_end = end & ~(page_size - 1u);
    *release_offset = begin;
    *release_bytes = aligned_end > begin ? aligned_end - begin : 0;
    return 1;
}

extern "C" int ds4_vulkan_execution_arena_init(
    ds4_vulkan_execution_arena *arena, uint64_t alignment) {
    if (!arena || alignment == 0 || alignment > UINT32_MAX) return 0;
    *arena = {};
    arena->alignment = static_cast<uint32_t>(alignment);
    return 1;
}

extern "C" int ds4_vulkan_execution_arena_add(
    ds4_vulkan_execution_arena *arena,
    uint32_t layer,
    uint32_t matrix,
    const ds4_vulkan_execution_artifact *artifact,
    uint32_t *entry_index) {
    if (!arena || !artifact || !artifact->data || artifact->bytes == 0 ||
        arena->alignment == 0 || arena->entry_count == UINT32_MAX)
        return 0;
    for (uint32_t i = 0; i < arena->entry_count; ++i)
        if (arena->entries[i].layer == layer && arena->entries[i].matrix == matrix)
            return 0; /* duplicate keys make a stale descriptor table unsafe */

    const uint64_t offset = align_up(arena->bytes, arena->alignment);
    if (offset == UINT64_MAX || artifact->bytes > UINT64_MAX - offset) return 0;
    const uint64_t needed = offset + artifact->bytes;
    if (needed > SIZE_MAX) return 0;
    if (needed > arena->capacity) {
        uint64_t capacity = arena->capacity ? arena->capacity : arena->alignment;
        while (capacity < needed) {
            if (capacity > UINT64_MAX / 2u) { capacity = needed; break; }
            capacity *= 2u;
        }
        uint8_t *data = static_cast<uint8_t *>(std::realloc(
            arena->data, static_cast<size_t>(capacity)));
        if (!data) return 0;
        if (capacity > arena->capacity)
            std::memset(data + arena->capacity, 0,
                        static_cast<size_t>(capacity - arena->capacity));
        arena->data = data;
        arena->capacity = capacity;
    }
    if (arena->entry_count == arena->entry_capacity) {
        uint32_t capacity = arena->entry_capacity ? arena->entry_capacity * 2u : 16u;
        if (capacity < arena->entry_capacity) return 0;
        auto *entries = static_cast<ds4_vulkan_execution_arena_entry *>(std::realloc(
            arena->entries, static_cast<size_t>(capacity) *
                sizeof(ds4_vulkan_execution_arena_entry)));
        if (!entries) return 0;
        arena->entries = entries;
        arena->entry_capacity = capacity;
    }
    std::memcpy(arena->data + offset, artifact->data, static_cast<size_t>(artifact->bytes));
    arena->entries[arena->entry_count] = {
        layer, matrix, artifact->format, 0, offset, artifact->bytes,
        artifact->in_dim, artifact->out_dim};
    if (entry_index) *entry_index = arena->entry_count;
    arena->entry_count++;
    arena->bytes = needed;
    return 1;
}

extern "C" const ds4_vulkan_execution_arena_entry *
ds4_vulkan_execution_arena_find(
    const ds4_vulkan_execution_arena *arena,
    uint32_t layer,
    uint32_t matrix) {
    if (!arena) return nullptr;
    for (uint32_t i = 0; i < arena->entry_count; ++i)
        if (arena->entries[i].layer == layer && arena->entries[i].matrix == matrix)
            return &arena->entries[i];
    return nullptr;
}

extern "C" void ds4_vulkan_execution_arena_free(
    ds4_vulkan_execution_arena *arena) {
    if (!arena) return;
    std::free(arena->data);
    std::free(arena->entries);
    *arena = {};
}
