#ifndef DS4_VULKAN_EXECUTION_ARTIFACT_H
#define DS4_VULKAN_EXECUTION_ARTIFACT_H

/*
 * Canonical load-time representation for BC-250 weight matrices.
 *
 * GGUF remains the source of truth on the host, but shaders must not be
 * coupled to its per-block record layout. This ABI describes a tile-major
 * artifact with independently aligned planes. A tile contains four output
 * rows and one 256-element K block; rows beyond out_dim are zero padded.
 * Source quant bits are copied, never numerically transformed.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ds4_vulkan_execution_format {
    DS4_VULKAN_EXEC_Q8_0 = 8,
    DS4_VULKAN_EXEC_Q2_K = 10,
    DS4_VULKAN_EXEC_IQ2_XXS = 16,
};

enum ds4_vulkan_execution_plane {
    DS4_VULKAN_EXEC_PLANE_SCALE = 0,
    DS4_VULKAN_EXEC_PLANE_PAYLOAD = 1,
    DS4_VULKAN_EXEC_PLANE_Q2_D = 2,
};

struct ds4_vulkan_execution_artifact {
    uint8_t *data;
    uint64_t bytes;
    uint64_t source_offset;
    uint64_t source_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    uint32_t format;
    uint32_t block_elements;
    uint32_t source_block_bytes;
    uint32_t source_blocks_per_tile;
    uint32_t blocks_per_row;
    uint32_t tile_rows;
    uint32_t tile_count;
    uint32_t alignment;
    uint32_t plane_count;
    uint64_t plane_offset[3];
    uint64_t plane_bytes[3];
    uint32_t plane_element_bytes[3];
};

/* A blade-local immutable arena. Entries are an offset table; the arena owns
 * only packed execution bytes, never a second raw GGUF copy. A Vulkan loader
 * can upload `data` once and bind each entry by offset/range. */
struct ds4_vulkan_execution_arena_entry {
    uint32_t layer;
    uint32_t matrix;
    uint32_t format;
    uint32_t reserved;
    uint64_t data_offset;
    uint64_t bytes;
    uint64_t in_dim;
    uint64_t out_dim;
};

struct ds4_vulkan_execution_arena {
    uint8_t *data;
    uint64_t bytes;
    uint64_t capacity;
    struct ds4_vulkan_execution_arena_entry *entries;
    uint32_t entry_count;
    uint32_t entry_capacity;
    uint32_t alignment;
};

int ds4_vulkan_execution_artifact_build(
    struct ds4_vulkan_execution_artifact *artifact,
    const void *model_map,
    uint64_t model_size,
    uint64_t source_offset,
    uint64_t in_dim,
    uint64_t out_dim,
    uint32_t format,
    uint64_t min_storage_buffer_offset_alignment);

void ds4_vulkan_execution_artifact_free(
    struct ds4_vulkan_execution_artifact *artifact);

uint64_t ds4_vulkan_execution_artifact_index(
    const struct ds4_vulkan_execution_artifact *artifact,
    uint32_t plane,
    uint32_t tile,
    uint32_t block,
    uint32_t row_in_tile);

int ds4_vulkan_execution_arena_init(
    struct ds4_vulkan_execution_arena *arena,
    uint64_t alignment);

int ds4_vulkan_execution_arena_add(
    struct ds4_vulkan_execution_arena *arena,
    uint32_t layer,
    uint32_t matrix,
    const struct ds4_vulkan_execution_artifact *artifact,
    uint32_t *entry_index);

const struct ds4_vulkan_execution_arena_entry *
ds4_vulkan_execution_arena_find(
    const struct ds4_vulkan_execution_arena *arena,
    uint32_t layer,
    uint32_t matrix);

void ds4_vulkan_execution_arena_free(struct ds4_vulkan_execution_arena *arena);

#ifdef __cplusplus
}
#endif

#endif
