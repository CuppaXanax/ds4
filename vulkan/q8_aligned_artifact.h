#ifndef DS4_VULKAN_Q8_ALIGNED_ARTIFACT_H
#define DS4_VULKAN_Q8_ALIGNED_ARTIFACT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ds4_vulkan_q8_aligned_artifact {
    uint8_t *data;
    uint64_t bytes;
    uint64_t source_offset;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t blocks_per_row;
    uint64_t scale_bytes;
    uint64_t payload_offset;
    uint64_t payload_bytes;
    uint64_t section_alignment;
};

int ds4_vulkan_q8_aligned_build(
    struct ds4_vulkan_q8_aligned_artifact *artifact, const void *model_map,
    uint64_t model_size, uint64_t source_offset, uint64_t in_dim,
    uint64_t out_dim, uint64_t min_storage_buffer_offset_alignment);
void ds4_vulkan_q8_aligned_free(struct ds4_vulkan_q8_aligned_artifact *artifact);

#ifdef __cplusplus
}
#endif

#endif