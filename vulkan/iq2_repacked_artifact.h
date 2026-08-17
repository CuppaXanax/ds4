#ifndef DS4_VULKAN_IQ2_REPACKED_ARTIFACT_H
#define DS4_VULKAN_IQ2_REPACKED_ARTIFACT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Execution layout for IQ2_XXS rows.  Each row contains, in order,
 * nb scale words, 8*nb transposed q words, and 8*nb transposed aux words.
 * The row stride is storage-alignment padded; source GGUF bytes are never
 * exposed to the shader. */
struct ds4_vulkan_iq2_repacked_artifact {
    uint8_t *data;
    uint64_t bytes;
    uint64_t source_offset;
    uint64_t in_dim;
    uint64_t out_dim;
    uint64_t experts;
    uint64_t blocks_per_row;
    uint64_t raw_row_bytes;
    uint64_t row_bytes;
    uint64_t section_alignment;
};

int ds4_vulkan_iq2_repacked_build(
    struct ds4_vulkan_iq2_repacked_artifact *artifact, const void *model_map,
    uint64_t model_size, uint64_t source_offset, uint64_t in_dim,
    uint64_t out_dim, uint64_t experts, uint64_t raw_row_bytes,
    uint64_t min_storage_buffer_offset_alignment);
void ds4_vulkan_iq2_repacked_free(
    struct ds4_vulkan_iq2_repacked_artifact *artifact);

#ifdef __cplusplus
}
#endif

#endif
