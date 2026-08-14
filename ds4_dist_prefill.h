#ifndef DS4_DIST_PREFILL_H
#define DS4_DIST_PREFILL_H

#include <stdint.h>

#define DS4_DIST_PREFILL_WIRE_TARGET_BYTES (8ull * 1024ull * 1024ull)
#define DS4_DIST_PREFILL_EXECUTION_TILE_TOKENS 256u

static inline uint32_t ds4_dist_prefill_chunk_policy(
        uint32_t prefill_cap,
        uint32_t requested,
        uint64_t hidden_values,
        uint32_t activation_bits,
        uint32_t compressor_alignment)
{
    if (requested != 0 || prefill_cap == 0) return requested;
    if (hidden_values == 0 || activation_bits == 0 || activation_bits % 8u != 0) {
        return prefill_cap;
    }

    const uint64_t bytes_per_value = activation_bits / 8u;
    uint64_t automatic;
    if (hidden_values > UINT64_MAX / bytes_per_value) {
        automatic = 1;
    } else {
        const uint64_t wire_bytes_per_token = hidden_values * bytes_per_value;
        automatic = DS4_DIST_PREFILL_WIRE_TARGET_BYTES / wire_bytes_per_token;
        if (automatic == 0) automatic = 1;
    }

    if (automatic > DS4_DIST_PREFILL_EXECUTION_TILE_TOKENS) {
        automatic = DS4_DIST_PREFILL_EXECUTION_TILE_TOKENS;
    }
    if (automatic > prefill_cap) automatic = prefill_cap;
    if (compressor_alignment > 1u && automatic >= compressor_alignment) {
        automatic -= automatic % compressor_alignment;
    }
    return (uint32_t)automatic;
}

#endif