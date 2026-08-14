#include "../ds4_dist_prefill.h"

#include <stdint.h>
#include <stdio.h>

static int failures = 0;

#define CHECK_EQ(actual, expected, name) do { \
    const uint32_t got = (actual); \
    if (got != (expected)) { \
        fprintf(stderr, "FAIL: %s: got %u, expected %u\n", \
                (name), got, (uint32_t)(expected)); \
        failures++; \
    } \
} while (0)

int main(void) {
    const uint64_t flash_hidden_values = 4u * 4096u;

    CHECK_EQ(ds4_dist_prefill_chunk_policy(
                     4096, 0, flash_hidden_values, 32, 128),
             256, "32-bit Flash uses a 16 MiB work item");
    CHECK_EQ(ds4_dist_prefill_chunk_policy(
                     4096, 0, flash_hidden_values, 16, 128),
             256, "16-bit Flash reaches the execution-tile cap");
    CHECK_EQ(ds4_dist_prefill_chunk_policy(
                     4096, 0, flash_hidden_values, 8, 128),
             256, "8-bit Flash remains execution-tile bounded");
    CHECK_EQ(ds4_dist_prefill_chunk_policy(
                     4096, 384, flash_hidden_values, 32, 128),
             384, "explicit override is authoritative");
    CHECK_EQ(ds4_dist_prefill_chunk_policy(
                     64, 0, flash_hidden_values, 32, 128),
             64, "automatic chunk is bounded by prefill capacity");
    CHECK_EQ(ds4_dist_prefill_chunk_policy(
                     200, 0, flash_hidden_values, 16, 128),
             128, "capacity is rounded to a compressor boundary");
    CHECK_EQ(ds4_dist_prefill_chunk_policy(
                     4096, 0, flash_hidden_values, 32, 96),
             192, "wire target is rounded to compressor alignment");
    CHECK_EQ(ds4_dist_prefill_chunk_policy(
                     4096, 0, 8u * 4096u, 32, 128),
             128, "wider hidden state reduces the wire-sized chunk");
    CHECK_EQ(ds4_dist_prefill_chunk_policy(
                     4096, 0, 0, 32, 128),
             4096, "missing model shape preserves session capacity");
    CHECK_EQ(ds4_dist_prefill_chunk_policy(
                     4096, 0, UINT64_MAX, 32, 128),
             1, "oversized payload still makes forward progress");

    if (failures != 0) {
        fprintf(stderr, "test_dist_prefill: %d failure(s)\n", failures);
        return 1;
    }
    fprintf(stderr, "test_dist_prefill: PASS\n");
    return 0;
}