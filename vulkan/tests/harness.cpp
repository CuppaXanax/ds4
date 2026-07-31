/* Vulkan kernel test harness.
 *
 * Run with ./run-kernel-tests.sh (needs GPU /dev/dri access).
 * Each registered test builds synthetic tensors, dispatches the real
 * Vulkan kernel through the ds4_gpu_* contract, reads the output back and
 * compares it to an inline CPU reference.  Prints PASS/FAIL per test.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "../../ds4_gpu.h"
#include "tests.h"

int main(void) {
    if (ds4_gpu_init() == 0) {
        fprintf(stderr, "harness: ds4_gpu_init failed (GPU not available?)\n");
        return 1;
    }
    int pass = 0, fail = 0;
    for (auto &t : kernel_test_registry()) {
        int r = t.fn();
        printf("[%s] %s\n", r == 0 ? "PASS" : "FAIL", t.name);
        fflush(stdout);
        if (r == 0) pass++; else fail++;
    }
    ds4_gpu_cleanup();
    printf("harness: %d passed, %d failed\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
