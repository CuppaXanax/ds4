#define DS4_TEST_HOOKS
#include "ds4_distributed.h"

#include <stdio.h>

int main(void) {
    if (ds4_test_distributed_local_output_route() != 0) {
        fprintf(stderr, "distributed local-output route test failed\n");
        return 1;
    }
    puts("distributed local-output route test passed");
    return 0;
}
