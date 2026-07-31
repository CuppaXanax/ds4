/* kernel tests registration — harness support.
 *
 * Each kernel test lives in its own translation unit under
 * vulkan/tests/tests/<name>.cpp and registers itself with REGISTER_TEST.
 */
#ifndef KERNEL_TESTS_H
#define KERNEL_TESTS_H

#include <vector>

struct kernel_test {
    const char *name;
    int (*fn)();              /* returns 0 on PASS, nonzero on FAIL */
};

std::vector<kernel_test> &kernel_test_registry();

struct kernel_test_registrar {
    kernel_test_registrar(const char *name, int (*fn)());
};

#define REGISTER_TEST(NAME, FN) \
    static kernel_test_registrar _reg_##NAME(#NAME, FN)

#endif /* KERNEL_TESTS_H */
