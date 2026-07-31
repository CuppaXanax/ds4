#include "tests.h"

static std::vector<kernel_test> g_tests;

std::vector<kernel_test> &kernel_test_registry() {
    return g_tests;
}

kernel_test_registrar::kernel_test_registrar(const char *name, int (*fn)()) {
    g_tests.push_back({name, fn});
}
