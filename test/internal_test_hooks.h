#ifndef LMMC_INTERNAL_TEST_HOOKS_H
#define LMMC_INTERNAL_TEST_HOOKS_H

#include <stddef.h>
#include <stdint.h>

void lmmc_memory_fail_after_for_test(size_t successful_allocations);

static inline void lmmc_memory_fail_reset_for_test(void) {
    lmmc_memory_fail_after_for_test(SIZE_MAX);
}

#endif
