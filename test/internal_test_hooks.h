#ifndef LMMC_INTERNAL_TEST_HOOKS_H
#define LMMC_INTERNAL_TEST_HOOKS_H

#include <stddef.h>
#include <stdint.h>

void lmmc_memory_fail_after_for_test(size_t successful_allocations);
lmmc_status_t lmmc_quad_gauss_hermite_with_iteration_limit_for_test(
    lmmc_quad_func_t f, void *ud, size_t order,
    size_t iteration_limit, lmmc_real_t *out);

static inline void lmmc_memory_fail_reset_for_test(void) {
    lmmc_memory_fail_after_for_test(SIZE_MAX);
}

#endif
