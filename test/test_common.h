#ifndef LMMC_TEST_COMMON_H
#define LMMC_TEST_COMMON_H

#include <math.h>

static int lmmc_test_nearly_equal(lmmc_real_t a, lmmc_real_t b, lmmc_real_t eps) {
    return fabs(a - b) <= eps;
}

#endif
