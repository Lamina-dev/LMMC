#ifndef LMMC_TEST_COMMON_H
#define LMMC_TEST_COMMON_H

#include <math.h>

static int lmmc_test_nearly_equal(double a, double b, double eps) {
    return fabs(a - b) <= eps;
}

#endif
