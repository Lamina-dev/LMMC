/**
 * @file test_common.h
 * @brief 测试工具：浮点近似比较等公共辅助宏 / 函数。
 *
 * @internal
 */
#ifndef LMMC_TEST_COMMON_H
#define LMMC_TEST_COMMON_H

#include <math.h>

static int lmmc_test_nearly_equal(lmmc_real_t a, lmmc_real_t b, lmmc_real_t eps) {
    return fabs(a - b) <= eps;
}

#endif
