/**
 * @file test_lmmp_stats.c
 * 针对 LMMC 中 LMMP 统计接口的单元测试。
 */
#include <stdio.h>
#include "test_common.h"
#include "lmmc/lmmc.h"

static void test_large_factorials(void **state) {
    (void)state;
    lmmc_real_t fac_150;
    lmmc_stats_factorial(&fac_150, 150);
    assert_true(fac_150 > 5.7e262 && fac_150 < 5.8e262);

    lmmc_real_t fac_170;
    lmmc_stats_factorial(&fac_170, 170);
    assert_true(fac_170 > 7.2e306 && fac_170 < 7.3e306);
}

static void test_large_permutations(void **state) {
    (void)state;
    lmmc_real_t npr_val;
    lmmc_stats_npr(&npr_val, 150, 5);
    assert_true(npr_val > 7.0e10 && npr_val < 7.2e10);
}

static void test_large_combinations(void **state) {
    (void)state;
    lmmc_real_t ncr_val;
    lmmc_stats_ncr(&ncr_val, 150, 5);
    assert_true(ncr_val > 5.8e8 && ncr_val < 6.0e8);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_large_factorials),
        cmocka_unit_test(test_large_permutations),
        cmocka_unit_test(test_large_combinations),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
