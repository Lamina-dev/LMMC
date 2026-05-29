/**
 * @file test_lammp_stats.c
 * 针对 LMMC 中 lammp stats 相关接口的单元测试。
 */
#include <stdio.h>
#include <assert.h>
#include "lmmc/lmmc.h"

int main(void) {
    printf("Testing LAMMP high-precision stats functions...\n");


    lmmc_real_t fac_150;
    lmmc_stats_factorial(&fac_150, 150);
    printf("150! = %e\n", fac_150);


    assert(fac_150 > 5.7e262 && fac_150 < 5.8e262);


    lmmc_real_t fac_170;
    lmmc_stats_factorial(&fac_170, 170);
    printf("170! = %e\n", fac_170);
    assert(fac_170 > 7.2e306 && fac_170 < 7.3e306);


    lmmc_real_t npr_val;
    lmmc_stats_nPr(&npr_val, 150, 5);
    printf("P(150, 5) = %e\n", npr_val);
    assert(npr_val > 7.0e10 && npr_val < 7.2e10);


    lmmc_real_t ncr_val;
    lmmc_stats_nCr(&ncr_val, 150, 5);
    printf("C(150, 5) = %e\n", ncr_val);
    assert(ncr_val > 5.8e8 && ncr_val < 6.0e8);

    printf("LAMMP Stats Tests Passed!\n");
    return 0;
}
