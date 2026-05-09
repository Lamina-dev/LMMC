#include <stdio.h>
#include <assert.h>
#include "lmmc/lmmc.h"

int main(void) {
    printf("Testing LAMMP high-precision stats functions...\n");

    // 1. Test 150! 
    // Double max is ~1.7e308, so 150! (~5.7e262) fits in double, but naive double
    // multiplication might lose precision in lower bits before conversion, 
    // whereas LAMMP calculates the exact integer and converts safely.
    lmmc_real_t fac_150;
    lmmc_stats_factorial(&fac_150, 150);
    printf("150! = %e\n", fac_150);
    
    // Basic bounds check (150! is approx 5.71338e262)
    assert(fac_150 > 5.7e262 && fac_150 < 5.8e262);

    // 2. Test 170! (approaches double precision limit)
    lmmc_real_t fac_170;
    lmmc_stats_factorial(&fac_170, 170);
    printf("170! = %e\n", fac_170);
    assert(fac_170 > 7.2e306 && fac_170 < 7.3e306);

    // 3. Test nPr(150, 5) = 150 * 149 * 148 * 147 * 146
    lmmc_real_t npr_val;
    lmmc_stats_nPr(&npr_val, 150, 5);
    printf("P(150, 5) = %e\n", npr_val);
    assert(npr_val > 7.0e10 && npr_val < 7.2e10);

    // 4. Test nCr(150, 5) = P(150, 5) / 120
    lmmc_real_t ncr_val;
    lmmc_stats_nCr(&ncr_val, 150, 5);
    printf("C(150, 5) = %e\n", ncr_val);
    assert(ncr_val > 5.8e8 && ncr_val < 6.0e8);

    printf("LAMMP Stats Tests Passed!\n");
    return 0;
}
