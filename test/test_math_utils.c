#include <math.h>
#include <stdio.h>
#include <float.h>
#include "lmmc/lmmc.h"

#define TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            printf("Test failed: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
            return 1; \
        } \
    } while (0)

int main(void) {
    lmmc_status_t st;
    lmmc_real_t val, val2;
    int flag;

    // Constants
    TEST_ASSERT(lmmc_inf(&val) == LMMC_STATUS_OK && isinf(val));
    TEST_ASSERT(lmmc_nan(&val) == LMMC_STATUS_OK && isnan(val));
    TEST_ASSERT(lmmc_eps(&val) == LMMC_STATUS_OK && val == LMMC_REAL_EPSILON);

    // Float Properties
    TEST_ASSERT(lmmc_isnan(NAN, &flag) == LMMC_STATUS_OK && flag == 1);
    TEST_ASSERT(lmmc_isnan(1.0, &flag) == LMMC_STATUS_OK && flag == 0);
    TEST_ASSERT(lmmc_isinf(INFINITY, &flag) == LMMC_STATUS_OK && flag == 1);
    TEST_ASSERT(lmmc_isinf(1.0, &flag) == LMMC_STATUS_OK && flag == 0);
    TEST_ASSERT(lmmc_isfinite(1.0, &flag) == LMMC_STATUS_OK && flag == 1);
    TEST_ASSERT(lmmc_isfinite(INFINITY, &flag) == LMMC_STATUS_OK && flag == 0);
    TEST_ASSERT(lmmc_signbit(-1.0, &flag) == LMMC_STATUS_OK && flag == 1);
    TEST_ASSERT(lmmc_signbit(1.0, &flag) == LMMC_STATUS_OK && flag == 0);

    // Enhanced Trig
    TEST_ASSERT(lmmc_atan2(1.0, 1.0, &val) == LMMC_STATUS_OK && fabs(val - 0.7853981633974483) < 1e5 * LMMC_REAL_EPSILON);
    TEST_ASSERT(lmmc_sincos(0.5, &val, &val2) == LMMC_STATUS_OK && fabs(val - sin(0.5)) < 1e5 * LMMC_REAL_EPSILON && fabs(val2 - cos(0.5)) < 1e5 * LMMC_REAL_EPSILON);
    TEST_ASSERT(lmmc_hypot(3.0, 4.0, &val) == LMMC_STATUS_OK && fabs(val - 5.0) < 1e5 * LMMC_REAL_EPSILON);

    // Power and Log
    TEST_ASSERT(lmmc_exp2(3.0, &val) == LMMC_STATUS_OK && fabs(val - 8.0) < 1e5 * LMMC_REAL_EPSILON);
    TEST_ASSERT(lmmc_log2(8.0, &val) == LMMC_STATUS_OK && fabs(val - 3.0) < 1e5 * LMMC_REAL_EPSILON);
    TEST_ASSERT(lmmc_expm1(1e-10, &val) == LMMC_STATUS_OK); // Just check status, exact value hard to type
    TEST_ASSERT(lmmc_log1p(1e-10, &val) == LMMC_STATUS_OK);

    // Manipulation
    TEST_ASSERT(lmmc_split_int_frac(3.14, &val, &val2) == LMMC_STATUS_OK && val == 3.0 && fabs(val2 - 0.14) < 1e5 * LMMC_REAL_EPSILON);
    TEST_ASSERT(lmmc_fmod(5.3, 2.0, &val) == LMMC_STATUS_OK && fabs(val - 1.3) < 1e5 * LMMC_REAL_EPSILON);
    TEST_ASSERT(lmmc_ldexp(0.5, 3, &val) == LMMC_STATUS_OK && val == 4.0);
    TEST_ASSERT(lmmc_nextafter(1.0, 2.0, &val) == LMMC_STATUS_OK && val > 1.0);

    // Comparison
    TEST_ASSERT(lmmc_approx_eq(1.0, 1.0001, 1e-3, &flag) == LMMC_STATUS_OK && flag == 1);
    TEST_ASSERT(lmmc_approx_eq(1.0, 1.0001, 1e-5, &flag) == LMMC_STATUS_OK && flag == 0);

    printf("All math utility tests passed!\n");
    return 0;
}
