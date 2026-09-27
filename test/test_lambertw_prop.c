#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "lmmc/lmmc.h"

#define RNG_SEED UINT64_C(0xDEADBEEF42)
#define NUM_TRIALS 2000
static uint64_t rng_state = RNG_SEED;
static double uniform(double lo, double hi) {
    uint64_t x = rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state = x;
    return lo + (hi - lo) * (double)((x * UINT64_C(2685821657736338717)) >> 11) * 0x1p-53;
}

/**
 * @brief 生成已知根作为参考，直接检验前向误差。
 * 输入 z 含 exp 和乘法舍入；在这些远离分支点的区间，
 * 一阶前向影响界为 8 eps |w/(1+w)|。
 */
static void certify_branch_value(int branch, int i, double reference, double z,
                                 double actual, lmmc_status_t status) {
    const double bound = 64.0 * DBL_EPSILON * fabs(reference) +
                         8.0 * DBL_EPSILON * fabs(reference / (1.0 + reference)) +
                         2.0 * nextafter(0.0, 1.0);
    if (status != LMMC_STATUS_OK || !isfinite(actual) || (branch == 0 ? actual < -1.0 : actual > -1.0) || !(fabs(actual - reference) <= bound)) {
        fail_msg("branch=%d trial=%d z=%.17g reference=%.17g actual=%.17g status=%d\n", branch, i, z, reference, actual, status);
    }
}

static void test_branch_values(void **state) {
    const int branch = *(const int *)*state;
    rng_state = RNG_SEED;
    int i;
    for (i = 0; i < NUM_TRIALS; ++i) {
        double reference, z, actual = 123.0;
        lmmc_status_t status;
        if (branch == -1) {
            reference = i % 2 ? uniform(-700.0, -2.0) : uniform(-2.0, -1.01);
        } else {
            switch (i % 4) {
            case 0:
                reference = uniform(-0.99, -0.5);
                break;
            case 1:
                reference = uniform(-0.5, 1.0);
                break;
            case 2:
                reference = exp(uniform(-700.0, -1.0));
                break;
            default:
                reference = uniform(1.0, 700.0);
                break;
            }
        }
        z = reference * exp(reference);
        status = branch == 0 ? lmmc_lambertw(z, &actual) : lmmc_lambertw_wm1(z, &actual);
        certify_branch_value(branch, i, reference, z, actual, status);
    }
}

int main(void) {
    static int branches[] = {0, -1};
    const struct CMUnitTest tests[] = {
        {"principal_branch", test_branch_values, NULL, NULL, &branches[0]},
        {"negative_branch", test_branch_values, NULL, NULL, &branches[1]},
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
