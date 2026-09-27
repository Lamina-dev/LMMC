/**
 * @file test_stats_moments.c
 * @brief 均值、方差与标准差测试。
 */
#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

struct test_fixture {
    lmmc_vec_t x;
    lmmc_vec_t y;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_vec_destroy(&fixture->x);
    lmmc_vec_destroy(&fixture->y);
    free(fixture);
    *state = NULL;
    return 0;
}

static void test_moments(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_status_t st;

    double mean, var, std;
    st = lmmc_vec_create(4, &fixture->x);
    assert_false(st != LMMC_STATUS_OK);
    LMMC_REAL_SET_D(&fixture->x.data[0], 1.0);
    LMMC_REAL_SET_D(&fixture->x.data[1], 2.0);
    LMMC_REAL_SET_D(&fixture->x.data[2], 3.0);
    LMMC_REAL_SET_D(&fixture->x.data[3], 4.0);
    st = lmmc_vec_mean(&fixture->x, &mean);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(mean, 2.5, 1e-12));

    st = lmmc_vec_variance_population(&fixture->x, &var);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(var, 1.25, 1e-12));

    st = lmmc_vec_variance_sample(&fixture->x, &var);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(var, 1.6666666666666667, 1e-12));

    st = lmmc_vec_stddev_population(&fixture->x, &std);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(std, 1.118033988749895, 1e-12));

    st = lmmc_vec_stddev_sample(&fixture->x, &std);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(std, 1.2909944487358056, 1e-12));
}
static void test_covariance_correlation(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_status_t st;

    double cov, corr;
    st = lmmc_vec_create(4, &fixture->x);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_vec_create(4, &fixture->y);
    assert_false(st != LMMC_STATUS_OK);

    LMMC_REAL_SET_D(&fixture->x.data[0], 1.0);
    LMMC_REAL_SET_D(&fixture->x.data[1], 2.0);
    LMMC_REAL_SET_D(&fixture->x.data[2], 3.0);
    LMMC_REAL_SET_D(&fixture->x.data[3], 4.0);

    LMMC_REAL_SET_D(&fixture->y.data[0], 2.0);
    LMMC_REAL_SET_D(&fixture->y.data[1], 4.0);
    LMMC_REAL_SET_D(&fixture->y.data[2], 6.0);
    LMMC_REAL_SET_D(&fixture->y.data[3], 8.0);
    st = lmmc_vec_covariance_population(&fixture->x, &fixture->y, &cov);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(cov, 2.5, 1e-12));

    st = lmmc_vec_covariance_sample(&fixture->x, &fixture->y, &cov);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(cov, 3.3333333333333335, 1e-12));

    st = lmmc_vec_correlation_population(&fixture->x, &fixture->y, &corr);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(corr, 1.0, 1e-12));

    st = lmmc_vec_correlation_sample(&fixture->x, &fixture->y, &corr);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(corr, 1.0, 1e-12));
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_moments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_covariance_correlation, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
