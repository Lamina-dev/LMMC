/**
 * @file test_stats_extended.c
 * 针对 LMMC 中 stats extended 相关接口的单元测试。
 */
#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

struct test_fixture {
    lmmc_vec_t v1;
    lmmc_vec_t v2;
    lmmc_vec_t big_v1;
    lmmc_mat_t data;
    lmmc_mat_t cov_mat;
    lmmc_mat_t corr_mat;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_vec_destroy(&fixture->v1);
    lmmc_vec_destroy(&fixture->v2);
    lmmc_vec_destroy(&fixture->big_v1);
    lmmc_mat_destroy(&fixture->data);
    lmmc_mat_destroy(&fixture->cov_mat);
    lmmc_mat_destroy(&fixture->corr_mat);
    free(fixture);
    *state = NULL;
    return 0;
}

static void test_factorial_values(void **state) {
    (void)state;

    lmmc_real_t val = 0.0;

    lmmc_stats_factorial(&val, 0);
    assert_true(lmmc_test_nearly_equal(val, 1.0, 1e-12));

    lmmc_stats_factorial(&val, 1);
    assert_true(lmmc_test_nearly_equal(val, 1.0, 1e-12));

    lmmc_stats_factorial(&val, 10);
    assert_true(lmmc_test_nearly_equal(val, 3628800.0, 1e-6));

    lmmc_stats_factorial(&val, 20);
    assert_true(lmmc_test_nearly_equal(val, 2432902008176640000.0, 1e6));
}

static void test_combination_values(void **state) {
    (void)state;

    lmmc_real_t val = 0.0;

    lmmc_stats_ncr(&val, 10, 3);
    assert_true(lmmc_test_nearly_equal(val, 120.0, 1e-10));

    lmmc_stats_ncr(&val, 20, 10);
    assert_true(lmmc_test_nearly_equal(val, 184756.0, 1e-6));

    lmmc_stats_ncr(&val, 5, 0);
    assert_true(lmmc_test_nearly_equal(val, 1.0, 1e-12));

    lmmc_stats_ncr(&val, 100, 0);
    assert_true(lmmc_test_nearly_equal(val, 1.0, 1e-12));

    lmmc_stats_ncr(&val, 5, 5);
    assert_true(lmmc_test_nearly_equal(val, 1.0, 1e-12));

    lmmc_stats_ncr(&val, 50, 50);
    assert_true(lmmc_test_nearly_equal(val, 1.0, 1e-12));
}

/**
 * @brief 用 Python 整数运算生成独立参考值：float(math.comb(n, n // 2)).hex()。
 * OverflowError 对应 INFINITY。
 */
static const struct {
    uint32_t n;
    double expected;
} combination_fixtures[] = {
    {1000u, 0x1.9d4965077dfecp+994},
    {1001u, 0x1.9cdfcdef36969p+995},
    {1002u, 0x1.9cdfcdef36969p+996},
    {1003u, 0x1.9c76879c1b403p+997},
    {1004u, 0x1.9c76879c1b403p+998},
    {1005u, 0x1.9c0d91a7670e8p+999},
    {1006u, 0x1.9c0d91a7670e8p+1000},
    {1007u, 0x1.9ba4ebab0bc7ap+1001},
    {1008u, 0x1.9ba4ebab0bc7ap+1002},
    {1009u, 0x1.9b3c9541b0447p+1003},
    {1010u, 0x1.9b3c9541b0447p+1004},
    {1011u, 0x1.9ad48e06aed40p+1005},
    {1012u, 0x1.9ad48e06aed40p+1006},
    {1013u, 0x1.9a6cd59613a46p+1007},
    {1014u, 0x1.9a6cd59613a46p+1008},
    {1015u, 0x1.9a056b8c9b2e8p+1009},
    {1016u, 0x1.9a056b8c9b2e8p+1010},
    {1017u, 0x1.999e4f87b0a7fp+1011},
    {1018u, 0x1.999e4f87b0a7fp+1012},
    {1019u, 0x1.993781256c779p+1013},
    {1020u, 0x1.993781256c779p+1014},
    {1021u, 0x1.98d1000492af9p+1015},
    {1022u, 0x1.98d1000492af9p+1016},
    {1023u, 0x1.986acbc4918afp+1017},
    {1024u, 0x1.986acbc4918afp+1018},
    {1025u, 0x1.9804e4057fef5p+1019},
    {1026u, 0x1.9804e4057fef5p+1020},
    {1027u, 0x1.979f48681bf35p+1021},
    {1028u, 0x1.979f48681bf35p+1022},
    {1029u, 0x1.9739f88dc9682p+1023},
    {1030u, INFINITY},
    {1031u, INFINITY},
    {1032u, INFINITY},
    {1033u, INFINITY},
    {1034u, INFINITY},
    {1035u, INFINITY},
    {1036u, INFINITY},
    {1037u, INFINITY},
    {1038u, INFINITY},
    {1039u, INFINITY},
    {1040u, INFINITY}};

static void test_combination_reference_boundaries(void **state) {
    (void)state;
    for (size_t i = 0; i < sizeof(combination_fixtures) / sizeof(combination_fixtures[0]); ++i) {
        const uint32_t n = combination_fixtures[i].n;
        double value, symmetric;
        lmmc_stats_ncr(&value, n, n / 2);
        lmmc_stats_ncr(&symmetric, n, n - n / 2);
        assert_true(value == symmetric);
        if (isfinite(combination_fixtures[i].expected)) {
            assert_true(isfinite(value));
            assert_true(lmmc_test_nearly_equal(value / combination_fixtures[i].expected, 1.0, 5e-13));
        } else {
            assert_true(isinf(value) && value > 0.0);
        }
    }
}

static void test_combination_representability(void **state) {
    (void)state;

    {
        double value, symmetric;
        lmmc_stats_ncr(&value, 0, 0);
        assert_true(value == 1.0);
        lmmc_stats_ncr(&value, UINT32_MAX, 0);
        assert_true(value == 1.0);
        lmmc_stats_ncr(&value, UINT32_MAX, UINT32_MAX);
        assert_true(value == 1.0);
        lmmc_stats_ncr(&value, UINT32_MAX, 1);
        assert_true(value == 4294967295.0);
        lmmc_stats_ncr(&value, UINT32_MAX, 2);
        assert_true(value == 0x1.fffffffa00000p+62);
        lmmc_stats_ncr(&symmetric, UINT32_MAX, UINT32_MAX - 2);
        assert_true(value == symmetric);
        lmmc_stats_ncr(&value, UINT32_MAX, UINT32_MAX / 2);
        assert_true(isinf(value) && value > 0.0);
        lmmc_stats_ncr(NULL, UINT32_MAX, UINT32_MAX / 2);
    }
}

static void test_permutation_values(void **state) {
    (void)state;

    lmmc_real_t val = 0.0;

    lmmc_stats_npr(&val, 5, 3);
    assert_true(lmmc_test_nearly_equal(val, 60.0, 1e-10));

    lmmc_stats_npr(&val, 10, 5);
    assert_true(lmmc_test_nearly_equal(val, 30240.0, 1e-6));
}

static void test_constant_variance(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    lmmc_real_t var = 0.0;

    st = lmmc_vec_create(50, &fixture->v1);
    assert_false(st != LMMC_STATUS_OK);

    for (size_t i = 0; i < 50; i++) {
        LMMC_REAL_SET_D(&fixture->v1.data[i], 7.5);
    }

    st = lmmc_vec_variance_population(&fixture->v1, &var);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(var, 0.0, 1e-12));
}

static void test_negative_correlation(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    lmmc_real_t corr = 0.0;

    st = lmmc_vec_create(5, &fixture->v1);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_vec_create(5, &fixture->v2);
    assert_false(st != LMMC_STATUS_OK);

    for (size_t i = 0; i < 5; i++) {
        LMMC_REAL_SET_D(&fixture->v1.data[i], (double)(i + 1));
        LMMC_REAL_SET_D(&fixture->v2.data[i], -(double)(i + 1));
    }

    st = lmmc_vec_correlation_sample(&fixture->v1, &fixture->v2, &corr);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(corr, -1.0, 1e-12));

    st = lmmc_vec_correlation_population(&fixture->v1, &fixture->v2, &corr);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(corr, -1.0, 1e-12));
}

static void test_large_sample_moments(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    lmmc_real_t mean_val = 0.0;
    lmmc_real_t var_val = 0.0;
    const size_t n = 1000;

    st = lmmc_vec_create(n, &fixture->big_v1);
    assert_false(st != LMMC_STATUS_OK);

    for (size_t i = 0; i < n; i++) {
        LMMC_REAL_SET_D(&fixture->big_v1.data[i], (double)(i + 1));
    }

    st = lmmc_vec_mean(&fixture->big_v1, &mean_val);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(mean_val, 500.5, 1e-10));

    st = lmmc_vec_variance_population(&fixture->big_v1, &var_val);
    assert_false(st != LMMC_STATUS_OK || !lmmc_test_nearly_equal(var_val, 83333.25, 1e-6));
}

static int covariance_is_symmetric(const lmmc_mat_t *covariance, size_t cols) {
    for (size_t i = 0; i < cols; i++) {
        for (size_t j = 0; j < cols; j++) {
            double cij = covariance->data[i * cols + j];
            double cji = covariance->data[j * cols + i];
            if (!lmmc_test_nearly_equal(cij, cji, 1e-12)) {
                return 0;
            }
        }
    }
    return 1;
}

static void test_covariance_symmetry(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    const size_t rows = 10;
    const size_t cols = 3;

    st = lmmc_mat_create(rows, cols, &fixture->data);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_mat_create(cols, cols, &fixture->cov_mat);
    assert_false(st != LMMC_STATUS_OK);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double val = (double)(i * cols + j) * 0.7 + (double)(j * j) * 1.3;
            LMMC_REAL_SET_D(&fixture->data.data[i * cols + j], val);
        }
    }

    st = lmmc_mat_covariance_population(&fixture->data, &fixture->cov_mat);
    assert_false(st != LMMC_STATUS_OK);

    assert_true(covariance_is_symmetric(&fixture->cov_mat, cols));

    st = lmmc_mat_covariance_sample(&fixture->data, &fixture->cov_mat);
    assert_false(st != LMMC_STATUS_OK);

    assert_true(covariance_is_symmetric(&fixture->cov_mat, cols));
}

static void test_correlation_diagonal(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    const size_t rows = 10;
    const size_t cols = 4;

    st = lmmc_mat_create(rows, cols, &fixture->data);
    assert_false(st != LMMC_STATUS_OK);
    st = lmmc_mat_create(cols, cols, &fixture->corr_mat);
    assert_false(st != LMMC_STATUS_OK);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            double val = (double)(i + 1) * (double)(j + 1) + (double)(i * i) * 0.1;
            LMMC_REAL_SET_D(&fixture->data.data[i * cols + j], val);
        }
    }

    st = lmmc_mat_correlation_population(&fixture->data, &fixture->corr_mat);
    assert_false(st != LMMC_STATUS_OK);

    for (size_t i = 0; i < cols; i++) {
        double diag_val = fixture->corr_mat.data[i * cols + i];
        assert_true(lmmc_test_nearly_equal(diag_val, 1.0, 1e-12));
    }

    st = lmmc_mat_correlation_sample(&fixture->data, &fixture->corr_mat);
    assert_false(st != LMMC_STATUS_OK);

    for (size_t i = 0; i < cols; i++) {
        double diag_val = fixture->corr_mat.data[i * cols + i];
        assert_true(lmmc_test_nearly_equal(diag_val, 1.0, 1e-12));
    }
}

static void test_impossible_combinations(void **state) {
    (void)state;

    lmmc_real_t val = 999.0;

    lmmc_stats_ncr(&val, 5, 6);
    assert_true(lmmc_test_nearly_equal(val, 0.0, 1e-12));

    lmmc_stats_ncr(&val, 3, 10);
    assert_true(lmmc_test_nearly_equal(val, 0.0, 1e-12));

    lmmc_stats_ncr(&val, 0, 1);
    assert_true(lmmc_test_nearly_equal(val, 0.0, 1e-12));
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_factorial_values, setup, teardown),
        cmocka_unit_test_setup_teardown(test_combination_values, setup, teardown),
        cmocka_unit_test_setup_teardown(test_combination_reference_boundaries, setup, teardown),
        cmocka_unit_test_setup_teardown(test_combination_representability, setup, teardown),
        cmocka_unit_test_setup_teardown(test_permutation_values, setup, teardown),
        cmocka_unit_test_setup_teardown(test_constant_variance, setup, teardown),
        cmocka_unit_test_setup_teardown(test_negative_correlation, setup, teardown),
        cmocka_unit_test_setup_teardown(test_large_sample_moments, setup, teardown),
        cmocka_unit_test_setup_teardown(test_covariance_symmetry, setup, teardown),
        cmocka_unit_test_setup_teardown(test_correlation_diagonal, setup, teardown),
        cmocka_unit_test_setup_teardown(test_impossible_combinations, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
