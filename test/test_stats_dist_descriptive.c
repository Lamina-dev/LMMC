/**
 * @file test_stats_dist_descriptive.c
 * @brief 描述性统计测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static int setup(void **state) {
    (void)state;
    assert_int_equal(lmmc_init(), LMMC_STATUS_OK);
    return 0;
}

static int teardown(void **state) {
    (void)state;
    assert_int_equal(lmmc_deinit(), LMMC_STATUS_OK);
    return 0;
}

static void test_median(void **state) {
    (void)state;
    lmmc_real_t data_odd[] = {5.0, 1.0, 3.0, 2.0, 4.0};
    lmmc_vec_t v_odd = {5, data_odd, 0};
    lmmc_real_t med;

    assert_true(lmmc_vec_median(&v_odd, &med) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(med, 3.0, 1e-12));

    lmmc_real_t data_even[] = {1.0, 2.0, 3.0, 4.0};
    lmmc_vec_t v_even = {4, data_even, 0};
    assert_true(lmmc_vec_median(&v_even, &med) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(med, 2.5, 1e-12));

    {
        lmmc_real_t data_extreme[] = {DBL_MAX, DBL_MAX};
        lmmc_vec_t v_extreme = {2, data_extreme, 0};
        assert_true(lmmc_vec_median(&v_extreme, &med) == LMMC_STATUS_OK);
        assert_true(med == DBL_MAX);
    }

    {
        lmmc_real_t data_nonfinite[] = {1.0, NAN, 3.0};
        lmmc_vec_t v_nonfinite = {3, data_nonfinite, 0};
        med = 17.0;
        assert_true(lmmc_vec_median(&v_nonfinite, &med) ==
                    LMMC_STATUS_NUMERICAL_FAILURE);
        assert_true(med == 17.0);
    }
}

static void test_quantile(void **state) {
    (void)state;
    lmmc_real_t data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    lmmc_vec_t v = {5, data, 0};
    lmmc_real_t q;

    assert_true(lmmc_vec_quantile(&v, 0.0, &q) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(q, 1.0, 1e-12));

    assert_true(lmmc_vec_quantile(&v, 1.0, &q) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(q, 5.0, 1e-12));

    assert_true(lmmc_vec_quantile(&v, 0.5, &q) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(q, 3.0, 1e-12));

    assert_true(lmmc_vec_quantile(&v, 0.25, &q) == LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(q, 2.0, 1e-12));

    q = 17.0;
    assert_true(lmmc_vec_quantile(&v, NAN, &q) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(q == 17.0);

    {
        lmmc_real_t data_nonfinite[] = {1.0, NAN, 3.0};
        lmmc_vec_t v_nonfinite = {3, data_nonfinite, 0};
        assert_true(lmmc_vec_quantile(&v_nonfinite, 0.5, &q) ==
                    LMMC_STATUS_NUMERICAL_FAILURE);
        assert_true(q == 17.0);
    }
}

static void test_histogram(void **state) {
    (void)state;
    lmmc_real_t data[] = {0.5, 1.5, 2.5, 3.5, 4.5};
    lmmc_vec_t v = {5, data, 0};
    lmmc_real_t edges[6];
    size_t counts[5];

    assert_true(lmmc_vec_histogram(&v, 5, edges, counts) == LMMC_STATUS_OK);
    for (size_t i = 0; i < 5; ++i) {
        assert_int_equal(counts[i], 1);
    }

    {
        lmmc_real_t extreme_data[] = {-DBL_MAX, DBL_MAX};
        lmmc_vec_t extreme = {2, extreme_data, 0};
        lmmc_real_t extreme_edges[3];
        size_t extreme_counts[2];

        assert_true(lmmc_vec_histogram(
                        &extreme, 2, extreme_edges, extreme_counts) ==
                    LMMC_STATUS_OK);
        assert_true(extreme_edges[0] == -DBL_MAX &&
                    extreme_edges[1] == 0.0 &&
                    extreme_edges[2] == DBL_MAX);
        assert_true(extreme_counts[0] == 1 && extreme_counts[1] == 1);
    }

    {
        lmmc_real_t nonfinite_data[] = {1.0, NAN};
        lmmc_vec_t nonfinite = {2, nonfinite_data, 0};
        lmmc_real_t preserved_edges[3] = {7.0, 8.0, 9.0};
        size_t preserved_counts[2] = {10, 11};

        assert_true(lmmc_vec_histogram(
                        &nonfinite, 2, preserved_edges, preserved_counts) ==
                    LMMC_STATUS_NUMERICAL_FAILURE);
        assert_true(preserved_edges[0] == 7.0 &&
                    preserved_edges[1] == 8.0 &&
                    preserved_edges[2] == 9.0 &&
                    preserved_counts[0] == 10 &&
                    preserved_counts[1] == 11);
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_median, setup, teardown),
        cmocka_unit_test_setup_teardown(test_quantile, setup, teardown),
        cmocka_unit_test_setup_teardown(test_histogram, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
