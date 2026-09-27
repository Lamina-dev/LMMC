/**
 * @file test_sparse_extended_norms.c
 * @brief 稀疏矩阵范数测试。
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#define TEST_EPS_TIGHT 1e-12
#define TEST_EPS_NORMAL 1e-10

#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

struct test_fixture {
    lmmc_sparse_mat_t diagonal_norm_sparse;
    lmmc_sparse_mat_t dense_norm_sparse2;
    lmmc_mat_t dense_norm_dense2;
    lmmc_sparse_mat_t extreme_norm_extreme;
    lmmc_sparse_builder_t *extreme_norm_builder;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_sparse_builder_destroy(fixture->extreme_norm_builder);
    lmmc_sparse_destroy(&fixture->extreme_norm_extreme);
    lmmc_mat_destroy(&fixture->dense_norm_dense2);
    lmmc_sparse_destroy(&fixture->dense_norm_sparse2);
    lmmc_sparse_destroy(&fixture->diagonal_norm_sparse);
    free(fixture);
    return 0;
}

static void test_diagonal_norm(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    double data[] = {3.0, 0.0, 0.0, 4.0};

    lmmc_real_t norm_sparse = 0.0;

    st = lmmc_test_build_sparse(data, 2, 2, &fixture->diagonal_norm_sparse);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_norm_fro(&fixture->diagonal_norm_sparse, &norm_sparse);
    assert_true(st == LMMC_STATUS_OK);

    assert_true(lmmc_test_nearly_equal(norm_sparse, 5.0, TEST_EPS_TIGHT));

    lmmc_sparse_destroy(&fixture->diagonal_norm_sparse);
}

static void test_dense_norm(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    double data2[] = {
        1.0, 2.0, 0.0,
        0.0, 3.0, 4.0,
        5.0, 0.0, 6.0};

    lmmc_real_t norm_s2 = 0.0, norm_d2 = 0.0;

    st = lmmc_test_build_sparse(data2, 3, 3, &fixture->dense_norm_sparse2);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_norm_fro(&fixture->dense_norm_sparse2, &norm_s2);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_mat_create(3, 3, &fixture->dense_norm_dense2);
    assert_true(st == LMMC_STATUS_OK);
    for (size_t i = 0; i < 9; i++)
        LMMC_REAL_SET_D(&fixture->dense_norm_dense2.data[i], data2[i]);

    st = lmmc_mat_norm_fro(&fixture->dense_norm_dense2, &norm_d2);
    assert_true(st == LMMC_STATUS_OK);

    assert_true(lmmc_test_nearly_equal(norm_s2, norm_d2, TEST_EPS_TIGHT));
    lmmc_mat_destroy(&fixture->dense_norm_dense2);
    lmmc_sparse_destroy(&fixture->dense_norm_sparse2);
}

static void test_extreme_norm(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    lmmc_real_t norm_s2 = 0.0;
    const double large_data[] = {1.0e308, 1.0e308};
    const double small_data[] = {1.0e-300, 1.0e-300};

    st = lmmc_test_build_sparse(large_data, 1, 2, &fixture->extreme_norm_extreme);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_norm_fro(&fixture->extreme_norm_extreme, &norm_s2);
    assert_true(((st == LMMC_STATUS_OK) && (isfinite(norm_s2))) && (fabs(norm_s2 / 1.0e308 - sqrt(2.0)) <= 1.0e-15));
    lmmc_sparse_destroy(&fixture->extreme_norm_extreme);

    {

        st = lmmc_sparse_builder_create(1, 2, 2, &fixture->extreme_norm_builder);
        if (st == LMMC_STATUS_OK) {
            st = lmmc_sparse_builder_add(
                fixture->extreme_norm_builder, 0, 0, small_data[0]);
        }
        if (st == LMMC_STATUS_OK) {
            st = lmmc_sparse_builder_add(
                fixture->extreme_norm_builder, 0, 1, small_data[1]);
        }
        if (st == LMMC_STATUS_OK) {
            st = lmmc_sparse_builder_build(
                fixture->extreme_norm_builder, LMMC_SPARSE_CSR, &fixture->extreme_norm_extreme);
        }
        lmmc_sparse_builder_destroy(fixture->extreme_norm_builder);
        fixture->extreme_norm_builder = NULL;
        assert_true(st == LMMC_STATUS_OK);
    }
    st = lmmc_sparse_norm_fro(&fixture->extreme_norm_extreme, &norm_s2);
    assert_true((st == LMMC_STATUS_OK) && (fabs(norm_s2 / 1.0e-300 - sqrt(2.0)) <= 1.0e-15));
    lmmc_sparse_destroy(&fixture->extreme_norm_extreme);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_diagonal_norm, setup, teardown),
        cmocka_unit_test_setup_teardown(test_dense_norm, setup, teardown),
        cmocka_unit_test_setup_teardown(test_extreme_norm, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
