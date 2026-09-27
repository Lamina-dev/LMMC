/**
 * @file test_sparse_extended_boundaries.c
 * @brief 扩展稀疏矩阵接口的边界测试。
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
    lmmc_sparse_mat_t dimensions_sa;
    lmmc_vec_t dimensions_x_bad;
    lmmc_vec_t dimensions_y_bad;
    lmmc_sparse_mat_t dimensions_sb;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_sparse_destroy(&fixture->dimensions_sb);
    lmmc_vec_destroy(&fixture->dimensions_y_bad);
    lmmc_vec_destroy(&fixture->dimensions_x_bad);
    lmmc_sparse_destroy(&fixture->dimensions_sa);
    free(fixture);
    return 0;
}

static void test_dimensions(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    double a_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    st = lmmc_test_build_sparse(a_data, 2, 3, &fixture->dimensions_sa);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_vec_create(2, &fixture->dimensions_x_bad);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_vec_create(2, &fixture->dimensions_y_bad);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_mat_vec_mul(&fixture->dimensions_sa, &fixture->dimensions_x_bad, &fixture->dimensions_y_bad);
    assert_true(st == LMMC_STATUS_DIMENSION_MISMATCH);

    lmmc_sparse_mat_t sc = {0};
    double b_data[] = {1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    st = lmmc_test_build_sparse(b_data, 2, 3, &fixture->dimensions_sb);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_mat_mat_mul_sparse(&fixture->dimensions_sa, &fixture->dimensions_sb, &sc);
    assert_true(st == LMMC_STATUS_DIMENSION_MISMATCH);
    lmmc_sparse_destroy(&fixture->dimensions_sb);

    lmmc_vec_destroy(&fixture->dimensions_y_bad);
    lmmc_vec_destroy(&fixture->dimensions_x_bad);
    lmmc_sparse_destroy(&fixture->dimensions_sa);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_dimensions, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
