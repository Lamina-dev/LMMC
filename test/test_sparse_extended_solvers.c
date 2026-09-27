/**
 * @file test_sparse_extended_solvers.c
 * @brief 稀疏矩阵扩展求解接口测试。
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
    lmmc_sparse_mat_t lu_a_csc;
    lmmc_sparse_lu_t *lu_lu;
    lmmc_vec_t lu_b;
    lmmc_vec_t lu_x;
    lmmc_sparse_mat_t cholesky_a_csc;
    lmmc_sparse_chol_t *cholesky_chol;
    lmmc_vec_t cholesky_b;
    lmmc_vec_t cholesky_x;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_vec_destroy(&fixture->cholesky_x);
    lmmc_vec_destroy(&fixture->cholesky_b);
    lmmc_sparse_chol_destroy(fixture->cholesky_chol);
    lmmc_sparse_destroy(&fixture->cholesky_a_csc);
    lmmc_vec_destroy(&fixture->lu_x);
    lmmc_vec_destroy(&fixture->lu_b);
    lmmc_sparse_lu_destroy(fixture->lu_lu);
    lmmc_sparse_destroy(&fixture->lu_a_csc);
    free(fixture);
    return 0;
}

static lmmc_status_t helper_build_sparse_csc(const double *data, size_t rows, size_t cols,
                                             lmmc_sparse_mat_t *out) {
    lmmc_sparse_builder_t *builder = NULL;
    lmmc_status_t st = lmmc_sparse_builder_create(rows, cols, rows * cols, &builder);
    if (st != LMMC_STATUS_OK)
        return st;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            if (fabs(data[i * cols + j]) > 1e-14) {
                st = lmmc_sparse_builder_add(builder, i, j, data[i * cols + j]);
                if (st != LMMC_STATUS_OK) {
                    lmmc_sparse_builder_destroy(builder);
                    return st;
                }
            }
        }
    }
    st = lmmc_sparse_builder_build(builder, LMMC_SPARSE_CSC, out);
    lmmc_sparse_builder_destroy(builder);
    return st;
}

static void test_lu(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    double a_data[] = {
        2.0, 0.0, 0.0,
        0.0, 3.0, 0.0,
        0.0, 0.0, 4.0};

    st = helper_build_sparse_csc(a_data, 3, 3, &fixture->lu_a_csc);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_lu_symbolic(&fixture->lu_a_csc, &fixture->lu_lu);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_lu_numeric(&fixture->lu_a_csc, fixture->lu_lu);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_vec_create(3, &fixture->lu_b);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_vec_create(3, &fixture->lu_x);
    assert_true(st == LMMC_STATUS_OK);

    fixture->lu_b.data[0] = 2.0;
    fixture->lu_b.data[1] = 6.0;
    fixture->lu_b.data[2] = 12.0;

    st = lmmc_sparse_lu_solve(fixture->lu_lu, &fixture->lu_b, &fixture->lu_x);
    assert_true(st == LMMC_STATUS_OK);

    assert_true(((lmmc_test_nearly_equal(fixture->lu_x.data[0], 1.0, TEST_EPS_NORMAL)) && (lmmc_test_nearly_equal(fixture->lu_x.data[1], 2.0, TEST_EPS_NORMAL))) && (lmmc_test_nearly_equal(fixture->lu_x.data[2], 3.0, TEST_EPS_NORMAL)));

    lmmc_vec_destroy(&fixture->lu_x);
    lmmc_vec_destroy(&fixture->lu_b);
    lmmc_sparse_lu_destroy(fixture->lu_lu);
    fixture->lu_lu = NULL;
    lmmc_sparse_destroy(&fixture->lu_a_csc);
}

static void test_cholesky(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    double a_data[] = {
        4.0, 1.0, 0.0,
        1.0, 4.0, 1.0,
        0.0, 1.0, 4.0};

    st = helper_build_sparse_csc(a_data, 3, 3, &fixture->cholesky_a_csc);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_symbolic(&fixture->cholesky_a_csc, &fixture->cholesky_chol);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_numeric(&fixture->cholesky_a_csc, fixture->cholesky_chol);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_vec_create(3, &fixture->cholesky_b);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_vec_create(3, &fixture->cholesky_x);
    assert_true(st == LMMC_STATUS_OK);

    fixture->cholesky_b.data[0] = 6.0;
    fixture->cholesky_b.data[1] = 12.0;
    fixture->cholesky_b.data[2] = 14.0;

    st = lmmc_sparse_chol_solve(fixture->cholesky_chol, &fixture->cholesky_b, &fixture->cholesky_x);
    assert_true(st == LMMC_STATUS_OK);

    assert_true(((lmmc_test_nearly_equal(fixture->cholesky_x.data[0], 1.0, TEST_EPS_NORMAL)) && (lmmc_test_nearly_equal(fixture->cholesky_x.data[1], 2.0, TEST_EPS_NORMAL))) && (lmmc_test_nearly_equal(fixture->cholesky_x.data[2], 3.0, TEST_EPS_NORMAL)));

    lmmc_vec_destroy(&fixture->cholesky_x);
    lmmc_vec_destroy(&fixture->cholesky_b);
    lmmc_sparse_chol_destroy(fixture->cholesky_chol);
    fixture->cholesky_chol = NULL;
    lmmc_sparse_destroy(&fixture->cholesky_a_csc);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_lu, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cholesky, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
