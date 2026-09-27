/**
 * @file test_sparse_extended_products.c
 * @brief 稀疏矩阵扩展乘积接口测试。
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
    lmmc_sparse_mat_t spmv_dense_sparse;
    lmmc_mat_t spmv_dense_dense_a;
    lmmc_vec_t spmv_dense_x;
    lmmc_vec_t spmv_dense_y_sparse;
    lmmc_vec_t spmv_dense_y_dense;
    lmmc_mat_t check_spgemm_dense_da;
    lmmc_mat_t check_spgemm_dense_db;
    lmmc_mat_t check_spgemm_dense_dc_dense;
    lmmc_mat_t check_spgemm_dense_dc_from_sparse;
    lmmc_sparse_mat_t spgemm_dense_sa;
    lmmc_sparse_mat_t spgemm_dense_sb;
    lmmc_sparse_mat_t spgemm_dense_sc;
    lmmc_sparse_mat_t diagonal_spmv_diag_sparse;
    lmmc_vec_t diagonal_spmv_x;
    lmmc_vec_t diagonal_spmv_y;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_vec_destroy(&fixture->diagonal_spmv_y);
    lmmc_vec_destroy(&fixture->diagonal_spmv_x);
    lmmc_sparse_destroy(&fixture->diagonal_spmv_diag_sparse);
    lmmc_sparse_destroy(&fixture->spgemm_dense_sc);
    lmmc_sparse_destroy(&fixture->spgemm_dense_sb);
    lmmc_sparse_destroy(&fixture->spgemm_dense_sa);
    lmmc_mat_destroy(&fixture->check_spgemm_dense_dc_from_sparse);
    lmmc_mat_destroy(&fixture->check_spgemm_dense_dc_dense);
    lmmc_mat_destroy(&fixture->check_spgemm_dense_db);
    lmmc_mat_destroy(&fixture->check_spgemm_dense_da);
    lmmc_vec_destroy(&fixture->spmv_dense_y_dense);
    lmmc_vec_destroy(&fixture->spmv_dense_y_sparse);
    lmmc_vec_destroy(&fixture->spmv_dense_x);
    lmmc_mat_destroy(&fixture->spmv_dense_dense_a);
    lmmc_sparse_destroy(&fixture->spmv_dense_sparse);
    free(fixture);
    return 0;
}

static void check_spmv_values(const lmmc_vec_t *sparse, const lmmc_vec_t *dense) {

    for (size_t i = 0; i < 3; ++i) {
        assert_true(lmmc_test_nearly_equal(sparse->data[i], dense->data[i], TEST_EPS_TIGHT));
    }
}

static void test_spmv_dense(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    double a_data[] = {
        2.0, 0.0, 1.0,
        0.0, 3.0, 0.0,
        4.0, 0.0, 5.0};
    double x_data[] = {1.0, 2.0, 3.0};

    st = lmmc_test_build_sparse(a_data, 3, 3, &fixture->spmv_dense_sparse);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_mat_create(3, 3, &fixture->spmv_dense_dense_a);
    assert_true(st == LMMC_STATUS_OK);
    for (size_t i = 0; i < 9; i++)
        LMMC_REAL_SET_D(&fixture->spmv_dense_dense_a.data[i], a_data[i]);

    st = lmmc_vec_create(3, &fixture->spmv_dense_x);
    assert_true(st == LMMC_STATUS_OK);
    for (size_t i = 0; i < 3; i++)
        LMMC_REAL_SET_D(&fixture->spmv_dense_x.data[i], x_data[i]);

    st = lmmc_vec_create(3, &fixture->spmv_dense_y_sparse);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_vec_create(3, &fixture->spmv_dense_y_dense);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_mat_vec_mul(&fixture->spmv_dense_sparse, &fixture->spmv_dense_x, &fixture->spmv_dense_y_sparse);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_mat_vec_mul(&fixture->spmv_dense_dense_a, &fixture->spmv_dense_x, &fixture->spmv_dense_y_dense);
    assert_true(st == LMMC_STATUS_OK);

    check_spmv_values(&fixture->spmv_dense_y_sparse, &fixture->spmv_dense_y_dense);

    lmmc_vec_destroy(&fixture->spmv_dense_y_dense);
    lmmc_vec_destroy(&fixture->spmv_dense_y_sparse);
    lmmc_vec_destroy(&fixture->spmv_dense_x);
    lmmc_mat_destroy(&fixture->spmv_dense_dense_a);
    lmmc_sparse_destroy(&fixture->spmv_dense_sparse);
}

static void check_spgemm_dense(struct test_fixture *fixture, const lmmc_sparse_mat_t *sc, const double *a_data, const double *b_data) {

    lmmc_status_t st;

    st = lmmc_mat_create(3, 3, &fixture->check_spgemm_dense_da);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_mat_create(3, 3, &fixture->check_spgemm_dense_db);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_mat_create(3, 3, &fixture->check_spgemm_dense_dc_dense);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_mat_create(3, 3, &fixture->check_spgemm_dense_dc_from_sparse);
    assert_true(st == LMMC_STATUS_OK);

    for (size_t i = 0; i < 9; i++)
        LMMC_REAL_SET_D(&fixture->check_spgemm_dense_da.data[i], a_data[i]);
    for (size_t i = 0; i < 9; i++)
        LMMC_REAL_SET_D(&fixture->check_spgemm_dense_db.data[i], b_data[i]);

    st = lmmc_mat_mul(&fixture->check_spgemm_dense_da, &fixture->check_spgemm_dense_db, &fixture->check_spgemm_dense_dc_dense);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_to_dense(sc, &fixture->check_spgemm_dense_dc_from_sparse);
    assert_true(st == LMMC_STATUS_OK);

    for (size_t i = 0; i < 9; i++) {
        assert_true(lmmc_test_nearly_equal(fixture->check_spgemm_dense_dc_from_sparse.data[i], fixture->check_spgemm_dense_dc_dense.data[i], TEST_EPS_TIGHT));
    }

    lmmc_mat_destroy(&fixture->check_spgemm_dense_dc_from_sparse);
    lmmc_mat_destroy(&fixture->check_spgemm_dense_dc_dense);
    lmmc_mat_destroy(&fixture->check_spgemm_dense_db);
    lmmc_mat_destroy(&fixture->check_spgemm_dense_da);
}

static void test_spgemm_dense(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    double a_data[] = {
        1.0, 0.0, 2.0,
        0.0, 3.0, 0.0,
        4.0, 0.0, 5.0};
    double b_data[] = {
        0.0, 1.0, 0.0,
        2.0, 0.0, 3.0,
        0.0, 4.0, 0.0};

    st = lmmc_test_build_sparse(a_data, 3, 3, &fixture->spgemm_dense_sa);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_test_build_sparse(b_data, 3, 3, &fixture->spgemm_dense_sb);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_mat_mat_mul_sparse(&fixture->spgemm_dense_sa, &fixture->spgemm_dense_sb, &fixture->spgemm_dense_sc);
    assert_true(st == LMMC_STATUS_OK);

    check_spgemm_dense(fixture, &fixture->spgemm_dense_sc, a_data, b_data);
    lmmc_sparse_destroy(&fixture->spgemm_dense_sc);
    lmmc_sparse_destroy(&fixture->spgemm_dense_sb);
    lmmc_sparse_destroy(&fixture->spgemm_dense_sa);
}

static void test_diagonal_spmv(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    double diag_data[] = {
        2.0, 0.0, 0.0,
        0.0, 3.0, 0.0,
        0.0, 0.0, 4.0};
    double x_vals[] = {1.0, 2.0, 3.0};

    st = lmmc_test_build_sparse(diag_data, 3, 3, &fixture->diagonal_spmv_diag_sparse);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_vec_create(3, &fixture->diagonal_spmv_x);
    assert_true(st == LMMC_STATUS_OK);
    for (size_t i = 0; i < 3; i++)
        LMMC_REAL_SET_D(&fixture->diagonal_spmv_x.data[i], x_vals[i]);

    st = lmmc_vec_create(3, &fixture->diagonal_spmv_y);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_mat_vec_mul(&fixture->diagonal_spmv_diag_sparse, &fixture->diagonal_spmv_x, &fixture->diagonal_spmv_y);
    assert_true(st == LMMC_STATUS_OK);

    assert_true(((lmmc_test_nearly_equal(fixture->diagonal_spmv_y.data[0], 2.0, TEST_EPS_TIGHT)) && (lmmc_test_nearly_equal(fixture->diagonal_spmv_y.data[1], 6.0, TEST_EPS_TIGHT))) && (lmmc_test_nearly_equal(fixture->diagonal_spmv_y.data[2], 12.0, TEST_EPS_TIGHT)));

    lmmc_vec_destroy(&fixture->diagonal_spmv_y);
    lmmc_vec_destroy(&fixture->diagonal_spmv_x);
    lmmc_sparse_destroy(&fixture->diagonal_spmv_diag_sparse);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_spmv_dense, setup, teardown),
        cmocka_unit_test_setup_teardown(test_spgemm_dense, setup, teardown),
        cmocka_unit_test_setup_teardown(test_diagonal_spmv, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
