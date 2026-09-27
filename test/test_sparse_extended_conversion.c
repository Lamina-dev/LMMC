/**
 * @file test_sparse_extended_conversion.c
 * @brief 稀疏矩阵扩展转换接口测试。
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
    lmmc_sparse_mat_t csr_csc_roundtrip_csr;
    lmmc_sparse_mat_t csr_csc_roundtrip_csc;
    lmmc_sparse_mat_t csr_csc_roundtrip_csr_back;
    lmmc_mat_t dense_roundtrip_dense_orig;
    lmmc_mat_t dense_roundtrip_dense_back;
    lmmc_sparse_mat_t dense_roundtrip_sparse;
    lmmc_sparse_mat_t transpose_roundtrip_orig;
    lmmc_sparse_mat_t transpose_roundtrip_trans;
    lmmc_sparse_mat_t transpose_roundtrip_trans_trans;
    lmmc_mat_t transpose_roundtrip_d_orig;
    lmmc_mat_t transpose_roundtrip_d_tt;
    lmmc_sparse_coo_t coo_formats_coo;
    lmmc_sparse_mat_t coo_formats_csr;
    lmmc_sparse_mat_t coo_formats_csc;
    lmmc_mat_t coo_formats_d_csr;
    lmmc_mat_t coo_formats_d_csc;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_destroy(&fixture->coo_formats_d_csc);
    lmmc_mat_destroy(&fixture->coo_formats_d_csr);
    lmmc_sparse_destroy(&fixture->coo_formats_csc);
    lmmc_sparse_destroy(&fixture->coo_formats_csr);
    lmmc_sparse_coo_destroy(&fixture->coo_formats_coo);
    lmmc_mat_destroy(&fixture->transpose_roundtrip_d_tt);
    lmmc_mat_destroy(&fixture->transpose_roundtrip_d_orig);
    lmmc_sparse_destroy(&fixture->transpose_roundtrip_trans_trans);
    lmmc_sparse_destroy(&fixture->transpose_roundtrip_trans);
    lmmc_sparse_destroy(&fixture->transpose_roundtrip_orig);
    lmmc_sparse_destroy(&fixture->dense_roundtrip_sparse);
    lmmc_mat_destroy(&fixture->dense_roundtrip_dense_back);
    lmmc_mat_destroy(&fixture->dense_roundtrip_dense_orig);
    lmmc_sparse_destroy(&fixture->csr_csc_roundtrip_csr_back);
    lmmc_sparse_destroy(&fixture->csr_csc_roundtrip_csc);
    lmmc_sparse_destroy(&fixture->csr_csc_roundtrip_csr);
    free(fixture);
    return 0;
}

static void test_csr_csc_roundtrip(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    double data[] = {
        4.0, 0.0, 2.0,
        0.0, 5.0, 1.0,
        3.0, 0.0, 7.0};

    st = lmmc_test_build_sparse(data, 3, 3, &fixture->csr_csc_roundtrip_csr);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_to_csc(&fixture->csr_csc_roundtrip_csr, &fixture->csr_csc_roundtrip_csc);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_to_csr(&fixture->csr_csc_roundtrip_csc, &fixture->csr_csc_roundtrip_csr_back);
    assert_true(st == LMMC_STATUS_OK);

    assert_true(((fixture->csr_csc_roundtrip_csr_back.nnz == fixture->csr_csc_roundtrip_csr.nnz) && (fixture->csr_csc_roundtrip_csr_back.rows == fixture->csr_csc_roundtrip_csr.rows)) && (fixture->csr_csc_roundtrip_csr_back.cols == fixture->csr_csc_roundtrip_csr.cols));
    for (size_t i = 0; i <= fixture->csr_csc_roundtrip_csr.rows; i++) {
        assert_true(fixture->csr_csc_roundtrip_csr_back.row_ptr[i] == fixture->csr_csc_roundtrip_csr.row_ptr[i]);
    }
    for (size_t i = 0; i < fixture->csr_csc_roundtrip_csr.nnz; i++) {
        assert_true((fixture->csr_csc_roundtrip_csr_back.col_idx[i] == fixture->csr_csc_roundtrip_csr.col_idx[i]) && (lmmc_test_nearly_equal(fixture->csr_csc_roundtrip_csr_back.values[i], fixture->csr_csc_roundtrip_csr.values[i], TEST_EPS_TIGHT)));
    }
    lmmc_sparse_destroy(&fixture->csr_csc_roundtrip_csr_back);
    lmmc_sparse_destroy(&fixture->csr_csc_roundtrip_csc);
    lmmc_sparse_destroy(&fixture->csr_csc_roundtrip_csr);
}

static void test_dense_roundtrip(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    double data[] = {
        1.0, 0.0, 3.0,
        0.0, 2.0, 0.0,
        4.0, 0.0, 5.0};

    st = lmmc_mat_create(3, 3, &fixture->dense_roundtrip_dense_orig);
    assert_true(st == LMMC_STATUS_OK);
    for (size_t i = 0; i < 9; i++)
        LMMC_REAL_SET_D(&fixture->dense_roundtrip_dense_orig.data[i], data[i]);

    st = lmmc_sparse_from_dense(&fixture->dense_roundtrip_dense_orig, 1e-14, &fixture->dense_roundtrip_sparse);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_mat_create(3, 3, &fixture->dense_roundtrip_dense_back);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_to_dense(&fixture->dense_roundtrip_sparse, &fixture->dense_roundtrip_dense_back);
    assert_true(st == LMMC_STATUS_OK);

    for (size_t i = 0; i < 9; i++) {
        assert_true(lmmc_test_nearly_equal(fixture->dense_roundtrip_dense_back.data[i], data[i], TEST_EPS_TIGHT));
    }
    lmmc_mat_destroy(&fixture->dense_roundtrip_dense_back);
    lmmc_sparse_destroy(&fixture->dense_roundtrip_sparse);
    lmmc_mat_destroy(&fixture->dense_roundtrip_dense_orig);
}

static void test_transpose_roundtrip(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    double data[] = {
        1.0, 0.0, 2.0,
        0.0, 3.0, 0.0,
        4.0, 0.0, 5.0};

    st = lmmc_test_build_sparse(data, 3, 3, &fixture->transpose_roundtrip_orig);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_transpose(&fixture->transpose_roundtrip_orig, &fixture->transpose_roundtrip_trans);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_transpose(&fixture->transpose_roundtrip_trans, &fixture->transpose_roundtrip_trans_trans);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_mat_create(3, 3, &fixture->transpose_roundtrip_d_orig);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_mat_create(3, 3, &fixture->transpose_roundtrip_d_tt);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_to_dense(&fixture->transpose_roundtrip_orig, &fixture->transpose_roundtrip_d_orig);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_to_dense(&fixture->transpose_roundtrip_trans_trans, &fixture->transpose_roundtrip_d_tt);
    assert_true(st == LMMC_STATUS_OK);

    for (size_t i = 0; i < 9; i++) {
        assert_true(lmmc_test_nearly_equal(fixture->transpose_roundtrip_d_orig.data[i], fixture->transpose_roundtrip_d_tt.data[i], TEST_EPS_TIGHT));
    }

    lmmc_mat_destroy(&fixture->transpose_roundtrip_d_tt);
    lmmc_mat_destroy(&fixture->transpose_roundtrip_d_orig);
    lmmc_sparse_destroy(&fixture->transpose_roundtrip_trans_trans);
    lmmc_sparse_destroy(&fixture->transpose_roundtrip_trans);
    lmmc_sparse_destroy(&fixture->transpose_roundtrip_orig);
}

static void test_coo_formats(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_sparse_coo_create(3, 3, 8, &fixture->coo_formats_coo);
    assert_true(st == LMMC_STATUS_OK);

    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo_formats_coo, 0, 0, 1.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo_formats_coo, 0, 2, 2.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo_formats_coo, 1, 1, 3.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo_formats_coo, 2, 0, 4.0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_sparse_coo_add_entry(&fixture->coo_formats_coo, 2, 2, 5.0), LMMC_STATUS_OK);

    st = lmmc_sparse_coo_to_csr(&fixture->coo_formats_coo, &fixture->coo_formats_csr);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_coo_to_csc(&fixture->coo_formats_coo, &fixture->coo_formats_csc);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_mat_create(3, 3, &fixture->coo_formats_d_csr);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_mat_create(3, 3, &fixture->coo_formats_d_csc);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_to_dense(&fixture->coo_formats_csr, &fixture->coo_formats_d_csr);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_to_dense(&fixture->coo_formats_csc, &fixture->coo_formats_d_csc);
    assert_true(st == LMMC_STATUS_OK);

    for (size_t i = 0; i < 9; i++) {
        assert_true(lmmc_test_nearly_equal(fixture->coo_formats_d_csr.data[i], fixture->coo_formats_d_csc.data[i], TEST_EPS_TIGHT));
    }

    lmmc_mat_destroy(&fixture->coo_formats_d_csc);
    lmmc_mat_destroy(&fixture->coo_formats_d_csr);
    lmmc_sparse_destroy(&fixture->coo_formats_csc);
    lmmc_sparse_destroy(&fixture->coo_formats_csr);
    lmmc_sparse_coo_destroy(&fixture->coo_formats_coo);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_csr_csc_roundtrip, setup, teardown),
        cmocka_unit_test_setup_teardown(test_dense_roundtrip, setup, teardown),
        cmocka_unit_test_setup_teardown(test_transpose_roundtrip, setup, teardown),
        cmocka_unit_test_setup_teardown(test_coo_formats, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
