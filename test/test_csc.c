/**
 * @file test_csc.c
 * 针对 LMMC 中 csc 相关接口的单元测试。
 */
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

struct test_fixture {
    lmmc_mat_t dense;
    lmmc_sparse_mat_t csr;
    lmmc_sparse_mat_t csc;
    lmmc_sparse_mat_t csr_back;
    lmmc_vec_t x;
    lmmc_vec_t y_csr;
    lmmc_vec_t y_csc;
    lmmc_sparse_mat_t csc_transpose_csc_t;
    lmmc_mat_t csc_transpose_t_dense;
};

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_destroy(&fixture->csc_transpose_t_dense);
    lmmc_sparse_destroy(&fixture->csc_transpose_csc_t);
    lmmc_mat_destroy(&fixture->dense);
    lmmc_sparse_destroy(&fixture->csr);
    lmmc_sparse_destroy(&fixture->csc);
    lmmc_sparse_destroy(&fixture->csr_back);
    lmmc_vec_destroy(&fixture->x);
    lmmc_vec_destroy(&fixture->y_csr);
    lmmc_vec_destroy(&fixture->y_csc);
    free(fixture);
    *state = NULL;
    return 0;
}

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    lmmc_status_t st;

    {
        lmmc_mat_t *dense = &fixture->dense;
        lmmc_sparse_mat_t *csr = &fixture->csr;
        lmmc_sparse_mat_t *csc = &fixture->csc;

        st = lmmc_mat_create(3, 3, dense);
        if (st != LMMC_STATUS_OK)
            goto cleanup;
        LMMC_REAL_SET_D(&dense->data[0], 1.0);
        LMMC_REAL_SET_D(&dense->data[1], 0.0);
        LMMC_REAL_SET_D(&dense->data[2], 2.0);
        LMMC_REAL_SET_D(&dense->data[3], 3.0);
        LMMC_REAL_SET_D(&dense->data[4], 4.0);
        LMMC_REAL_SET_D(&dense->data[5], 0.0);
        LMMC_REAL_SET_D(&dense->data[6], 0.0);
        LMMC_REAL_SET_D(&dense->data[7], 5.0);
        LMMC_REAL_SET_D(&dense->data[8], 6.0);

        st = lmmc_sparse_from_dense(dense, 1e-14, csr);
        if (st != LMMC_STATUS_OK)
            goto cleanup;

        st = lmmc_sparse_to_csc(csr, csc);
        if (st != LMMC_STATUS_OK)
            goto cleanup;
    }
    return 0;

cleanup:
    teardown(state);
    return st;
}

static void test_csr_to_csc(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_sparse_mat_t *csc = &fixture->csc;

    assert_true(csc->format == LMMC_SPARSE_CSC);
}

static void test_csc_spmv_equivalence(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_sparse_mat_t *csr = &fixture->csr;
    lmmc_sparse_mat_t *csc = &fixture->csc;
    lmmc_vec_t *x = &fixture->x;
    lmmc_vec_t *y_csr = &fixture->y_csr;
    lmmc_vec_t *y_csc = &fixture->y_csc;

    lmmc_status_t st;

    st = lmmc_vec_create(3, x);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_vec_create(3, y_csr);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_vec_create(3, y_csc);
    assert_true(st == LMMC_STATUS_OK);
    LMMC_REAL_SET_D(&x->data[0], 1.0);
    LMMC_REAL_SET_D(&x->data[1], 1.0);
    LMMC_REAL_SET_D(&x->data[2], 1.0);

    st = lmmc_sparse_mat_vec_mul(csr, x, y_csr);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_mat_vec_mul(csc, x, y_csc);
    assert_true(st == LMMC_STATUS_OK);

    for (size_t i = 0; i < 3; ++i) {
        assert_true(lmmc_test_nearly_equal(y_csr->data[i], y_csc->data[i], 1e-12));
    }
}

static void test_csc_roundtrip(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_sparse_mat_t *csr = &fixture->csr;
    lmmc_sparse_mat_t *csc = &fixture->csc;
    lmmc_sparse_mat_t *csr_back = &fixture->csr_back;

    lmmc_status_t st;

    st = lmmc_sparse_to_csr(csc, csr_back);
    assert_true(st == LMMC_STATUS_OK);

    assert_true(((csr_back->nnz == csr->nnz) && (csr_back->rows == csr->rows)) && (csr_back->cols == csr->cols));

    for (size_t i = 0; i <= csr->rows; ++i) {
        assert_true(csr_back->row_ptr[i] == csr->row_ptr[i]);
    }

    for (size_t i = 0; i < csr->nnz; ++i) {
        assert_true((csr_back->col_idx[i] == csr->col_idx[i]) && (lmmc_test_nearly_equal(csr_back->values[i], csr->values[i], 1e-12)));
    }
}

static void test_csc_transpose(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *dense = &fixture->dense;
    lmmc_sparse_mat_t *csc = &fixture->csc;

    lmmc_status_t st;

    st = lmmc_sparse_transpose(csc, &fixture->csc_transpose_csc_t);
    assert_true(st == LMMC_STATUS_OK);
    assert_true(((fixture->csc_transpose_csc_t.format == LMMC_SPARSE_CSC) && (fixture->csc_transpose_csc_t.rows == 3)) && (fixture->csc_transpose_csc_t.cols == 3));

    assert_int_equal(lmmc_mat_create(3, 3, &fixture->csc_transpose_t_dense), LMMC_STATUS_OK);
    st = lmmc_sparse_to_dense(&fixture->csc_transpose_csc_t, &fixture->csc_transpose_t_dense);
    assert_true(st == LMMC_STATUS_OK);

    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            assert_true(lmmc_test_nearly_equal(fixture->csc_transpose_t_dense.data[i * fixture->csc_transpose_t_dense.stride + j], dense->data[j * dense->stride + i], 1e-12));
        }
    }
    lmmc_mat_destroy(&fixture->csc_transpose_t_dense);

    lmmc_sparse_destroy(&fixture->csc_transpose_csc_t);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_csr_to_csc, setup, teardown),
        cmocka_unit_test_setup_teardown(test_csc_spmv_equivalence, setup, teardown),
        cmocka_unit_test_setup_teardown(test_csc_roundtrip, setup, teardown),
        cmocka_unit_test_setup_teardown(test_csc_transpose, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
