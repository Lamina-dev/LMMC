/**
 * @file test_sparse.c
 * 针对 LMMC 中 sparse 相关接口的单元测试。
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
    lmmc_mat_t dense_t;
    lmmc_mat_t zero_dense;
    lmmc_mat_t bmat;
    lmmc_mat_t cmat;
    lmmc_mat_t bbad;
    lmmc_mat_t bzero;
    lmmc_mat_t czero;
    lmmc_sparse_mat_t sparse;
    lmmc_sparse_mat_t sparse_t;
    lmmc_sparse_mat_t zero_sparse;
    lmmc_sparse_mat_t bad_sparse;
    lmmc_vec_t x;
    lmmc_vec_t y;
    lmmc_vec_t zx;
    lmmc_vec_t zy;
    lmmc_vec_t x_bad;
    lmmc_mat_t canonical_storage_boundary_dense;
    lmmc_vec_t canonical_storage_boundary_rhs;
    lmmc_vec_t canonical_storage_boundary_solution;
    lmmc_sparse_mat_t canonical_storage_boundary_valid;
    lmmc_sparse_lu_t *canonical_storage_boundary_lu;
    lmmc_sparse_chol_t *canonical_storage_boundary_chol;
};

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_sparse_chol_destroy(fixture->canonical_storage_boundary_chol);
    lmmc_sparse_lu_destroy(fixture->canonical_storage_boundary_lu);
    lmmc_sparse_destroy(&fixture->canonical_storage_boundary_valid);
    lmmc_vec_destroy(&fixture->canonical_storage_boundary_solution);
    lmmc_vec_destroy(&fixture->canonical_storage_boundary_rhs);
    lmmc_mat_destroy(&fixture->canonical_storage_boundary_dense);
    lmmc_mat_destroy(&fixture->dense);
    lmmc_mat_destroy(&fixture->zero_dense);
    lmmc_vec_destroy(&fixture->x);
    lmmc_vec_destroy(&fixture->y);
    lmmc_sparse_destroy(&fixture->bad_sparse);
    lmmc_sparse_destroy(&fixture->zero_sparse);
    lmmc_sparse_destroy(&fixture->sparse);
    lmmc_sparse_destroy(&fixture->sparse_t);
    lmmc_vec_destroy(&fixture->zx);
    lmmc_vec_destroy(&fixture->zy);
    lmmc_vec_destroy(&fixture->x_bad);
    lmmc_mat_destroy(&fixture->dense_t);
    lmmc_mat_destroy(&fixture->bmat);
    lmmc_mat_destroy(&fixture->cmat);
    lmmc_mat_destroy(&fixture->bbad);
    lmmc_mat_destroy(&fixture->bzero);
    lmmc_mat_destroy(&fixture->czero);
    free(fixture);
    *state = NULL;
    return 0;
}

static lmmc_status_t create_sparse_product_operands(struct test_fixture *fixture) {
    lmmc_status_t st = lmmc_mat_create(3, 3, &fixture->dense);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_create(3, &fixture->x);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_create(3, &fixture->y);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    LMMC_REAL_SET_D(&fixture->dense.data[0], 4.0);
    LMMC_REAL_SET_D(&fixture->dense.data[1], 0.0);
    LMMC_REAL_SET_D(&fixture->dense.data[2], 0.0);
    LMMC_REAL_SET_D(&fixture->dense.data[3], 0.0);
    LMMC_REAL_SET_D(&fixture->dense.data[4], 5.0);
    LMMC_REAL_SET_D(&fixture->dense.data[5], 1.0);
    LMMC_REAL_SET_D(&fixture->dense.data[6], 2.0);
    LMMC_REAL_SET_D(&fixture->dense.data[7], 0.0);
    LMMC_REAL_SET_D(&fixture->dense.data[8], 3.0);
    LMMC_REAL_SET_D(&fixture->x.data[0], 1.0);
    LMMC_REAL_SET_D(&fixture->x.data[1], 2.0);
    LMMC_REAL_SET_D(&fixture->x.data[2], -1.0);
    return lmmc_sparse_from_dense(&fixture->dense, 1e-14, &fixture->sparse);
}

static lmmc_status_t create_zero_operands(struct test_fixture *fixture) {
    lmmc_status_t st = lmmc_mat_create(4, 4, &fixture->zero_dense);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_sparse_from_dense(&fixture->zero_dense, 0.0, &fixture->zero_sparse);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_create(4, &fixture->zx);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_create(4, &fixture->zy);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    return lmmc_vec_fill(&fixture->zx, 3.0);
}

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    lmmc_status_t st = create_sparse_product_operands(fixture);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_sparse_transpose(&fixture->sparse, &fixture->sparse_t);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_mat_create(3, 3, &fixture->dense_t);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_sparse_to_dense(&fixture->sparse_t, &fixture->dense_t);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_mat_create(3, 2, &fixture->bmat);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_mat_create(3, 2, &fixture->cmat);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    LMMC_REAL_SET_D(&fixture->bmat.data[0], 1.0);
    LMMC_REAL_SET_D(&fixture->bmat.data[1], 2.0);
    LMMC_REAL_SET_D(&fixture->bmat.data[2], 0.0);
    LMMC_REAL_SET_D(&fixture->bmat.data[3], 1.0);
    LMMC_REAL_SET_D(&fixture->bmat.data[4], 3.0);
    LMMC_REAL_SET_D(&fixture->bmat.data[5], -1.0);
    st = create_zero_operands(fixture);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    return 0;

cleanup:
    teardown(state);
    return st;
}

static void test_sparse_matrix_vector_product(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_sparse_mat_t *sparse = &fixture->sparse;
    lmmc_vec_t *x = &fixture->x;
    lmmc_vec_t *y = &fixture->y;
    lmmc_status_t st;
    st = lmmc_sparse_mat_vec_mul(sparse, x, y);
    assert_true(st == LMMC_STATUS_OK);

    assert_true(((lmmc_test_nearly_equal(y->data[0], 4.0, 1e-12)) && (lmmc_test_nearly_equal(y->data[1], 9.0, 1e-12))) && (lmmc_test_nearly_equal(y->data[2], -1.0, 1e-12)));
}

static void test_transpose_values(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *dense_t = &fixture->dense_t;

    static const lmmc_real_t expected[] = {
        4.0, 0.0, 2.0,
        0.0, 5.0, 0.0,
        0.0, 1.0, 3.0};
    assert_int_equal(lmmc_sparse_transpose(&fixture->sparse, NULL), LMMC_STATUS_INVALID_ARGUMENT);
    for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i) {
        assert_true(lmmc_test_nearly_equal(dense_t->data[i], expected[i], 1e-12));
    }
}

static void test_sparse_dense_matrix_product(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *bmat = &fixture->bmat;
    lmmc_mat_t *cmat = &fixture->cmat;
    lmmc_sparse_mat_t *sparse = &fixture->sparse;
    lmmc_status_t st;
    static const lmmc_real_t expected[] = {
        4.0, 8.0,
        3.0, 4.0,
        11.0, 1.0};
    st = lmmc_sparse_mat_mat_mul_dense(sparse, bmat, cmat);
    assert_true(st == LMMC_STATUS_OK);

    for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i) {
        assert_true(lmmc_test_nearly_equal(cmat->data[i], expected[i], 1e-12));
    }
}

static void test_sparse_input_errors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *cmat = &fixture->cmat;
    lmmc_mat_t *bbad = &fixture->bbad;
    lmmc_sparse_mat_t *sparse = &fixture->sparse;
    lmmc_sparse_mat_t *bad_sparse = &fixture->bad_sparse;

    lmmc_status_t st;

    st = lmmc_mat_create(2, 2, bbad);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_mat_mat_mul_dense(sparse, bbad, cmat);
    assert_true(st == LMMC_STATUS_DIMENSION_MISMATCH);

    st = lmmc_sparse_create_csr(0, 3, 0, bad_sparse);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    {
        size_t bad_row_ptr[3] = {0, 1, 0};
        size_t bad_col_idx[1] = {0};
        double bad_values[1] = {1.0};
        st = lmmc_sparse_wrap_csr(2, 2, 1, bad_row_ptr, bad_col_idx, bad_values, bad_sparse);
        assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
    }
}

static void test_zero_sparse_vector_product(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_sparse_mat_t *zero_sparse = &fixture->zero_sparse;
    lmmc_vec_t *zx = &fixture->zx;
    lmmc_vec_t *zy = &fixture->zy;
    lmmc_status_t st;
    assert_int_equal(fixture->zero_sparse.nnz, 0);
    st = lmmc_sparse_mat_vec_mul(zero_sparse, zx, zy);
    assert_true(st == LMMC_STATUS_OK);
    assert_true((((lmmc_test_nearly_equal(zy->data[0], 0.0, 1e-12)) && (lmmc_test_nearly_equal(zy->data[1], 0.0, 1e-12))) && (lmmc_test_nearly_equal(zy->data[2], 0.0, 1e-12))) && (lmmc_test_nearly_equal(zy->data[3], 0.0, 1e-12)));
}

static void test_zero_sparse_matrix_product(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *bzero = &fixture->bzero;
    lmmc_mat_t *czero = &fixture->czero;
    lmmc_sparse_mat_t *zero_sparse = &fixture->zero_sparse;

    lmmc_status_t st;

    st = lmmc_mat_create(4, 2, bzero);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_mat_create(4, 2, czero);
    assert_true(st == LMMC_STATUS_OK);

    LMMC_REAL_SET_D(&bzero->data[0], 1.0);
    LMMC_REAL_SET_D(&bzero->data[1], 0.0);
    LMMC_REAL_SET_D(&bzero->data[2], 2.0);
    LMMC_REAL_SET_D(&bzero->data[3], 1.0);
    LMMC_REAL_SET_D(&bzero->data[4], 3.0);
    LMMC_REAL_SET_D(&bzero->data[5], 2.0);
    LMMC_REAL_SET_D(&bzero->data[6], 4.0);
    LMMC_REAL_SET_D(&bzero->data[7], 3.0);

    st = lmmc_sparse_mat_mat_mul_dense(zero_sparse, bzero, czero);
    assert_true(st == LMMC_STATUS_OK);
    for (size_t i = 0; i < 8; ++i) {
        assert_true(lmmc_test_nearly_equal(czero->data[i], 0.0, 1e-12));
    }
}

static void test_sparse_vector_shape_error(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_sparse_mat_t *zero_sparse = &fixture->zero_sparse;
    lmmc_vec_t *zy = &fixture->zy;
    lmmc_vec_t *x_bad = &fixture->x_bad;

    lmmc_status_t st;

    st = lmmc_vec_create(3, x_bad);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_mat_vec_mul(zero_sparse, x_bad, zy);
    assert_true(st == LMMC_STATUS_DIMENSION_MISMATCH);
}

static void assert_damaged_consumers_reject(const lmmc_sparse_mat_t *bad,
                                            const lmmc_sparse_mat_t *valid, lmmc_mat_t *dense, lmmc_vec_t *rhs,
                                            lmmc_vec_t *solution, lmmc_sparse_lu_t *lu, lmmc_sparse_chol_t *chol) {

    lmmc_sparse_mat_t output = {0};
    lmmc_sparse_lu_t *new_lu = NULL;
    lmmc_sparse_chol_t *new_chol = NULL;
    lmmc_precond_t precond = {0};
    lmmc_itersolve_result_t result = {0};
    double norm = 23;
    assert_true(lmmc_sparse_to_dense(bad, dense) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_sparse_diag(bad, solution) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_sparse_norm_fro(bad, &norm) == LMMC_STATUS_INVALID_ARGUMENT && norm == 23);
    assert_true(lmmc_sparse_mat_vec_mul(bad, rhs, solution) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_sparse_mat_mat_mul_dense(bad, dense, dense) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_sparse_add(1, bad, 1, valid, &output) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_sparse_add(1, valid, 1, bad, &output) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_sparse_mat_mat_mul_sparse(bad, valid, &output) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_sparse_mat_mat_mul_sparse(valid, bad, &output) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_sparse_to_csr(bad, &output) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_sparse_to_csc(bad, &output) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_sparse_transpose(bad, &output) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_sparse_lu_symbolic(bad, &new_lu) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_sparse_chol_symbolic(bad, &new_chol) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_sparse_lu_numeric(bad, lu) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_sparse_chol_numeric(bad, chol) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_precond_create_jacobi(bad, &precond) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_precond_create_ilu0(bad, &precond) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_precond_create_ilut(bad, 0.0, 2, &precond) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_cg_solve(bad, rhs, NULL, NULL, solution, &result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_bicgstab_solve(bad, rhs, NULL, NULL, solution, &result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_gmres_solve(bad, rhs, NULL, NULL, solution, &result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_minres_solve(bad, rhs, NULL, NULL, solution, &result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_lsqr_solve(bad, rhs, NULL, solution, &result) == LMMC_STATUS_INVALID_ARGUMENT);
}

static void damage_sparse_descriptor(int damage, lmmc_sparse_mat_t *bad,
                                     size_t *ptr, size_t *idx) {
    switch (damage) {
    case 0: {
        idx[1] = 0;
        break;
    }
    case 1: {
        idx[0] = 1;
        idx[1] = 0;
        break;
    }
    case 2: {
        idx[1] = 2;
        break;
    }
    case 3: {
        ptr[0] = 1;
        break;
    }
    case 4: {
        ptr[1] = 3;
        break;
    }
    case 5: {
        ptr[2] = 1;
        break;
    }
    case 6: {
        bad->row_ptr = NULL;
        break;
    }
    case 7: {
        bad->col_idx = NULL;
        break;
    }
    case 8: {
        bad->values = NULL;
        break;
    }
    case 9: {
        bad->rows = 0;
        break;
    }
    case 10: {
        bad->format = (lmmc_sparse_format_t)7;
        break;
    }
    }
}

static void check_sparse_wrap_atomicity(const lmmc_sparse_mat_t *bad,
                                        const lmmc_sparse_mat_t *valid, size_t *ptr, size_t *idx, double *values,
                                        lmmc_status_t (*wrap)(size_t, size_t, size_t, size_t *, size_t *, double *, lmmc_sparse_mat_t *)) {

    const lmmc_sparse_mat_t before = *valid;
    lmmc_sparse_mat_t output = before;
    const size_t ptr_before[3] = {ptr[0], ptr[1], ptr[2]};
    const size_t idx_before[2] = {idx[0], idx[1]};
    const double values_before[2] = {values[0], values[1]};
    assert_int_equal(
        wrap(bad->rows, bad->cols, bad->nnz, bad->row_ptr,
             bad->col_idx, bad->values, &output),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(output.rows == before.rows);
    assert_true(output.cols == before.cols);
    assert_true(output.nnz == before.nnz);
    assert_true(output.row_ptr == before.row_ptr);
    assert_true(output.col_idx == before.col_idx);
    assert_true(output.values == before.values);
    assert_true(output.format == before.format);
    assert_true(output.owns_data == before.owns_data);
    for (size_t i = 0; i < 3; ++i) {
        assert_true(ptr[i] == ptr_before[i]);
    }
    for (size_t i = 0; i < 2; ++i) {
        assert_true(idx[i] == idx_before[i]);
        assert_true(values[i] == values_before[i]);
    }
}

static void test_canonical_storage_boundary(void **state) {
    struct test_fixture *fixture = *state;

    size_t valid_ptr[] = {0, 1, 2}, valid_idx[] = {0, 1};
    double valid_values[] = {2, 3};

    assert_true(lmmc_mat_create(2, 2, &fixture->canonical_storage_boundary_dense) == LMMC_STATUS_OK);
    assert_true(lmmc_vec_create(2, &fixture->canonical_storage_boundary_rhs) == LMMC_STATUS_OK);
    assert_true(lmmc_vec_create(2, &fixture->canonical_storage_boundary_solution) == LMMC_STATUS_OK);
    fixture->canonical_storage_boundary_rhs.data[0] = 2;
    fixture->canonical_storage_boundary_rhs.data[1] = 3;
    for (int format = 0; format < 2; ++format) {

        lmmc_status_t (*wrap)(size_t, size_t, size_t, size_t *, size_t *, double *, lmmc_sparse_mat_t *) =
            format ? lmmc_sparse_wrap_csc : lmmc_sparse_wrap_csr;
        assert_true(wrap(2, 2, 2, valid_ptr, valid_idx, valid_values, &fixture->canonical_storage_boundary_valid) == LMMC_STATUS_OK);
        assert_true(fixture->canonical_storage_boundary_valid.row_ptr == valid_ptr && fixture->canonical_storage_boundary_valid.col_idx == valid_idx &&
                    fixture->canonical_storage_boundary_valid.values == valid_values && !fixture->canonical_storage_boundary_valid.owns_data);
        assert_true(lmmc_sparse_lu_symbolic(&fixture->canonical_storage_boundary_valid, &fixture->canonical_storage_boundary_lu) == LMMC_STATUS_OK);
        assert_true(lmmc_sparse_chol_symbolic(&fixture->canonical_storage_boundary_valid, &fixture->canonical_storage_boundary_chol) == LMMC_STATUS_OK);
        for (int damage = 0; damage < 11; ++damage) {
            size_t ptr[] = {0, 2, 2}, idx[] = {0, 1};
            double values[] = {2, 3};
            lmmc_sparse_mat_t bad = fixture->canonical_storage_boundary_valid;
            bad.row_ptr = ptr;
            bad.col_idx = idx;
            bad.values = values;
            damage_sparse_descriptor(damage, &bad, ptr, idx);
            if (damage != 10) {
                check_sparse_wrap_atomicity(&bad, &fixture->canonical_storage_boundary_valid, ptr, idx, values, wrap);
            }
            assert_damaged_consumers_reject(&bad, &fixture->canonical_storage_boundary_valid,
                                            &fixture->canonical_storage_boundary_dense, &fixture->canonical_storage_boundary_rhs, &fixture->canonical_storage_boundary_solution, fixture->canonical_storage_boundary_lu, fixture->canonical_storage_boundary_chol);
        }
        lmmc_sparse_lu_destroy(fixture->canonical_storage_boundary_lu);
        fixture->canonical_storage_boundary_lu = NULL;
        lmmc_sparse_chol_destroy(fixture->canonical_storage_boundary_chol);
        fixture->canonical_storage_boundary_chol = NULL;
        lmmc_sparse_destroy(&fixture->canonical_storage_boundary_valid);
        assert_true(valid_ptr[1] == 1 && valid_idx[1] == 1 && valid_values[1] == 3);
        size_t empty_ptr[] = {0, 0, 0};
        assert_true(wrap(2, 2, 0, empty_ptr, NULL, NULL, &fixture->canonical_storage_boundary_valid) == LMMC_STATUS_OK);
        assert_true(lmmc_sparse_to_dense(&fixture->canonical_storage_boundary_valid, &fixture->canonical_storage_boundary_dense) == LMMC_STATUS_OK);
        assert_true(fixture->canonical_storage_boundary_dense.data[0] == 0 && fixture->canonical_storage_boundary_dense.data[fixture->canonical_storage_boundary_dense.stride + 1] == 0);
        lmmc_sparse_destroy(&fixture->canonical_storage_boundary_valid);
    }
    lmmc_mat_destroy(&fixture->canonical_storage_boundary_dense);
    lmmc_vec_destroy(&fixture->canonical_storage_boundary_rhs);
    lmmc_vec_destroy(&fixture->canonical_storage_boundary_solution);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_canonical_storage_boundary, setup, teardown),
        cmocka_unit_test_setup_teardown(test_sparse_matrix_vector_product, setup, teardown),
        cmocka_unit_test_setup_teardown(test_transpose_values, setup, teardown),
        cmocka_unit_test_setup_teardown(test_sparse_dense_matrix_product, setup, teardown),
        cmocka_unit_test_setup_teardown(test_sparse_input_errors, setup, teardown),
        cmocka_unit_test_setup_teardown(test_zero_sparse_vector_product, setup, teardown),
        cmocka_unit_test_setup_teardown(test_zero_sparse_matrix_product, setup, teardown),
        cmocka_unit_test_setup_teardown(test_sparse_vector_shape_error, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
