/**
 * @file test_sparse_cholesky.c
 * 针对 LMMC 中 sparse cholesky 相关接口的单元测试。
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"
#include "internal_test_hooks.h"

#include <stdlib.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

struct test_fixture {
    lmmc_sparse_mat_t A;
    lmmc_sparse_coo_t coo;
    lmmc_sparse_chol_t *chol;
    lmmc_sparse_chol_t *second;
    lmmc_vec_t b;
    lmmc_vec_t x;
    lmmc_vec_t x2;
};

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    *state = fixture;
    assert_non_null(fixture);
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_vec_destroy(&fixture->x2);
    lmmc_vec_destroy(&fixture->x);
    lmmc_vec_destroy(&fixture->b);
    lmmc_sparse_chol_destroy(fixture->second);
    lmmc_sparse_chol_destroy(fixture->chol);
    lmmc_sparse_destroy(&fixture->A);
    lmmc_sparse_coo_destroy(&fixture->coo);
    free(fixture);
    *state = NULL;
    return 0;
}

static lmmc_status_t build_sparse_spd(const lmmc_real_t *data, size_t n,
                                      lmmc_sparse_mat_t *out) {
    lmmc_sparse_builder_t *builder = NULL;
    lmmc_status_t st;

    st = lmmc_sparse_builder_create(n, n, n * n, &builder);
    if (st != LMMC_STATUS_OK)
        return st;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            lmmc_real_t val = data[i * n + j];
            if (val != 0.0) {
                st = lmmc_sparse_builder_add(builder, i, j, val);
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

static lmmc_real_t repeated_spd_entry(size_t variant, size_t row, size_t col) {
    if (row == col) {
        return 12.0 + (lmmc_real_t)row + (lmmc_real_t)variant;
    }
    return 0.05 * (lmmc_real_t)(1 + ((row + col + variant) % 5));
}

static void set_repeated_spd_values(lmmc_sparse_mat_t *matrix, size_t variant) {
    assert_int_equal(matrix->format, LMMC_SPARSE_CSC);
    for (size_t col = 0; col < matrix->cols; ++col) {
        for (size_t p = matrix->row_ptr[col]; p < matrix->row_ptr[col + 1]; ++p) {
            matrix->values[p] =
                repeated_spd_entry(variant, matrix->col_idx[p], col);
        }
    }
}

static void set_repeated_spd_rhs(size_t variant, const lmmc_vec_t *expected,
                                 lmmc_vec_t *rhs) {
    for (size_t row = 0; row < expected->size; ++row) {
        lmmc_real_t value = 0.0;
        for (size_t col = 0; col < expected->size; ++col) {
            value += repeated_spd_entry(variant, row, col) *
                     expected->data[col];
        }
        rhs->data[row] = value;
    }
}

static double repeated_spd_residual(size_t variant, const lmmc_vec_t *rhs,
                                    const lmmc_vec_t *solution) {
    double residual_sq = 0.0;
    for (size_t row = 0; row < solution->size; ++row) {
        double value = 0.0;
        for (size_t col = 0; col < solution->size; ++col) {
            value += repeated_spd_entry(variant, row, col) *
                     solution->data[col];
        }
        const double residual = value - rhs->data[row];
        residual_sq += residual * residual;
    }
    return sqrt(residual_sq);
}

static void test_basic_3x3(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_real_t A_data[] = {
        4.0, 12.0, -16.0,
        12.0, 37.0, -43.0,
        -16.0, -43.0, 98.0};

    lmmc_status_t st;

    st = build_sparse_spd(A_data, 3, &fixture->A);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_symbolic(&fixture->A, &fixture->chol);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_numeric(&fixture->A, fixture->chol);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_vec_create(3, &fixture->b);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_vec_create(3, &fixture->x);
    assert_true(st == LMMC_STATUS_OK);

    fixture->b.data[0] = -20.0;
    fixture->b.data[1] = -43.0;
    fixture->b.data[2] = 192.0;

    st = lmmc_sparse_chol_solve(fixture->chol, &fixture->b, &fixture->x);
    assert_true(st == LMMC_STATUS_OK);

    assert_true(lmmc_test_nearly_equal(fixture->x.data[0], 1.0, 1e-10));
    assert_true(lmmc_test_nearly_equal(fixture->x.data[1], 2.0, 1e-10));
    assert_true(lmmc_test_nearly_equal(fixture->x.data[2], 3.0, 1e-10));
}

static void test_identity_2x2(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_real_t A_data[] = {
        1.0, 0.0,
        0.0, 1.0};

    lmmc_status_t st;

    st = build_sparse_spd(A_data, 2, &fixture->A);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_symbolic(&fixture->A, &fixture->chol);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_numeric(&fixture->A, fixture->chol);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_vec_create(2, &fixture->b);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_vec_create(2, &fixture->x);
    assert_true(st == LMMC_STATUS_OK);

    fixture->b.data[0] = 3.0;
    fixture->b.data[1] = 7.0;

    st = lmmc_sparse_chol_solve(fixture->chol, &fixture->b, &fixture->x);
    assert_true(st == LMMC_STATUS_OK);

    assert_true(lmmc_test_nearly_equal(fixture->x.data[0], 3.0, 1e-10));
    assert_true(lmmc_test_nearly_equal(fixture->x.data[1], 7.0, 1e-10));
}

static void test_not_positive_definite(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_real_t A_data[] = {
        1.0, 2.0,
        2.0, 1.0};

    lmmc_status_t st;

    st = build_sparse_spd(A_data, 2, &fixture->A);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_symbolic(&fixture->A, &fixture->chol);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_numeric(&fixture->A, fixture->chol);
    assert_true(st == LMMC_STATUS_NOT_POSITIVE_DEFINITE);
}

static void test_non_square_matrix(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;

    st = lmmc_sparse_create_csc(2, 3, 0, &fixture->A);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_symbolic(&fixture->A, &fixture->chol);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_null_pointers(void **state) {
    (void)state;

    lmmc_sparse_chol_t *chol = NULL;
    lmmc_status_t st;

    st = lmmc_sparse_chol_symbolic(NULL, &chol);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_sparse_mat_t A = {0};
    A.rows = 2;
    A.cols = 2;
    st = lmmc_sparse_chol_symbolic(&A, NULL);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_vec_t b = {0}, x = {0};
    st = lmmc_sparse_chol_solve(NULL, &b, &x);
    assert_true(st == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_sparse_chol_destroy(NULL);
}

static void test_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_real_t A_data[] = {
        2.0, 1.0,
        1.0, 2.0};

    lmmc_status_t st;

    st = build_sparse_spd(A_data, 2, &fixture->A);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_symbolic(&fixture->A, &fixture->chol);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_numeric(&fixture->A, fixture->chol);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_vec_create(3, &fixture->b);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_vec_create(2, &fixture->x);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_solve(fixture->chol, &fixture->b, &fixture->x);
    assert_true(st == LMMC_STATUS_DIMENSION_MISMATCH);
}

static void test_4x4_spd(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_real_t A_data[] = {
        3.0, 1.0, 1.0, 1.0,
        1.0, 4.0, 1.0, 1.0,
        1.0, 1.0, 5.0, 1.0,
        1.0, 1.0, 1.0, 6.0};

    lmmc_status_t st;

    st = build_sparse_spd(A_data, 4, &fixture->A);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_symbolic(&fixture->A, &fixture->chol);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_numeric(&fixture->A, fixture->chol);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_vec_create(4, &fixture->b);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_vec_create(4, &fixture->x);
    assert_true(st == LMMC_STATUS_OK);

    fixture->b.data[0] = 12.0;
    fixture->b.data[1] = 16.0;
    fixture->b.data[2] = 22.0;
    fixture->b.data[3] = 30.0;

    st = lmmc_sparse_chol_solve(fixture->chol, &fixture->b, &fixture->x);
    assert_true(st == LMMC_STATUS_OK);

    assert_true(lmmc_test_nearly_equal(fixture->x.data[0], 1.0, 1e-10));
    assert_true(lmmc_test_nearly_equal(fixture->x.data[1], 2.0, 1e-10));
    assert_true(lmmc_test_nearly_equal(fixture->x.data[2], 3.0, 1e-10));
    assert_true(lmmc_test_nearly_equal(fixture->x.data[3], 4.0, 1e-10));
}

static void create_disconnected_duplicate_matrix(struct test_fixture *fixture, lmmc_sparse_mat_t *matrix) {

    lmmc_status_t st;
    const size_t col_ptr[5] = {0, 1, 4, 7, 8};
    const size_t row_idx[8] = {0, 1, 2, 2, 1, 1, 2, 3};
    const lmmc_real_t values[8] = {4.0, 3.0, 0.5, 0.5, 0.5, 0.5, 3.0, 5.0};

    st = lmmc_sparse_coo_create(4, 4, 8, &fixture->coo);
    assert_true(st == LMMC_STATUS_OK);
    for (size_t col = 0; col < 4 && st == LMMC_STATUS_OK; ++col) {
        for (size_t p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
            st = lmmc_sparse_coo_add_entry(&fixture->coo, row_idx[p], col, values[p]);
            if (st != LMMC_STATUS_OK) {
                break;
            }
        }
    }
    if (st == LMMC_STATUS_OK) {
        st = lmmc_sparse_coo_to_csc(&fixture->coo, matrix);
    }
    lmmc_sparse_coo_destroy(&fixture->coo);
    assert_true(st == LMMC_STATUS_OK);
    return;
}

static void test_disconnected_duplicate_pattern(void **state) {
    struct test_fixture *fixture = *state;

    lmmc_status_t st;
    const lmmc_real_t rhs[4] = {4.0, 9.0, 11.0, 20.0};

    create_disconnected_duplicate_matrix(fixture, &fixture->A);

    st = lmmc_sparse_chol_symbolic(&fixture->A, &fixture->chol);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_chol_numeric(&fixture->A, fixture->chol);
    assert_true(st == LMMC_STATUS_OK);

    st = lmmc_sparse_chol_symbolic(&fixture->A, &fixture->second);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_chol_numeric(&fixture->A, fixture->second);
    assert_true(st == LMMC_STATUS_OK);

    assert_true(((lmmc_vec_create(4, &fixture->b) == LMMC_STATUS_OK) && (lmmc_vec_create(4, &fixture->x) == LMMC_STATUS_OK)) && (lmmc_vec_create(4, &fixture->x2) == LMMC_STATUS_OK));
    memcpy(fixture->b.data, rhs, sizeof(rhs));
    st = lmmc_sparse_chol_solve(fixture->chol, &fixture->b, &fixture->x);
    assert_true(st == LMMC_STATUS_OK);
    st = lmmc_sparse_chol_solve(fixture->second, &fixture->b, &fixture->x2);
    assert_true(st == LMMC_STATUS_OK);
    for (size_t i = 0; i < 4; ++i) {
        assert_true(lmmc_test_nearly_equal(fixture->x.data[i], (lmmc_real_t)(i + 1), 1e-10));
        assert_true(lmmc_test_nearly_equal(fixture->x.data[i], fixture->x2.data[i], 1e-12));
    }
}

static void test_repeated_numeric_reuses_factor_storage(void **state) {
    struct test_fixture *fixture = *state;
    const size_t n = 8;
    lmmc_real_t matrix_data[64];

    for (size_t row = 0; row < n; ++row) {
        for (size_t col = 0; col < n; ++col) {
            matrix_data[row * n + col] = repeated_spd_entry(0, row, col);
        }
    }
    assert_int_equal(
        build_sparse_spd(matrix_data, n, &fixture->A),
        LMMC_STATUS_OK);
    assert_int_equal(
        lmmc_sparse_chol_symbolic(&fixture->A, &fixture->chol),
        LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &fixture->b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &fixture->x), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &fixture->x2), LMMC_STATUS_OK);
    for (size_t i = 0; i < n; ++i) {
        fixture->x2.data[i] = (lmmc_real_t)(i + 1);
    }

    for (size_t variant = 0; variant < 3; ++variant) {
        set_repeated_spd_values(&fixture->A, variant);
        set_repeated_spd_rhs(variant, &fixture->x2, &fixture->b);

        /* A warmed CSC factorization needs only its per-call workspace. */
        if (variant == 1) {
            lmmc_memory_fail_after_for_test(1);
        }
        const lmmc_status_t numeric_status =
            lmmc_sparse_chol_numeric(&fixture->A, fixture->chol);
        lmmc_memory_fail_reset_for_test();
        assert_int_equal(numeric_status, LMMC_STATUS_OK);

        assert_int_equal(
            lmmc_sparse_chol_solve(fixture->chol, &fixture->b, &fixture->x),
            LMMC_STATUS_OK);
        assert_true(
            repeated_spd_residual(variant, &fixture->b, &fixture->x) <=
            1e-10);
        for (size_t i = 0; i < n; ++i) {
            assert_true(lmmc_test_nearly_equal(
                fixture->x.data[i], fixture->x2.data[i], 1e-10));
        }
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_basic_3x3, setup, teardown),
        cmocka_unit_test_setup_teardown(test_identity_2x2, setup, teardown),
        cmocka_unit_test_setup_teardown(test_not_positive_definite, setup, teardown),
        cmocka_unit_test_setup_teardown(test_non_square_matrix, setup, teardown),
        cmocka_unit_test(test_null_pointers),
        cmocka_unit_test_setup_teardown(test_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_4x4_spd, setup, teardown),
        cmocka_unit_test_setup_teardown(test_disconnected_duplicate_pattern, setup, teardown),
        cmocka_unit_test_setup_teardown(test_repeated_numeric_reuses_factor_storage, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
