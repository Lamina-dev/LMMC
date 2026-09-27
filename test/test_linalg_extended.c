/**
 * @file test_linalg_extended.c
 * 针对 LMMC 中 linalg extended 相关接口的单元测试。
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#define TEST_EPS_NORMAL 1e-10
#define TEST_EPS_TIGHT 1e-12

typedef struct {
    lmmc_mat_t a;
    lmmc_vec_t b;
    lmmc_vec_t x;
} linalg_fixture_t;

static int setup_linalg(void **state) {
    linalg_fixture_t *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown_linalg(void **state) {
    linalg_fixture_t *fixture = *state;
    lmmc_vec_destroy(&fixture->x);
    lmmc_vec_destroy(&fixture->b);
    lmmc_mat_destroy(&fixture->a);
    free(fixture);
    return 0;
}

static double compute_residual(const double *A, size_t n, const double *x, const double *b) {
    double max_res = 0.0;
    for (size_t i = 0; i < n; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < n; j++) {
            sum += A[i * n + j] * x[j];
        }
        double res = fabs(sum - b[i]);
        if (res > max_res) {
            max_res = res;
        }
    }
    return max_res;
}

static void test_lu_two_by_two(void **state) {
    linalg_fixture_t *fixture = *state;

    lmmc_mat_t *a = &fixture->a;
    lmmc_vec_t *b = &fixture->b, *x = &fixture->x;
    size_t piv[2];
    double A_orig[] = {2.0, 1.0, 5.0, 7.0};
    double b_vals[] = {11.0, 13.0};

    assert_int_equal(lmmc_mat_create(2, 2, a), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(2, b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(2, x), LMMC_STATUS_OK);

    for (size_t i = 0; i < 4; i++)
        a->data[i] = A_orig[i];
    for (size_t i = 0; i < 2; i++)
        b->data[i] = b_vals[i];

    lmmc_status_t st = lmmc_lu_decompose_inplace(a, piv, NULL);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_lu_solve(a, piv, b, x);
    assert_int_equal(st, LMMC_STATUS_OK);

    double residual = compute_residual(A_orig, 2, x->data, b_vals);
    assert_false(residual > TEST_EPS_NORMAL);
}

static void test_lu_three_by_three(void **state) {
    linalg_fixture_t *fixture = *state;

    lmmc_mat_t *a = &fixture->a;
    lmmc_vec_t *b = &fixture->b, *x = &fixture->x;
    size_t piv[3];
    double A_orig[] = {3.0, 2.0, -1.0, 2.0, -2.0, 4.0, -1.0, 0.5, -1.0};
    double b_vals[] = {1.0, -2.0, 0.0};

    assert_int_equal(lmmc_mat_create(3, 3, a), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(3, b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(3, x), LMMC_STATUS_OK);
    for (size_t i = 0; i < 9; i++)
        a->data[i] = A_orig[i];
    for (size_t i = 0; i < 3; i++)
        b->data[i] = b_vals[i];

    lmmc_status_t st = lmmc_lu_decompose_inplace(a, piv, NULL);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_lu_solve(a, piv, b, x);
    assert_int_equal(st, LMMC_STATUS_OK);

    double residual = compute_residual(A_orig, 3, x->data, b_vals);
    assert_false(residual > TEST_EPS_NORMAL);
}

static void test_lu_five_by_five(void **state) {
    linalg_fixture_t *fixture = *state;

    lmmc_mat_t *a = &fixture->a;
    lmmc_vec_t *b = &fixture->b, *x = &fixture->x;
    size_t piv[5];
    double A_orig[] = {
        5.0, 7.0, 6.0, 5.0, 1.0,
        7.0, 10.0, 8.0, 7.0, 2.0,
        6.0, 8.0, 10.0, 9.0, 3.0,
        5.0, 7.0, 9.0, 10.0, 4.0,
        1.0, 2.0, 3.0, 4.0, 5.0};

    double x_true[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double b_vals[5];
    for (size_t i = 0; i < 5; i++) {
        b_vals[i] = 0.0;
        for (size_t j = 0; j < 5; j++)
            b_vals[i] += A_orig[i * 5 + j] * x_true[j];
    }

    assert_int_equal(lmmc_mat_create(5, 5, a), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(5, b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(5, x), LMMC_STATUS_OK);
    for (size_t i = 0; i < 25; i++)
        a->data[i] = A_orig[i];
    for (size_t i = 0; i < 5; i++)
        b->data[i] = b_vals[i];

    lmmc_status_t st = lmmc_lu_decompose_inplace(a, piv, NULL);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_lu_solve(a, piv, b, x);
    assert_int_equal(st, LMMC_STATUS_OK);

    double residual = compute_residual(A_orig, 5, x->data, b_vals);
    assert_false(residual > TEST_EPS_NORMAL);
}

static void test_lu_diagonally_dominant(void **state) {
    linalg_fixture_t *fixture = *state;

    lmmc_mat_t *a = &fixture->a;
    lmmc_vec_t *b = &fixture->b, *x = &fixture->x;
    size_t piv[10];

    double A_orig[100];
    double x_true[10];
    double b_vals[10];
    for (size_t i = 0; i < 10; i++) {
        x_true[i] = (double)(i + 1);
        for (size_t j = 0; j < 10; j++) {
            if (i == j) {
                A_orig[i * 10 + j] = 20.0;
            } else
                A_orig[i * 10 + j] = 1.0 / (1.0 + (double)((i > j) ? (i - j) : (j - i)));
        }
    }
    for (size_t i = 0; i < 10; i++) {
        b_vals[i] = 0.0;
        for (size_t j = 0; j < 10; j++)
            b_vals[i] += A_orig[i * 10 + j] * x_true[j];
    }

    assert_int_equal(lmmc_mat_create(10, 10, a), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(10, b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(10, x), LMMC_STATUS_OK);
    for (size_t i = 0; i < 100; i++)
        a->data[i] = A_orig[i];
    for (size_t i = 0; i < 10; i++)
        b->data[i] = b_vals[i];

    lmmc_status_t st = lmmc_lu_decompose_inplace(a, piv, NULL);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_lu_solve(a, piv, b, x);
    assert_int_equal(st, LMMC_STATUS_OK);

    double residual = compute_residual(A_orig, 10, x->data, b_vals);
    assert_false(residual > TEST_EPS_NORMAL);
}

static void test_cholesky_solution(void **state) {
    linalg_fixture_t *fixture = *state;

    lmmc_mat_t *a = &fixture->a;
    lmmc_vec_t *b = &fixture->b, *x = &fixture->x;

    double A_orig[] = {4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0};
    double x_true[] = {1.0, 2.0, 3.0};
    double b_vals[3];
    for (size_t i = 0; i < 3; i++) {
        b_vals[i] = 0.0;
        for (size_t j = 0; j < 3; j++)
            b_vals[i] += A_orig[i * 3 + j] * x_true[j];
    }

    assert_int_equal(lmmc_mat_create(3, 3, a), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(3, b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(3, x), LMMC_STATUS_OK);
    for (size_t i = 0; i < 9; i++)
        a->data[i] = A_orig[i];
    for (size_t i = 0; i < 3; i++)
        b->data[i] = b_vals[i];

    lmmc_status_t st = lmmc_cholesky_decompose_inplace(a);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_cholesky_solve(a, b, x);
    assert_int_equal(st, LMMC_STATUS_OK);

    double residual = compute_residual(A_orig, 3, x->data, b_vals);
    assert_false(residual > TEST_EPS_NORMAL);
}

static void test_qr_overdetermined_solution(void **state) {
    linalg_fixture_t *fixture = *state;

    lmmc_mat_t *a = &fixture->a;
    lmmc_vec_t *b = &fixture->b, *x = &fixture->x;

    double tau[2];

    assert_int_equal(lmmc_mat_create(5, 2, a), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(5, b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(2, x), LMMC_STATUS_OK);

    for (size_t i = 0; i < 5; i++) {
        a->data[i * 2 + 0] = 1.0;
        a->data[i * 2 + 1] = (double)i;
    }
    b->data[0] = 1.0;
    b->data[1] = 3.0;
    b->data[2] = 5.0;
    b->data[3] = 7.0;
    b->data[4] = 9.0;

    lmmc_status_t st = lmmc_qr_decompose_inplace(a, tau, 2);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_qr_solve(a, tau, b, x);
    assert_int_equal(st, LMMC_STATUS_OK);

    assert_false(!lmmc_test_nearly_equal(x->data[0], 1.0, TEST_EPS_NORMAL) ||
                 !lmmc_test_nearly_equal(x->data[1], 2.0, TEST_EPS_NORMAL));
}

static void test_lu_identity(void **state) {
    linalg_fixture_t *fixture = *state;

    lmmc_mat_t *a = &fixture->a;
    size_t piv[3];

    assert_int_equal(lmmc_mat_create(3, 3, a), LMMC_STATUS_OK);

    for (size_t i = 0; i < 9; i++)
        a->data[i] = 0.0;
    a->data[0] = 1.0;
    a->data[4] = 1.0;
    a->data[8] = 1.0;

    lmmc_status_t st = lmmc_lu_decompose_inplace(a, piv, NULL);
    assert_int_equal(st, LMMC_STATUS_OK);

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            double expected = (i == j) ? 1.0 : 0.0;
            assert_true(lmmc_test_nearly_equal(a->data[i * 3 + j], expected, TEST_EPS_TIGHT));
        }
    }
}

static void test_cholesky_diagonal(void **state) {
    linalg_fixture_t *fixture = *state;

    lmmc_mat_t *a = &fixture->a;
    double diag_vals[] = {4.0, 9.0, 16.0, 25.0};

    assert_int_equal(lmmc_mat_create(4, 4, a), LMMC_STATUS_OK);
    for (size_t i = 0; i < 16; i++)
        a->data[i] = 0.0;
    for (size_t i = 0; i < 4; i++)
        a->data[i * 4 + i] = diag_vals[i];

    lmmc_status_t st = lmmc_cholesky_decompose_inplace(a);
    assert_int_equal(st, LMMC_STATUS_OK);

    for (size_t i = 0; i < 4; i++) {
        double expected_diag = sqrt(diag_vals[i]);
        assert_true(lmmc_test_nearly_equal(a->data[i * 4 + i], expected_diag, TEST_EPS_TIGHT));

        for (size_t j = 0; j < 4; j++) {
            if (j != i) {
                assert_true(lmmc_test_nearly_equal(a->data[i * 4 + j], 0.0, TEST_EPS_TIGHT));
            }
        }
    }
}

static void test_lu_singular(void **state) {
    linalg_fixture_t *fixture = *state;

    lmmc_mat_t *a = &fixture->a;
    size_t piv[3];

    assert_int_equal(lmmc_mat_create(3, 3, a), LMMC_STATUS_OK);

    a->data[0] = 1.0;
    a->data[1] = 2.0;
    a->data[2] = 3.0;
    a->data[3] = 2.0;
    a->data[4] = 4.0;
    a->data[5] = 6.0;
    a->data[6] = 3.0;
    a->data[7] = 6.0;
    a->data[8] = 9.0;

    lmmc_status_t st = lmmc_lu_decompose_inplace(a, piv, NULL);
    assert_int_equal(st, LMMC_STATUS_SINGULAR_MATRIX);
}

static void test_cholesky_indefinite(void **state) {
    linalg_fixture_t *fixture = *state;

    lmmc_mat_t *a = &fixture->a;

    assert_int_equal(lmmc_mat_create(3, 3, a), LMMC_STATUS_OK);

    a->data[0] = 1.0;
    a->data[1] = 2.0;
    a->data[2] = 3.0;
    a->data[3] = 2.0;
    a->data[4] = 1.0;
    a->data[5] = 2.0;
    a->data[6] = 3.0;
    a->data[7] = 2.0;
    a->data[8] = 1.0;

    lmmc_status_t st = lmmc_cholesky_decompose_inplace(a);
    assert_int_equal(st, LMMC_STATUS_NOT_POSITIVE_DEFINITE);
}

static void multiply_lu_factors(lmmc_mat_t *a, double *LU) {
    double L[16], U[16];
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            if (i == j) {
                L[i * 4 + j] = 1.0;
                U[i * 4 + j] = a->data[i * 4 + j];
            } else if (i > j) {
                L[i * 4 + j] = a->data[i * 4 + j];
                U[i * 4 + j] = 0.0;
            } else {
                L[i * 4 + j] = 0.0;
                U[i * 4 + j] = a->data[i * 4 + j];
            }
        }
    }

    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            LU[i * 4 + j] = 0.0;
            for (size_t k = 0; k < 4; k++)
                LU[i * 4 + j] += L[i * 4 + k] * U[k * 4 + j];
        }
    }
}

static void test_lu_reconstruction(void **state) {
    linalg_fixture_t *fixture = *state;

    lmmc_mat_t *a = &fixture->a;
    size_t piv[4];
    size_t swap_count = 0;
    double A_orig[] = {
        2.0, 1.0, 1.0, 0.0,
        4.0, 3.0, 3.0, 1.0,
        8.0, 7.0, 9.0, 5.0,
        6.0, 7.0, 9.0, 8.0};

    assert_int_equal(lmmc_mat_create(4, 4, a), LMMC_STATUS_OK);
    for (size_t i = 0; i < 16; i++)
        a->data[i] = A_orig[i];

    lmmc_status_t st = lmmc_lu_decompose_inplace(a, piv, &swap_count);
    assert_int_equal(st, LMMC_STATUS_OK);

    double LU[16];
    multiply_lu_factors(a, LU);

    double PA[16];
    memcpy(PA, A_orig, sizeof(PA));

    for (size_t k = 0; k < 4; k++) {
        if (piv[k] != k) {
            for (size_t j = 0; j < 4; j++) {
                double tmp = PA[k * 4 + j];
                PA[k * 4 + j] = PA[piv[k] * 4 + j];
                PA[piv[k] * 4 + j] = tmp;
            }
        }
    }

    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 4; j++) {
            assert_true(lmmc_test_nearly_equal(PA[i * 4 + j], LU[i * 4 + j], TEST_EPS_NORMAL));
        }
    }
}

static void test_hilbert_solution(void **state) {
    linalg_fixture_t *fixture = *state;

    lmmc_mat_t *a = &fixture->a;
    lmmc_vec_t *b = &fixture->b, *x = &fixture->x;
    size_t piv[5];
    size_t n = 5;

    assert_int_equal(lmmc_mat_create(n, n, a), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, x), LMMC_STATUS_OK);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            a->data[i * n + j] = 1.0 / (double)(i + j + 1);
        }
        b->data[i] = 1.0;
    }

    lmmc_status_t st = lmmc_lu_decompose_inplace(a, piv, NULL);

    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_lu_solve(a, piv, b, x);
    assert_int_equal(st, LMMC_STATUS_OK);

    for (size_t i = 0; i < n; i++) {
        assert_true(isfinite(x->data[i]));
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_lu_two_by_two, setup_linalg, teardown_linalg),
        cmocka_unit_test_setup_teardown(test_lu_three_by_three, setup_linalg, teardown_linalg),
        cmocka_unit_test_setup_teardown(test_lu_five_by_five, setup_linalg, teardown_linalg),
        cmocka_unit_test_setup_teardown(test_lu_diagonally_dominant, setup_linalg, teardown_linalg),
        cmocka_unit_test_setup_teardown(test_cholesky_solution, setup_linalg, teardown_linalg),
        cmocka_unit_test_setup_teardown(test_qr_overdetermined_solution, setup_linalg, teardown_linalg),
        cmocka_unit_test_setup_teardown(test_lu_identity, setup_linalg, teardown_linalg),
        cmocka_unit_test_setup_teardown(test_cholesky_diagonal, setup_linalg, teardown_linalg),
        cmocka_unit_test_setup_teardown(test_lu_singular, setup_linalg, teardown_linalg),
        cmocka_unit_test_setup_teardown(test_cholesky_indefinite, setup_linalg, teardown_linalg),
        cmocka_unit_test_setup_teardown(test_lu_reconstruction, setup_linalg, teardown_linalg),
        cmocka_unit_test_setup_teardown(test_hilbert_solution, setup_linalg, teardown_linalg),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
