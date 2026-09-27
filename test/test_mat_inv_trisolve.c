/**
 * @file test_mat_inv_trisolve.c
 * 矩阵求逆与三角求解单元测试。
 */
#include <stdlib.h>
#include <math.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#define NUM_INV_TRIALS 20
#define NUM_TRI_TRIALS 20

typedef struct {
    lmmc_rng_t *inverse_rng;
    lmmc_rng_t *triangular_rng;
    int triangular_trial;
    lmmc_mat_t A, A_inv, product, identity, diff, T;
    lmmc_vec_t b, x, Tx, residual;
} solve_fixture;

static int teardown_trial(void **state) {
    solve_fixture *fixture = *state;
    lmmc_vec_destroy(&fixture->residual);
    lmmc_vec_destroy(&fixture->Tx);
    lmmc_vec_destroy(&fixture->x);
    lmmc_vec_destroy(&fixture->b);
    lmmc_mat_destroy(&fixture->T);
    lmmc_mat_destroy(&fixture->diff);
    lmmc_mat_destroy(&fixture->identity);
    lmmc_mat_destroy(&fixture->product);
    lmmc_mat_destroy(&fixture->A_inv);
    lmmc_mat_destroy(&fixture->A);
    return 0;
}

static int teardown_group(void **state) {
    solve_fixture *fixture = *state;
    if (fixture == NULL)
        return 0;
    teardown_trial(state);
    lmmc_rng_destroy(fixture->triangular_rng);
    lmmc_rng_destroy(fixture->inverse_rng);
    free(fixture);
    return 0;
}

static int setup_group(void **state) {
    solve_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    if (lmmc_rng_create(&fixture->inverse_rng) != LMMC_STATUS_OK ||
        lmmc_rng_create(&fixture->triangular_rng) != LMMC_STATUS_OK) {
        teardown_group(state);
        *state = NULL;
        return -1;
    }
    assert_int_equal(lmmc_rng_seed(fixture->inverse_rng, 12345), LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(fixture->triangular_rng, 67890), LMMC_STATUS_OK);
    return 0;
}

/** Fill matrix with random values in [-1, 1]. */
static void fill_random_matrix(lmmc_rng_t *rng, lmmc_mat_t *mat) {
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            lmmc_real_t val;
            assert_int_equal(lmmc_rng_uniform(rng, -1.0, 1.0, &val), LMMC_STATUS_OK);
            mat->data[i * mat->stride + j] = val;
        }
    }
}

/** The Frobenius condition estimate bounds the 2-norm condition number. */
static double estimate_condition_number(const lmmc_mat_t *A, const lmmc_mat_t *A_inv) {
    lmmc_real_t norm_A, norm_Ainv;
    assert_int_equal(lmmc_mat_norm_fro(A, &norm_A), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_norm_fro(A_inv, &norm_Ainv), LMMC_STATUS_OK);
    return norm_A * norm_Ainv;
}

/** Verify ||A*A_inv - I||_F <= 1e-8 * kappa(A). */
static void check_inverse_residual_bound(solve_fixture *fixture) {
    size_t n = fixture->A.rows;
    assert_int_equal(lmmc_mat_create(n, n, &fixture->diff), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_sub(&fixture->product, &fixture->identity,
                                  &fixture->diff),
                     LMMC_STATUS_OK);
    lmmc_real_t residual_norm;
    assert_int_equal(lmmc_mat_norm_fro(&fixture->diff, &residual_norm), LMMC_STATUS_OK);
    double kappa = estimate_condition_number(&fixture->A, &fixture->A_inv);
    double tol = 1e-8 * kappa;
    assert_true(isfinite(residual_norm));
    assert_true(residual_norm <= tol);
}

static void test_inverse_trial(void **state) {
    solve_fixture *fixture = *state;
    lmmc_rng_t *rng = fixture->inverse_rng;
    lmmc_real_t u;
    assert_int_equal(lmmc_rng_uniform(rng, 3.0, 11.0, &u), LMMC_STATUS_OK);
    size_t n = (size_t)u;
    if (n < 3)
        n = 3;
    if (n > 10)
        n = 10;

    assert_int_equal(lmmc_mat_create(n, n, &fixture->A), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_create(n, n, &fixture->A_inv), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_create(n, n, &fixture->product), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_identity(n, &fixture->identity), LMMC_STATUS_OK);
    fill_random_matrix(rng, &fixture->A);
    /* Diagonal dominance ensures non-singularity. */
    for (size_t i = 0; i < n; i++)
        fixture->A.data[i * fixture->A.stride + i] += (lmmc_real_t)(n + 1);
    assert_int_equal(lmmc_mat_inv(&fixture->A, &fixture->A_inv), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_mul(&fixture->A, &fixture->A_inv,
                                  &fixture->product),
                     LMMC_STATUS_OK);
    check_inverse_residual_bound(fixture);
}

static void fill_random_triangular_system(lmmc_rng_t *rng, size_t n, int upper,
                                          lmmc_mat_t *T, lmmc_vec_t *b) {
    assert_int_equal(lmmc_mat_fill(T, 0.0), LMMC_STATUS_OK);
    for (size_t i = 0; i < n; i++) {
        size_t begin = upper ? i : 0;
        size_t end = upper ? n : i + 1;
        for (size_t j = begin; j < end; j++) {
            lmmc_real_t val;
            assert_int_equal(lmmc_rng_uniform(rng, -2.0, 2.0, &val), LMMC_STATUS_OK);
            T->data[i * T->stride + j] = val;
        }
        lmmc_real_t diag_sign;
        assert_int_equal(lmmc_rng_uniform(rng, 0.0, 1.0, &diag_sign), LMMC_STATUS_OK);
        T->data[i * T->stride + i] += diag_sign > 0.5 ? 2.0 : -2.0;
    }
    for (size_t i = 0; i < n; i++) {
        lmmc_real_t val;
        assert_int_equal(lmmc_rng_uniform(rng, -5.0, 5.0, &val), LMMC_STATUS_OK);
        b->data[i] = val;
    }
}

/** Verify ||Tx-b||_2 <= 1e-10 * (||T||_F * ||x||_2 + ||b||_2). */
static void check_triangular_residual_bound(solve_fixture *fixture) {
    assert_int_equal(lmmc_mat_vec_mul(&fixture->T, &fixture->x,
                                      &fixture->Tx),
                     LMMC_STATUS_OK);
    for (size_t i = 0; i < fixture->b.size; i++)
        fixture->residual.data[i] = fixture->Tx.data[i] - fixture->b.data[i];
    lmmc_real_t residual_norm, T_norm, x_norm, b_norm;
    assert_int_equal(lmmc_vec_norm2(&fixture->residual, &residual_norm), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_norm_fro(&fixture->T, &T_norm), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_norm2(&fixture->x, &x_norm), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_norm2(&fixture->b, &b_norm), LMMC_STATUS_OK);
    double tol = 1e-10 * (T_norm * x_norm + b_norm);
    assert_true(isfinite(residual_norm));
    assert_true(residual_norm <= tol);
}

static void test_triangular_trial(void **state) {
    solve_fixture *fixture = *state;
    lmmc_rng_t *rng = fixture->triangular_rng;
    int upper = fixture->triangular_trial++ % 2 == 0;
    lmmc_real_t u;
    assert_int_equal(lmmc_rng_uniform(rng, 3.0, 11.0, &u), LMMC_STATUS_OK);
    size_t n = (size_t)u;
    if (n < 3)
        n = 3;
    if (n > 10)
        n = 10;
    assert_int_equal(lmmc_mat_create(n, n, &fixture->T), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &fixture->b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &fixture->x), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &fixture->Tx), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, &fixture->residual), LMMC_STATUS_OK);
    fill_random_triangular_system(rng, n, upper, &fixture->T, &fixture->b);
    assert_int_equal(lmmc_solve_triangular(&fixture->T, upper, 0,
                                           &fixture->b, &fixture->x),
                     LMMC_STATUS_OK);
    check_triangular_residual_bound(fixture);
}

static void test_singular_matrix(void **state) {
    solve_fixture *fixture = *state;
    lmmc_mat_t *A = &fixture->A;
    lmmc_mat_t *A_inv = &fixture->A_inv;
    assert_int_equal(lmmc_mat_create(3, 3, A), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_create(3, 3, A_inv), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_fill(A_inv, 99.0), LMMC_STATUS_OK);
    /* Row 2 = 2 * row 0. */
    const lmmc_real_t values[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 2.0, 4.0, 6.0};
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 3; j++)
            A->data[i * A->stride + j] = values[i * 3 + j];
    assert_int_equal(lmmc_mat_inv(A, A_inv), LMMC_STATUS_SINGULAR_MATRIX);
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 3; j++)
            assert_true(lmmc_test_nearly_equal(A_inv->data[i * A_inv->stride + j],
                                               99.0, 1e-15));
}

static void test_unit_upper_triangular(void **state) {
    solve_fixture *fixture = *state;
    lmmc_mat_t *T = &fixture->T;
    lmmc_vec_t *b = &fixture->b;
    lmmc_vec_t *x = &fixture->x;
    size_t n = 4;
    assert_int_equal(lmmc_mat_create(n, n, T), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, x), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_fill(T, 0.0), LMMC_STATUS_OK);
    T->data[0 * T->stride + 0] = 999.0;
    T->data[0 * T->stride + 1] = 2.0;
    T->data[0 * T->stride + 2] = 3.0;
    T->data[0 * T->stride + 3] = 1.0;
    T->data[1 * T->stride + 1] = 888.0;
    T->data[1 * T->stride + 2] = -1.0;
    T->data[1 * T->stride + 3] = 2.0;
    T->data[2 * T->stride + 2] = 777.0;
    T->data[2 * T->stride + 3] = 4.0;
    T->data[3 * T->stride + 3] = 666.0;
    b->data[0] = 10.0;
    b->data[1] = 5.0;
    b->data[2] = 3.0;
    b->data[3] = 1.0;
    assert_int_equal(lmmc_solve_triangular(T, 1, 1, b, x), LMMC_STATUS_OK);
    /* T_unit = [[1,2,3,1],[0,1,-1,2],[0,0,1,4],[0,0,0,1]]. */
    const double expected_x[] = {8.0, 2.0, -1.0, 1.0};
    for (size_t i = 0; i < n; i++)
        assert_true(lmmc_test_nearly_equal(x->data[i], expected_x[i], 1e-12));
}

static void test_unit_lower_triangular(void **state) {
    solve_fixture *fixture = *state;
    lmmc_mat_t *T = &fixture->T;
    lmmc_vec_t *b = &fixture->b;
    lmmc_vec_t *x = &fixture->x;
    size_t n = 3;
    assert_int_equal(lmmc_mat_create(n, n, T), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(n, x), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_fill(T, 0.0), LMMC_STATUS_OK);
    T->data[0 * T->stride + 0] = 100.0;
    T->data[1 * T->stride + 0] = 3.0;
    T->data[1 * T->stride + 1] = 200.0;
    T->data[2 * T->stride + 0] = -1.0;
    T->data[2 * T->stride + 1] = 2.0;
    T->data[2 * T->stride + 2] = 300.0;
    b->data[0] = 4.0;
    b->data[1] = 7.0;
    b->data[2] = 1.0;
    assert_int_equal(lmmc_solve_triangular(T, 0, 1, b, x), LMMC_STATUS_OK);
    /* T_unit = [[1,0,0],[3,1,0],[-1,2,1]]. */
    const double expected_x[] = {4.0, -5.0, 15.0};
    for (size_t i = 0; i < n; i++)
        assert_true(lmmc_test_nearly_equal(x->data[i], expected_x[i], 1e-12));
}

int main(void) {
    struct CMUnitTest tests[NUM_INV_TRIALS + NUM_TRI_TRIALS + 3];
    for (size_t i = 0; i < NUM_INV_TRIALS; i++)
        tests[i] = (struct CMUnitTest)cmocka_unit_test_teardown(test_inverse_trial, teardown_trial);
    for (size_t i = 0; i < NUM_TRI_TRIALS; i++)
        tests[NUM_INV_TRIALS + i] = (struct CMUnitTest)cmocka_unit_test_teardown(test_triangular_trial, teardown_trial);
    tests[NUM_INV_TRIALS + NUM_TRI_TRIALS] =
        (struct CMUnitTest)cmocka_unit_test_teardown(test_singular_matrix, teardown_trial);
    tests[NUM_INV_TRIALS + NUM_TRI_TRIALS + 1] =
        (struct CMUnitTest)cmocka_unit_test_teardown(test_unit_upper_triangular, teardown_trial);
    tests[NUM_INV_TRIALS + NUM_TRI_TRIALS + 2] =
        (struct CMUnitTest)cmocka_unit_test_teardown(test_unit_lower_triangular, teardown_trial);
    return cmocka_run_group_tests(tests, setup_group, teardown_group);
}
