/**
 * @file test_gemm_accuracy_prop.c
 * GEMM 精度属性测试。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

/**
 * @brief Naive triple-loop GEMM reference implementation.
 *
 * Computes C_ref = alpha * op(A) * op(B) + beta * C0
 * where op(X) = X if trans == 0, X^T if trans != 0.
 *
 * @param alpha   Scalar multiplier for A*B product.
 * @param A       Input matrix A.
 * @param transA  Non-zero means transpose A.
 * @param B       Input matrix B.
 * @param transB  Non-zero means transpose B.
 * @param beta    Scalar multiplier for C0.
 * @param C0      Original C matrix (before GEMM).
 * @param C_ref   Output reference result (pre-created, same dims as C).
 */
static void naive_gemm(lmmc_real_t alpha, const lmmc_mat_t *A, int transA,
                       const lmmc_mat_t *B, int transB, lmmc_real_t beta,
                       const lmmc_mat_t *C0, lmmc_mat_t *C_ref) {
    size_t M = transA ? A->cols : A->rows;
    size_t K = transA ? A->rows : A->cols;
    size_t N = transB ? B->rows : B->cols;

    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < K; k++) {
                double a_ik;
                if (transA) {
                    a_ik = A->data[k * A->stride + i];
                } else {
                    a_ik = A->data[i * A->stride + k];
                }
                double b_kj;
                if (transB) {
                    b_kj = B->data[j * B->stride + k];
                } else {
                    b_kj = B->data[k * B->stride + j];
                }
                sum += a_ik * b_kj;
            }
            C_ref->data[i * C_ref->stride + j] =
                alpha * sum + beta * C0->data[i * C0->stride + j];
        }
    }
}

/**
 * @brief Generate a random scalar value from a set of interesting values.
 *
 * Returns 0, 1, -1, or a random value in [-5, 5] to cover edge cases.
 */
static lmmc_real_t random_scalar(lmmc_rng_t *rng) {
    lmmc_real_t u;
    assert_int_equal(lmmc_rng_uniform(rng, 0.0, 1.0, &u),
                     LMMC_STATUS_OK);
    if (u < 0.15) {
        return 0.0;
    }
    if (u < 0.30) {
        return 1.0;
    }
    if (u < 0.40) {
        return -1.0;
    }
    lmmc_real_t val;
    assert_int_equal(lmmc_rng_uniform(rng, -5.0, 5.0, &val),
                     LMMC_STATUS_OK);
    return val;
}

/**
 * @brief Fill a matrix with random values in [-1, 1].
 */
static void fill_random_matrix(lmmc_rng_t *rng, lmmc_mat_t *mat) {
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            lmmc_real_t val;
            assert_int_equal(lmmc_rng_uniform(rng, -1.0, 1.0, &val),
                             LMMC_STATUS_OK);
            mat->data[i * mat->stride + j] = val;
        }
    }
}

typedef struct {
    lmmc_rng_t *rng;
    lmmc_mat_t A, B, C, C0, C_ref;
    size_t M, K, N;
    int trial, transA, transB;
    lmmc_real_t alpha, beta;
} gemm_trial_t;

static void create_gemm_matrices(gemm_trial_t *f) {
    size_t A_rows = f->transA ? f->K : f->M;
    size_t A_cols = f->transA ? f->M : f->K;
    size_t B_rows = f->transB ? f->N : f->K;
    size_t B_cols = f->transB ? f->K : f->N;
    assert_int_equal(lmmc_mat_create(A_rows, A_cols, &f->A), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_create(B_rows, B_cols, &f->B), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_create(f->M, f->N, &f->C), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_create(f->M, f->N, &f->C0), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_create(f->M, f->N, &f->C_ref), LMMC_STATUS_OK);
}

static void compare_gemm_entries(gemm_trial_t *f, double tolerance) {
    for (size_t i = 0; i < f->M; i++) {
        for (size_t j = 0; j < f->N; j++) {
            const double actual = f->C.data[i * f->C.stride + j];
            const double expected = f->C_ref.data[i * f->C_ref.stride + j];
            if (!lmmc_test_nearly_equal(actual, expected, tolerance)) {
                fail_msg("trial %d (M=%" PRIuMAX ", K=%" PRIuMAX ", N=%" PRIuMAX ", transA=%d, transB=%d, alpha=%.3f, beta=%.3f): "
                         "C[%" PRIuMAX ",%" PRIuMAX "] = %.17g, reference = %.17g, tolerance = %.6e",
                         f->trial, (uintmax_t)(f->M), (uintmax_t)(f->K), (uintmax_t)(f->N), f->transA, f->transB, f->alpha, f->beta, (uintmax_t)(i), (uintmax_t)(j), actual, expected, tolerance);
            }
        }
    }
}

static int invalid_reference_norm(double norm) {
    return !isfinite(norm) || norm < 0.0;
}

static int invalid_gemm_reference_inputs(const gemm_trial_t *f,
                                         double norm_A, double norm_B, double norm_C0) {
    return !isfinite(f->alpha) || !isfinite(f->beta) ||
           invalid_reference_norm(norm_A) || invalid_reference_norm(norm_B) ||
           invalid_reference_norm(norm_C0);
}

static void check_gemm_reference(gemm_trial_t *f) {
    const double norm_A = lmmc_test_frobenius_norm(&f->A);
    const double norm_B = lmmc_test_frobenius_norm(&f->B);
    const double norm_C0 = lmmc_test_frobenius_norm(&f->C0);
    if (invalid_gemm_reference_inputs(f, norm_A, norm_B, norm_C0)) {
        fail_msg("trial %d: invalid scalar or reference norm", f->trial);
    }

    const double tolerance = 1e-10 *
                             (1.0 + fabs(f->alpha) * norm_A * norm_B + fabs(f->beta) * norm_C0);
    if (!isfinite(tolerance) || tolerance < 0.0) {
        fail_msg("trial %d: invalid tolerance %.6e", f->trial, tolerance);
    }

    compare_gemm_entries(f, tolerance);
}

static int setup_gemm(void **state) {
    gemm_trial_t *f = calloc(1, sizeof(*f));
    assert_non_null(f);
    *state = f;
    return 0;
}

static void release_gemm_matrices(gemm_trial_t *f) {
    lmmc_mat_destroy(&f->C_ref);
    lmmc_mat_destroy(&f->C0);
    lmmc_mat_destroy(&f->C);
    lmmc_mat_destroy(&f->B);
    lmmc_mat_destroy(&f->A);
}

static int teardown_gemm(void **state) {
    gemm_trial_t *f = *state;
    release_gemm_matrices(f);
    lmmc_rng_destroy(f->rng);
    free(f);
    return 0;
}

static void test_gemm_accuracy(void **state) {
    gemm_trial_t *f = *state;
    const size_t sizes[][3] = {
        {1, 1, 1},
        {2, 2, 2},
        {3, 3, 3},
        {4, 4, 4},
        {5, 5, 5},
        {8, 8, 8},
        {10, 10, 10},
        {16, 16, 16},
        {20, 20, 20},
        {2, 5, 3},
        {5, 2, 7},
        {1, 10, 1},
        {10, 1, 10},
        {7, 4, 9},
        {15, 8, 12},
    };
    assert_int_equal(lmmc_rng_create(&f->rng), LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(f->rng, UINT64_C(0x47454D4D)),
                     LMMC_STATUS_OK);

    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++) {
        f->M = sizes[si][0];
        f->K = sizes[si][1];
        f->N = sizes[si][2];
        for (f->trial = 1; f->trial <= 8; f->trial++) {
            lmmc_real_t u;
            assert_int_equal(lmmc_rng_uniform(f->rng, 0.0, 1.0, &u),
                             LMMC_STATUS_OK);
            f->transA = (u > 0.5) ? 1 : 0;
            assert_int_equal(lmmc_rng_uniform(f->rng, 0.0, 1.0, &u),
                             LMMC_STATUS_OK);
            f->transB = (u > 0.5) ? 1 : 0;
            f->alpha = random_scalar(f->rng);
            f->beta = random_scalar(f->rng);
            create_gemm_matrices(f);
            fill_random_matrix(f->rng, &f->A);
            fill_random_matrix(f->rng, &f->B);
            fill_random_matrix(f->rng, &f->C0);
            assert_int_equal(lmmc_mat_copy(&f->C0, &f->C), LMMC_STATUS_OK);
            naive_gemm(f->alpha, &f->A, f->transA, &f->B, f->transB,
                       f->beta, &f->C0, &f->C_ref);
            assert_int_equal(lmmc_mat_gemm(f->alpha, &f->A, f->transA, &f->B,
                                           f->transB, f->beta, &f->C),
                             LMMC_STATUS_OK);
            check_gemm_reference(f);
            release_gemm_matrices(f);
        }
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_gemm_accuracy, setup_gemm, teardown_gemm),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
