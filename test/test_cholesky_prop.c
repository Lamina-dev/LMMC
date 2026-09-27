/**
 * @file test_cholesky_prop.c
 * Cholesky 分解重构精度属性测试。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

/**
 * @brief Generate a random SPD matrix of size n x n.
 *
 * Constructs A = B * B^T + eps * I where B is n x n with entries in [-1, 1].
 * This guarantees A is symmetric positive definite.
 *
 * @param rng   Random number generator.
 * @param n     Matrix dimension.
 * @param eps   Diagonal regularization (e.g. 0.01).
 * @param out_a Output matrix (must be pre-created as n x n).
 * @return LMMC_STATUS_OK on success.
 */
static lmmc_status_t generate_random_spd(lmmc_rng_t *rng, size_t n, double eps, lmmc_mat_t *out_a) {
    lmmc_mat_t b = {0};
    lmmc_mat_t bt = {0};
    lmmc_status_t st;

    st = lmmc_mat_create(n, n, &b);
    if (st != LMMC_STATUS_OK)
        return st;

    st = lmmc_mat_create(n, n, &bt);
    if (st != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&b);
        return st;
    }

    /* Fill B with random values in [-1, 1] */
    for (size_t i = 0; i < n * n; i++) {
        lmmc_real_t val;
        lmmc_rng_uniform(rng, -1.0, 1.0, &val);
        b.data[i] = val;
    }

    /* Compute B^T */
    st = lmmc_mat_transpose_to(&b, &bt);
    if (st != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&bt);
        lmmc_mat_destroy(&b);
        return st;
    }

    /* Compute A = B * B^T */
    st = lmmc_mat_mul(&b, &bt, out_a);
    if (st != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&bt);
        lmmc_mat_destroy(&b);
        return st;
    }

    /* Add eps * I to ensure strict positive definiteness */
    for (size_t i = 0; i < n; i++) {
        out_a->data[i * out_a->stride + i] += eps;
    }

    lmmc_mat_destroy(&bt);
    lmmc_mat_destroy(&b);
    return LMMC_STATUS_OK;
}

/**
 * @brief Compute the Frobenius norm of the difference between two matrices.
 *
 * @param a First matrix.
 * @param b Second matrix (same dimensions as a).
 * @return The Frobenius norm ||a - b||_F.
 */
static double frobenius_norm_diff(const lmmc_mat_t *a, const lmmc_mat_t *b) {
    double sum = 0.0;
    for (size_t i = 0; i < a->rows; i++) {
        for (size_t j = 0; j < a->cols; j++) {
            double diff = a->data[i * a->stride + j] - b->data[i * b->stride + j];
            sum += diff * diff;
        }
    }
    return sqrt(sum);
}

/**
 * @brief Reconstruct L * L^T from the lower-triangular Cholesky factor.
 *
 * After lmmc_cholesky_decompose_inplace, the lower triangle of the matrix
 * contains L. This function computes L * L^T into out_reconstructed.
 *
 * @param l_mat            Matrix containing L in its lower triangle.
 * @param n                Dimension.
 * @param out_reconstructed Output matrix (pre-created n x n).
 */
static void reconstruct_from_cholesky(const lmmc_mat_t *l_mat, size_t n, lmmc_mat_t *out_reconstructed) {
    /* Zero out the result */
    for (size_t i = 0; i < n * n; i++) {
        out_reconstructed->data[i] = 0.0;
    }

    /* Compute L * L^T manually using only the lower triangle of l_mat */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j <= i; j++) {
            /* (L * L^T)[i][j] = sum_k L[i][k] * L[j][k] for k = 0..min(i,j) */
            double sum = 0.0;
            for (size_t k = 0; k <= j; k++) {
                double l_ik = l_mat->data[i * l_mat->stride + k];
                double l_jk = l_mat->data[j * l_mat->stride + k];
                sum += l_ik * l_jk;
            }
            out_reconstructed->data[i * out_reconstructed->stride + j] = sum;
            out_reconstructed->data[j * out_reconstructed->stride + i] = sum;
        }
    }
}

typedef struct {
    lmmc_rng_t *rng;
    lmmc_mat_t a, a_copy, reconstructed;
} cholesky_fixture_t;

static int setup_cholesky(void **state) {
    cholesky_fixture_t *f = calloc(1, sizeof(*f));
    assert_non_null(f);
    *state = f;
    return 0;
}

static void release_cholesky_matrices(cholesky_fixture_t *f) {
    lmmc_mat_destroy(&f->reconstructed);
    lmmc_mat_destroy(&f->a_copy);
    lmmc_mat_destroy(&f->a);
}

static int teardown_cholesky(void **state) {
    cholesky_fixture_t *f = *state;
    release_cholesky_matrices(f);
    lmmc_rng_destroy(f->rng);
    free(f);
    return 0;
}

static void test_cholesky_reconstruction(void **state) {
    cholesky_fixture_t *f = *state;
    const size_t sizes[] = {2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20};
    assert_int_equal(lmmc_rng_create(&f->rng), LMMC_STATUS_OK);
    lmmc_rng_seed(f->rng, UINT64_C(0x43484F4C));

    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++) {
        const size_t n = sizes[si];
        for (int trial = 1; trial <= 10; trial++) {
            assert_int_equal(lmmc_mat_create(n, n, &f->a), LMMC_STATUS_OK);
            assert_int_equal(lmmc_mat_create(n, n, &f->a_copy), LMMC_STATUS_OK);
            assert_int_equal(lmmc_mat_create(n, n, &f->reconstructed), LMMC_STATUS_OK);
            assert_int_equal(generate_random_spd(f->rng, n, 0.01, &f->a), LMMC_STATUS_OK);
            assert_int_equal(lmmc_mat_copy(&f->a, &f->a_copy), LMMC_STATUS_OK);
            assert_int_equal(lmmc_cholesky_decompose_inplace(&f->a), LMMC_STATUS_OK);
            reconstruct_from_cholesky(&f->a, n, &f->reconstructed);

            const double diff_norm = frobenius_norm_diff(&f->reconstructed, &f->a_copy);
            lmmc_real_t a_norm;
            assert_int_equal(lmmc_mat_norm_fro(&f->a_copy, &a_norm), LMMC_STATUS_OK);
            const double tolerance = 1e-10 * a_norm;
            if (!lmmc_test_nearly_equal(diff_norm, 0.0, tolerance)) {
                fail_msg("trial %d (n=%" PRIuMAX "): ||L*L^T - A||_F = %.6e, bound = %.6e", trial, (uintmax_t)(n), diff_norm, tolerance);
            }
            release_cholesky_matrices(f);
        }
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_cholesky_reconstruction,
                                        setup_cholesky, teardown_cholesky),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
