/**
 * @file test_mat_inv_trisolve.c
 * 矩阵求逆与三角求解单元测试。
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#define NUM_INV_TRIALS 20
#define NUM_TRI_TRIALS 20

static int test_failures = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: "); \
        printf(__VA_ARGS__); \
        printf("\n"); \
        test_failures++; \
        return 1; \
    } \
} while(0)

/**
 * @brief Fill matrix with random values in [-1, 1].
 */
static void fill_random_matrix(lmmc_rng_t* rng, lmmc_mat_t* mat) {
    for (size_t i = 0; i < mat->rows; i++) {
        for (size_t j = 0; j < mat->cols; j++) {
            lmmc_real_t val;
            lmmc_rng_uniform(rng, -1.0, 1.0, &val);
            mat->data[i * mat->stride + j] = val;
        }
    }
}

/**
 * @brief Estimate condition number κ(A) = ‖A‖_F * ‖A_inv‖_F.
 * This is an upper bound on the 2-norm condition number.
 */
static double estimate_condition_number(const lmmc_mat_t* A, const lmmc_mat_t* A_inv) {
    lmmc_real_t norm_A, norm_Ainv;
    lmmc_mat_norm_fro(A, &norm_A);
    lmmc_mat_norm_fro(A_inv, &norm_Ainv);
    return norm_A * norm_Ainv;
}

/**
 * @brief Test matrix inversion for random non-singular matrices.
 * Verifies ‖A·A_inv - I‖_F ≤ 1e-8 * κ(A).
 */
static int test_mat_inv_accuracy(void) {
    lmmc_rng_t* rng = NULL;
    lmmc_status_t st;
    int rc = 0;

    printf("Test: Matrix inversion accuracy for random non-singular matrices\n");

    st = lmmc_rng_create(&rng);
    CHECK(st == LMMC_STATUS_OK, "Failed to create RNG");
    lmmc_rng_seed(rng, 12345);

    for (int trial = 0; trial < NUM_INV_TRIALS; trial++) {
        /* Random size from 3 to 10 */
        lmmc_real_t u;
        lmmc_rng_uniform(rng, 3.0, 11.0, &u);
        size_t n = (size_t)u;
        if (n < 3) n = 3;
        if (n > 10) n = 10;

        lmmc_mat_t A = {0}, A_inv = {0}, product = {0}, identity = {0};

        st = lmmc_mat_create(n, n, &A);
        if (st != LMMC_STATUS_OK) { rc = 1; goto trial_cleanup; }
        st = lmmc_mat_create(n, n, &A_inv);
        if (st != LMMC_STATUS_OK) { rc = 1; goto trial_cleanup; }
        st = lmmc_mat_create(n, n, &product);
        if (st != LMMC_STATUS_OK) { rc = 1; goto trial_cleanup; }
        st = lmmc_mat_identity(n, &identity);
        if (st != LMMC_STATUS_OK) { rc = 1; goto trial_cleanup; }

        /* Generate random matrix and add diagonal dominance to ensure non-singularity */
        fill_random_matrix(rng, &A);
        for (size_t i = 0; i < n; i++) {
            A.data[i * A.stride + i] += (lmmc_real_t)(n + 1);
        }

        /* Compute inverse */
        st = lmmc_mat_inv(&A, &A_inv);
        if (st != LMMC_STATUS_OK) {
            printf("  FAIL: lmmc_mat_inv returned %d for trial %d (n=%zu)\n",
                   (int)st, trial, n);
            rc = 1;
            goto trial_cleanup;
        }

        /* Compute A * A_inv */
        st = lmmc_mat_mul(&A, &A_inv, &product);
        if (st != LMMC_STATUS_OK) {
            printf("  FAIL: lmmc_mat_mul returned %d for trial %d\n", (int)st, trial);
            rc = 1;
            goto trial_cleanup;
        }

        /* Compute ‖A*A_inv - I‖_F */
        lmmc_mat_t diff = {0};
        st = lmmc_mat_create(n, n, &diff);
        if (st != LMMC_STATUS_OK) { rc = 1; goto trial_cleanup; }
        st = lmmc_mat_sub(&product, &identity, &diff);
        if (st != LMMC_STATUS_OK) {
            lmmc_mat_destroy(&diff);
            rc = 1;
            goto trial_cleanup;
        }

        lmmc_real_t residual_norm;
        st = lmmc_mat_norm_fro(&diff, &residual_norm);
        lmmc_mat_destroy(&diff);
        if (st != LMMC_STATUS_OK) { rc = 1; goto trial_cleanup; }

        /* Estimate condition number */
        double kappa = estimate_condition_number(&A, &A_inv);
        double tol = 1e-8 * kappa;

        if (residual_norm > tol) {
            printf("  FAIL: trial %d (n=%zu): ‖A*A_inv - I‖_F = %.3e > 1e-8 * κ(A) = %.3e\n",
                   trial, n, (double)residual_norm, tol);
            rc = 1;
            goto trial_cleanup;
        }

trial_cleanup:
        lmmc_mat_destroy(&identity);
        lmmc_mat_destroy(&product);
        lmmc_mat_destroy(&A_inv);
        lmmc_mat_destroy(&A);
        if (rc != 0) break;
    }

    lmmc_rng_destroy(rng);
    if (rc == 0) printf("  PASS\n");
    return rc;
}

/**
 * @brief Test triangular solve for random upper and lower triangular matrices.
 * Verifies ‖Tx - b‖_2 ≤ 1e-10 * (‖T‖_F * ‖x‖_2 + ‖b‖_2).
 */
static int test_triangular_solve_accuracy(void) {
    lmmc_rng_t* rng = NULL;
    lmmc_status_t st;
    int rc = 0;

    printf("Test: Triangular solve accuracy\n");

    st = lmmc_rng_create(&rng);
    CHECK(st == LMMC_STATUS_OK, "Failed to create RNG");
    lmmc_rng_seed(rng, 67890);

    for (int trial = 0; trial < NUM_TRI_TRIALS; trial++) {
        /* Random size from 3 to 10 */
        lmmc_real_t u;
        lmmc_rng_uniform(rng, 3.0, 11.0, &u);
        size_t n = (size_t)u;
        if (n < 3) n = 3;
        if (n > 10) n = 10;

        /* Alternate between upper and lower triangular */
        int upper = (trial % 2 == 0) ? 1 : 0;

        lmmc_mat_t T = {0};
        lmmc_vec_t b = {0}, x = {0}, Tx = {0}, residual_vec = {0};

        st = lmmc_mat_create(n, n, &T);
        if (st != LMMC_STATUS_OK) { rc = 1; goto tri_cleanup; }
        st = lmmc_vec_create(n, &b);
        if (st != LMMC_STATUS_OK) { rc = 1; goto tri_cleanup; }
        st = lmmc_vec_create(n, &x);
        if (st != LMMC_STATUS_OK) { rc = 1; goto tri_cleanup; }
        st = lmmc_vec_create(n, &Tx);
        if (st != LMMC_STATUS_OK) { rc = 1; goto tri_cleanup; }
        st = lmmc_vec_create(n, &residual_vec);
        if (st != LMMC_STATUS_OK) { rc = 1; goto tri_cleanup; }

        /* Fill T as triangular with non-zero diagonal */
        st = lmmc_mat_fill(&T, 0.0);
        if (st != LMMC_STATUS_OK) { rc = 1; goto tri_cleanup; }

        if (upper) {
            for (size_t i = 0; i < n; i++) {
                for (size_t j = i; j < n; j++) {
                    lmmc_real_t val;
                    lmmc_rng_uniform(rng, -2.0, 2.0, &val);
                    T.data[i * T.stride + j] = val;
                }
                /* Ensure non-zero diagonal */
                lmmc_real_t diag_sign;
                lmmc_rng_uniform(rng, 0.0, 1.0, &diag_sign);
                T.data[i * T.stride + i] = (diag_sign > 0.5) ? 
                    (T.data[i * T.stride + i] + 2.0) : (T.data[i * T.stride + i] - 2.0);
            }
        } else {
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j <= i; j++) {
                    lmmc_real_t val;
                    lmmc_rng_uniform(rng, -2.0, 2.0, &val);
                    T.data[i * T.stride + j] = val;
                }
                /* Ensure non-zero diagonal */
                lmmc_real_t diag_sign;
                lmmc_rng_uniform(rng, 0.0, 1.0, &diag_sign);
                T.data[i * T.stride + i] = (diag_sign > 0.5) ? 
                    (T.data[i * T.stride + i] + 2.0) : (T.data[i * T.stride + i] - 2.0);
            }
        }

        /* Generate random RHS */
        for (size_t i = 0; i < n; i++) {
            lmmc_real_t val;
            lmmc_rng_uniform(rng, -5.0, 5.0, &val);
            b.data[i] = val;
        }

        /* Solve Tx = b */
        st = lmmc_solve_triangular(&T, upper, 0, &b, &x);
        if (st != LMMC_STATUS_OK) {
            printf("  FAIL: lmmc_solve_triangular returned %d for trial %d (n=%zu, upper=%d)\n",
                   (int)st, trial, n, upper);
            rc = 1;
            goto tri_cleanup;
        }

        /* Compute Tx */
        st = lmmc_mat_vec_mul(&T, &x, &Tx);
        if (st != LMMC_STATUS_OK) {
            printf("  FAIL: lmmc_mat_vec_mul returned %d for trial %d\n", (int)st, trial);
            rc = 1;
            goto tri_cleanup;
        }

        /* Compute residual = Tx - b */
        for (size_t i = 0; i < n; i++) {
            residual_vec.data[i] = Tx.data[i] - b.data[i];
        }

        lmmc_real_t residual_norm, T_norm, x_norm, b_norm;
        lmmc_vec_norm2(&residual_vec, &residual_norm);
        lmmc_mat_norm_fro(&T, &T_norm);
        lmmc_vec_norm2(&x, &x_norm);
        lmmc_vec_norm2(&b, &b_norm);

        double tol = 1e-10 * (T_norm * x_norm + b_norm);

        if (residual_norm > tol) {
            printf("  FAIL: trial %d (n=%zu, upper=%d): ‖Tx-b‖_2 = %.3e > tol = %.3e\n",
                   trial, n, upper, (double)residual_norm, tol);
            rc = 1;
            goto tri_cleanup;
        }

tri_cleanup:
        lmmc_vec_destroy(&residual_vec);
        lmmc_vec_destroy(&Tx);
        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&b);
        lmmc_mat_destroy(&T);
        if (rc != 0) break;
    }

    lmmc_rng_destroy(rng);
    if (rc == 0) printf("  PASS\n");
    return rc;
}

/**
 * @brief Test that lmmc_mat_inv returns LMMC_STATUS_SINGULAR_MATRIX for a singular matrix.
 */
static int test_singular_matrix(void) {
    lmmc_status_t st;
    lmmc_mat_t A = {0}, A_inv = {0};
    int rc = 0;

    printf("Test: Singular matrix returns LMMC_STATUS_SINGULAR_MATRIX\n");

    /* Create a 3x3 singular matrix (row 2 = 2 * row 1) */
    st = lmmc_mat_create(3, 3, &A);
    CHECK(st == LMMC_STATUS_OK, "Failed to create A");
    st = lmmc_mat_create(3, 3, &A_inv);
    CHECK(st == LMMC_STATUS_OK, "Failed to create A_inv");

    /* Fill A_inv with known values to verify it's not modified */
    st = lmmc_mat_fill(&A_inv, 99.0);
    CHECK(st == LMMC_STATUS_OK, "Failed to fill A_inv");

    /* Singular matrix: row 2 = 2 * row 0 */
    A.data[0 * A.stride + 0] = 1.0;
    A.data[0 * A.stride + 1] = 2.0;
    A.data[0 * A.stride + 2] = 3.0;
    A.data[1 * A.stride + 0] = 4.0;
    A.data[1 * A.stride + 1] = 5.0;
    A.data[1 * A.stride + 2] = 6.0;
    A.data[2 * A.stride + 0] = 2.0;
    A.data[2 * A.stride + 1] = 4.0;
    A.data[2 * A.stride + 2] = 6.0;

    st = lmmc_mat_inv(&A, &A_inv);
    if (st != LMMC_STATUS_SINGULAR_MATRIX) {
        printf("  FAIL: Expected LMMC_STATUS_SINGULAR_MATRIX (%d), got %d\n",
               (int)LMMC_STATUS_SINGULAR_MATRIX, (int)st);
        rc = 1;
    }

    /* Verify A_inv was not modified (should still be 99.0) */
    if (rc == 0) {
        for (size_t i = 0; i < 3; i++) {
            for (size_t j = 0; j < 3; j++) {
                if (!lmmc_test_nearly_equal(A_inv.data[i * A_inv.stride + j], 99.0, 1e-15)) {
                    printf("  FAIL: A_inv was modified at (%zu,%zu)\n", i, j);
                    rc = 1;
                    break;
                }
            }
            if (rc != 0) break;
        }
    }

    lmmc_mat_destroy(&A_inv);
    lmmc_mat_destroy(&A);
    if (rc == 0) printf("  PASS\n");
    return rc;
}

/**
 * @brief Test unit diagonal mode for triangular solve.
 * When diag_unit=1, the diagonal of T is treated as 1 regardless of actual values.
 */
static int test_unit_diagonal_triangular(void) {
    lmmc_status_t st;
    int rc = 0;

    printf("Test: Unit diagonal mode for triangular solve\n");

    /* Test upper triangular with unit diagonal */
    {
        lmmc_mat_t T = {0};
        lmmc_vec_t b = {0}, x = {0}, Tx_unit = {0};
        size_t n = 4;

        st = lmmc_mat_create(n, n, &T);
        CHECK(st == LMMC_STATUS_OK, "Failed to create T");
        st = lmmc_vec_create(n, &b);
        CHECK(st == LMMC_STATUS_OK, "Failed to create b");
        st = lmmc_vec_create(n, &x);
        CHECK(st == LMMC_STATUS_OK, "Failed to create x");
        st = lmmc_vec_create(n, &Tx_unit);
        CHECK(st == LMMC_STATUS_OK, "Failed to create Tx_unit");

        /* Upper triangular with arbitrary diagonal (will be ignored) */
        lmmc_mat_fill(&T, 0.0);
        T.data[0 * T.stride + 0] = 999.0;  /* ignored in unit mode */
        T.data[0 * T.stride + 1] = 2.0;
        T.data[0 * T.stride + 2] = 3.0;
        T.data[0 * T.stride + 3] = 1.0;
        T.data[1 * T.stride + 1] = 888.0;  /* ignored */
        T.data[1 * T.stride + 2] = -1.0;
        T.data[1 * T.stride + 3] = 2.0;
        T.data[2 * T.stride + 2] = 777.0;  /* ignored */
        T.data[2 * T.stride + 3] = 4.0;
        T.data[3 * T.stride + 3] = 666.0;  /* ignored */

        /* Set b */
        b.data[0] = 10.0;
        b.data[1] = 5.0;
        b.data[2] = 3.0;
        b.data[3] = 1.0;

        /* Solve with unit diagonal */
        st = lmmc_solve_triangular(&T, 1, 1, &b, &x);
        if (st != LMMC_STATUS_OK) {
            printf("  FAIL: lmmc_solve_triangular (unit upper) returned %d\n", (int)st);
            rc = 1;
            goto upper_cleanup;
        }

        /* Verify: manually compute T_unit * x where T_unit has 1s on diagonal */
        /* T_unit = [[1,2,3,1],[0,1,-1,2],[0,0,1,4],[0,0,0,1]] */
        /* x[3] = b[3] = 1 */
        /* x[2] = b[2] - 4*x[3] = 3 - 4 = -1 */
        /* x[1] = b[1] - (-1)*x[2] - 2*x[3] = 5 + (-1) - 2 = 2 */
        /* x[0] = b[0] - 2*x[1] - 3*x[2] - 1*x[3] = 10 - 4 + 3 - 1 = 8 */
        double expected_x[] = {8.0, 2.0, -1.0, 1.0};

        for (size_t i = 0; i < n; i++) {
            if (!lmmc_test_nearly_equal(x.data[i], expected_x[i], 1e-12)) {
                printf("  FAIL: x[%zu] = %.6f, expected %.6f\n",
                       i, (double)x.data[i], expected_x[i]);
                rc = 1;
                goto upper_cleanup;
            }
        }

upper_cleanup:
        lmmc_vec_destroy(&Tx_unit);
        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&b);
        lmmc_mat_destroy(&T);
    }

    /* Test lower triangular with unit diagonal */
    if (rc == 0) {
        lmmc_mat_t T = {0};
        lmmc_vec_t b = {0}, x = {0};
        size_t n = 3;

        st = lmmc_mat_create(n, n, &T);
        CHECK(st == LMMC_STATUS_OK, "Failed to create T (lower)");
        st = lmmc_vec_create(n, &b);
        CHECK(st == LMMC_STATUS_OK, "Failed to create b (lower)");
        st = lmmc_vec_create(n, &x);
        CHECK(st == LMMC_STATUS_OK, "Failed to create x (lower)");

        /* Lower triangular with arbitrary diagonal (will be ignored) */
        lmmc_mat_fill(&T, 0.0);
        T.data[0 * T.stride + 0] = 100.0;  /* ignored */
        T.data[1 * T.stride + 0] = 3.0;
        T.data[1 * T.stride + 1] = 200.0;  /* ignored */
        T.data[2 * T.stride + 0] = -1.0;
        T.data[2 * T.stride + 1] = 2.0;
        T.data[2 * T.stride + 2] = 300.0;  /* ignored */

        /* Set b */
        b.data[0] = 4.0;
        b.data[1] = 7.0;
        b.data[2] = 1.0;

        /* Solve with unit diagonal */
        st = lmmc_solve_triangular(&T, 0, 1, &b, &x);
        if (st != LMMC_STATUS_OK) {
            printf("  FAIL: lmmc_solve_triangular (unit lower) returned %d\n", (int)st);
            rc = 1;
            goto lower_cleanup;
        }

        /* Verify: T_unit = [[1,0,0],[3,1,0],[-1,2,1]] */
        /* x[0] = b[0] = 4 */
        /* x[1] = b[1] - 3*x[0] = 7 - 12 = -5 */
        /* x[2] = b[2] - (-1)*x[0] - 2*x[1] = 1 + 4 + 10 = 15 */
        double expected_x[] = {4.0, -5.0, 15.0};

        for (size_t i = 0; i < n; i++) {
            if (!lmmc_test_nearly_equal(x.data[i], expected_x[i], 1e-12)) {
                printf("  FAIL: x[%zu] = %.6f, expected %.6f\n",
                       i, (double)x.data[i], expected_x[i]);
                rc = 1;
                goto lower_cleanup;
            }
        }

lower_cleanup:
        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&b);
        lmmc_mat_destroy(&T);
    }

    if (rc == 0) printf("  PASS\n");
    return rc;
}

int main(void) {
    int rc = 0;

    printf("=== Matrix Inversion and Triangular Solve Unit Tests ===\n\n");

    rc |= test_mat_inv_accuracy();
    rc |= test_triangular_solve_accuracy();
    rc |= test_singular_matrix();
    rc |= test_unit_diagonal_triangular();

    printf("\n");
    if (rc == 0) {
        printf("All tests PASSED.\n");
    } else {
        printf("Some tests FAILED.\n");
    }

    return rc;
}
