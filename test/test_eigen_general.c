/**
 * @file test_eigen_general.c
 * @brief Property-based tests for general eigenvalue decomposition.
 *
 * Implements Property 9 from the design doc:
 *   For any real square matrix A, the returned (real[i], imag[i]) eigenvalue
 *   pairs should satisfy the eigenvalue equation, and complex eigenvalues
 *   should come in conjugate pairs.
 *
 * Tests:
 *   1. Diagonal matrices: eigenvalues equal diagonal entries
 *   2. Block-diagonal matrices: eigenvalues match the blocks
 *   3. Complex eigenvalue case: 2x2 rotation matrix has e^(±i*theta)
 *   4. Conjugate pairs: when imag != 0 there must be a matching -imag entry
 *   5. Random small (2x2, 3x3) matrices: verify p(lambda) = det(A - lambda*I)
 *      is approximately zero for each returned eigenvalue
 *
 * Validates: Requirements 6.1, 6.4
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/status.h"
#include "test_common.h"

#define TOL 1e-8

#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: " msg "\n", ##__VA_ARGS__); \
        return 1; \
    } \
} while (0)

/* ========================================================================
 * Helpers: complex arithmetic and characteristic polynomial evaluation.
 * Use plain double tuples to avoid <complex.h> portability concerns on MSVC.
 * ======================================================================== */

typedef struct { double re; double im; } cplx_t;

static cplx_t cplx_make(double r, double i) {
    cplx_t z;
    z.re = r;
    z.im = i;
    return z;
}

static cplx_t cplx_mul(cplx_t a, cplx_t b) {
    cplx_t r;
    r.re = a.re * b.re - a.im * b.im;
    r.im = a.re * b.im + a.im * b.re;
    return r;
}

static cplx_t cplx_add(cplx_t a, cplx_t b) {
    cplx_t r;
    r.re = a.re + b.re;
    r.im = a.im + b.im;
    return r;
}

static double cplx_abs(cplx_t a) {
    return sqrt(a.re * a.re + a.im * a.im);
}

/*
 * Evaluate a real-coefficient polynomial p(x) = c[0] + c[1]*x + ... + c[deg]*x^deg
 * at a complex point lambda using Horner's scheme.
 */
static cplx_t poly_eval_complex(const double *coeffs, int degree, cplx_t lambda) {
    cplx_t acc = cplx_make(coeffs[degree], 0.0);
    int k;
    for (k = degree - 1; k >= 0; k--) {
        acc = cplx_mul(acc, lambda);
        acc.re += coeffs[k];
    }
    return acc;
}

/*
 * Build coefficients of the characteristic polynomial p(lambda) = det(A - lambda*I)
 * for a real n x n matrix (n = 2 or 3). Coefficients are stored low-degree first:
 *   coeffs[0] + coeffs[1]*lambda + ... + coeffs[n]*lambda^n.
 * Returns 0 on success, 1 if n is unsupported.
 */
static int build_char_poly(const lmmc_real_t *A, size_t n, double *coeffs) {
    if (n == 2) {
        double a = A[0], b = A[1];
        double c = A[2], d = A[3];
        double tr = a + d;
        double det = a * d - b * c;
        /* det(A - lambda*I) = lambda^2 - tr*lambda + det */
        coeffs[0] = det;
        coeffs[1] = -tr;
        coeffs[2] = 1.0;
        return 0;
    }
    if (n == 3) {
        double a11 = A[0], a12 = A[1], a13 = A[2];
        double a21 = A[3], a22 = A[4], a23 = A[5];
        double a31 = A[6], a32 = A[7], a33 = A[8];
        double tr = a11 + a22 + a33;
        /* sum of 2x2 principal minors */
        double m11 = a22 * a33 - a23 * a32;
        double m22 = a11 * a33 - a13 * a31;
        double m33 = a11 * a22 - a12 * a21;
        double c1 = m11 + m22 + m33;
        /* determinant */
        double det = a11 * (a22 * a33 - a23 * a32)
                   - a12 * (a21 * a33 - a23 * a31)
                   + a13 * (a21 * a32 - a22 * a31);
        /* det(A - lambda*I) = -lambda^3 + tr*lambda^2 - c1*lambda + det */
        coeffs[0] = det;
        coeffs[1] = -c1;
        coeffs[2] = tr;
        coeffs[3] = -1.0;
        return 0;
    }
    return 1;
}

/*
 * Verify that every returned eigenvalue is a root of the characteristic polynomial.
 *
 * The matrix scale is used to normalize tolerance so that |p(lambda)| ~ 0 is
 * checked relative to a reasonable scale (sum of |a_ij| raised to power n).
 */
static int verify_char_poly_roots(const lmmc_real_t *A, size_t n,
                                  const lmmc_eigen_gen_result_t *result,
                                  double tol)
{
    double coeffs[4];
    if (build_char_poly(A, n, coeffs) != 0) {
        printf("  FAIL: unsupported dimension n=%zu\n", n);
        return 1;
    }

    /* Scale used to make tolerance meaningful for matrices with non-unit entries. */
    double anorm = 0.0;
    for (size_t i = 0; i < n * n; i++) {
        if (fabs(A[i]) > anorm) anorm = fabs(A[i]);
    }
    double scale = 1.0;
    for (size_t k = 0; k < n; k++) scale *= (1.0 + anorm);
    double scaled_tol = tol * (scale + 1.0);

    for (size_t i = 0; i < n; i++) {
        cplx_t lambda = cplx_make(result->real_parts.data[i],
                                  result->imag_parts.data[i]);
        cplx_t p = poly_eval_complex(coeffs, (int)n, lambda);
        double mag = cplx_abs(p);
        if (mag > scaled_tol) {
            printf("  FAIL: |p(lambda_%zu)| = %.3e exceeds tol %.3e "
                   "(lambda = %.6f + %.6f i)\n",
                   i, mag, scaled_tol, lambda.re, lambda.im);
            return 1;
        }
    }
    return 0;
}

/*
 * Verify that every nonzero imaginary part has a matching opposite-sign entry,
 * confirming that complex eigenvalues come in conjugate pairs.
 */
static int verify_conjugate_pairs(const lmmc_eigen_gen_result_t *result, double tol)
{
    size_t n = result->real_parts.size;
    int *matched = (int *)calloc(n, sizeof(int));
    if (!matched) {
        printf("  FAIL: allocation failure in verify_conjugate_pairs\n");
        return 1;
    }

    for (size_t i = 0; i < n; i++) {
        if (matched[i]) continue;
        double im = result->imag_parts.data[i];
        if (fabs(im) <= tol) {
            matched[i] = 1;
            continue;
        }
        double re = result->real_parts.data[i];
        int found = 0;
        for (size_t j = 0; j < n; j++) {
            if (matched[j] || j == i) continue;
            double rj = result->real_parts.data[j];
            double ij = result->imag_parts.data[j];
            if (fabs(re - rj) <= tol && fabs(im + ij) <= tol) {
                matched[i] = 1;
                matched[j] = 1;
                found = 1;
                break;
            }
        }
        if (!found) {
            printf("  FAIL: no conjugate partner for lambda_%zu = %.6f + %.6f i\n",
                   i, re, im);
            free(matched);
            return 1;
        }
    }
    free(matched);
    return 0;
}

/* Convenience helper: run the solver and run both invariants. */
static int run_property_check(const lmmc_real_t *A, size_t n, const char *label)
{
    lmmc_mat_t mat;
    lmmc_status_t st = lmmc_mat_create(n, n, &mat);
    CHECK(st == LMMC_STATUS_OK, "[%s] mat_create failed: %d", label, (int)st);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            mat.data[i * mat.stride + j] = A[i * n + j];
        }
    }

    lmmc_eigen_gen_result_t result;
    memset(&result, 0, sizeof(result));
    st = lmmc_eigen_general(&mat, &result);
    if (st != LMMC_STATUS_OK) {
        printf("  FAIL: [%s] lmmc_eigen_general returned %d\n", label, (int)st);
        lmmc_mat_destroy(&mat);
        return 1;
    }

    int rc = 0;
    if (verify_char_poly_roots(A, n, &result, TOL) != 0) {
        printf("    (in test: %s)\n", label);
        rc = 1;
    }
    if (rc == 0 && verify_conjugate_pairs(&result, TOL) != 0) {
        printf("    (in test: %s)\n", label);
        rc = 1;
    }

    lmmc_eigen_gen_result_destroy(&result);
    lmmc_mat_destroy(&mat);
    return rc;
}

/* ========================================================================
 * Test 1: Diagonal matrices - eigenvalues equal diagonal entries.
 * ======================================================================== */
static int test_diagonal_2x2(void)
{
    lmmc_real_t A[] = {
        3.0, 0.0,
        0.0, 7.0
    };
    return run_property_check(A, 2, "diagonal_2x2");
}

static int test_diagonal_3x3(void)
{
    lmmc_real_t A[] = {
       -1.5, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 4.5
    };
    return run_property_check(A, 3, "diagonal_3x3");
}

/* ========================================================================
 * Test 2: Block-diagonal matrices - eigenvalues match the blocks.
 *
 * A = block_diag(B1, [c]) where B1 is a 2x2 block with known eigenvalues
 * and [c] is a 1x1 block.
 * Use B1 = [[2, 1], [1, 2]] with eigenvalues 1, 3. Combined 3x3 spectrum: {1, 3, c}.
 * ======================================================================== */
static int test_block_diagonal_3x3(void)
{
    lmmc_real_t A[] = {
        2.0, 1.0, 0.0,
        1.0, 2.0, 0.0,
        0.0, 0.0, 5.0
    };
    return run_property_check(A, 3, "block_diagonal_3x3");
}

/* ========================================================================
 * Test 3: Complex eigenvalue case - 2x2 rotation matrix.
 * R(theta) = [[cos, -sin], [sin, cos]] has eigenvalues cos +/- i sin = e^(±i theta).
 * ======================================================================== */
static int test_rotation_2x2(void)
{
    double theta = 0.7;
    double c = cos(theta);
    double s = sin(theta);
    lmmc_real_t A[] = {
         c, -s,
         s,  c
    };
    return run_property_check(A, 2, "rotation_2x2");
}

/* ========================================================================
 * Test 4: Verify conjugate pairs explicitly via known-complex 2x2 case.
 * Already covered by test_rotation_2x2's conjugate check, but exercise an
 * additional skew-symmetric matrix [[0, -2], [2, 0]] -> eigenvalues +/- 2i.
 * ======================================================================== */
static int test_skew_symmetric_2x2(void)
{
    lmmc_real_t A[] = {
        0.0, -2.0,
        2.0,  0.0
    };
    return run_property_check(A, 2, "skew_symmetric_2x2");
}

/* ========================================================================
 * Test 5: Random small matrices - characteristic polynomial property.
 * Use deterministic seeded "random" matrices to keep tests reproducible.
 * ======================================================================== */
static unsigned long lcg_state = 12345UL;
static double next_uniform(double lo, double hi)
{
    /* Numerical Recipes style linear congruential generator. */
    lcg_state = lcg_state * 1664525UL + 1013904223UL;
    double u = (double)(lcg_state & 0xFFFFFFFFUL) / 4294967296.0;
    return lo + (hi - lo) * u;
}

static int test_random_2x2_matrices(void)
{
    int trials = 8;
    int i;
    char label[32];
    lcg_state = 12345UL;
    for (i = 0; i < trials; i++) {
        lmmc_real_t A[4];
        A[0] = (lmmc_real_t)next_uniform(-2.0, 2.0);
        A[1] = (lmmc_real_t)next_uniform(-2.0, 2.0);
        A[2] = (lmmc_real_t)next_uniform(-2.0, 2.0);
        A[3] = (lmmc_real_t)next_uniform(-2.0, 2.0);
        snprintf(label, sizeof(label), "random_2x2_#%d", i);
        if (run_property_check(A, 2, label) != 0) return 1;
    }
    return 0;
}

static int test_random_3x3_matrices(void)
{
    int trials = 8;
    int i, k;
    char label[32];
    lcg_state = 67890UL;
    for (i = 0; i < trials; i++) {
        lmmc_real_t A[9];
        for (k = 0; k < 9; k++) {
            A[k] = (lmmc_real_t)next_uniform(-2.0, 2.0);
        }
        snprintf(label, sizeof(label), "random_3x3_#%d", i);
        if (run_property_check(A, 3, label) != 0) return 1;
    }
    return 0;
}

/* ========================================================================
 * Test: Input validation.
 * ======================================================================== */
static int test_invalid_inputs(void)
{
    lmmc_eigen_gen_result_t result;
    memset(&result, 0, sizeof(result));
    lmmc_status_t st;

    st = lmmc_eigen_general(NULL, &result);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "NULL input matrix should return INVALID_ARGUMENT, got %d", (int)st);

    lmmc_mat_t mat;
    st = lmmc_mat_create(3, 3, &mat);
    CHECK(st == LMMC_STATUS_OK, "mat_create failed");
    st = lmmc_eigen_general(&mat, NULL);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "NULL out_result should return INVALID_ARGUMENT, got %d", (int)st);
    lmmc_mat_destroy(&mat);

    /* Non-square matrix */
    st = lmmc_mat_create(2, 3, &mat);
    CHECK(st == LMMC_STATUS_OK, "mat_create non-square failed");
    st = lmmc_eigen_general(&mat, &result);
    CHECK(st == LMMC_STATUS_INVALID_ARGUMENT,
          "non-square input should return INVALID_ARGUMENT, got %d", (int)st);
    lmmc_mat_destroy(&mat);
    return 0;
}

/* ========================================================================
 * Main
 * ======================================================================== */
typedef int (*test_func_t)(void);

typedef struct {
    const char *name;
    test_func_t func;
} test_entry_t;

int main(void)
{
    test_entry_t tests[] = {
        {"invalid_inputs",            test_invalid_inputs},
        {"diagonal_2x2",              test_diagonal_2x2},
        {"diagonal_3x3",              test_diagonal_3x3},
        {"block_diagonal_3x3",        test_block_diagonal_3x3},
        {"rotation_2x2",              test_rotation_2x2},
        {"skew_symmetric_2x2",        test_skew_symmetric_2x2},
        {"random_2x2_matrices",       test_random_2x2_matrices},
        {"random_3x3_matrices",       test_random_3x3_matrices},
    };

    size_t n_tests = sizeof(tests) / sizeof(tests[0]);
    size_t n_passed = 0;
    size_t n_failed = 0;

    printf("=== General Eigenvalue Decomposition Property Tests ===\n");
    printf("Property 9: Validates Requirements 6.1, 6.4\n\n");

    for (size_t i = 0; i < n_tests; i++) {
        printf("[%zu/%zu] %s ... ", i + 1, n_tests, tests[i].name);
        if (tests[i].func() == 0) {
            printf("PASS\n");
            n_passed++;
        } else {
            printf("FAIL\n");
            n_failed++;
        }
    }

    printf("\n=== Results: %zu passed, %zu failed ===\n", n_passed, n_failed);
    return (n_failed > 0) ? 1 : 0;
}
