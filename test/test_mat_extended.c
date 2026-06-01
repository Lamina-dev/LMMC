/**
 * @file test_mat_extended.c
 * @brief Property-based and unit tests for extended matrix algebra operations.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>

#include "lmmc/lmmc.h"
#include "test_common.h"

#define PBT_ITERATIONS 100

static int test_failures = 0;
static int test_count = 0;

#define REPORT(name, result) do { \
    test_count++; \
    if (result) { \
        printf("  FAIL: %s\n", name); \
        test_failures++; \
    } else { \
        printf("  PASS: %s\n", name); \
    } \
} while (0)

static double rand_double(double lo, double hi)
{
    return ((double)rand() / RAND_MAX) * (hi - lo) + lo;
}

static int test_property10_cross_product_orthogonality(void)
{
    int i;
    double eps = 1e-9;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        lmmc_vec_t a, b, c;
        lmmc_status_t st;
        double dot_ca, dot_cb;
        size_t j;

        st = lmmc_vec_create(3, &a);
        if (st != LMMC_STATUS_OK) { printf("    vec_create(a) failed at iter %d\n", i); return 1; }
        st = lmmc_vec_create(3, &b);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&a); return 1; }
        st = lmmc_vec_create(3, &c);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&a); lmmc_vec_destroy(&b); return 1; }

        /* Fill a and b with random values in [-10, 10] */
        for (j = 0; j < 3; j++) {
            a.data[j] = rand_double(-10.0, 10.0);
            b.data[j] = rand_double(-10.0, 10.0);
        }

        /* c = cross(a, b) */
        st = lmmc_vec_cross(&a, &b, &c);
        if (st != LMMC_STATUS_OK) {
            printf("    vec_cross failed at iter %d, status=%d\n", i, (int)st);
            lmmc_vec_destroy(&a); lmmc_vec_destroy(&b); lmmc_vec_destroy(&c);
            return 1;
        }

        /* dot(c, a) should be ~0 */
        st = lmmc_vec_dot(&c, &a, &dot_ca);
        if (st != LMMC_STATUS_OK) {
            printf("    vec_dot(c,a) failed at iter %d\n", i);
            lmmc_vec_destroy(&a); lmmc_vec_destroy(&b); lmmc_vec_destroy(&c);
            return 1;
        }

        /* dot(c, b) should be ~0 */
        st = lmmc_vec_dot(&c, &b, &dot_cb);
        if (st != LMMC_STATUS_OK) {
            printf("    vec_dot(c,b) failed at iter %d\n", i);
            lmmc_vec_destroy(&a); lmmc_vec_destroy(&b); lmmc_vec_destroy(&c);
            return 1;
        }

        if (!lmmc_test_nearly_equal(dot_ca, 0.0, eps)) {
            printf("    dot(cross(a,b), a) != 0 at iter %d: got %g\n", i, dot_ca);
            lmmc_vec_destroy(&a); lmmc_vec_destroy(&b); lmmc_vec_destroy(&c);
            return 1;
        }

        if (!lmmc_test_nearly_equal(dot_cb, 0.0, eps)) {
            printf("    dot(cross(a,b), b) != 0 at iter %d: got %g\n", i, dot_cb);
            lmmc_vec_destroy(&a); lmmc_vec_destroy(&b); lmmc_vec_destroy(&c);
            return 1;
        }

        lmmc_vec_destroy(&a);
        lmmc_vec_destroy(&b);
        lmmc_vec_destroy(&c);
    }
    return 0;
}

static int test_property11_cross_product_self_annihilation(void)
{
    int i;
    double eps = 1e-15;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        lmmc_vec_t a, c;
        lmmc_status_t st;
        size_t j;

        st = lmmc_vec_create(3, &a);
        if (st != LMMC_STATUS_OK) { printf("    vec_create(a) failed at iter %d\n", i); return 1; }
        st = lmmc_vec_create(3, &c);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&a); return 1; }

        /* Fill a with random values in [-10, 10] */
        for (j = 0; j < 3; j++) {
            a.data[j] = rand_double(-10.0, 10.0);
        }

        /* c = cross(a, a) */
        st = lmmc_vec_cross(&a, &a, &c);
        if (st != LMMC_STATUS_OK) {
            printf("    vec_cross(a,a) failed at iter %d, status=%d\n", i, (int)st);
            lmmc_vec_destroy(&a); lmmc_vec_destroy(&c);
            return 1;
        }

        /* c should be zero vector */
        for (j = 0; j < 3; j++) {
            if (!lmmc_test_nearly_equal(c.data[j], 0.0, eps)) {
                printf("    cross(a,a)[%zu] != 0 at iter %d: got %g\n", j, i, c.data[j]);
                lmmc_vec_destroy(&a); lmmc_vec_destroy(&c);
                return 1;
            }
        }

        lmmc_vec_destroy(&a);
        lmmc_vec_destroy(&c);
    }
    return 0;
}

static int test_property12_mat_pow_exponent_addition(void)
{
    int i;
    double eps = 1e-6;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        lmmc_mat_t A, pow_mn, pow_m, pow_n, product;
        lmmc_status_t st;
        int m, n;
        size_t r, c;

        /* Generate small exponents m, n in [1, 3] */
        m = (rand() % 3) + 1;
        n = (rand() % 3) + 1;

        st = lmmc_mat_create(2, 2, &A);
        if (st != LMMC_STATUS_OK) { printf("    mat_create(A) failed at iter %d\n", i); return 1; }

        /* Fill A with random values and add diagonal dominance */
        for (r = 0; r < 2; r++) {
            for (c = 0; c < 2; c++) {
                A.data[r * A.stride + c] = rand_double(-1.0, 1.0);
            }
            A.data[r * A.stride + r] += 5.0;  /* diagonal dominance */
        }

        st = lmmc_mat_create(2, 2, &pow_mn);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&A); return 1; }
        st = lmmc_mat_create(2, 2, &pow_m);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&A); lmmc_mat_destroy(&pow_mn); return 1; }
        st = lmmc_mat_create(2, 2, &pow_n);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&A); lmmc_mat_destroy(&pow_mn); lmmc_mat_destroy(&pow_m); return 1; }
        st = lmmc_mat_create(2, 2, &product);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&A); lmmc_mat_destroy(&pow_mn); lmmc_mat_destroy(&pow_m); lmmc_mat_destroy(&pow_n); return 1; }

        /* pow(A, m+n) */
        st = lmmc_mat_pow(&A, m + n, &pow_mn);
        if (st != LMMC_STATUS_OK) {
            printf("    mat_pow(A, m+n) failed at iter %d, status=%d\n", i, (int)st);
            lmmc_mat_destroy(&A); lmmc_mat_destroy(&pow_mn);
            lmmc_mat_destroy(&pow_m); lmmc_mat_destroy(&pow_n); lmmc_mat_destroy(&product);
            return 1;
        }

        /* pow(A, m) */
        st = lmmc_mat_pow(&A, m, &pow_m);
        if (st != LMMC_STATUS_OK) {
            printf("    mat_pow(A, m) failed at iter %d, status=%d\n", i, (int)st);
            lmmc_mat_destroy(&A); lmmc_mat_destroy(&pow_mn);
            lmmc_mat_destroy(&pow_m); lmmc_mat_destroy(&pow_n); lmmc_mat_destroy(&product);
            return 1;
        }

        /* pow(A, n) */
        st = lmmc_mat_pow(&A, n, &pow_n);
        if (st != LMMC_STATUS_OK) {
            printf("    mat_pow(A, n) failed at iter %d, status=%d\n", i, (int)st);
            lmmc_mat_destroy(&A); lmmc_mat_destroy(&pow_mn);
            lmmc_mat_destroy(&pow_m); lmmc_mat_destroy(&pow_n); lmmc_mat_destroy(&product);
            return 1;
        }

        /* product = pow(A, m) * pow(A, n) */
        st = lmmc_mat_mul(&pow_m, &pow_n, &product);
        if (st != LMMC_STATUS_OK) {
            printf("    mat_mul failed at iter %d, status=%d\n", i, (int)st);
            lmmc_mat_destroy(&A); lmmc_mat_destroy(&pow_mn);
            lmmc_mat_destroy(&pow_m); lmmc_mat_destroy(&pow_n); lmmc_mat_destroy(&product);
            return 1;
        }

        /* Check pow(A, m+n) ~ pow(A, m) * pow(A, n) */
        for (r = 0; r < 2; r++) {
            for (c = 0; c < 2; c++) {
                double got = pow_mn.data[r * pow_mn.stride + c];
                double expected = product.data[r * product.stride + c];
                if (!lmmc_test_nearly_equal(got, expected, eps * fabs(expected) + eps)) {
                    printf("    pow exponent addition failed at iter %d [%zu][%zu]: got %g, expected %g\n",
                           i, r, c, got, expected);
                    lmmc_mat_destroy(&A); lmmc_mat_destroy(&pow_mn);
                    lmmc_mat_destroy(&pow_m); lmmc_mat_destroy(&pow_n); lmmc_mat_destroy(&product);
                    return 1;
                }
            }
        }

        lmmc_mat_destroy(&A);
        lmmc_mat_destroy(&pow_mn);
        lmmc_mat_destroy(&pow_m);
        lmmc_mat_destroy(&pow_n);
        lmmc_mat_destroy(&product);
    }
    return 0;
}

static int test_property13_mat_rdiv_roundtrip(void)
{
    int i;
    double eps = 1e-7;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        lmmc_mat_t A, B, X, recovered;
        lmmc_status_t st;
        size_t n = 2;
        size_t m = (size_t)(rand() % 3) + 1;  /* B is m x n */
        size_t r, c;

        st = lmmc_mat_create(n, n, &A);
        if (st != LMMC_STATUS_OK) { printf("    mat_create(A) failed at iter %d\n", i); return 1; }
        st = lmmc_mat_create(m, n, &B);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&A); return 1; }
        st = lmmc_mat_create(m, n, &X);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&A); lmmc_mat_destroy(&B); return 1; }
        st = lmmc_mat_create(m, n, &recovered);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&A); lmmc_mat_destroy(&B); lmmc_mat_destroy(&X); return 1; }

        /* Generate non-singular A by adding large diagonal */
        for (r = 0; r < n; r++) {
            for (c = 0; c < n; c++) {
                A.data[r * A.stride + c] = rand_double(-1.0, 1.0);
            }
            A.data[r * A.stride + r] += 5.0;
        }

        /* Fill B with random values */
        for (r = 0; r < m; r++) {
            for (c = 0; c < n; c++) {
                B.data[r * B.stride + c] = rand_double(-5.0, 5.0);
            }
        }

        /* X = rdiv(B, A) = B * A^{-1} */
        st = lmmc_mat_rdiv(&B, &A, &X);
        if (st != LMMC_STATUS_OK) {
            printf("    mat_rdiv failed at iter %d, status=%d\n", i, (int)st);
            lmmc_mat_destroy(&A); lmmc_mat_destroy(&B);
            lmmc_mat_destroy(&X); lmmc_mat_destroy(&recovered);
            return 1;
        }

        /* recovered = X * A, should ~ B */
        st = lmmc_mat_mul(&X, &A, &recovered);
        if (st != LMMC_STATUS_OK) {
            printf("    mat_mul(X, A) failed at iter %d, status=%d\n", i, (int)st);
            lmmc_mat_destroy(&A); lmmc_mat_destroy(&B);
            lmmc_mat_destroy(&X); lmmc_mat_destroy(&recovered);
            return 1;
        }

        /* Check recovered ~ B */
        for (r = 0; r < m; r++) {
            for (c = 0; c < n; c++) {
                double got = recovered.data[r * recovered.stride + c];
                double expected = B.data[r * B.stride + c];
                if (!lmmc_test_nearly_equal(got, expected, eps * fabs(expected) + eps)) {
                    printf("    rdiv round-trip failed at iter %d [%zu][%zu]: got %g, expected %g\n",
                           i, r, c, got, expected);
                    lmmc_mat_destroy(&A); lmmc_mat_destroy(&B);
                    lmmc_mat_destroy(&X); lmmc_mat_destroy(&recovered);
                    return 1;
                }
            }
        }

        lmmc_mat_destroy(&A);
        lmmc_mat_destroy(&B);
        lmmc_mat_destroy(&X);
        lmmc_mat_destroy(&recovered);
    }
    return 0;
}

static int test_property14_mat_norm_transpose_duality(void)
{
    int i;
    double eps = 1e-12;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        size_t rows = (size_t)(rand() % 5) + 1;
        size_t cols = (size_t)(rand() % 5) + 1;
        lmmc_mat_t A, AT;
        lmmc_status_t st;
        double norm1_A, norm_inf_AT;
        size_t r, c;

        st = lmmc_mat_create(rows, cols, &A);
        if (st != LMMC_STATUS_OK) { printf("    mat_create(A) failed at iter %d\n", i); return 1; }
        st = lmmc_mat_create(cols, rows, &AT);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&A); return 1; }

        /* Fill A with random values */
        for (r = 0; r < rows; r++) {
            for (c = 0; c < cols; c++) {
                A.data[r * A.stride + c] = rand_double(-10.0, 10.0);
            }
        }

        /* Compute A^T */
        st = lmmc_mat_transpose_to(&A, &AT);
        if (st != LMMC_STATUS_OK) {
            printf("    mat_transpose_to failed at iter %d, status=%d\n", i, (int)st);
            lmmc_mat_destroy(&A); lmmc_mat_destroy(&AT);
            return 1;
        }

        /* norm1(A) */
        st = lmmc_mat_norm1(&A, &norm1_A);
        if (st != LMMC_STATUS_OK) {
            printf("    mat_norm1 failed at iter %d, status=%d\n", i, (int)st);
            lmmc_mat_destroy(&A); lmmc_mat_destroy(&AT);
            return 1;
        }

        /* norm_inf(A^T) */
        st = lmmc_mat_norm_inf(&AT, &norm_inf_AT);
        if (st != LMMC_STATUS_OK) {
            printf("    mat_norm_inf failed at iter %d, status=%d\n", i, (int)st);
            lmmc_mat_destroy(&A); lmmc_mat_destroy(&AT);
            return 1;
        }

        /* norm1(A) should equal norm_inf(A^T) */
        if (!lmmc_test_nearly_equal(norm1_A, norm_inf_AT, eps)) {
            printf("    norm1(A) != norm_inf(A^T) at iter %d: %g vs %g\n",
                   i, norm1_A, norm_inf_AT);
            lmmc_mat_destroy(&A); lmmc_mat_destroy(&AT);
            return 1;
        }

        /* Both should be non-negative */
        if (norm1_A < 0.0 || norm_inf_AT < 0.0) {
            printf("    Negative norm at iter %d: norm1=%g, norm_inf=%g\n",
                   i, norm1_A, norm_inf_AT);
            lmmc_mat_destroy(&A); lmmc_mat_destroy(&AT);
            return 1;
        }

        lmmc_mat_destroy(&A);
        lmmc_mat_destroy(&AT);
    }
    return 0;
}

static int test_property15_mat_rank_upper_bound(void)
{
    int i;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        size_t rows = (size_t)(rand() % 5) + 1;
        size_t cols = (size_t)(rand() % 5) + 1;
        size_t min_dim = rows < cols ? rows : cols;
        lmmc_mat_t A;
        lmmc_status_t st;
        size_t rank;
        size_t r, c;

        st = lmmc_mat_create(rows, cols, &A);
        if (st != LMMC_STATUS_OK) { printf("    mat_create(A) failed at iter %d\n", i); return 1; }

        /* Fill A with random values */
        for (r = 0; r < rows; r++) {
            for (c = 0; c < cols; c++) {
                A.data[r * A.stride + c] = rand_double(-10.0, 10.0);
            }
        }

        /* Compute rank with default tolerance (tol <= 0) */
        st = lmmc_mat_rank(&A, -1.0, &rank);
        if (st != LMMC_STATUS_OK) {
            printf("    mat_rank failed at iter %d, status=%d\n", i, (int)st);
            lmmc_mat_destroy(&A);
            return 1;
        }

        /* rank should be <= min(rows, cols) */
        if (rank > min_dim) {
            printf("    rank(%zu x %zu) = %zu > min(%zu, %zu) = %zu at iter %d\n",
                   rows, cols, rank, rows, cols, min_dim, i);
            lmmc_mat_destroy(&A);
            return 1;
        }

        lmmc_mat_destroy(&A);
    }
    return 0;
}

/* ---- Unit Tests ---- */

/* Unit test: mat_pow with n=0 returns identity */
static int test_unit_pow_zero(void)
{
    lmmc_mat_t A, result;
    lmmc_status_t st;
    double eps = 1e-15;
    size_t r, c;

    st = lmmc_mat_create(3, 3, &A);
    if (st != LMMC_STATUS_OK) { printf("    mat_create(A) failed\n"); return 1; }
    st = lmmc_mat_create(3, 3, &result);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&A); return 1; }

    /* Fill A with arbitrary values */
    for (r = 0; r < 3; r++) {
        for (c = 0; c < 3; c++) {
            A.data[r * A.stride + c] = (double)(r * 3 + c + 1);
        }
    }

    /* A^0 = I */
    st = lmmc_mat_pow(&A, 0, &result);
    if (st != LMMC_STATUS_OK) {
        printf("    mat_pow(A, 0) failed, status=%d\n", (int)st);
        lmmc_mat_destroy(&A); lmmc_mat_destroy(&result);
        return 1;
    }

    /* Check result is identity */
    for (r = 0; r < 3; r++) {
        for (c = 0; c < 3; c++) {
            double expected = (r == c) ? 1.0 : 0.0;
            double got = result.data[r * result.stride + c];
            if (!lmmc_test_nearly_equal(got, expected, eps)) {
                printf("    A^0[%zu][%zu] = %g, expected %g\n", r, c, got, expected);
                lmmc_mat_destroy(&A); lmmc_mat_destroy(&result);
                return 1;
            }
        }
    }

    lmmc_mat_destroy(&A);
    lmmc_mat_destroy(&result);
    return 0;
}

/* Unit test: mat_pow with n=1 returns copy of A */
static int test_unit_pow_one(void)
{
    lmmc_mat_t A, result;
    lmmc_status_t st;
    double eps = 1e-15;
    size_t r, c;

    st = lmmc_mat_create(3, 3, &A);
    if (st != LMMC_STATUS_OK) { printf("    mat_create(A) failed\n"); return 1; }
    st = lmmc_mat_create(3, 3, &result);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&A); return 1; }

    /* Fill A with known values */
    for (r = 0; r < 3; r++) {
        for (c = 0; c < 3; c++) {
            A.data[r * A.stride + c] = (double)(r * 3 + c + 1);
        }
    }

    /* A^1 = A */
    st = lmmc_mat_pow(&A, 1, &result);
    if (st != LMMC_STATUS_OK) {
        printf("    mat_pow(A, 1) failed, status=%d\n", (int)st);
        lmmc_mat_destroy(&A); lmmc_mat_destroy(&result);
        return 1;
    }

    /* Check result equals A */
    for (r = 0; r < 3; r++) {
        for (c = 0; c < 3; c++) {
            double expected = A.data[r * A.stride + c];
            double got = result.data[r * result.stride + c];
            if (!lmmc_test_nearly_equal(got, expected, eps)) {
                printf("    A^1[%zu][%zu] = %g, expected %g\n", r, c, got, expected);
                lmmc_mat_destroy(&A); lmmc_mat_destroy(&result);
                return 1;
            }
        }
    }

    lmmc_mat_destroy(&A);
    lmmc_mat_destroy(&result);
    return 0;
}

/* Unit test: mat_pow with singular matrix and negative exponent */
static int test_unit_pow_singular_negative(void)
{
    lmmc_mat_t A, result;
    lmmc_status_t st;

    st = lmmc_mat_create(2, 2, &A);
    if (st != LMMC_STATUS_OK) { printf("    mat_create(A) failed\n"); return 1; }
    st = lmmc_mat_create(2, 2, &result);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&A); return 1; }

    /* Singular matrix: row 2 = 2 * row 1 */
    A.data[0 * A.stride + 0] = 1.0; A.data[0 * A.stride + 1] = 2.0;
    A.data[1 * A.stride + 0] = 2.0; A.data[1 * A.stride + 1] = 4.0;

    /* A^{-1} should fail with SINGULAR_MATRIX */
    st = lmmc_mat_pow(&A, -1, &result);
    if (st != LMMC_STATUS_SINGULAR_MATRIX) {
        printf("    Expected SINGULAR_MATRIX for singular A^{-1}, got %d\n", (int)st);
        lmmc_mat_destroy(&A); lmmc_mat_destroy(&result);
        return 1;
    }

    lmmc_mat_destroy(&A);
    lmmc_mat_destroy(&result);
    return 0;
}

/* Unit test: rank of zero matrix is 0 */
static int test_unit_rank_zero_matrix(void)
{
    lmmc_mat_t A;
    lmmc_status_t st;
    size_t rank;

    st = lmmc_mat_create(3, 4, &A);
    if (st != LMMC_STATUS_OK) { printf("    mat_create(A) failed\n"); return 1; }

    /* A is already zero-initialized by lmmc_mat_create */
    st = lmmc_mat_rank(&A, -1.0, &rank);
    if (st != LMMC_STATUS_OK) {
        printf("    mat_rank failed for zero matrix, status=%d\n", (int)st);
        lmmc_mat_destroy(&A);
        return 1;
    }

    if (rank != 0) {
        printf("    rank(zero matrix) = %zu, expected 0\n", rank);
        lmmc_mat_destroy(&A);
        return 1;
    }

    lmmc_mat_destroy(&A);
    return 0;
}

/* Unit test: rank of identity matrix is n */
static int test_unit_rank_identity(void)
{
    lmmc_mat_t I;
    lmmc_status_t st;
    size_t rank;

    st = lmmc_mat_identity(4, &I);
    if (st != LMMC_STATUS_OK) { printf("    mat_identity failed\n"); return 1; }

    st = lmmc_mat_rank(&I, -1.0, &rank);
    if (st != LMMC_STATUS_OK) {
        printf("    mat_rank failed for identity, status=%d\n", (int)st);
        lmmc_mat_destroy(&I);
        return 1;
    }

    if (rank != 4) {
        printf("    rank(I_4) = %zu, expected 4\n", rank);
        lmmc_mat_destroy(&I);
        return 1;
    }

    lmmc_mat_destroy(&I);
    return 0;
}

/* Unit test: cross product dimension error (size != 3) */
static int test_unit_cross_dimension_error(void)
{
    lmmc_vec_t a, b, c;
    lmmc_status_t st;

    st = lmmc_vec_create(4, &a);
    if (st != LMMC_STATUS_OK) { printf("    vec_create(a) failed\n"); return 1; }
    st = lmmc_vec_create(3, &b);
    if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&a); return 1; }
    st = lmmc_vec_create(3, &c);
    if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&a); lmmc_vec_destroy(&b); return 1; }

    /* a has size 4, should fail */
    st = lmmc_vec_cross(&a, &b, &c);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        printf("    Expected DIMENSION_MISMATCH for size-4 cross, got %d\n", (int)st);
        lmmc_vec_destroy(&a); lmmc_vec_destroy(&b); lmmc_vec_destroy(&c);
        return 1;
    }

    lmmc_vec_destroy(&a);
    lmmc_vec_destroy(&b);
    lmmc_vec_destroy(&c);
    return 0;
}

/* ---- Main ---- */

int main(void)
{
    srand(12345);
    lmmc_init();

    printf("=== Extended Matrix Algebra Property Tests ===\n");
    REPORT("Property 10: Cross product orthogonality",
           test_property10_cross_product_orthogonality());
    REPORT("Property 11: Cross product self-annihilation",
           test_property11_cross_product_self_annihilation());
    REPORT("Property 12: Matrix power exponent addition",
           test_property12_mat_pow_exponent_addition());
    REPORT("Property 13: Matrix right-division round-trip",
           test_property13_mat_rdiv_roundtrip());
    REPORT("Property 14: Matrix norm transpose duality",
           test_property14_mat_norm_transpose_duality());
    REPORT("Property 15: Matrix rank upper bound",
           test_property15_mat_rank_upper_bound());

    printf("\n=== Extended Matrix Algebra Unit Tests ===\n");
    REPORT("Unit: mat_pow n=0 returns identity",
           test_unit_pow_zero());
    REPORT("Unit: mat_pow n=1 returns copy",
           test_unit_pow_one());
    REPORT("Unit: mat_pow singular matrix negative exponent",
           test_unit_pow_singular_negative());
    REPORT("Unit: rank of zero matrix is 0",
           test_unit_rank_zero_matrix());
    REPORT("Unit: rank of identity matrix is n",
           test_unit_rank_identity());
    REPORT("Unit: cross product dimension error",
           test_unit_cross_dimension_error());

    printf("\n=== Results: %d/%d passed ===\n",
           test_count - test_failures, test_count);

    lmmc_deinit();
    return test_failures > 0 ? 1 : 0;
}
