/**
 * @file test_elementwise.c
 * @brief Property-based and unit tests for element-wise vector/matrix operations.
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

/**
 * Generate a random non-zero double in [lo, hi] with |value| >= min_abs.
 */
static double rand_nonzero(double lo, double hi, double min_abs)
{
    double v;
    do {
        v = rand_double(lo, hi);
    } while (fabs(v) < min_abs);
    return v;
}

static int test_property7_vec_hadamard_div_roundtrip(void)
{
    int i;
    double eps = 1e-10;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        size_t size = (size_t)(rand() % 20) + 1;
        size_t j;
        lmmc_vec_t a, b, product, recovered;
        lmmc_status_t st;

        st = lmmc_vec_create(size, &a);
        if (st != LMMC_STATUS_OK) { printf("    vec_create(a) failed at iter %d\n", i); return 1; }
        st = lmmc_vec_create(size, &b);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&a); printf("    vec_create(b) failed at iter %d\n", i); return 1; }
        st = lmmc_vec_create(size, &product);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&a); lmmc_vec_destroy(&b); printf("    vec_create(product) failed at iter %d\n", i); return 1; }
        st = lmmc_vec_create(size, &recovered);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&a); lmmc_vec_destroy(&b); lmmc_vec_destroy(&product); printf("    vec_create(recovered) failed at iter %d\n", i); return 1; }

        /* Fill a with random values in [-10, 10] */
        for (j = 0; j < size; j++) {
            a.data[j] = rand_double(-10.0, 10.0);
        }

        /* Fill b with non-zero values in [0.1, 10.0] (positive to avoid sign issues) */
        for (j = 0; j < size; j++) {
            b.data[j] = rand_nonzero(0.1, 10.0, 0.1);
        }

        /* product = hadamard(a, b) */
        st = lmmc_vec_hadamard(&a, &b, &product);
        if (st != LMMC_STATUS_OK) {
            printf("    vec_hadamard failed at iter %d, status=%d\n", i, (int)st);
            lmmc_vec_destroy(&a); lmmc_vec_destroy(&b);
            lmmc_vec_destroy(&product); lmmc_vec_destroy(&recovered);
            return 1;
        }

        /* recovered = elementwise_div(product, b) */
        st = lmmc_vec_elementwise_div(&product, &b, &recovered);
        if (st != LMMC_STATUS_OK) {
            printf("    vec_elementwise_div failed at iter %d, status=%d\n", i, (int)st);
            lmmc_vec_destroy(&a); lmmc_vec_destroy(&b);
            lmmc_vec_destroy(&product); lmmc_vec_destroy(&recovered);
            return 1;
        }

        /* Check recovered ~ a */
        for (j = 0; j < size; j++) {
            if (!lmmc_test_nearly_equal(recovered.data[j], a.data[j], eps)) {
                printf("    Round-trip failed at iter %d, elem %zu: got %g, expected %g\n",
                       i, j, recovered.data[j], a.data[j]);
                lmmc_vec_destroy(&a); lmmc_vec_destroy(&b);
                lmmc_vec_destroy(&product); lmmc_vec_destroy(&recovered);
                return 1;
            }
        }

        lmmc_vec_destroy(&a);
        lmmc_vec_destroy(&b);
        lmmc_vec_destroy(&product);
        lmmc_vec_destroy(&recovered);
    }
    return 0;
}

static int test_property8_mat_hadamard_div_roundtrip(void)
{
    int i;
    double eps = 1e-10;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        size_t rows = (size_t)(rand() % 10) + 1;
        size_t cols = (size_t)(rand() % 10) + 1;
        size_t r, c;
        lmmc_mat_t a, b, product, recovered;
        lmmc_status_t st;

        st = lmmc_mat_create(rows, cols, &a);
        if (st != LMMC_STATUS_OK) { printf("    mat_create(a) failed at iter %d\n", i); return 1; }
        st = lmmc_mat_create(rows, cols, &b);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); printf("    mat_create(b) failed at iter %d\n", i); return 1; }
        st = lmmc_mat_create(rows, cols, &product);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); lmmc_mat_destroy(&b); printf("    mat_create(product) failed at iter %d\n", i); return 1; }
        st = lmmc_mat_create(rows, cols, &recovered);
        if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); lmmc_mat_destroy(&b); lmmc_mat_destroy(&product); printf("    mat_create(recovered) failed at iter %d\n", i); return 1; }

        /* Fill a with random values in [-10, 10] */
        for (r = 0; r < rows; r++) {
            for (c = 0; c < cols; c++) {
                a.data[r * a.stride + c] = rand_double(-10.0, 10.0);
            }
        }

        /* Fill b with non-zero values in [0.1, 10.0] */
        for (r = 0; r < rows; r++) {
            for (c = 0; c < cols; c++) {
                b.data[r * b.stride + c] = rand_nonzero(0.1, 10.0, 0.1);
            }
        }

        /* product = hadamard(a, b) */
        st = lmmc_mat_hadamard(&a, &b, &product);
        if (st != LMMC_STATUS_OK) {
            printf("    mat_hadamard failed at iter %d, status=%d\n", i, (int)st);
            lmmc_mat_destroy(&a); lmmc_mat_destroy(&b);
            lmmc_mat_destroy(&product); lmmc_mat_destroy(&recovered);
            return 1;
        }

        /* recovered = elementwise_div(product, b) */
        st = lmmc_mat_elementwise_div(&product, &b, &recovered);
        if (st != LMMC_STATUS_OK) {
            printf("    mat_elementwise_div failed at iter %d, status=%d\n", i, (int)st);
            lmmc_mat_destroy(&a); lmmc_mat_destroy(&b);
            lmmc_mat_destroy(&product); lmmc_mat_destroy(&recovered);
            return 1;
        }

        /* Check recovered ~ a */
        for (r = 0; r < rows; r++) {
            for (c = 0; c < cols; c++) {
                double got = recovered.data[r * recovered.stride + c];
                double expected = a.data[r * a.stride + c];
                if (!lmmc_test_nearly_equal(got, expected, eps)) {
                    printf("    Round-trip failed at iter %d, elem [%zu][%zu]: got %g, expected %g\n",
                           i, r, c, got, expected);
                    lmmc_mat_destroy(&a); lmmc_mat_destroy(&b);
                    lmmc_mat_destroy(&product); lmmc_mat_destroy(&recovered);
                    return 1;
                }
            }
        }

        lmmc_mat_destroy(&a);
        lmmc_mat_destroy(&b);
        lmmc_mat_destroy(&product);
        lmmc_mat_destroy(&recovered);
    }
    return 0;
}

static int test_property9_vec_cmp_complementarity(void)
{
    int i;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        size_t size = (size_t)(rand() % 20) + 1;
        size_t j;
        lmmc_vec_t a, b;
        int* gt_out = NULL;
        int* le_out = NULL;
        lmmc_status_t st;

        st = lmmc_vec_create(size, &a);
        if (st != LMMC_STATUS_OK) { printf("    vec_create(a) failed at iter %d\n", i); return 1; }
        st = lmmc_vec_create(size, &b);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&a); printf("    vec_create(b) failed at iter %d\n", i); return 1; }

        /* Fill a and b with random values in [-10, 10] */
        for (j = 0; j < size; j++) {
            a.data[j] = rand_double(-10.0, 10.0);
            b.data[j] = rand_double(-10.0, 10.0);
        }

        /* Allocate output arrays */
        gt_out = (int*)malloc(size * sizeof(int));
        le_out = (int*)malloc(size * sizeof(int));
        if (gt_out == NULL || le_out == NULL) {
            printf("    malloc failed at iter %d\n", i);
            free(gt_out); free(le_out);
            lmmc_vec_destroy(&a); lmmc_vec_destroy(&b);
            return 1;
        }

        /* Compute cmp_gt(a, b) */
        st = lmmc_vec_cmp_gt(&a, &b, gt_out);
        if (st != LMMC_STATUS_OK) {
            printf("    vec_cmp_gt failed at iter %d, status=%d\n", i, (int)st);
            free(gt_out); free(le_out);
            lmmc_vec_destroy(&a); lmmc_vec_destroy(&b);
            return 1;
        }

        /* Compute cmp_le(a, b) */
        st = lmmc_vec_cmp_le(&a, &b, le_out);
        if (st != LMMC_STATUS_OK) {
            printf("    vec_cmp_le failed at iter %d, status=%d\n", i, (int)st);
            free(gt_out); free(le_out);
            lmmc_vec_destroy(&a); lmmc_vec_destroy(&b);
            return 1;
        }

        /* Check complementarity: gt[i] + le[i] == 1 for all i */
        for (j = 0; j < size; j++) {
            if (gt_out[j] + le_out[j] != 1) {
                printf("    Complementarity failed at iter %d, elem %zu: gt=%d, le=%d (a=%g, b=%g)\n",
                       i, j, gt_out[j], le_out[j], a.data[j], b.data[j]);
                free(gt_out); free(le_out);
                lmmc_vec_destroy(&a); lmmc_vec_destroy(&b);
                return 1;
            }
        }

        free(gt_out);
        free(le_out);
        lmmc_vec_destroy(&a);
        lmmc_vec_destroy(&b);
    }
    return 0;
}

/* ---- Unit Tests ---- */

/* Unit test: zero divisor scan for vector elementwise_div */
static int test_unit_zero_divisor_vec(void)
{
    lmmc_vec_t a, b, c;
    lmmc_status_t st;

    st = lmmc_vec_create(4, &a);
    if (st != LMMC_STATUS_OK) { printf("    vec_create(a) failed\n"); return 1; }
    st = lmmc_vec_create(4, &b);
    if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&a); printf("    vec_create(b) failed\n"); return 1; }
    st = lmmc_vec_create(4, &c);
    if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&a); lmmc_vec_destroy(&b); printf("    vec_create(c) failed\n"); return 1; }

    /* Fill a with non-zero values */
    a.data[0] = 1.0; a.data[1] = 2.0; a.data[2] = 3.0; a.data[3] = 4.0;

    /* b has a zero element at index 2 */
    b.data[0] = 1.0; b.data[1] = 2.0; b.data[2] = 0.0; b.data[3] = 4.0;

    st = lmmc_vec_elementwise_div(&a, &b, &c);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        printf("    Expected NUMERICAL_FAILURE for zero divisor, got %d\n", (int)st);
        lmmc_vec_destroy(&a); lmmc_vec_destroy(&b); lmmc_vec_destroy(&c);
        return 1;
    }

    lmmc_vec_destroy(&a);
    lmmc_vec_destroy(&b);
    lmmc_vec_destroy(&c);
    return 0;
}

/* Unit test: zero divisor scan for matrix elementwise_div */
static int test_unit_zero_divisor_mat(void)
{
    lmmc_mat_t a, b, c;
    lmmc_status_t st;

    st = lmmc_mat_create(2, 3, &a);
    if (st != LMMC_STATUS_OK) { printf("    mat_create(a) failed\n"); return 1; }
    st = lmmc_mat_create(2, 3, &b);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); printf("    mat_create(b) failed\n"); return 1; }
    st = lmmc_mat_create(2, 3, &c);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); lmmc_mat_destroy(&b); printf("    mat_create(c) failed\n"); return 1; }

    /* Fill a with values */
    lmmc_mat_fill(&a, 5.0);

    /* Fill b with non-zero values, then set one to zero */
    lmmc_mat_fill(&b, 2.0);
    b.data[1 * b.stride + 1] = 0.0;  /* zero at (1,1) */

    st = lmmc_mat_elementwise_div(&a, &b, &c);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE) {
        printf("    Expected NUMERICAL_FAILURE for zero divisor in matrix, got %d\n", (int)st);
        lmmc_mat_destroy(&a); lmmc_mat_destroy(&b); lmmc_mat_destroy(&c);
        return 1;
    }

    lmmc_mat_destroy(&a);
    lmmc_mat_destroy(&b);
    lmmc_mat_destroy(&c);
    return 0;
}

/* Unit test: dimension mismatch for vector operations */
static int test_unit_dimension_mismatch_vec(void)
{
    lmmc_vec_t a, b, c;
    int out_cmp[5];
    lmmc_status_t st;

    st = lmmc_vec_create(3, &a);
    if (st != LMMC_STATUS_OK) { printf("    vec_create(a) failed\n"); return 1; }
    st = lmmc_vec_create(5, &b);
    if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&a); printf("    vec_create(b) failed\n"); return 1; }
    st = lmmc_vec_create(3, &c);
    if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&a); lmmc_vec_destroy(&b); printf("    vec_create(c) failed\n"); return 1; }

    /* Hadamard with mismatched sizes */
    st = lmmc_vec_hadamard(&a, &b, &c);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        printf("    vec_hadamard expected DIMENSION_MISMATCH, got %d\n", (int)st);
        lmmc_vec_destroy(&a); lmmc_vec_destroy(&b); lmmc_vec_destroy(&c);
        return 1;
    }

    /* elementwise_div with mismatched sizes */
    st = lmmc_vec_elementwise_div(&a, &b, &c);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        printf("    vec_elementwise_div expected DIMENSION_MISMATCH, got %d\n", (int)st);
        lmmc_vec_destroy(&a); lmmc_vec_destroy(&b); lmmc_vec_destroy(&c);
        return 1;
    }

    /* cmp_gt with mismatched sizes */
    st = lmmc_vec_cmp_gt(&a, &b, out_cmp);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        printf("    vec_cmp_gt expected DIMENSION_MISMATCH, got %d\n", (int)st);
        lmmc_vec_destroy(&a); lmmc_vec_destroy(&b); lmmc_vec_destroy(&c);
        return 1;
    }

    /* cmp_le with mismatched sizes */
    st = lmmc_vec_cmp_le(&a, &b, out_cmp);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        printf("    vec_cmp_le expected DIMENSION_MISMATCH, got %d\n", (int)st);
        lmmc_vec_destroy(&a); lmmc_vec_destroy(&b); lmmc_vec_destroy(&c);
        return 1;
    }

    lmmc_vec_destroy(&a);
    lmmc_vec_destroy(&b);
    lmmc_vec_destroy(&c);
    return 0;
}

/* Unit test: dimension mismatch for matrix operations */
static int test_unit_dimension_mismatch_mat(void)
{
    lmmc_mat_t a, b, c;
    lmmc_status_t st;

    st = lmmc_mat_create(2, 3, &a);
    if (st != LMMC_STATUS_OK) { printf("    mat_create(a) failed\n"); return 1; }
    st = lmmc_mat_create(3, 2, &b);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); printf("    mat_create(b) failed\n"); return 1; }
    st = lmmc_mat_create(2, 3, &c);
    if (st != LMMC_STATUS_OK) { lmmc_mat_destroy(&a); lmmc_mat_destroy(&b); printf("    mat_create(c) failed\n"); return 1; }

    /* Hadamard with mismatched dimensions */
    st = lmmc_mat_hadamard(&a, &b, &c);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        printf("    mat_hadamard expected DIMENSION_MISMATCH, got %d\n", (int)st);
        lmmc_mat_destroy(&a); lmmc_mat_destroy(&b); lmmc_mat_destroy(&c);
        return 1;
    }

    /* elementwise_div with mismatched dimensions */
    st = lmmc_mat_elementwise_div(&a, &b, &c);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        printf("    mat_elementwise_div expected DIMENSION_MISMATCH, got %d\n", (int)st);
        lmmc_mat_destroy(&a); lmmc_mat_destroy(&b); lmmc_mat_destroy(&c);
        return 1;
    }

    lmmc_mat_destroy(&a);
    lmmc_mat_destroy(&b);
    lmmc_mat_destroy(&c);
    return 0;
}

/* Unit test: aliasing — hadamard(a, b, a) overwrites input a correctly */
static int test_unit_aliasing(void)
{
    lmmc_vec_t a, b;
    lmmc_status_t st;
    double eps = 1e-15;

    st = lmmc_vec_create(4, &a);
    if (st != LMMC_STATUS_OK) { printf("    vec_create(a) failed\n"); return 1; }
    st = lmmc_vec_create(4, &b);
    if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&a); printf("    vec_create(b) failed\n"); return 1; }

    /* Set known values */
    a.data[0] = 2.0; a.data[1] = 3.0; a.data[2] = 4.0; a.data[3] = 5.0;
    b.data[0] = 1.5; b.data[1] = 2.5; b.data[2] = 3.5; b.data[3] = 4.5;

    /* Expected results: a[i] * b[i] */
    double expected[4];
    expected[0] = 2.0 * 1.5;
    expected[1] = 3.0 * 2.5;
    expected[2] = 4.0 * 3.5;
    expected[3] = 5.0 * 4.5;

    /* hadamard(a, b, a) — output overwrites input a */
    st = lmmc_vec_hadamard(&a, &b, &a);
    if (st != LMMC_STATUS_OK) {
        printf("    vec_hadamard aliasing failed, status=%d\n", (int)st);
        lmmc_vec_destroy(&a); lmmc_vec_destroy(&b);
        return 1;
    }

    /* Verify results */
    {
        size_t j;
        for (j = 0; j < 4; j++) {
            if (!lmmc_test_nearly_equal(a.data[j], expected[j], eps)) {
                printf("    Aliasing result mismatch at elem %zu: got %g, expected %g\n",
                       j, a.data[j], expected[j]);
                lmmc_vec_destroy(&a); lmmc_vec_destroy(&b);
                return 1;
            }
        }
    }

    lmmc_vec_destroy(&a);
    lmmc_vec_destroy(&b);
    return 0;
}

/* ---- Main ---- */

int main(void)
{
    srand(12345);
    lmmc_init();

    printf("=== Element-wise Operations Property Tests ===\n");
    REPORT("Property 7: Vector Hadamard-division round-trip",
           test_property7_vec_hadamard_div_roundtrip());
    REPORT("Property 8: Matrix Hadamard-division round-trip",
           test_property8_mat_hadamard_div_roundtrip());
    REPORT("Property 9: Vector comparison complementarity",
           test_property9_vec_cmp_complementarity());

    printf("\n=== Element-wise Operations Unit Tests ===\n");
    REPORT("Unit: Zero divisor scan (vector)",
           test_unit_zero_divisor_vec());
    REPORT("Unit: Zero divisor scan (matrix)",
           test_unit_zero_divisor_mat());
    REPORT("Unit: Dimension mismatch (vector)",
           test_unit_dimension_mismatch_vec());
    REPORT("Unit: Dimension mismatch (matrix)",
           test_unit_dimension_mismatch_mat());
    REPORT("Unit: Aliasing (hadamard output overwrites input)",
           test_unit_aliasing());

    printf("\n=== Results: %d/%d passed ===\n",
           test_count - test_failures, test_count);

    lmmc_deinit();
    return test_failures > 0 ? 1 : 0;
}
