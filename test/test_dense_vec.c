/**
 * @file test_dense_vec.c
 * @brief Property-based tests for BLAS Level 1 vector operations in dense.h.
 *
 * Property 3: 向量 axpy 线性组合正确性
 *   For any vectors x, y and scalar alpha, after lmmc_vec_axpy,
 *   each element y[i] should equal alpha * x[i] + old_y[i].
 *
 * Property 4: 向量范数非负性与一致性
 *   For any non-zero vector x, norm2(x) > 0 and norm_inf(x) > 0;
 *   for zero vector, both are 0.
 *   Also norm_inf(x) <= norm2(x) <= sqrt(n) * norm_inf(x).
 *
 * Property 5: 向量 copy/swap 数据保持
 *   copy(x, dst) makes dst equal to x;
 *   swap(x, y) exchanges their data.
 *
 * Validates: Requirements 2.1, 2.2, 2.4, 2.5, 2.6
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "lmmc/config.h"
#include "lmmc/dense.h"

/* Number of random iterations for property tests */
#define NUM_ITERATIONS 150

/* Tolerance for floating-point comparisons */
#define TOL 1e-12

static int test_failures = 0;

#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("FAIL: " msg "\n", ##__VA_ARGS__); \
        test_failures++; \
        return 1; \
    } \
} while (0)

/* Generate a random double in [-range, range] */
static double rand_double(double range)
{
    return ((double)rand() / (double)RAND_MAX) * 2.0 * range - range;
}

/* Generate a random vector size in [min_size, max_size] */
static size_t rand_vec_size(size_t min_size, size_t max_size)
{
    return min_size + (size_t)(rand() % (int)(max_size - min_size + 1));
}

/* Fill a vector with random values in [-range, range] */
static void fill_random_vec(lmmc_vec_t* v, double range)
{
    for (size_t i = 0; i < v->size; ++i) {
        v->data[i] = rand_double(range);
    }
}

/* Fill a vector with zeros */
static void fill_zero_vec(lmmc_vec_t* v)
{
    for (size_t i = 0; i < v->size; ++i) {
        v->data[i] = 0.0;
    }
}

/* ========================================================================
 * Property 3: 向量 axpy 线性组合正确性
 * Validates: Requirements 2.4
 * ======================================================================== */

/**
 * Test: For random vectors x, y and scalar alpha,
 * after lmmc_vec_axpy(alpha, x, y), y[i] == alpha * x[i] + old_y[i].
 */
static int test_axpy_correctness(void)
{
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t n = rand_vec_size(1, 64);
        lmmc_vec_t x, y;
        lmmc_vec_create(n, &x);
        lmmc_vec_create(n, &y);

        double range = (iter < 10) ? 1e15 : 100.0;
        fill_random_vec(&x, range);
        fill_random_vec(&y, range);

        /* Save original y */
        lmmc_vec_t old_y;
        lmmc_vec_create(n, &old_y);
        for (size_t i = 0; i < n; ++i) {
            old_y.data[i] = y.data[i];
        }

        lmmc_real_t alpha = rand_double(10.0);

        lmmc_status_t status = lmmc_vec_axpy(alpha, &x, &y);
        CHECK(status == LMMC_STATUS_OK,
              "axpy returned error %d (iter=%d, n=%zu)", (int)status, iter, n);

        /* Verify each element */
        for (size_t i = 0; i < n; ++i) {
            double expected = alpha * x.data[i] + old_y.data[i];
            double diff = fabs(y.data[i] - expected);
            double scale = fabs(expected) + 1.0;
            CHECK(diff <= TOL * scale,
                  "axpy mismatch at [%zu]: got %g, expected %g (diff=%g, iter=%d)",
                  i, y.data[i], expected, diff, iter);
        }

        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&y);
        lmmc_vec_destroy(&old_y);
    }
    return 0;
}

/**
 * Test: Edge case - single-element vector axpy.
 */
static int test_axpy_single_element(void)
{
    lmmc_vec_t x, y;
    lmmc_vec_create(1, &x);
    lmmc_vec_create(1, &y);

    x.data[0] = 3.0;
    y.data[0] = 7.0;
    lmmc_real_t alpha = 2.0;

    lmmc_status_t status = lmmc_vec_axpy(alpha, &x, &y);
    CHECK(status == LMMC_STATUS_OK, "axpy single element returned error");
    CHECK(fabs(y.data[0] - 13.0) < TOL,
          "axpy single element: expected 13.0, got %g", y.data[0]);

    lmmc_vec_destroy(&x);
    lmmc_vec_destroy(&y);
    return 0;
}

/**
 * Test: axpy with alpha = 0 should leave y unchanged.
 */
static int test_axpy_alpha_zero(void)
{
    size_t n = 10;
    lmmc_vec_t x, y;
    lmmc_vec_create(n, &x);
    lmmc_vec_create(n, &y);
    fill_random_vec(&x, 100.0);
    fill_random_vec(&y, 100.0);

    /* Save original y */
    double old_y[10];
    for (size_t i = 0; i < n; ++i) old_y[i] = y.data[i];

    lmmc_real_t alpha = 0.0;
    lmmc_status_t status = lmmc_vec_axpy(alpha, &x, &y);
    CHECK(status == LMMC_STATUS_OK, "axpy alpha=0 returned error");

    for (size_t i = 0; i < n; ++i) {
        CHECK(y.data[i] == old_y[i],
              "axpy alpha=0: y[%zu] changed from %g to %g", i, old_y[i], y.data[i]);
    }

    lmmc_vec_destroy(&x);
    lmmc_vec_destroy(&y);
    return 0;
}

/* ========================================================================
 * Property 4: 向量范数非负性与一致性
 * Validates: Requirements 2.1, 2.2
 * ======================================================================== */

/**
 * Test: For any non-zero vector x, norm2(x) > 0 and norm_inf(x) > 0.
 * For zero vector, both are 0.
 * Also: norm_inf(x) <= norm2(x) <= sqrt(n) * norm_inf(x).
 */
static int test_norm_properties(void)
{
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t n = rand_vec_size(1, 64);
        lmmc_vec_t x;
        lmmc_vec_create(n, &x);

        double range = (iter < 10) ? 1e10 : 50.0;
        fill_random_vec(&x, range);

        lmmc_real_t norm2_val, norm_inf_val;
        lmmc_status_t s1 = lmmc_vec_norm2(&x, &norm2_val);
        lmmc_status_t s2 = lmmc_vec_norm_inf(&x, &norm_inf_val);
        CHECK(s1 == LMMC_STATUS_OK, "norm2 returned error %d", (int)s1);
        CHECK(s2 == LMMC_STATUS_OK, "norm_inf returned error %d", (int)s2);

        /* Non-negativity */
        CHECK(norm2_val >= 0.0,
              "norm2 should be >= 0, got %g (iter=%d)", norm2_val, iter);
        CHECK(norm_inf_val >= 0.0,
              "norm_inf should be >= 0, got %g (iter=%d)", norm_inf_val, iter);

        /* Check if vector is non-zero */
        int is_nonzero = 0;
        for (size_t i = 0; i < n; ++i) {
            if (x.data[i] != 0.0) { is_nonzero = 1; break; }
        }

        if (is_nonzero) {
            CHECK(norm2_val > 0.0,
                  "norm2 of non-zero vec should be > 0 (iter=%d)", iter);
            CHECK(norm_inf_val > 0.0,
                  "norm_inf of non-zero vec should be > 0 (iter=%d)", iter);
        }

        /* Consistency: norm_inf <= norm2 <= sqrt(n) * norm_inf */
        double sqrt_n = sqrt((double)n);
        CHECK(norm_inf_val <= norm2_val + TOL * norm2_val,
              "norm_inf (%g) should be <= norm2 (%g) (iter=%d)",
              norm_inf_val, norm2_val, iter);
        CHECK(norm2_val <= sqrt_n * norm_inf_val + TOL * norm2_val,
              "norm2 (%g) should be <= sqrt(n)*norm_inf (%g) (iter=%d, n=%zu)",
              norm2_val, sqrt_n * norm_inf_val, iter, n);

        lmmc_vec_destroy(&x);
    }
    return 0;
}

/**
 * Test: Zero vector norms should both be 0.
 */
static int test_norm_zero_vector(void)
{
    int iter;

    for (iter = 0; iter < 10; iter++) {
        size_t n = rand_vec_size(1, 32);
        lmmc_vec_t x;
        lmmc_vec_create(n, &x);
        fill_zero_vec(&x);

        lmmc_real_t norm2_val, norm_inf_val;
        lmmc_vec_norm2(&x, &norm2_val);
        lmmc_vec_norm_inf(&x, &norm_inf_val);

        CHECK(norm2_val == 0.0,
              "norm2 of zero vec should be 0, got %g (n=%zu)", norm2_val, n);
        CHECK(norm_inf_val == 0.0,
              "norm_inf of zero vec should be 0, got %g (n=%zu)", norm_inf_val, n);

        lmmc_vec_destroy(&x);
    }
    return 0;
}

/**
 * Test: Single-element vector norm consistency.
 */
static int test_norm_single_element(void)
{
    lmmc_vec_t x;
    lmmc_vec_create(1, &x);
    x.data[0] = -5.0;

    lmmc_real_t norm2_val, norm_inf_val;
    lmmc_vec_norm2(&x, &norm2_val);
    lmmc_vec_norm_inf(&x, &norm_inf_val);

    /* For single element, norm2 == norm_inf == |x[0]| */
    CHECK(fabs(norm2_val - 5.0) < TOL,
          "norm2 of [-5] should be 5, got %g", norm2_val);
    CHECK(fabs(norm_inf_val - 5.0) < TOL,
          "norm_inf of [-5] should be 5, got %g", norm_inf_val);

    lmmc_vec_destroy(&x);
    return 0;
}

/* ========================================================================
 * Property 5: 向量 copy/swap 数据保持
 * Validates: Requirements 2.5, 2.6
 * ======================================================================== */

/**
 * Test: copy(x, dst) makes dst equal to x element-wise.
 */
static int test_copy_correctness(void)
{
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t n = rand_vec_size(1, 64);
        lmmc_vec_t x, dst;
        lmmc_vec_create(n, &x);
        lmmc_vec_create(n, &dst);

        double range = (iter < 10) ? 1e15 : 100.0;
        fill_random_vec(&x, range);
        fill_random_vec(&dst, range); /* dst has different initial data */

        lmmc_status_t status = lmmc_vec_copy(&x, &dst);
        CHECK(status == LMMC_STATUS_OK,
              "copy returned error %d (iter=%d)", (int)status, iter);

        for (size_t i = 0; i < n; ++i) {
            CHECK(dst.data[i] == x.data[i],
                  "copy mismatch at [%zu]: dst=%g, x=%g (iter=%d)",
                  i, dst.data[i], x.data[i], iter);
        }

        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&dst);
    }
    return 0;
}

/**
 * Test: swap(x, y) exchanges their data completely.
 */
static int test_swap_correctness(void)
{
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        size_t n = rand_vec_size(1, 64);
        lmmc_vec_t x, y;
        lmmc_vec_create(n, &x);
        lmmc_vec_create(n, &y);

        double range = (iter < 10) ? 1e15 : 100.0;
        fill_random_vec(&x, range);
        fill_random_vec(&y, range);

        /* Save original data */
        double* orig_x = (double*)malloc(n * sizeof(double));
        double* orig_y = (double*)malloc(n * sizeof(double));
        for (size_t i = 0; i < n; ++i) {
            orig_x[i] = x.data[i];
            orig_y[i] = y.data[i];
        }

        lmmc_status_t status = lmmc_vec_swap(&x, &y);
        CHECK(status == LMMC_STATUS_OK,
              "swap returned error %d (iter=%d)", (int)status, iter);

        /* After swap: x should have orig_y, y should have orig_x */
        for (size_t i = 0; i < n; ++i) {
            CHECK(x.data[i] == orig_y[i],
                  "swap: x[%zu]=%g should be orig_y=%g (iter=%d)",
                  i, x.data[i], orig_y[i], iter);
            CHECK(y.data[i] == orig_x[i],
                  "swap: y[%zu]=%g should be orig_x=%g (iter=%d)",
                  i, y.data[i], orig_x[i], iter);
        }

        free(orig_x);
        free(orig_y);
        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&y);
    }
    return 0;
}

/**
 * Test: Edge case - copy/swap with single-element vectors.
 */
static int test_copy_swap_single_element(void)
{
    lmmc_vec_t x, y;
    lmmc_vec_create(1, &x);
    lmmc_vec_create(1, &y);

    x.data[0] = 42.0;
    y.data[0] = -99.0;

    /* Test copy */
    lmmc_vec_t dst;
    lmmc_vec_create(1, &dst);
    dst.data[0] = 0.0;
    lmmc_vec_copy(&x, &dst);
    CHECK(dst.data[0] == 42.0,
          "copy single: dst should be 42.0, got %g", dst.data[0]);

    /* Test swap */
    lmmc_vec_swap(&x, &y);
    CHECK(x.data[0] == -99.0,
          "swap single: x should be -99.0, got %g", x.data[0]);
    CHECK(y.data[0] == 42.0,
          "swap single: y should be 42.0, got %g", y.data[0]);

    lmmc_vec_destroy(&x);
    lmmc_vec_destroy(&y);
    lmmc_vec_destroy(&dst);
    return 0;
}

/**
 * Test: Double swap should restore original data.
 */
static int test_swap_double_swap(void)
{
    int iter;

    for (iter = 0; iter < 50; iter++) {
        size_t n = rand_vec_size(2, 32);
        lmmc_vec_t x, y;
        lmmc_vec_create(n, &x);
        lmmc_vec_create(n, &y);
        fill_random_vec(&x, 100.0);
        fill_random_vec(&y, 100.0);

        /* Save originals */
        double* orig_x = (double*)malloc(n * sizeof(double));
        double* orig_y = (double*)malloc(n * sizeof(double));
        for (size_t i = 0; i < n; ++i) {
            orig_x[i] = x.data[i];
            orig_y[i] = y.data[i];
        }

        /* Swap twice */
        lmmc_vec_swap(&x, &y);
        lmmc_vec_swap(&x, &y);

        /* Should be back to original */
        for (size_t i = 0; i < n; ++i) {
            CHECK(x.data[i] == orig_x[i],
                  "double swap: x[%zu] not restored (iter=%d)", i, iter);
            CHECK(y.data[i] == orig_y[i],
                  "double swap: y[%zu] not restored (iter=%d)", i, iter);
        }

        free(orig_x);
        free(orig_y);
        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&y);
    }
    return 0;
}

/* ========================================================================
 * Main
 * ======================================================================== */

int main(void)
{
    int rc = 0;

    srand((unsigned int)time(NULL));

    printf("=== Property 3: 向量 axpy 线性组合正确性 ===\n");
    printf("  Validates: Requirements 2.4\n\n");

    if (test_axpy_correctness()) { rc = 1; printf("  [FAIL] axpy correctness\n"); }
    else { printf("  [PASS] axpy correctness (%d iterations)\n", NUM_ITERATIONS); }

    if (test_axpy_single_element()) { rc = 1; printf("  [FAIL] axpy single element\n"); }
    else { printf("  [PASS] axpy single element\n"); }

    if (test_axpy_alpha_zero()) { rc = 1; printf("  [FAIL] axpy alpha=0\n"); }
    else { printf("  [PASS] axpy alpha=0\n"); }

    printf("\n=== Property 4: 向量范数非负性与一致性 ===\n");
    printf("  Validates: Requirements 2.1, 2.2\n\n");

    if (test_norm_properties()) { rc = 1; printf("  [FAIL] norm properties\n"); }
    else { printf("  [PASS] norm properties (%d iterations)\n", NUM_ITERATIONS); }

    if (test_norm_zero_vector()) { rc = 1; printf("  [FAIL] norm zero vector\n"); }
    else { printf("  [PASS] norm zero vector\n"); }

    if (test_norm_single_element()) { rc = 1; printf("  [FAIL] norm single element\n"); }
    else { printf("  [PASS] norm single element\n"); }

    printf("\n=== Property 5: 向量 copy/swap 数据保持 ===\n");
    printf("  Validates: Requirements 2.5, 2.6\n\n");

    if (test_copy_correctness()) { rc = 1; printf("  [FAIL] copy correctness\n"); }
    else { printf("  [PASS] copy correctness (%d iterations)\n", NUM_ITERATIONS); }

    if (test_swap_correctness()) { rc = 1; printf("  [FAIL] swap correctness\n"); }
    else { printf("  [PASS] swap correctness (%d iterations)\n", NUM_ITERATIONS); }

    if (test_copy_swap_single_element()) { rc = 1; printf("  [FAIL] copy/swap single\n"); }
    else { printf("  [PASS] copy/swap single element\n"); }

    if (test_swap_double_swap()) { rc = 1; printf("  [FAIL] double swap\n"); }
    else { printf("  [PASS] double swap restore\n"); }

    printf("\n");
    if (rc == 0) {
        printf("All BLAS L1 vector property tests PASSED.\n");
    } else {
        printf("Some BLAS L1 vector property tests FAILED.\n");
    }

    return rc;
}
