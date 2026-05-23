/**
 * @file test_internal.c
 * @brief Property-based tests for src/internal.h utility functions.
 *
 * Property 1: Integer overflow detection correctness
 *   For any two size_t values a and b, if a*b would overflow size_t,
 *   lmmc_safe_mul_size should return failure (0); if no overflow,
 *   return success (1) with result == a*b. Same for addition.
 *
 * Property 2: Real utility function mathematical properties
 *   For any lmmc_real_t values: abs(x) >= 0, max(a,b) >= a and >= b,
 *   min(a,b) <= a and <= b, clamp(x, lo, hi) is in [lo, hi],
 *   swap exchanges values.
 *
 * Validates: Requirements 1.1, 1.3
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "lmmc/config.h"
#include "../src/internal.h"

/* Number of random iterations for property tests */
#define NUM_ITERATIONS 200

static int test_failures = 0;

#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("FAIL: " msg "\n", ##__VA_ARGS__); \
        test_failures++; \
        return 1; \
    } \
} while (0)

/* Simple PRNG helper to generate random size_t values */
static size_t rand_size(void)
{
    size_t val = 0;
    size_t i;
    /* Build a random size_t from multiple rand() calls */
    for (i = 0; i < sizeof(size_t); i++) {
        val = (val << 8) | (size_t)(rand() & 0xFF);
    }
    return val;
}

/* Generate a random double in [-range, range] */
static double rand_double(double range)
{
    return ((double)rand() / (double)RAND_MAX) * 2.0 * range - range;
}

/* Generate a random non-negative double in [0, range] */
static double rand_positive(double range)
{
    return ((double)rand() / (double)RAND_MAX) * range;
}

/* ========================================================================
 * Property 1: Integer overflow detection correctness
 * Validates: Requirements 1.1
 * ======================================================================== */

/**
 * Test: lmmc_safe_mul_size overflow detection near SIZE_MAX boundaries.
 */
static int test_safe_mul_overflow(void)
{
    int i;
    size_t result;
    int ret;

    /* Known overflow cases */
    ret = lmmc_safe_mul_size(SIZE_MAX, 2, &result);
    CHECK(ret == 0, "SIZE_MAX * 2 should overflow");

    ret = lmmc_safe_mul_size(SIZE_MAX, SIZE_MAX, &result);
    CHECK(ret == 0, "SIZE_MAX * SIZE_MAX should overflow");

    ret = lmmc_safe_mul_size(SIZE_MAX / 2 + 1, 2, &result);
    CHECK(ret == 0, "(SIZE_MAX/2 + 1) * 2 should overflow");

    ret = lmmc_safe_mul_size(SIZE_MAX / 2 + 2, 2, &result);
    CHECK(ret == 0, "(SIZE_MAX/2 + 2) * 2 should overflow");

    /* Boundary: SIZE_MAX / 2 * 2 should NOT overflow (since SIZE_MAX is odd, SIZE_MAX/2 rounds down) */
    ret = lmmc_safe_mul_size(SIZE_MAX / 2, 2, &result);
    CHECK(ret == 1, "(SIZE_MAX/2) * 2 should not overflow");
    CHECK(result == (SIZE_MAX / 2) * 2, "(SIZE_MAX/2) * 2 result mismatch");

    /* Random property: for random a, b, check consistency */
    for (i = 0; i < NUM_ITERATIONS; i++) {
        size_t a = rand_size();
        size_t b = rand_size();

        ret = lmmc_safe_mul_size(a, b, &result);

        if (a == 0 || b == 0) {
            /* Zero case: never overflows */
            CHECK(ret == 1, "mul(%zu, %zu) should succeed (zero operand)", a, b);
            CHECK(result == 0, "mul(%zu, %zu) result should be 0", a, b);
        } else if (a > SIZE_MAX / b) {
            /* Overflow case */
            CHECK(ret == 0, "mul(%zu, %zu) should report overflow", a, b);
        } else {
            /* No overflow case */
            CHECK(ret == 1, "mul(%zu, %zu) should succeed", a, b);
            CHECK(result == a * b, "mul(%zu, %zu) result mismatch", a, b);
        }
    }

    return 0;
}

/**
 * Test: lmmc_safe_mul_size normal (non-overflow) cases.
 */
static int test_safe_mul_normal(void)
{
    size_t result;
    int ret;

    /* Zero cases */
    ret = lmmc_safe_mul_size(0, 0, &result);
    CHECK(ret == 1 && result == 0, "0 * 0 should be 0");

    ret = lmmc_safe_mul_size(0, SIZE_MAX, &result);
    CHECK(ret == 1 && result == 0, "0 * SIZE_MAX should be 0");

    ret = lmmc_safe_mul_size(SIZE_MAX, 0, &result);
    CHECK(ret == 1 && result == 0, "SIZE_MAX * 0 should be 0");

    /* Identity */
    ret = lmmc_safe_mul_size(1, SIZE_MAX, &result);
    CHECK(ret == 1 && result == SIZE_MAX, "1 * SIZE_MAX should be SIZE_MAX");

    ret = lmmc_safe_mul_size(SIZE_MAX, 1, &result);
    CHECK(ret == 1 && result == SIZE_MAX, "SIZE_MAX * 1 should be SIZE_MAX");

    /* Small values */
    ret = lmmc_safe_mul_size(100, 200, &result);
    CHECK(ret == 1 && result == 20000, "100 * 200 should be 20000");

    return 0;
}

/**
 * Test: lmmc_safe_add_size overflow detection near SIZE_MAX boundaries.
 */
static int test_safe_add_overflow(void)
{
    int i;
    size_t result;
    int ret;

    /* Known overflow cases */
    ret = lmmc_safe_add_size(SIZE_MAX, 1, &result);
    CHECK(ret == 0, "SIZE_MAX + 1 should overflow");

    ret = lmmc_safe_add_size(SIZE_MAX, SIZE_MAX, &result);
    CHECK(ret == 0, "SIZE_MAX + SIZE_MAX should overflow");

    ret = lmmc_safe_add_size(SIZE_MAX / 2 + 1, SIZE_MAX / 2 + 1, &result);
    CHECK(ret == 0, "(SIZE_MAX/2+1) + (SIZE_MAX/2+1) should overflow");

    /* Boundary: SIZE_MAX + 0 should NOT overflow */
    ret = lmmc_safe_add_size(SIZE_MAX, 0, &result);
    CHECK(ret == 1, "SIZE_MAX + 0 should not overflow");
    CHECK(result == SIZE_MAX, "SIZE_MAX + 0 result mismatch");

    /* Boundary: (SIZE_MAX - 1) + 1 should NOT overflow */
    ret = lmmc_safe_add_size(SIZE_MAX - 1, 1, &result);
    CHECK(ret == 1, "(SIZE_MAX-1) + 1 should not overflow");
    CHECK(result == SIZE_MAX, "(SIZE_MAX-1) + 1 result mismatch");

    /* Random property: for random a, b, check consistency */
    for (i = 0; i < NUM_ITERATIONS; i++) {
        size_t a = rand_size();
        size_t b = rand_size();

        ret = lmmc_safe_add_size(a, b, &result);

        if (a > SIZE_MAX - b) {
            /* Overflow case */
            CHECK(ret == 0, "add(%zu, %zu) should report overflow", a, b);
        } else {
            /* No overflow case */
            CHECK(ret == 1, "add(%zu, %zu) should succeed", a, b);
            CHECK(result == a + b, "add(%zu, %zu) result mismatch", a, b);
        }
    }

    return 0;
}

/**
 * Test: lmmc_safe_add_size normal (non-overflow) cases.
 */
static int test_safe_add_normal(void)
{
    size_t result;
    int ret;

    /* Zero cases */
    ret = lmmc_safe_add_size(0, 0, &result);
    CHECK(ret == 1 && result == 0, "0 + 0 should be 0");

    ret = lmmc_safe_add_size(0, 42, &result);
    CHECK(ret == 1 && result == 42, "0 + 42 should be 42");

    ret = lmmc_safe_add_size(42, 0, &result);
    CHECK(ret == 1 && result == 42, "42 + 0 should be 42");

    /* Small values */
    ret = lmmc_safe_add_size(100, 200, &result);
    CHECK(ret == 1 && result == 300, "100 + 200 should be 300");

    return 0;
}

/* ========================================================================
 * Property 2: Real utility function mathematical properties
 * Validates: Requirements 1.3
 * ======================================================================== */

/**
 * Test: lmmc_abs(x) >= 0 for all x, including negative values and zero.
 */
static int test_abs_property(void)
{
    int i;

    /* Edge cases */
    CHECK(lmmc_abs(0.0) == 0.0, "abs(0) should be 0");
    CHECK(lmmc_abs(-0.0) == 0.0, "abs(-0) should be 0");
    CHECK(lmmc_abs(1.0) == 1.0, "abs(1) should be 1");
    CHECK(lmmc_abs(-1.0) == 1.0, "abs(-1) should be 1");
    CHECK(lmmc_abs(-123.456) == 123.456, "abs(-123.456) should be 123.456");

    /* Property: abs(x) >= 0 for random values */
    for (i = 0; i < NUM_ITERATIONS; i++) {
        lmmc_real_t x = rand_double(1e10);
        lmmc_real_t ax = lmmc_abs(x);
        CHECK(ax >= 0.0, "abs(%g) = %g should be >= 0", x, ax);

        /* Also: abs(x) == abs(-x) */
        CHECK(ax == lmmc_abs(-x), "abs(%g) should equal abs(%g)", x, -x);
    }

    return 0;
}

/**
 * Test: lmmc_max(a, b) >= a and lmmc_max(a, b) >= b.
 */
static int test_max_property(void)
{
    int i;

    /* Edge cases */
    CHECK(lmmc_max(0.0, 0.0) == 0.0, "max(0,0) should be 0");
    CHECK(lmmc_max(-1.0, 1.0) == 1.0, "max(-1,1) should be 1");
    CHECK(lmmc_max(1.0, -1.0) == 1.0, "max(1,-1) should be 1");
    CHECK(lmmc_max(-5.0, -3.0) == -3.0, "max(-5,-3) should be -3");

    /* Property: max(a,b) >= a and max(a,b) >= b */
    for (i = 0; i < NUM_ITERATIONS; i++) {
        lmmc_real_t a = rand_double(1e10);
        lmmc_real_t b = rand_double(1e10);
        lmmc_real_t m = lmmc_max(a, b);

        CHECK(m >= a, "max(%g, %g) = %g should be >= %g", a, b, m, a);
        CHECK(m >= b, "max(%g, %g) = %g should be >= %g", a, b, m, b);

        /* max(a,b) should equal either a or b */
        CHECK(m == a || m == b, "max(%g, %g) = %g should equal one of them", a, b, m);
    }

    return 0;
}

/**
 * Test: lmmc_min(a, b) <= a and lmmc_min(a, b) <= b.
 */
static int test_min_property(void)
{
    int i;

    /* Edge cases */
    CHECK(lmmc_min(0.0, 0.0) == 0.0, "min(0,0) should be 0");
    CHECK(lmmc_min(-1.0, 1.0) == -1.0, "min(-1,1) should be -1");
    CHECK(lmmc_min(1.0, -1.0) == -1.0, "min(1,-1) should be -1");
    CHECK(lmmc_min(-5.0, -3.0) == -5.0, "min(-5,-3) should be -5");

    /* Property: min(a,b) <= a and min(a,b) <= b */
    for (i = 0; i < NUM_ITERATIONS; i++) {
        lmmc_real_t a = rand_double(1e10);
        lmmc_real_t b = rand_double(1e10);
        lmmc_real_t m = lmmc_min(a, b);

        CHECK(m <= a, "min(%g, %g) = %g should be <= %g", a, b, m, a);
        CHECK(m <= b, "min(%g, %g) = %g should be <= %g", a, b, m, b);

        /* min(a,b) should equal either a or b */
        CHECK(m == a || m == b, "min(%g, %g) = %g should equal one of them", a, b, m);
    }

    return 0;
}

/**
 * Test: lmmc_clamp(x, lo, hi) is in [lo, hi] for all x.
 */
static int test_clamp_property(void)
{
    int i;

    /* Edge cases */
    CHECK(lmmc_clamp(5.0, 0.0, 10.0) == 5.0, "clamp(5, 0, 10) should be 5");
    CHECK(lmmc_clamp(-5.0, 0.0, 10.0) == 0.0, "clamp(-5, 0, 10) should be 0");
    CHECK(lmmc_clamp(15.0, 0.0, 10.0) == 10.0, "clamp(15, 0, 10) should be 10");
    CHECK(lmmc_clamp(0.0, 0.0, 0.0) == 0.0, "clamp(0, 0, 0) should be 0");
    CHECK(lmmc_clamp(-100.0, -50.0, -10.0) == -50.0, "clamp(-100, -50, -10) should be -50");
    CHECK(lmmc_clamp(-30.0, -50.0, -10.0) == -30.0, "clamp(-30, -50, -10) should be -30");
    CHECK(lmmc_clamp(0.0, -50.0, -10.0) == -10.0, "clamp(0, -50, -10) should be -10");

    /* Property: clamp(x, lo, hi) in [lo, hi] */
    for (i = 0; i < NUM_ITERATIONS; i++) {
        lmmc_real_t lo = rand_double(1e5);
        lmmc_real_t hi = lo + rand_positive(1e5); /* ensure hi >= lo */
        lmmc_real_t x = rand_double(2e5);
        lmmc_real_t c = lmmc_clamp(x, lo, hi);

        CHECK(c >= lo, "clamp(%g, %g, %g) = %g should be >= lo", x, lo, hi, c);
        CHECK(c <= hi, "clamp(%g, %g, %g) = %g should be <= hi", x, lo, hi, c);

        /* If x is already in range, clamp should return x */
        if (x >= lo && x <= hi) {
            CHECK(c == x, "clamp(%g, %g, %g) = %g should be x when in range", x, lo, hi, c);
        }
    }

    return 0;
}

/**
 * Test: lmmc_swap exchanges values correctly.
 */
static int test_swap_property(void)
{
    int i;

    /* Edge cases */
    {
        lmmc_real_t a = 1.0, b = 2.0;
        lmmc_swap(&a, &b);
        CHECK(a == 2.0 && b == 1.0, "swap(1,2) should give (2,1)");
    }
    {
        lmmc_real_t a = -5.0, b = 5.0;
        lmmc_swap(&a, &b);
        CHECK(a == 5.0 && b == -5.0, "swap(-5,5) should give (5,-5)");
    }
    {
        lmmc_real_t a = 0.0, b = 0.0;
        lmmc_swap(&a, &b);
        CHECK(a == 0.0 && b == 0.0, "swap(0,0) should give (0,0)");
    }

    /* Property: after swap, values are exchanged */
    for (i = 0; i < NUM_ITERATIONS; i++) {
        lmmc_real_t orig_a = rand_double(1e10);
        lmmc_real_t orig_b = rand_double(1e10);
        lmmc_real_t a = orig_a;
        lmmc_real_t b = orig_b;

        lmmc_swap(&a, &b);

        CHECK(a == orig_b, "after swap, a should be original b (%g != %g)", a, orig_b);
        CHECK(b == orig_a, "after swap, b should be original a (%g != %g)", b, orig_a);
    }

    return 0;
}

/* ========================================================================
 * lmmc_is_finite tests (part of Property 2 coverage)
 * ======================================================================== */

/**
 * Test: lmmc_is_finite returns true for normal values, false for NaN/Inf.
 */
static int test_is_finite_property(void)
{
    int i;

    /* Normal finite values */
    {
        lmmc_real_t val = 0.0;
        CHECK(lmmc_is_finite(&val), "0.0 should be finite");
    }
    {
        lmmc_real_t val = 1.0;
        CHECK(lmmc_is_finite(&val), "1.0 should be finite");
    }
    {
        lmmc_real_t val = -1.0;
        CHECK(lmmc_is_finite(&val), "-1.0 should be finite");
    }
    {
        lmmc_real_t val = 1e308;
        CHECK(lmmc_is_finite(&val), "1e308 should be finite");
    }
    {
        lmmc_real_t val = -1e308;
        CHECK(lmmc_is_finite(&val), "-1e308 should be finite");
    }
    {
        lmmc_real_t val = 1e-308;
        CHECK(lmmc_is_finite(&val), "1e-308 (subnormal) should be finite");
    }

    /* NaN */
    {
        lmmc_real_t val = NAN;
        CHECK(!lmmc_is_finite(&val), "NaN should NOT be finite");
    }
    {
        /* Generate NaN at runtime to avoid MSVC compile-time division error */
        volatile lmmc_real_t zero = 0.0;
        lmmc_real_t val = zero / zero;
        CHECK(!lmmc_is_finite(&val), "0.0/0.0 should NOT be finite");
    }

    /* Infinity */
    {
        lmmc_real_t val = INFINITY;
        CHECK(!lmmc_is_finite(&val), "+Inf should NOT be finite");
    }
    {
        lmmc_real_t val = -INFINITY;
        CHECK(!lmmc_is_finite(&val), "-Inf should NOT be finite");
    }
    {
        /* Generate Inf at runtime to avoid MSVC compile-time division error */
        volatile lmmc_real_t zero = 0.0;
        lmmc_real_t val = 1.0 / zero;
        CHECK(!lmmc_is_finite(&val), "1.0/0.0 should NOT be finite");
    }
    {
        volatile lmmc_real_t zero = 0.0;
        lmmc_real_t val = -1.0 / zero;
        CHECK(!lmmc_is_finite(&val), "-1.0/0.0 should NOT be finite");
    }

    /* Random finite values should all pass */
    for (i = 0; i < NUM_ITERATIONS; i++) {
        lmmc_real_t val = rand_double(1e15);
        CHECK(lmmc_is_finite(&val), "random value %g should be finite", val);
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

    printf("=== Property 1: Integer overflow detection correctness ===\n");
    printf("  Validates: Requirements 1.1\n\n");

    if (test_safe_mul_overflow()) { rc = 1; printf("  [FAIL] safe_mul overflow\n"); }
    else { printf("  [PASS] safe_mul overflow\n"); }

    if (test_safe_mul_normal()) { rc = 1; printf("  [FAIL] safe_mul normal\n"); }
    else { printf("  [PASS] safe_mul normal\n"); }

    if (test_safe_add_overflow()) { rc = 1; printf("  [FAIL] safe_add overflow\n"); }
    else { printf("  [PASS] safe_add overflow\n"); }

    if (test_safe_add_normal()) { rc = 1; printf("  [FAIL] safe_add normal\n"); }
    else { printf("  [PASS] safe_add normal\n"); }

    printf("\n=== Property 2: Real utility function mathematical properties ===\n");
    printf("  Validates: Requirements 1.3\n\n");

    if (test_abs_property()) { rc = 1; printf("  [FAIL] abs property\n"); }
    else { printf("  [PASS] abs property\n"); }

    if (test_max_property()) { rc = 1; printf("  [FAIL] max property\n"); }
    else { printf("  [PASS] max property\n"); }

    if (test_min_property()) { rc = 1; printf("  [FAIL] min property\n"); }
    else { printf("  [PASS] min property\n"); }

    if (test_clamp_property()) { rc = 1; printf("  [FAIL] clamp property\n"); }
    else { printf("  [PASS] clamp property\n"); }

    if (test_swap_property()) { rc = 1; printf("  [FAIL] swap property\n"); }
    else { printf("  [PASS] swap property\n"); }

    if (test_is_finite_property()) { rc = 1; printf("  [FAIL] is_finite property\n"); }
    else { printf("  [PASS] is_finite property\n"); }

    printf("\n");
    if (rc == 0) {
        printf("All internal.h property tests PASSED.\n");
    } else {
        printf("Some internal.h property tests FAILED.\n");
    }

    return rc;
}
