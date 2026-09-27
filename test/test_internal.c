#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

/**
 * @file test_internal.c
 * 针对 LMMC 中 internal 相关接口的单元测试。
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "lmmc/config.h"
#include "internal.h"

#define NUM_ITERATIONS 200

static size_t rand_size(void) {
    size_t val = 0;
    size_t i;

    for (i = 0; i < sizeof(size_t); i++) {
        val = (val << 8) | (size_t)(rand() & 0xFF);
    }
    return val;
}

static double rand_double(double range) {
    return ((double)rand() / (double)RAND_MAX) * 2.0 * range - range;
}

static double rand_positive(double range) {
    return ((double)rand() / (double)RAND_MAX) * range;
}

static void test_safe_mul_overflow(void **state) {
    (void)state;
    int i;
    size_t result;
    int ret;

    ret = lmmc_safe_mul_size(SIZE_MAX, 2, &result);
    assert_int_equal(ret, 0);

    ret = lmmc_safe_mul_size(SIZE_MAX, SIZE_MAX, &result);
    assert_int_equal(ret, 0);

    ret = lmmc_safe_mul_size(SIZE_MAX / 2 + 1, 2, &result);
    assert_int_equal(ret, 0);

    ret = lmmc_safe_mul_size(SIZE_MAX / 2 + 2, 2, &result);
    assert_int_equal(ret, 0);

    ret = lmmc_safe_mul_size(SIZE_MAX / 2, 2, &result);
    assert_int_equal(ret, 1);
    assert_int_equal(result, (SIZE_MAX / 2) * 2);

    for (i = 0; i < NUM_ITERATIONS; i++) {
        size_t a = rand_size();
        size_t b = rand_size();

        ret = lmmc_safe_mul_size(a, b, &result);

        if (a == 0 || b == 0) {

            assert_int_equal(ret, 1);
            assert_int_equal(result, 0);
        } else if (a > SIZE_MAX / b) {

            assert_int_equal(ret, 0);
        } else {

            assert_int_equal(ret, 1);
            assert_int_equal(result, a * b);
        }
    }
}

static void test_safe_mul_normal(void **state) {
    (void)state;
    size_t result;
    int ret;

    ret = lmmc_safe_mul_size(0, 0, &result);
    assert_int_equal(ret, 1);
    assert_int_equal(result, 0);

    ret = lmmc_safe_mul_size(0, SIZE_MAX, &result);
    assert_int_equal(ret, 1);
    assert_int_equal(result, 0);

    ret = lmmc_safe_mul_size(SIZE_MAX, 0, &result);
    assert_int_equal(ret, 1);
    assert_int_equal(result, 0);

    ret = lmmc_safe_mul_size(1, SIZE_MAX, &result);
    assert_int_equal(ret, 1);
    assert_int_equal(result, SIZE_MAX);

    ret = lmmc_safe_mul_size(SIZE_MAX, 1, &result);
    assert_int_equal(ret, 1);
    assert_int_equal(result, SIZE_MAX);

    ret = lmmc_safe_mul_size(100, 200, &result);
    assert_int_equal(ret, 1);
    assert_int_equal(result, 20000);
}

static void test_safe_add_overflow(void **state) {
    (void)state;
    int i;
    size_t result;
    int ret;

    ret = lmmc_safe_add_size(SIZE_MAX, 1, &result);
    assert_int_equal(ret, 0);

    ret = lmmc_safe_add_size(SIZE_MAX, SIZE_MAX, &result);
    assert_int_equal(ret, 0);

    ret = lmmc_safe_add_size(SIZE_MAX / 2 + 1, SIZE_MAX / 2 + 1, &result);
    assert_int_equal(ret, 0);

    ret = lmmc_safe_add_size(SIZE_MAX, 0, &result);
    assert_int_equal(ret, 1);
    assert_int_equal(result, SIZE_MAX);

    ret = lmmc_safe_add_size(SIZE_MAX - 1, 1, &result);
    assert_int_equal(ret, 1);
    assert_int_equal(result, SIZE_MAX);

    for (i = 0; i < NUM_ITERATIONS; i++) {
        size_t a = rand_size();
        size_t b = rand_size();

        ret = lmmc_safe_add_size(a, b, &result);

        if (a > SIZE_MAX - b) {

            assert_int_equal(ret, 0);
        } else {

            assert_int_equal(ret, 1);
            assert_int_equal(result, a + b);
        }
    }
}

static void test_safe_add_normal(void **state) {
    (void)state;
    size_t result;
    int ret;

    ret = lmmc_safe_add_size(0, 0, &result);
    assert_true(ret == 1 && result == 0);

    ret = lmmc_safe_add_size(0, 42, &result);
    assert_true(ret == 1 && result == 42);

    ret = lmmc_safe_add_size(42, 0, &result);
    assert_true(ret == 1 && result == 42);

    ret = lmmc_safe_add_size(100, 200, &result);
    assert_true(ret == 1 && result == 300);
}

static void test_abs_property(void **state) {
    (void)state;
    int i;

    assert_true(lmmc_abs(0.0) == 0.0);
    assert_true(lmmc_abs(-0.0) == 0.0);
    assert_true(lmmc_abs(1.0) == 1.0);
    assert_true(lmmc_abs(-1.0) == 1.0);
    assert_true(lmmc_abs(-123.456) == 123.456);

    for (i = 0; i < NUM_ITERATIONS; i++) {
        lmmc_real_t x = rand_double(1e10);
        lmmc_real_t ax = lmmc_abs(x);
        assert_true(ax >= 0.0);

        assert_true(ax == lmmc_abs(-x));
    }
}

static void test_max_property(void **state) {
    (void)state;
    int i;

    assert_true(lmmc_max(0.0, 0.0) == 0.0);
    assert_true(lmmc_max(-1.0, 1.0) == 1.0);
    assert_true(lmmc_max(1.0, -1.0) == 1.0);
    assert_true(lmmc_max(-5.0, -3.0) == -3.0);

    for (i = 0; i < NUM_ITERATIONS; i++) {
        lmmc_real_t a = rand_double(1e10);
        lmmc_real_t b = rand_double(1e10);
        lmmc_real_t m = lmmc_max(a, b);

        assert_true(m >= a);
        assert_true(m >= b);

        assert_true(m == a || m == b);
    }
}

static void test_min_property(void **state) {
    (void)state;
    int i;

    assert_true(lmmc_min(0.0, 0.0) == 0.0);
    assert_true(lmmc_min(-1.0, 1.0) == -1.0);
    assert_true(lmmc_min(1.0, -1.0) == -1.0);
    assert_true(lmmc_min(-5.0, -3.0) == -5.0);

    for (i = 0; i < NUM_ITERATIONS; i++) {
        lmmc_real_t a = rand_double(1e10);
        lmmc_real_t b = rand_double(1e10);
        lmmc_real_t m = lmmc_min(a, b);

        assert_true(m <= a);
        assert_true(m <= b);

        assert_true(m == a || m == b);
    }
}

static void test_clamp_property(void **state) {
    (void)state;
    int i;

    assert_true(lmmc_clamp(5.0, 0.0, 10.0) == 5.0);
    assert_true(lmmc_clamp(-5.0, 0.0, 10.0) == 0.0);
    assert_true(lmmc_clamp(15.0, 0.0, 10.0) == 10.0);
    assert_true(lmmc_clamp(0.0, 0.0, 0.0) == 0.0);
    assert_true(lmmc_clamp(-100.0, -50.0, -10.0) == -50.0);
    assert_true(lmmc_clamp(-30.0, -50.0, -10.0) == -30.0);
    assert_true(lmmc_clamp(0.0, -50.0, -10.0) == -10.0);

    for (i = 0; i < NUM_ITERATIONS; i++) {
        lmmc_real_t lo = rand_double(1e5);
        lmmc_real_t hi = lo + rand_positive(1e5);
        lmmc_real_t x = rand_double(2e5);
        lmmc_real_t c = lmmc_clamp(x, lo, hi);

        assert_true(c >= lo);
        assert_true(c <= hi);

        if (x >= lo && x <= hi) {
            assert_true(c == x);
        }
    }
}

static void test_swap_property(void **state) {
    (void)state;
    int i;

    {
        lmmc_real_t a = 1.0, b = 2.0;
        lmmc_swap(&a, &b);
        assert_true(a == 2.0 && b == 1.0);
    }
    {
        lmmc_real_t a = -5.0, b = 5.0;
        lmmc_swap(&a, &b);
        assert_true(a == 5.0 && b == -5.0);
    }
    {
        lmmc_real_t a = 0.0, b = 0.0;
        lmmc_swap(&a, &b);
        assert_true(a == 0.0 && b == 0.0);
    }

    for (i = 0; i < NUM_ITERATIONS; i++) {
        lmmc_real_t orig_a = rand_double(1e10);
        lmmc_real_t orig_b = rand_double(1e10);
        lmmc_real_t a = orig_a;
        lmmc_real_t b = orig_b;

        lmmc_swap(&a, &b);

        assert_true(a == orig_b);
        assert_true(b == orig_a);
    }
}

static void test_is_finite_property(void **state) {
    (void)state;
    int i;

    lmmc_real_t finite_values[] = {0.0, 1.0, -1.0, 1e308, -1e308, 1e-308};
    for (size_t j = 0; j < sizeof(finite_values) / sizeof(finite_values[0]); ++j) {
        assert_true(lmmc_is_finite(&finite_values[j]));
    }

    lmmc_real_t nonfinite_values[] = {NAN, INFINITY, -INFINITY};
    for (size_t j = 0; j < sizeof(nonfinite_values) / sizeof(nonfinite_values[0]); ++j) {
        assert_false(lmmc_is_finite(&nonfinite_values[j]));
    }
    {
        volatile lmmc_real_t zero = 0.0;
        lmmc_real_t val = zero / zero;
        assert_false(lmmc_is_finite(&val));
    }
    {

        volatile lmmc_real_t zero = 0.0;
        lmmc_real_t val = 1.0 / zero;
        assert_false(lmmc_is_finite(&val));
    }
    {
        volatile lmmc_real_t zero = 0.0;
        lmmc_real_t val = -1.0 / zero;
        assert_false(lmmc_is_finite(&val));
    }

    for (i = 0; i < NUM_ITERATIONS; i++) {
        lmmc_real_t val = rand_double(1e15);
        assert_true(lmmc_is_finite(&val));
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_safe_mul_overflow),
        cmocka_unit_test(test_safe_mul_normal),
        cmocka_unit_test(test_safe_add_overflow),
        cmocka_unit_test(test_safe_add_normal),
        cmocka_unit_test(test_abs_property),
        cmocka_unit_test(test_max_property),
        cmocka_unit_test(test_min_property),
        cmocka_unit_test(test_clamp_property),
        cmocka_unit_test(test_swap_property),
        cmocka_unit_test(test_is_finite_property),
    };
    srand(0x494E544Cu);
    return cmocka_run_group_tests(tests, NULL, NULL);
}
