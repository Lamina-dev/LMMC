/**
 * @file test_dense_extended_vector_operations.c
 * @brief 稠密向量扩展运算测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#define TEST_EPS_TIGHT 1e-12
#define TEST_EPS_NORMAL 1e-10
#define TEST_EPS_LOOSE 1e-6

struct test_fixture {
    struct test_axpy_resources {
        lmmc_vec_t x;
        lmmc_vec_t y;
    } test_axpy;
    struct test_swap_resources {
        lmmc_vec_t x;
        lmmc_vec_t y;
    } test_swap;
};

static void test_axpy(void **state) {
    struct test_fixture *fixture = *state;
    struct test_axpy_resources *resources = &fixture->test_axpy;

    lmmc_status_t st;
    size_t n = 5;

    st = lmmc_vec_create(n, &resources->x);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_vec_create(n, &resources->y);
    assert_int_equal(st, LMMC_STATUS_OK);

    for (size_t i = 0; i < n; i++) {
        resources->x.data[i] = (lmmc_real_t)(i + 1);
        resources->y.data[i] = (lmmc_real_t)(10 + i);
    }

    lmmc_real_t orig_y[5];
    memcpy(orig_y, resources->y.data, n * sizeof(lmmc_real_t));

    st = lmmc_vec_axpy(0.0, &resources->x, &resources->y);
    assert_int_equal(st, LMMC_STATUS_OK);
    for (size_t i = 0; i < n; i++) {
        if (!lmmc_test_nearly_equal(resources->y.data[i], orig_y[i], TEST_EPS_TIGHT)) {
            fail_msg("5.8 FAIL: axpy alpha=0 changed y[%" PRIuMAX "]\n", (uintmax_t)(i));
        }
    }

    st = lmmc_vec_axpy(1.0, &resources->x, &resources->y);
    assert_int_equal(st, LMMC_STATUS_OK);
    for (size_t i = 0; i < n; i++) {
        lmmc_real_t expected = resources->x.data[i] + orig_y[i];
        if (!lmmc_test_nearly_equal(resources->y.data[i], expected, TEST_EPS_TIGHT)) {
            fail_msg("5.8 FAIL: axpy alpha=1: y[%" PRIuMAX "]=%g, expected %g\n", (uintmax_t)(i), resources->y.data[i], expected);
        }
    }

    lmmc_vec_destroy(&resources->x);
    lmmc_vec_destroy(&resources->y);
    return;
}

static void test_swap(void **state) {
    struct test_fixture *fixture = *state;
    struct test_swap_resources *resources = &fixture->test_swap;

    lmmc_status_t st;
    size_t n = 6;

    st = lmmc_vec_create(n, &resources->x);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_vec_create(n, &resources->y);
    assert_int_equal(st, LMMC_STATUS_OK);

    for (size_t i = 0; i < n; i++) {
        resources->x.data[i] = (lmmc_real_t)(i + 1);
        resources->y.data[i] = (lmmc_real_t)(100 + i);
    }

    lmmc_real_t orig_x[6], orig_y[6];
    memcpy(orig_x, resources->x.data, n * sizeof(lmmc_real_t));
    memcpy(orig_y, resources->y.data, n * sizeof(lmmc_real_t));

    st = lmmc_vec_swap(&resources->x, &resources->y);
    assert_int_equal(st, LMMC_STATUS_OK);

    for (size_t i = 0; i < n; i++) {
        if (!lmmc_test_nearly_equal(resources->x.data[i], orig_y[i], TEST_EPS_TIGHT) ||
            !lmmc_test_nearly_equal(resources->y.data[i], orig_x[i], TEST_EPS_TIGHT)) {
            fail_msg("5.10 FAIL: swap incorrect at index %" PRIuMAX "\n", (uintmax_t)(i));
        }
    }

    lmmc_vec_destroy(&resources->x);
    lmmc_vec_destroy(&resources->y);
    return;
}

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_vec_destroy(&fixture->test_axpy.x);
    lmmc_vec_destroy(&fixture->test_axpy.y);
    lmmc_vec_destroy(&fixture->test_swap.x);
    lmmc_vec_destroy(&fixture->test_swap.y);
    free(fixture);
    *state = NULL;
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_axpy, setup, teardown),
        cmocka_unit_test_setup_teardown(test_swap, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
