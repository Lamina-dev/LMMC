/**
 * @file test_dense_extended_transpose.c
 * @brief 稠密矩阵扩展接口单元测试。
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
    struct test_rectangular_transpose_resources {
        lmmc_mat_t a;
        lmmc_mat_t at;
        lmmc_mat_t att;
    } test_rectangular_transpose;
    struct test_square_transpose_resources {
        lmmc_mat_t s;
        lmmc_mat_t st2;
        lmmc_mat_t stt;
    } test_square_transpose;
};

static void test_rectangular_transpose(void **state) {
    struct test_fixture *fixture = *state;
    struct test_rectangular_transpose_resources *resources = &fixture->test_rectangular_transpose;

    lmmc_status_t st;

    st = lmmc_mat_create(3, 4, &resources->a);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_mat_create(4, 3, &resources->at);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_mat_create(3, 4, &resources->att);
    assert_int_equal(st, LMMC_STATUS_OK);

    for (size_t i = 0; i < 12; i++)
        resources->a.data[i] = (lmmc_real_t)(i + 1);

    st = lmmc_mat_transpose_to(&resources->a, &resources->at);
    assert_int_equal(st, LMMC_STATUS_OK);

    st = lmmc_mat_transpose_to(&resources->at, &resources->att);
    assert_int_equal(st, LMMC_STATUS_OK);

    for (size_t i = 0; i < 12; i++) {
        if (!lmmc_test_nearly_equal(resources->att.data[i], resources->a.data[i], TEST_EPS_TIGHT)) {
            fail_msg("5.5 FAIL: Transpose roundtrip failed at index %" PRIuMAX "\n", (uintmax_t)(i));
        }
    }

    lmmc_mat_destroy(&resources->a);
    lmmc_mat_destroy(&resources->at);
    lmmc_mat_destroy(&resources->att);
    return;
}

static void test_square_transpose(void **state) {
    struct test_fixture *fixture = *state;
    struct test_square_transpose_resources *resources = &fixture->test_square_transpose;

    lmmc_status_t st;

    st = lmmc_mat_create(5, 5, &resources->s);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_mat_create(5, 5, &resources->st2);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_mat_create(5, 5, &resources->stt);
    assert_int_equal(st, LMMC_STATUS_OK);

    for (size_t i = 0; i < 25; i++)
        resources->s.data[i] = (lmmc_real_t)(i * 3 - 7);

    st = lmmc_mat_transpose_to(&resources->s, &resources->st2);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_mat_transpose_to(&resources->st2, &resources->stt);
    assert_int_equal(st, LMMC_STATUS_OK);

    for (size_t i = 0; i < 25; i++) {
        if (!lmmc_test_nearly_equal(resources->stt.data[i], resources->s.data[i], TEST_EPS_TIGHT)) {
            fail_msg("5.5 FAIL: Square transpose roundtrip failed at %" PRIuMAX "\n", (uintmax_t)(i));
        }
    }

    lmmc_mat_destroy(&resources->s);
    lmmc_mat_destroy(&resources->st2);
    lmmc_mat_destroy(&resources->stt);
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
    lmmc_mat_destroy(&fixture->test_rectangular_transpose.a);
    lmmc_mat_destroy(&fixture->test_rectangular_transpose.at);
    lmmc_mat_destroy(&fixture->test_rectangular_transpose.att);
    lmmc_mat_destroy(&fixture->test_square_transpose.s);
    lmmc_mat_destroy(&fixture->test_square_transpose.st2);
    lmmc_mat_destroy(&fixture->test_square_transpose.stt);
    free(fixture);
    *state = NULL;
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_rectangular_transpose, setup, teardown),
        cmocka_unit_test_setup_teardown(test_square_transpose, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
