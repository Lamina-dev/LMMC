/**
 * @file test_dense_extended_invariants.c
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
    struct test_determinants_resources {
        lmmc_mat_t eye;
        lmmc_mat_t sing;
    } test_determinants;
    struct test_traces_resources {
        lmmc_mat_t a;
        lmmc_mat_t eye;
    } test_traces;
};

static void test_determinants(void **state) {
    struct test_fixture *fixture = *state;
    struct test_determinants_resources *resources = &fixture->test_determinants;

    lmmc_status_t st;

    st = lmmc_mat_identity(4, &resources->eye);
    assert_int_equal(st, LMMC_STATUS_OK);

    lmmc_real_t det_val = 0.0;
    st = lmmc_mat_det(&resources->eye, &det_val);
    if (st != LMMC_STATUS_OK) {
        fail_msg("5.6 FAIL: det(I) returned error %d\n", (int)st);
    }
    if (!lmmc_test_nearly_equal(det_val, 1.0, TEST_EPS_TIGHT)) {
        fail_msg("5.6 FAIL: det(I) = %g, expected 1.0\n", det_val);
    }
    lmmc_mat_destroy(&resources->eye);

    st = lmmc_mat_create(3, 3, &resources->sing);
    assert_int_equal(st, LMMC_STATUS_OK);

    resources->sing.data[0] = 1;
    resources->sing.data[1] = 2;
    resources->sing.data[2] = 3;
    resources->sing.data[3] = 4;
    resources->sing.data[4] = 5;
    resources->sing.data[5] = 6;
    resources->sing.data[6] = 7;
    resources->sing.data[7] = 8;
    resources->sing.data[8] = 9;

    st = lmmc_mat_det(&resources->sing, &det_val);
    assert_int_equal(st, LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(
        det_val, 0.0, TEST_EPS_NORMAL));
    lmmc_mat_destroy(&resources->sing);
    return;
}

static void test_traces(void **state) {
    struct test_fixture *fixture = *state;
    struct test_traces_resources *resources = &fixture->test_traces;

    lmmc_status_t st;

    st = lmmc_mat_create(4, 4, &resources->a);
    assert_int_equal(st, LMMC_STATUS_OK);

    for (size_t i = 0; i < 16; i++)
        resources->a.data[i] = (lmmc_real_t)(i + 1);

    lmmc_real_t trace_val = 0.0;
    st = lmmc_mat_trace(&resources->a, &trace_val);
    if (st != LMMC_STATUS_OK) {
        fail_msg("5.7 FAIL: trace returned error %d\n", (int)st);
    }

    lmmc_real_t expected_trace = 1.0 + 6.0 + 11.0 + 16.0;
    if (!lmmc_test_nearly_equal(trace_val, expected_trace, TEST_EPS_TIGHT)) {
        fail_msg("5.7 FAIL: trace = %g, expected %g\n", trace_val, expected_trace);
    }

    lmmc_mat_destroy(&resources->a);

    st = lmmc_mat_identity(7, &resources->eye);
    assert_int_equal(st, LMMC_STATUS_OK);

    st = lmmc_mat_trace(&resources->eye, &trace_val);
    assert_int_equal(st, LMMC_STATUS_OK);
    if (!lmmc_test_nearly_equal(trace_val, 7.0, TEST_EPS_TIGHT)) {
        fail_msg("5.7 FAIL: trace(I_7) = %g, expected 7.0\n", trace_val);
    }
    lmmc_mat_destroy(&resources->eye);
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
    lmmc_mat_destroy(&fixture->test_determinants.eye);
    lmmc_mat_destroy(&fixture->test_determinants.sing);
    lmmc_mat_destroy(&fixture->test_traces.a);
    lmmc_mat_destroy(&fixture->test_traces.eye);
    free(fixture);
    *state = NULL;
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_determinants, setup, teardown),
        cmocka_unit_test_setup_teardown(test_traces, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
