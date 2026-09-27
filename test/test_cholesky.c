/**
 * @file test_cholesky.c
 * 针对 LMMC 中 cholesky 相关接口的单元测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

struct test_fixture {
    lmmc_mat_t a;
    lmmc_vec_t b;
    lmmc_vec_t x;
};

static void test_cholesky_factors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *a = &fixture->a;
    lmmc_status_t st;

    static const struct {
        size_t index;
        lmmc_real_t value;
    } factors[] = {
        {0, 2.0}, {3, 6.0}, {4, 1.0}, {6, -8.0}, {7, 5.0}, {8, 3.0}};
    st = lmmc_cholesky_decompose_inplace(a);
    if (st != LMMC_STATUS_OK) {
        fail_msg("Cholesky decomposition failed: %s\n", lmmc_status_string(st));
    }

    for (size_t i = 0; i < sizeof(factors) / sizeof(factors[0]); ++i) {
        if (!lmmc_test_nearly_equal(a->data[factors[i].index], factors[i].value, 1e-10)) {
            fail_msg("Cholesky L factors incorrect\n");
        }
    }
}

static void test_cholesky_solution(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *a = &fixture->a;
    lmmc_vec_t *b = &fixture->b;
    lmmc_vec_t *x = &fixture->x;
    lmmc_status_t st;

    st = lmmc_cholesky_solve(a, b, x);
    if (st != LMMC_STATUS_OK) {
        fail_msg("Cholesky solve failed: %s\n", lmmc_status_string(st));
    }

    if (!lmmc_test_nearly_equal(x->data[0], 1.0, 1e-10) ||
        !lmmc_test_nearly_equal(x->data[1], 2.0, 1e-10) ||
        !lmmc_test_nearly_equal(x->data[2], 3.0, 1e-10)) {
        fail_msg("Cholesky solution incorrect: [%f, %f, %f]\n", x->data[0], x->data[1], x->data[2]);
    }
}

static void test_indefinite_rejection(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *a = &fixture->a;
    lmmc_status_t st;

    st = lmmc_mat_create(2, 2, a);
    assert_int_equal(st, LMMC_STATUS_OK);
    LMMC_REAL_SET_D(&a->data[0], 1.0);
    LMMC_REAL_SET_D(&a->data[1], 2.0);
    LMMC_REAL_SET_D(&a->data[2], 2.0);
    LMMC_REAL_SET_D(&a->data[3], 1.0);
    st = lmmc_cholesky_decompose_inplace(a);
    if (st != LMMC_STATUS_NOT_POSITIVE_DEFINITE) {
        fail_msg("Expected NOT_POSITIVE_DEFINITE for non-positive definite matrix, got %s\n", lmmc_status_string(st));
    }
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_destroy(&fixture->a);
    lmmc_vec_destroy(&fixture->b);
    lmmc_vec_destroy(&fixture->x);
    free(fixture);
    *state = NULL;
    return 0;
}

static int setup_empty(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    *state = fixture;
    if (fixture == NULL) {
        return -1;
    }
    return 0;
}

static int setup_spd(void **state) {
    if (setup_empty(state) != 0) {
        return -1;
    }
    struct test_fixture *fixture = *state;
    lmmc_status_t st = lmmc_mat_create(3, 3, &fixture->a);
    if (st != LMMC_STATUS_OK) {
        teardown(state);
        return st;
    }
    LMMC_REAL_SET_D(&fixture->a.data[0], 4.0);
    LMMC_REAL_SET_D(&fixture->a.data[1], 12.0);
    LMMC_REAL_SET_D(&fixture->a.data[2], -16.0);
    LMMC_REAL_SET_D(&fixture->a.data[3], 12.0);
    LMMC_REAL_SET_D(&fixture->a.data[4], 37.0);
    LMMC_REAL_SET_D(&fixture->a.data[5], -43.0);
    LMMC_REAL_SET_D(&fixture->a.data[6], -16.0);
    LMMC_REAL_SET_D(&fixture->a.data[7], -43.0);
    LMMC_REAL_SET_D(&fixture->a.data[8], 98.0);
    return 0;
}

static int setup_solution(void **state) {
    if (setup_spd(state) != 0) {
        return -1;
    }
    struct test_fixture *fixture = *state;
    lmmc_status_t st = lmmc_cholesky_decompose_inplace(&fixture->a);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(3, &fixture->b);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(3, &fixture->x);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    LMMC_REAL_SET_D(&fixture->b.data[0], -20.0);
    LMMC_REAL_SET_D(&fixture->b.data[1], -43.0);
    LMMC_REAL_SET_D(&fixture->b.data[2], 192.0);
    return 0;
cleanup:
    teardown(state);
    return st;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_cholesky_factors, setup_spd, teardown),
        cmocka_unit_test_setup_teardown(test_cholesky_solution, setup_solution, teardown),
        cmocka_unit_test_setup_teardown(test_indefinite_rejection, setup_empty, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
