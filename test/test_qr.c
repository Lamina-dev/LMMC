/**
 * @file test_qr.c
 * 针对 LMMC 中 qr 相关接口的单元测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdlib.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

struct test_fixture {
    lmmc_mat_t a;
    lmmc_vec_t b;
    lmmc_vec_t x;
    double tau[2];
};

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
    return fixture ? 0 : -1;
}

static lmmc_status_t prepare_qr_input(struct test_fixture *fixture) {
    lmmc_mat_t *a = &fixture->a;
    lmmc_status_t st = lmmc_mat_create(4, 2, a);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_create(2, &fixture->x);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    LMMC_REAL_SET_D(&a->data[0], 1.0);
    LMMC_REAL_SET_D(&a->data[1], 0.0);
    LMMC_REAL_SET_D(&a->data[2], 1.0);
    LMMC_REAL_SET_D(&a->data[3], 1.0);
    LMMC_REAL_SET_D(&a->data[4], 1.0);
    LMMC_REAL_SET_D(&a->data[5], 2.0);
    LMMC_REAL_SET_D(&a->data[6], 1.0);
    LMMC_REAL_SET_D(&a->data[7], 3.0);
    return lmmc_qr_decompose_inplace(a, fixture->tau, 2);
}

static int setup_least_squares(void **state) {
    if (setup_empty(state) != 0) {
        return -1;
    }
    struct test_fixture *fixture = *state;
    lmmc_status_t st = prepare_qr_input(fixture);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(4, &fixture->b);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    LMMC_REAL_SET_D(&fixture->b.data[0], 1.0);
    LMMC_REAL_SET_D(&fixture->b.data[1], 3.0);
    LMMC_REAL_SET_D(&fixture->b.data[2], 5.0);
    LMMC_REAL_SET_D(&fixture->b.data[3], 7.0);
    return 0;

cleanup:
    teardown(state);
    return st;
}

static int setup_rhs_shape(void **state) {
    if (setup_empty(state) != 0) {
        return -1;
    }
    struct test_fixture *fixture = *state;
    lmmc_status_t st = prepare_qr_input(fixture);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(3, &fixture->b);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    return 0;

cleanup:
    teardown(state);
    return st;
}

static void test_short_householder_storage(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *a = &fixture->a;
    double tau_short[1];
    lmmc_status_t st;

    st = lmmc_mat_create(3, 2, a);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_qr_decompose_inplace(a, tau_short, 1);
    assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_qr_least_squares(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *a = &fixture->a;
    lmmc_vec_t *b = &fixture->b;
    lmmc_vec_t *x = &fixture->x;
    double *tau = fixture->tau;
    lmmc_status_t st;

    st = lmmc_qr_solve(a, tau, b, x);
    assert_int_equal(st, LMMC_STATUS_OK);

    assert_false(!lmmc_test_nearly_equal(x->data[0], 1.0, 1e-9) ||
                 !lmmc_test_nearly_equal(x->data[1], 2.0, 1e-9));
}

static void test_qr_rhs_shape(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *a = &fixture->a;
    lmmc_vec_t *x = &fixture->x;
    lmmc_vec_t *b = &fixture->b;
    double *tau = fixture->tau;
    lmmc_status_t st;

    st = lmmc_qr_solve(a, tau, b, x);
    assert_int_equal(st, LMMC_STATUS_DIMENSION_MISMATCH);
}

static void test_qr_rank_deficiency(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *a = &fixture->a;
    lmmc_vec_t *b = &fixture->b;
    lmmc_vec_t *x = &fixture->x;
    double *tau = fixture->tau;
    lmmc_status_t st;

    st = lmmc_mat_create(3, 2, a);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_vec_create(3, b);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_vec_create(2, x);
    assert_int_equal(st, LMMC_STATUS_OK);

    LMMC_REAL_SET_D(&a->data[0], 1.0);
    LMMC_REAL_SET_D(&a->data[1], 2.0);
    LMMC_REAL_SET_D(&a->data[2], 0.0);
    LMMC_REAL_SET_D(&a->data[3], 0.0);
    LMMC_REAL_SET_D(&a->data[4], 0.0);
    LMMC_REAL_SET_D(&a->data[5], 0.0);

    LMMC_REAL_SET_D(&b->data[0], 1.0);
    LMMC_REAL_SET_D(&b->data[1], 0.0);
    LMMC_REAL_SET_D(&b->data[2], 0.0);

    st = lmmc_qr_decompose_inplace(a, tau, 2);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_qr_solve(a, tau, b, x);
    assert_int_equal(st, LMMC_STATUS_SINGULAR_MATRIX);
}


int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_short_householder_storage, setup_empty, teardown),
        cmocka_unit_test_setup_teardown(test_qr_least_squares, setup_least_squares, teardown),
        cmocka_unit_test_setup_teardown(test_qr_rhs_shape, setup_rhs_shape, teardown),
        cmocka_unit_test_setup_teardown(test_qr_rank_deficiency, setup_empty, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
