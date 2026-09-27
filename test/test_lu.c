/**
 * @file test_lu.c
 * 针对 LMMC 中 lu 相关接口的单元测试。
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
    lmmc_mat_t other;
    lmmc_vec_t b;
    lmmc_vec_t x;
    size_t piv[3];
};

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_destroy(&fixture->a);
    lmmc_mat_destroy(&fixture->other);
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

static lmmc_status_t prepare_lu_input(struct test_fixture *fixture) {
    lmmc_mat_t *a = &fixture->a;
    lmmc_status_t st = lmmc_mat_create(3, 3, a);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_vec_create(3, &fixture->x);
    if (st != LMMC_STATUS_OK) {
        return st;
    }

    LMMC_REAL_SET_D(&a->data[0], 3.0);
    LMMC_REAL_SET_D(&a->data[1], 2.0);
    LMMC_REAL_SET_D(&a->data[2], -1.0);
    LMMC_REAL_SET_D(&a->data[3], 2.0);
    LMMC_REAL_SET_D(&a->data[4], -2.0);
    LMMC_REAL_SET_D(&a->data[5], 4.0);
    LMMC_REAL_SET_D(&a->data[6], -1.0);
    LMMC_REAL_SET_D(&a->data[7], 0.5);
    LMMC_REAL_SET_D(&a->data[8], -1.0);
    return lmmc_lu_decompose_inplace(a, fixture->piv, NULL);
}

static int setup_solution(void **state) {
    if (setup_empty(state) != 0) {
        return -1;
    }
    struct test_fixture *fixture = *state;
    lmmc_status_t st = prepare_lu_input(fixture);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(3, &fixture->b);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    LMMC_REAL_SET_D(&fixture->b.data[0], 1.0);
    LMMC_REAL_SET_D(&fixture->b.data[1], -2.0);
    LMMC_REAL_SET_D(&fixture->b.data[2], 0.0);
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
    lmmc_status_t st = prepare_lu_input(fixture);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    st = lmmc_vec_create(2, &fixture->b);
    if (st != LMMC_STATUS_OK) {
        goto cleanup;
    }
    return 0;

cleanup:
    teardown(state);
    return st;
}

static void test_lu_solution(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *a = &fixture->a;
    lmmc_vec_t *b = &fixture->b;
    lmmc_vec_t *x = &fixture->x;
    size_t *piv = fixture->piv;
    lmmc_status_t st;

    st = lmmc_lu_solve(a, piv, b, x);
    assert_int_equal(st, LMMC_STATUS_OK);

    assert_false(!lmmc_test_nearly_equal(x->data[0], 1.0, 1e-10) ||
                 !lmmc_test_nearly_equal(x->data[1], -2.0, 1e-10) ||
                 !lmmc_test_nearly_equal(x->data[2], -2.0, 1e-10));
}

static void test_lu_factorization_errors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *singular = &fixture->a;
    lmmc_mat_t *rect = &fixture->other;
    size_t *piv = fixture->piv;
    lmmc_status_t st;

    st = lmmc_mat_create(2, 2, singular);
    assert_int_equal(st, LMMC_STATUS_OK);
    LMMC_REAL_SET_D(&singular->data[0], 1.0);
    LMMC_REAL_SET_D(&singular->data[1], 2.0);
    LMMC_REAL_SET_D(&singular->data[2], 2.0);
    LMMC_REAL_SET_D(&singular->data[3], 4.0);
    st = lmmc_lu_decompose_inplace(singular, piv, NULL);
    assert_int_equal(st, LMMC_STATUS_SINGULAR_MATRIX);

    st = lmmc_mat_create(2, 3, rect);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_lu_decompose_inplace(rect, piv, NULL);
    assert_int_equal(st, LMMC_STATUS_DIMENSION_MISMATCH);
}

static void test_lu_rhs_shape(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_t *a = &fixture->a;
    lmmc_vec_t *x = &fixture->x;
    lmmc_vec_t *b = &fixture->b;
    size_t *piv = fixture->piv;
    lmmc_status_t st;

    st = lmmc_lu_solve(a, piv, b, x);
    assert_int_equal(st, LMMC_STATUS_DIMENSION_MISMATCH);
}


int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_lu_solution, setup_solution, teardown),
        cmocka_unit_test_setup_teardown(test_lu_factorization_errors, setup_empty, teardown),
        cmocka_unit_test_setup_teardown(test_lu_rhs_shape, setup_rhs_shape, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
