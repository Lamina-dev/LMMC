/**
 * @file test_dense_unit.c
 * 针对 LMMC 中 dense unit 相关接口的单元测试。
 */
#include <inttypes.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/status.h"

#define TOL 1e-12

struct test_fixture {
    lmmc_vec_t x;
    lmmc_vec_t y;
    lmmc_mat_t a;
    lmmc_mat_t b;
    lmmc_mat_t c;
};

static void test_vec_norm2_null(void **state) {
    (void)state;
    lmmc_real_t result;
    lmmc_status_t s = lmmc_vec_norm2(NULL, &result);
    if (!(s == LMMC_STATUS_INVALID_ARGUMENT)) {
        fail_msg("vec_norm2(NULL, ...) should return INVALID_ARGUMENT, got %d", (int)s);
    }
}

static void test_vec_axpy_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;

    assert_int_equal(lmmc_vec_create(3, &fixture->x), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(5, &fixture->y), LMMC_STATUS_OK);

    lmmc_real_t alpha = 1.0;
    lmmc_status_t s = lmmc_vec_axpy(alpha, &fixture->x, &fixture->y);
    if (!(s == LMMC_STATUS_DIMENSION_MISMATCH)) {
        fail_msg("vec_axpy with mismatched sizes should return DIMENSION_MISMATCH, got %d", (int)s);
    }
}

static void test_vec_copy_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;

    assert_int_equal(lmmc_vec_create(4, &fixture->x), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(2, &fixture->y), LMMC_STATUS_OK);

    lmmc_status_t s = lmmc_vec_copy(&fixture->x, &fixture->y);
    if (!(s == LMMC_STATUS_DIMENSION_MISMATCH)) {
        fail_msg("vec_copy with mismatched sizes should return DIMENSION_MISMATCH, got %d", (int)s);
    }
}

static void test_vec_swap_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;

    assert_int_equal(lmmc_vec_create(6, &fixture->x), LMMC_STATUS_OK);
    assert_int_equal(lmmc_vec_create(3, &fixture->y), LMMC_STATUS_OK);

    lmmc_status_t s = lmmc_vec_swap(&fixture->x, &fixture->y);
    if (!(s == LMMC_STATUS_DIMENSION_MISMATCH)) {
        fail_msg("vec_swap with mismatched sizes should return DIMENSION_MISMATCH, got %d", (int)s);
    }
}

static void test_mat_add_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;

    assert_int_equal(lmmc_mat_create(2, 3, &fixture->a), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_create(3, 2, &fixture->b), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_create(2, 3, &fixture->c), LMMC_STATUS_OK);

    lmmc_status_t s = lmmc_mat_add(&fixture->a, &fixture->b, &fixture->c);
    if (!(s == LMMC_STATUS_DIMENSION_MISMATCH)) {
        fail_msg("mat_add with mismatched dims should return DIMENSION_MISMATCH, got %d", (int)s);
    }
}

static void test_mat_trace_non_square(void **state) {
    struct test_fixture *fixture = *state;

    assert_int_equal(lmmc_mat_create(2, 3, &fixture->a), LMMC_STATUS_OK);

    lmmc_real_t trace;
    lmmc_status_t s = lmmc_mat_trace(&fixture->a, &trace);
    if (!(s == LMMC_STATUS_INVALID_ARGUMENT)) {
        fail_msg("mat_trace on non-square should return INVALID_ARGUMENT, got %d", (int)s);
    }
}

static void test_mat_det_non_square(void **state) {
    struct test_fixture *fixture = *state;

    assert_int_equal(lmmc_mat_create(3, 2, &fixture->a), LMMC_STATUS_OK);

    lmmc_real_t det;
    lmmc_status_t s = lmmc_mat_det(&fixture->a, &det);
    if (!(s == LMMC_STATUS_INVALID_ARGUMENT)) {
        fail_msg("mat_det on non-square should return INVALID_ARGUMENT, got %d", (int)s);
    }
}

static void test_vec_norm2_known(void **state) {
    struct test_fixture *fixture = *state;

    assert_int_equal(lmmc_vec_create(2, &fixture->x), LMMC_STATUS_OK);
    fixture->x.data[0] = 3.0;
    fixture->x.data[1] = 4.0;

    lmmc_real_t norm;
    lmmc_status_t s = lmmc_vec_norm2(&fixture->x, &norm);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("norm2 returned error %d", (int)s);
    }
    if (!(fabs(norm - 5.0) < TOL)) {
        fail_msg("norm2([3,4]) should be 5.0, got %g", norm);
    }
}

static void test_vec_asum_known(void **state) {
    struct test_fixture *fixture = *state;

    assert_int_equal(lmmc_vec_create(3, &fixture->x), LMMC_STATUS_OK);
    fixture->x.data[0] = -1.0;
    fixture->x.data[1] = 2.0;
    fixture->x.data[2] = -3.0;

    lmmc_real_t asum;
    lmmc_status_t s = lmmc_vec_asum(&fixture->x, &asum);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("asum returned error %d", (int)s);
    }
    if (!(fabs(asum - 6.0) < TOL)) {
        fail_msg("asum([-1,2,-3]) should be 6.0, got %g", asum);
    }
}

static void test_vec_iamax_known(void **state) {
    struct test_fixture *fixture = *state;

    assert_int_equal(lmmc_vec_create(3, &fixture->x), LMMC_STATUS_OK);
    fixture->x.data[0] = 1.0;
    fixture->x.data[1] = -5.0;
    fixture->x.data[2] = 3.0;

    size_t idx;
    lmmc_status_t s = lmmc_vec_iamax(&fixture->x, &idx);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("iamax returned error %d", (int)s);
    }
    if (!(idx == 1)) {
        fail_msg("iamax([1,-5,3]) should be 1, got %" PRIuMAX, (uintmax_t)(idx));
    }
}

static void test_mat_trace_known(void **state) {
    struct test_fixture *fixture = *state;

    assert_int_equal(lmmc_mat_create(2, 2, &fixture->a), LMMC_STATUS_OK);
    fixture->a.data[0 * fixture->a.stride + 0] = 1.0;
    fixture->a.data[0 * fixture->a.stride + 1] = 2.0;
    fixture->a.data[1 * fixture->a.stride + 0] = 3.0;
    fixture->a.data[1 * fixture->a.stride + 1] = 4.0;

    lmmc_real_t trace;
    lmmc_status_t s = lmmc_mat_trace(&fixture->a, &trace);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("trace returned error %d", (int)s);
    }
    if (!(fabs(trace - 5.0) < TOL)) {
        fail_msg("trace([[1,2],[3,4]]) should be 5.0, got %g", trace);
    }
}

static void test_mat_det_known(void **state) {
    struct test_fixture *fixture = *state;

    assert_int_equal(lmmc_mat_create(2, 2, &fixture->a), LMMC_STATUS_OK);
    fixture->a.data[0 * fixture->a.stride + 0] = 1.0;
    fixture->a.data[0 * fixture->a.stride + 1] = 2.0;
    fixture->a.data[1 * fixture->a.stride + 0] = 3.0;
    fixture->a.data[1 * fixture->a.stride + 1] = 4.0;

    lmmc_real_t det;
    lmmc_status_t s = lmmc_mat_det(&fixture->a, &det);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("det returned error %d", (int)s);
    }
    if (!(fabs(det - (-2.0)) < TOL)) {
        fail_msg("det([[1,2],[3,4]]) should be -2.0, got %g", det);
    }
}

static void test_mat_det_cancelled_overflow(void **state) {
    (void)state;
    lmmc_real_t values[] = {1e200, 1e200, 1e200, 1e200};
    lmmc_mat_t matrix = {2, 2, 2, values, 0};
    lmmc_real_t det = 42.0;

    assert_int_equal(lmmc_mat_det(&matrix, &det), LMMC_STATUS_OK);
    assert_true(det == 0.0);
}

static void test_mat_det_finite_after_overflow(void **state) {
    (void)state;
    const lmmc_real_t large = 1e160;
    const lmmc_real_t next = nextafter(large, INFINITY);
    lmmc_real_t values[] = {large, large, large, next};
    lmmc_mat_t matrix = {2, 2, 2, values, 0};
    lmmc_real_t det = 42.0;
    const lmmc_real_t expected = large * (next - large);

    assert_int_equal(lmmc_mat_det(&matrix, &det), LMMC_STATUS_OK);
    assert_true(isfinite(det) && fabs(det / expected - 1.0) < 1e-12);
}

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    *state = fixture;
    assert_non_null(fixture);
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_vec_destroy(&fixture->x);
    lmmc_vec_destroy(&fixture->y);
    lmmc_mat_destroy(&fixture->a);
    lmmc_mat_destroy(&fixture->b);
    lmmc_mat_destroy(&fixture->c);
    free(fixture);
    *state = NULL;
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_vec_norm2_null),
        cmocka_unit_test_setup_teardown(test_vec_axpy_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_vec_copy_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_vec_swap_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_mat_add_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_mat_trace_non_square, setup, teardown),
        cmocka_unit_test_setup_teardown(test_mat_det_non_square, setup, teardown),
        cmocka_unit_test_setup_teardown(test_vec_norm2_known, setup, teardown),
        cmocka_unit_test_setup_teardown(test_vec_asum_known, setup, teardown),
        cmocka_unit_test_setup_teardown(test_vec_iamax_known, setup, teardown),
        cmocka_unit_test_setup_teardown(test_mat_trace_known, setup, teardown),
        cmocka_unit_test_setup_teardown(test_mat_det_known, setup, teardown),
        cmocka_unit_test(test_mat_det_cancelled_overflow),
        cmocka_unit_test(test_mat_det_finite_after_overflow),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
