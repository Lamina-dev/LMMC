/**
 * @file test_mat_extended_boundaries.c
 * @brief 扩展矩阵代数的属性与边界测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "lmmc/lmmc.h"
#include "test_common.h"

#define PBT_ITERATIONS 100

struct test_fixture {
    struct test_unit_pow_zero_resources {
        lmmc_mat_t A;
        lmmc_mat_t result;
    } test_unit_pow_zero;
    struct test_unit_pow_one_resources {
        lmmc_mat_t A;
        lmmc_mat_t result;
    } test_unit_pow_one;
    struct test_unit_pow_singular_negative_resources {
        lmmc_mat_t A;
        lmmc_mat_t result;
    } test_unit_pow_singular_negative;
    struct test_unit_rank_zero_matrix_resources {
        lmmc_mat_t A;
    } test_unit_rank_zero_matrix;
    struct test_unit_rank_identity_resources {
        lmmc_mat_t I;
    } test_unit_rank_identity;
    struct test_unit_cross_dimension_error_resources {
        lmmc_vec_t a;
        lmmc_vec_t b;
        lmmc_vec_t c;
    } test_unit_cross_dimension_error;
};

static void test_unit_pow_zero(void **state) {
    struct test_fixture *fixture = *state;
    struct test_unit_pow_zero_resources *resources = &fixture->test_unit_pow_zero;

    lmmc_status_t st;
    double eps = 1e-15;
    size_t r, c;

    st = lmmc_mat_create(3, 3, &resources->A);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    mat_create(A) failed\n");
    }
    st = lmmc_mat_create(3, 3, &resources->result);
    assert_int_equal(st, LMMC_STATUS_OK);

    for (r = 0; r < 3; r++) {
        for (c = 0; c < 3; c++) {
            resources->A.data[r * resources->A.stride + c] = (double)(r * 3 + c + 1);
        }
    }

    st = lmmc_mat_pow(&resources->A, 0, &resources->result);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    mat_pow(A, 0) failed, status=%d\n", (int)st);
    }

    for (r = 0; r < 3; r++) {
        for (c = 0; c < 3; c++) {
            double expected = (r == c) ? 1.0 : 0.0;
            double got = resources->result.data[r * resources->result.stride + c];
            if (!lmmc_test_nearly_equal(got, expected, eps)) {
                fail_msg("    A^0[%" PRIuMAX "][%" PRIuMAX "] = %g, expected %g\n", (uintmax_t)(r), (uintmax_t)(c), got, expected);
            }
        }
    }

    lmmc_mat_destroy(&resources->A);
    lmmc_mat_destroy(&resources->result);
}

static void test_unit_pow_one(void **state) {
    struct test_fixture *fixture = *state;
    struct test_unit_pow_one_resources *resources = &fixture->test_unit_pow_one;

    lmmc_status_t st;
    double eps = 1e-15;
    size_t r, c;

    st = lmmc_mat_create(3, 3, &resources->A);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    mat_create(A) failed\n");
    }
    st = lmmc_mat_create(3, 3, &resources->result);
    assert_int_equal(st, LMMC_STATUS_OK);

    for (r = 0; r < 3; r++) {
        for (c = 0; c < 3; c++) {
            resources->A.data[r * resources->A.stride + c] = (double)(r * 3 + c + 1);
        }
    }

    st = lmmc_mat_pow(&resources->A, 1, &resources->result);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    mat_pow(A, 1) failed, status=%d\n", (int)st);
    }

    for (r = 0; r < 3; r++) {
        for (c = 0; c < 3; c++) {
            double expected = resources->A.data[r * resources->A.stride + c];
            double got = resources->result.data[r * resources->result.stride + c];
            if (!lmmc_test_nearly_equal(got, expected, eps)) {
                fail_msg("    A^1[%" PRIuMAX "][%" PRIuMAX "] = %g, expected %g\n", (uintmax_t)(r), (uintmax_t)(c), got, expected);
            }
        }
    }

    lmmc_mat_destroy(&resources->A);
    lmmc_mat_destroy(&resources->result);
}

static void test_unit_pow_singular_negative(void **state) {
    struct test_fixture *fixture = *state;
    struct test_unit_pow_singular_negative_resources *resources = &fixture->test_unit_pow_singular_negative;

    lmmc_status_t st;

    st = lmmc_mat_create(2, 2, &resources->A);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    mat_create(A) failed\n");
    }
    st = lmmc_mat_create(2, 2, &resources->result);
    assert_int_equal(st, LMMC_STATUS_OK);

    resources->A.data[0 * resources->A.stride + 0] = 1.0;
    resources->A.data[0 * resources->A.stride + 1] = 2.0;
    resources->A.data[1 * resources->A.stride + 0] = 2.0;
    resources->A.data[1 * resources->A.stride + 1] = 4.0;

    st = lmmc_mat_pow(&resources->A, -1, &resources->result);
    if (st != LMMC_STATUS_SINGULAR_MATRIX) {
        fail_msg("    Expected SINGULAR_MATRIX for singular A^{-1}, got %d\n", (int)st);
    }

    lmmc_mat_destroy(&resources->A);
    lmmc_mat_destroy(&resources->result);
}

static void test_unit_rank_zero_matrix(void **state) {
    struct test_fixture *fixture = *state;
    struct test_unit_rank_zero_matrix_resources *resources = &fixture->test_unit_rank_zero_matrix;

    lmmc_status_t st;
    size_t rank;

    st = lmmc_mat_create(3, 4, &resources->A);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    mat_create(A) failed\n");
    }

    st = lmmc_mat_rank(&resources->A, -1.0, &rank);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    mat_rank failed for zero matrix, status=%d\n", (int)st);
    }

    if (rank != 0) {
        fail_msg("    rank(zero matrix) = %" PRIuMAX ", expected 0\n", (uintmax_t)(rank));
    }

    lmmc_mat_destroy(&resources->A);
}

static void test_unit_rank_identity(void **state) {
    struct test_fixture *fixture = *state;
    struct test_unit_rank_identity_resources *resources = &fixture->test_unit_rank_identity;

    lmmc_status_t st;
    size_t rank;

    st = lmmc_mat_identity(4, &resources->I);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    mat_identity failed\n");
    }

    st = lmmc_mat_rank(&resources->I, -1.0, &rank);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    mat_rank failed for identity, status=%d\n", (int)st);
    }

    if (rank != 4) {
        fail_msg("    rank(I_4) = %" PRIuMAX ", expected 4\n", (uintmax_t)(rank));
    }

    lmmc_mat_destroy(&resources->I);
}

static void test_unit_cross_dimension_error(void **state) {
    struct test_fixture *fixture = *state;
    struct test_unit_cross_dimension_error_resources *resources = &fixture->test_unit_cross_dimension_error;

    lmmc_status_t st;

    st = lmmc_vec_create(4, &resources->a);
    if (st != LMMC_STATUS_OK) {
        fail_msg("    vec_create(a) failed\n");
    }
    st = lmmc_vec_create(3, &resources->b);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_vec_create(3, &resources->c);
    assert_int_equal(st, LMMC_STATUS_OK);

    st = lmmc_vec_cross(&resources->a, &resources->b, &resources->c);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        fail_msg("    Expected DIMENSION_MISMATCH for size-4 cross, got %d\n", (int)st);
    }

    lmmc_vec_destroy(&resources->a);
    lmmc_vec_destroy(&resources->b);
    lmmc_vec_destroy(&resources->c);
}

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_destroy(&fixture->test_unit_pow_zero.A);
    lmmc_mat_destroy(&fixture->test_unit_pow_zero.result);
    lmmc_mat_destroy(&fixture->test_unit_pow_one.A);
    lmmc_mat_destroy(&fixture->test_unit_pow_one.result);
    lmmc_mat_destroy(&fixture->test_unit_pow_singular_negative.A);
    lmmc_mat_destroy(&fixture->test_unit_pow_singular_negative.result);
    lmmc_mat_destroy(&fixture->test_unit_rank_zero_matrix.A);
    lmmc_mat_destroy(&fixture->test_unit_rank_identity.I);
    lmmc_vec_destroy(&fixture->test_unit_cross_dimension_error.a);
    lmmc_vec_destroy(&fixture->test_unit_cross_dimension_error.b);
    lmmc_vec_destroy(&fixture->test_unit_cross_dimension_error.c);
    free(fixture);
    *state = NULL;
    return 0;
}

static int group_setup(void **state) {
    (void)state;
    assert_int_equal(lmmc_init(), LMMC_STATUS_OK);
    return 0;
}

static int group_teardown(void **state) {
    (void)state;
    assert_int_equal(lmmc_deinit(), LMMC_STATUS_OK);
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_unit_pow_zero, setup, teardown),
        cmocka_unit_test_setup_teardown(test_unit_pow_one, setup, teardown),
        cmocka_unit_test_setup_teardown(test_unit_pow_singular_negative, setup, teardown),
        cmocka_unit_test_setup_teardown(test_unit_rank_zero_matrix, setup, teardown),
        cmocka_unit_test_setup_teardown(test_unit_rank_identity, setup, teardown),
        cmocka_unit_test_setup_teardown(test_unit_cross_dimension_error, setup, teardown),
    };
    return cmocka_run_group_tests(tests, group_setup, group_teardown);
}
