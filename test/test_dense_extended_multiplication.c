/**
 * @file test_dense_extended_multiplication.c
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
    struct test_identity_sizes_resources {
        lmmc_mat_t a;
        lmmc_mat_t b;
        lmmc_mat_t c;
    } test_identity_sizes;
    struct test_identity_sides_resources {
        lmmc_mat_t a;
        lmmc_mat_t eye;
        lmmc_mat_t c2;
    } test_identity_sides;
    struct test_rectangular_dimensions_resources {
        lmmc_mat_t a35;
        lmmc_mat_t b53;
        lmmc_mat_t c33;
        lmmc_mat_t bad;
    } test_rectangular_dimensions;
    struct test_finite_multiplication_resources {
        lmmc_mat_t a;
        lmmc_mat_t b;
        lmmc_mat_t c;
    } test_finite_multiplication;
};

static void test_identity_sizes(void **state) {
    struct test_fixture *fixture = *state;
    struct test_identity_sizes_resources *resources = &fixture->test_identity_sizes;

    lmmc_status_t st;
    size_t sizes[] = {1, 2, 3, 10, 100};
    size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (size_t si = 0; si < num_sizes; si++) {
        size_t n = sizes[si];

        st = lmmc_mat_create(n, n, &resources->a);
        assert_int_equal(st, LMMC_STATUS_OK);
        st = lmmc_mat_create(n, n, &resources->b);
        assert_int_equal(st, LMMC_STATUS_OK);
        st = lmmc_mat_create(n, n, &resources->c);
        assert_int_equal(st, LMMC_STATUS_OK);

        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                resources->a.data[i * n + j] = (lmmc_real_t)(i + j + 1);
                resources->b.data[i * n + j] = (i == j) ? 1.0 : 0.0;
            }
        }

        st = lmmc_mat_mul(&resources->a, &resources->b, &resources->c);
        assert_int_equal(st, LMMC_STATUS_OK);

        for (size_t i = 0; i < n * n; i++) {
            if (!lmmc_test_nearly_equal(resources->c.data[i], resources->a.data[i], TEST_EPS_TIGHT)) {
                fail_msg("5.1 FAIL: size=%" PRIuMAX ", A*I != A at index %" PRIuMAX "\n", (uintmax_t)(n), (uintmax_t)(i));
            }
        }

        lmmc_mat_destroy(&resources->a);
        lmmc_mat_destroy(&resources->b);
        lmmc_mat_destroy(&resources->c);
    }
    return;
}

static void test_identity_sides(void **state) {
    struct test_fixture *fixture = *state;
    struct test_identity_sides_resources *resources = &fixture->test_identity_sides;

    lmmc_status_t st;
    size_t n = 5;

    st = lmmc_mat_create(n, n, &resources->a);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_mat_identity(n, &resources->eye);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_mat_create(n, n, &resources->c2);
    assert_int_equal(st, LMMC_STATUS_OK);

    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            resources->a.data[i * n + j] = (lmmc_real_t)((i + 1) * 10 + j + 1);


    st = lmmc_mat_mul(&resources->eye, &resources->a, &resources->c2);
    assert_int_equal(st, LMMC_STATUS_OK);

    for (size_t i = 0; i < n * n; i++) {
        if (!lmmc_test_nearly_equal(
                resources->c2.data[i], resources->a.data[i],
                TEST_EPS_TIGHT)) {
            fail_msg(
                "5.2 FAIL: I*A != A at index %" PRIuMAX "\n",
                (uintmax_t)i);
        }
    }

    lmmc_mat_destroy(&resources->a);
    lmmc_mat_destroy(&resources->eye);
    lmmc_mat_destroy(&resources->c2);
    return;
}

static void test_rectangular_dimensions(void **state) {
    struct test_fixture *fixture = *state;
    struct test_rectangular_dimensions_resources *resources = &fixture->test_rectangular_dimensions;

    lmmc_status_t st;

    st = lmmc_mat_create(3, 5, &resources->a35);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_mat_create(5, 3, &resources->b53);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_mat_create(3, 3, &resources->c33);
    assert_int_equal(st, LMMC_STATUS_OK);

    for (size_t i = 0; i < 15; i++)
        resources->a35.data[i] = (lmmc_real_t)(i + 1);
    for (size_t i = 0; i < 15; i++)
        resources->b53.data[i] = (lmmc_real_t)(i + 1);

    st = lmmc_mat_mul(&resources->a35, &resources->b53, &resources->c33);
    if (st != LMMC_STATUS_OK) {
        fail_msg("5.3 FAIL: Valid non-square mul returned error %d\n", (int)st);
    }

    if (!lmmc_test_nearly_equal(resources->c33.data[0], 135.0, TEST_EPS_TIGHT)) {
        fail_msg("5.3 FAIL: C[0][0] = %g, expected 135\n", resources->c33.data[0]);
    }

    st = lmmc_mat_create(3, 3, &resources->bad);
    assert_int_equal(st, LMMC_STATUS_OK);

    st = lmmc_mat_mul(&resources->a35, &resources->bad, &resources->c33);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        fail_msg("5.3 FAIL: Dimension mismatch not detected, got %d\n", (int)st);
    }

    lmmc_mat_destroy(&resources->a35);
    lmmc_mat_destroy(&resources->b53);
    lmmc_mat_destroy(&resources->c33);
    lmmc_mat_destroy(&resources->bad);
    return;
}

static void test_finite_multiplication(void **state) {
    struct test_fixture *fixture = *state;
    struct test_finite_multiplication_resources *resources = &fixture->test_finite_multiplication;

    lmmc_status_t st;

    st = lmmc_mat_create(2, 2, &resources->a);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_mat_create(2, 2, &resources->b);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_mat_create(2, 2, &resources->c);
    assert_int_equal(st, LMMC_STATUS_OK);

    resources->a.data[0] = 1e300;
    resources->a.data[1] = 0.0;
    resources->a.data[2] = 0.0;
    resources->a.data[3] = 1e300;

    resources->b.data[0] = 1.0;
    resources->b.data[1] = 0.0;
    resources->b.data[2] = 0.0;
    resources->b.data[3] = 1.0;

    st = lmmc_mat_mul(&resources->a, &resources->b, &resources->c);
    assert_int_equal(st, LMMC_STATUS_OK);
    if (!lmmc_test_nearly_equal(resources->c.data[0], 1e300, 1e285) ||
        !lmmc_test_nearly_equal(resources->c.data[3], 1e300, 1e285)) {
        fail_msg("5.4 FAIL: Large value multiplication failed\n");
    }

    resources->a.data[0] = 1e-300;
    resources->a.data[1] = 0.0;
    resources->a.data[2] = 0.0;
    resources->a.data[3] = 1e-300;

    st = lmmc_mat_mul(&resources->a, &resources->b, &resources->c);
    assert_int_equal(st, LMMC_STATUS_OK);
    if (!lmmc_test_nearly_equal(resources->c.data[0], 1e-300, 1e-314) ||
        !lmmc_test_nearly_equal(resources->c.data[3], 1e-300, 1e-314)) {
        fail_msg("5.4 FAIL: Small value multiplication failed\n");
    }

    if (!isfinite(resources->c.data[0]) || !isfinite(resources->c.data[3])) {
        fail_msg("5.4 FAIL: Non-finite result detected\n");
    }

    lmmc_mat_destroy(&resources->a);
    lmmc_mat_destroy(&resources->b);
    lmmc_mat_destroy(&resources->c);
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
    lmmc_mat_destroy(&fixture->test_identity_sizes.a);
    lmmc_mat_destroy(&fixture->test_identity_sizes.b);
    lmmc_mat_destroy(&fixture->test_identity_sizes.c);
    lmmc_mat_destroy(&fixture->test_identity_sides.a);
    lmmc_mat_destroy(&fixture->test_identity_sides.eye);
    lmmc_mat_destroy(&fixture->test_identity_sides.c2);
    lmmc_mat_destroy(&fixture->test_rectangular_dimensions.a35);
    lmmc_mat_destroy(&fixture->test_rectangular_dimensions.b53);
    lmmc_mat_destroy(&fixture->test_rectangular_dimensions.c33);
    lmmc_mat_destroy(&fixture->test_rectangular_dimensions.bad);
    lmmc_mat_destroy(&fixture->test_finite_multiplication.a);
    lmmc_mat_destroy(&fixture->test_finite_multiplication.b);
    lmmc_mat_destroy(&fixture->test_finite_multiplication.c);
    free(fixture);
    *state = NULL;
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_identity_sizes, setup, teardown),
        cmocka_unit_test_setup_teardown(test_identity_sides, setup, teardown),
        cmocka_unit_test_setup_teardown(test_rectangular_dimensions, setup, teardown),
        cmocka_unit_test_setup_teardown(test_finite_multiplication, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
