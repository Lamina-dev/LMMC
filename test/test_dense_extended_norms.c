/**
 * @file test_dense_extended_norms.c
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
    struct test_zero_norms_resources {
        lmmc_vec_t z;
    } test_zero_norms;
    struct test_unit_norms_resources {
        lmmc_vec_t u;
    } test_unit_norms;
    struct test_signed_norms_resources {
        lmmc_vec_t v;
    } test_signed_norms;
    struct test_frobenius_resources {
        lmmc_mat_t eye;
        lmmc_mat_t m;
    } test_frobenius;
    struct test_extreme_vector_norm_resources {
        lmmc_vec_t extreme;
    } test_extreme_vector_norm;
    struct test_extreme_matrix_norm_resources {
        lmmc_mat_t extreme_matrix;
    } test_extreme_matrix_norm;
};

static void test_zero_norms(void **state) {
    struct test_fixture *fixture = *state;
    struct test_zero_norms_resources *resources = &fixture->test_zero_norms;

    lmmc_status_t st;
    size_t n = 5;

    st = lmmc_vec_create(n, &resources->z);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_vec_fill(&resources->z, 0.0);
    assert_int_equal(st, LMMC_STATUS_OK);

    lmmc_real_t norm2 = -1.0, norminf = -1.0, asum = -1.0;
    st = lmmc_vec_norm2(&resources->z, &norm2);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_vec_norm_inf(&resources->z, &norminf);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_vec_asum(&resources->z, &asum);
    assert_int_equal(st, LMMC_STATUS_OK);

    if (!lmmc_test_nearly_equal(norm2, 0.0, TEST_EPS_TIGHT) ||
        !lmmc_test_nearly_equal(norminf, 0.0, TEST_EPS_TIGHT) ||
        !lmmc_test_nearly_equal(asum, 0.0, TEST_EPS_TIGHT)) {
        fail_msg("5.11 FAIL: Zero vector norms not zero: L2=%g, inf=%g, asum=%g\n", norm2, norminf, asum);
    }
    lmmc_vec_destroy(&resources->z);
    return;
}

static void test_unit_norms(void **state) {
    struct test_fixture *fixture = *state;
    struct test_unit_norms_resources *resources = &fixture->test_unit_norms;

    lmmc_status_t st;
    lmmc_real_t norm2, norminf, asum;

    st = lmmc_vec_create(4, &resources->u);
    assert_int_equal(st, LMMC_STATUS_OK);
    resources->u.data[0] = 0.5;
    resources->u.data[1] = 0.5;
    resources->u.data[2] = 0.5;
    resources->u.data[3] = 0.5;

    st = lmmc_vec_norm2(&resources->u, &norm2);
    assert_int_equal(st, LMMC_STATUS_OK);
    if (!lmmc_test_nearly_equal(norm2, 1.0, TEST_EPS_TIGHT)) {
        fail_msg("5.11 FAIL: Unit vector L2 norm = %g, expected 1.0\n", norm2);
    }

    st = lmmc_vec_norm_inf(&resources->u, &norminf);
    assert_int_equal(st, LMMC_STATUS_OK);
    if (!lmmc_test_nearly_equal(norminf, 0.5, TEST_EPS_TIGHT)) {
        fail_msg("5.11 FAIL: Unit vector inf norm = %g, expected 0.5\n", norminf);
    }

    st = lmmc_vec_asum(&resources->u, &asum);
    assert_int_equal(st, LMMC_STATUS_OK);
    if (!lmmc_test_nearly_equal(asum, 2.0, TEST_EPS_TIGHT)) {
        fail_msg("5.11 FAIL: Unit vector asum = %g, expected 2.0\n", asum);
    }

    lmmc_vec_destroy(&resources->u);
    return;
}

static void test_signed_norms(void **state) {
    struct test_fixture *fixture = *state;
    struct test_signed_norms_resources *resources = &fixture->test_signed_norms;

    lmmc_status_t st;
    lmmc_real_t norm2, norminf, asum;

    st = lmmc_vec_create(2, &resources->v);
    assert_int_equal(st, LMMC_STATUS_OK);
    resources->v.data[0] = 3.0;
    resources->v.data[1] = -4.0;

    st = lmmc_vec_norm2(&resources->v, &norm2);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_vec_norm_inf(&resources->v, &norminf);
    assert_int_equal(st, LMMC_STATUS_OK);
    st = lmmc_vec_asum(&resources->v, &asum);
    assert_int_equal(st, LMMC_STATUS_OK);

    if (!lmmc_test_nearly_equal(norm2, 5.0, TEST_EPS_TIGHT)) {
        fail_msg("5.11 FAIL: [3,-4] L2=%g, expected 5.0\n", norm2);
    }
    if (!lmmc_test_nearly_equal(norminf, 4.0, TEST_EPS_TIGHT)) {
        fail_msg("5.11 FAIL: [3,-4] inf=%g, expected 4.0\n", norminf);
    }
    if (!lmmc_test_nearly_equal(asum, 7.0, TEST_EPS_TIGHT)) {
        fail_msg("5.11 FAIL: [3,-4] asum=%g, expected 7.0\n", asum);
    }

    lmmc_vec_destroy(&resources->v);
    return;
}

static void test_frobenius(void **state) {
    struct test_fixture *fixture = *state;
    struct test_frobenius_resources *resources = &fixture->test_frobenius;

    lmmc_status_t st;

    size_t sizes[] = {1, 2, 3, 5, 10};
    for (size_t si = 0; si < 5; si++) {
        size_t n = sizes[si];

        st = lmmc_mat_identity(n, &resources->eye);
        assert_int_equal(st, LMMC_STATUS_OK);

        lmmc_real_t fnorm = 0.0;
        st = lmmc_mat_norm_fro(&resources->eye, &fnorm);
        assert_int_equal(st, LMMC_STATUS_OK);

        lmmc_real_t expected = sqrt((double)n);
        if (!lmmc_test_nearly_equal(fnorm, expected, TEST_EPS_TIGHT)) {
            fail_msg("5.12 FAIL: ||I_%" PRIuMAX "||_F = %g, expected %g\n", (uintmax_t)(n), fnorm, expected);
        }
        lmmc_mat_destroy(&resources->eye);
    }

    st = lmmc_mat_create(2, 2, &resources->m);
    assert_int_equal(st, LMMC_STATUS_OK);
    resources->m.data[0] = 1.0;
    resources->m.data[1] = 2.0;
    resources->m.data[2] = 3.0;
    resources->m.data[3] = 4.0;

    lmmc_real_t fnorm = 0.0;
    st = lmmc_mat_norm_fro(&resources->m, &fnorm);
    assert_int_equal(st, LMMC_STATUS_OK);
    if (!lmmc_test_nearly_equal(fnorm, sqrt(30.0), TEST_EPS_TIGHT)) {
        fail_msg("5.12 FAIL: ||[[1,2],[3,4]]||_F = %g, expected %g\n", fnorm, sqrt(30.0));
    }
    lmmc_mat_destroy(&resources->m);
    return;
}

static void test_extreme_vector_norm(void **state) {
    struct test_fixture *fixture = *state;
    struct test_extreme_vector_norm_resources *resources = &fixture->test_extreme_vector_norm;

    lmmc_status_t st;

    lmmc_real_t norm = -1.0;
    const lmmc_real_t large = 1.0e308;
    const lmmc_real_t small = 1.0e-300;
    st = lmmc_vec_create(2, &resources->extreme);
    assert_int_equal(st, LMMC_STATUS_OK);
    resources->extreme.data[0] = large;
    resources->extreme.data[1] = large;
    st = lmmc_vec_norm2(&resources->extreme, &norm);
    assert_int_equal(st, LMMC_STATUS_OK);
    if (!isfinite(norm) ||
        fabs(norm / large - sqrt(2.0)) > 1.0e-15) {
        fail_msg("5.12 FAIL: large finite vector norm=%g\n", norm);
    }
    resources->extreme.data[0] = small;
    resources->extreme.data[1] = small;
    st = lmmc_vec_norm2(&resources->extreme, &norm);
    assert_int_equal(st, LMMC_STATUS_OK);
    if (!isfinite(norm) ||
        fabs(norm / small - sqrt(2.0)) > 1.0e-15) {
        fail_msg("5.12 FAIL: small nonzero vector norm=%g\n", norm);
    }
    lmmc_vec_destroy(&resources->extreme);
    return;
}

static void test_extreme_matrix_norm(void **state) {
    struct test_fixture *fixture = *state;
    struct test_extreme_matrix_norm_resources *resources = &fixture->test_extreme_matrix_norm;

    lmmc_status_t st;

    lmmc_real_t norm = -1.0;
    const lmmc_real_t large = 1.0e308;
    const lmmc_real_t small = 1.0e-300;
    st = lmmc_mat_create(1, 2, &resources->extreme_matrix);
    assert_int_equal(st, LMMC_STATUS_OK);
    resources->extreme_matrix.data[0] = large;
    resources->extreme_matrix.data[1] = large;
    st = lmmc_mat_norm_fro(&resources->extreme_matrix, &norm);
    assert_int_equal(st, LMMC_STATUS_OK);
    if (!isfinite(norm) ||
        fabs(norm / large - sqrt(2.0)) > 1.0e-15) {
        fail_msg("5.12 FAIL: large finite matrix norm=%g\n", norm);
    }
    resources->extreme_matrix.data[0] = small;
    resources->extreme_matrix.data[1] = small;
    st = lmmc_mat_norm_fro(&resources->extreme_matrix, &norm);
    assert_int_equal(st, LMMC_STATUS_OK);
    if (!isfinite(norm) ||
        fabs(norm / small - sqrt(2.0)) > 1.0e-15) {
        fail_msg("5.12 FAIL: small nonzero matrix norm=%g\n", norm);
    }
    lmmc_mat_destroy(&resources->extreme_matrix);
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
    lmmc_vec_destroy(&fixture->test_zero_norms.z);
    lmmc_vec_destroy(&fixture->test_unit_norms.u);
    lmmc_vec_destroy(&fixture->test_signed_norms.v);
    lmmc_mat_destroy(&fixture->test_frobenius.eye);
    lmmc_mat_destroy(&fixture->test_frobenius.m);
    lmmc_vec_destroy(&fixture->test_extreme_vector_norm.extreme);
    lmmc_mat_destroy(&fixture->test_extreme_matrix_norm.extreme_matrix);
    free(fixture);
    *state = NULL;
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_zero_norms, setup, teardown),
        cmocka_unit_test_setup_teardown(test_unit_norms, setup, teardown),
        cmocka_unit_test_setup_teardown(test_signed_norms, setup, teardown),
        cmocka_unit_test_setup_teardown(test_frobenius, setup, teardown),
        cmocka_unit_test_setup_teardown(test_extreme_vector_norm, setup, teardown),
        cmocka_unit_test_setup_teardown(test_extreme_matrix_norm, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
