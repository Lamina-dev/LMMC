/**
 * @file test_eigen_extended_boundaries.c
 * @brief 特征分解扩展接口的边界测试。
 */
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/status.h"
#include "test_common.h"

#define TOL 1e-10
#define TOL_LOOSE 1e-8
#define MAT_ELEM(mat, i, j) ((mat)->data[(i) * (mat)->stride + (j)])
struct test_fixture {
    struct test_scalar_symmetric_resources {
        lmmc_mat_t mat;
        lmmc_eigen_sym_result_t result;
    } test_scalar_symmetric;
    struct test_scalar_general_resources {
        lmmc_mat_t mat;
        lmmc_eigen_gen_result_t result;
    } test_scalar_general;
    struct test_scalar_svd_resources {
        lmmc_mat_t mat;
        lmmc_svd_result_t result;
    } test_scalar_svd;
    struct test_scalar_pinv_resources {
        lmmc_mat_t mat;
        lmmc_mat_t pinv;
    } test_scalar_pinv;
    struct test_scalar_condition_resources {
        lmmc_mat_t mat;
    } test_scalar_condition;
    struct test_error_handling_resources {
        lmmc_eigen_sym_result_t sym_result;
        lmmc_eigen_gen_result_t gen_result;
        lmmc_svd_result_t svd_result;
        lmmc_mat_t mat;
        lmmc_mat_t pinv;
    } test_error_handling;
};

static void test_scalar_symmetric(void **state) {
    struct test_fixture *fixture = *state;
    struct test_scalar_symmetric_resources *resources = &fixture->test_scalar_symmetric;

    assert_int_equal(lmmc_mat_create(1, 1, &resources->mat), LMMC_STATUS_OK);
    resources->mat.data[0] = 7.5;

    lmmc_status_t s = lmmc_eigen_symmetric(&resources->mat, &resources->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("1x1 sym eigen should succeed");
    }
    if (!(fabs(resources->result.eigenvalues.data[0] - 7.5) < TOL)) {
        fail_msg("1x1 eigenvalue should be 7.5, got %f", resources->result.eigenvalues.data[0]);
    }
    if (!(fabs(resources->result.eigenvectors.data[0] - 1.0) < TOL)) {
        fail_msg("1x1 eigenvector should be 1.0, got %f", resources->result.eigenvectors.data[0]);
    }

    lmmc_eigen_sym_result_destroy(&resources->result);
    lmmc_mat_destroy(&resources->mat);
}
static void test_scalar_general(void **state) {
    struct test_fixture *fixture = *state;
    struct test_scalar_general_resources *resources = &fixture->test_scalar_general;

    assert_int_equal(lmmc_mat_create(1, 1, &resources->mat), LMMC_STATUS_OK);
    resources->mat.data[0] = -3.0;

    lmmc_status_t s = lmmc_eigen_general(&resources->mat, &resources->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("1x1 gen eigen should succeed");
    }
    if (!(fabs(resources->result.real_parts.data[0] - (-3.0)) < TOL)) {
        fail_msg("1x1 gen eigenvalue real should be -3.0, got %f", resources->result.real_parts.data[0]);
    }
    if (!(fabs(resources->result.imag_parts.data[0]) < TOL)) {
        fail_msg("1x1 gen eigenvalue imag should be 0, got %f", resources->result.imag_parts.data[0]);
    }

    lmmc_eigen_gen_result_destroy(&resources->result);
    lmmc_mat_destroy(&resources->mat);
}
static void test_scalar_svd(void **state) {
    struct test_fixture *fixture = *state;
    struct test_scalar_svd_resources *resources = &fixture->test_scalar_svd;

    assert_int_equal(lmmc_mat_create(1, 1, &resources->mat), LMMC_STATUS_OK);
    resources->mat.data[0] = -4.0;

    lmmc_status_t s = lmmc_svd(&resources->mat, &resources->result);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("1x1 SVD should succeed");
    }
    if (!(fabs(resources->result.sigma.data[0] - 4.0) < TOL)) {
        fail_msg("1x1 SVD sigma should be 4.0, got %f", resources->result.sigma.data[0]);
    }

    lmmc_svd_result_destroy(&resources->result);
    lmmc_mat_destroy(&resources->mat);
}
static void test_scalar_pinv(void **state) {
    struct test_fixture *fixture = *state;
    struct test_scalar_pinv_resources *resources = &fixture->test_scalar_pinv;

    assert_int_equal(lmmc_mat_create(1, 1, &resources->mat), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_create(1, 1, &resources->pinv), LMMC_STATUS_OK);
    resources->mat.data[0] = 2.0;

    lmmc_status_t s = lmmc_pinv(&resources->mat, 0.0, &resources->pinv);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("1x1 pinv should succeed");
    }
    if (!(fabs(resources->pinv.data[0] - 0.5) < TOL)) {
        fail_msg("1x1 pinv should be 0.5, got %f", resources->pinv.data[0]);
    }

    lmmc_mat_destroy(&resources->pinv);
    lmmc_mat_destroy(&resources->mat);
}
static void test_scalar_condition(void **state) {
    struct test_fixture *fixture = *state;
    struct test_scalar_condition_resources *resources = &fixture->test_scalar_condition;

    assert_int_equal(lmmc_mat_create(1, 1, &resources->mat), LMMC_STATUS_OK);
    resources->mat.data[0] = 5.0;

    lmmc_real_t cond_val;
    lmmc_status_t s = lmmc_cond(&resources->mat, &cond_val);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("1x1 cond should succeed");
    }
    if (!(fabs(cond_val - 1.0) < TOL)) {
        fail_msg("1x1 cond should be 1.0, got %f", cond_val);
    }

    lmmc_mat_destroy(&resources->mat);
}
static void test_error_handling(void **state) {
    struct test_fixture *fixture = *state;
    struct test_error_handling_resources *resources = &fixture->test_error_handling;

    lmmc_status_t s;

    s = lmmc_eigen_symmetric(NULL, &resources->sym_result);
    if (!(s == LMMC_STATUS_INVALID_ARGUMENT)) {
        fail_msg("eigen_symmetric(NULL) should return INVALID_ARGUMENT, got %d", (int)s);
    }

    {

        assert_int_equal(lmmc_mat_create(2, 2, &resources->mat), LMMC_STATUS_OK);
        s = lmmc_eigen_symmetric(&resources->mat, NULL);
        if (!(s == LMMC_STATUS_INVALID_ARGUMENT)) {
            fail_msg("eigen_symmetric(mat, NULL) should return INVALID_ARGUMENT, got %d", (int)s);
        }
        lmmc_mat_destroy(&resources->mat);
    }

    {

        assert_int_equal(lmmc_mat_create(2, 3, &resources->mat), LMMC_STATUS_OK);
        s = lmmc_eigen_symmetric(&resources->mat, &resources->sym_result);
        if (!(s == LMMC_STATUS_INVALID_ARGUMENT)) {
            fail_msg("eigen_symmetric(2x3) should return INVALID_ARGUMENT, got %d", (int)s);
        }
        lmmc_mat_destroy(&resources->mat);
    }

    s = lmmc_eigen_general(NULL, &resources->gen_result);
    if (!(s == LMMC_STATUS_INVALID_ARGUMENT)) {
        fail_msg("eigen_general(NULL) should return INVALID_ARGUMENT, got %d", (int)s);
    }

    {

        assert_int_equal(lmmc_mat_create(3, 2, &resources->mat), LMMC_STATUS_OK);
        s = lmmc_eigen_general(&resources->mat, &resources->gen_result);
        if (!(s == LMMC_STATUS_INVALID_ARGUMENT)) {
            fail_msg("eigen_general(3x2) should return INVALID_ARGUMENT, got %d", (int)s);
        }
        lmmc_mat_destroy(&resources->mat);
    }

    s = lmmc_svd(NULL, &resources->svd_result);
    if (!(s == LMMC_STATUS_INVALID_ARGUMENT)) {
        fail_msg("svd(NULL) should return INVALID_ARGUMENT, got %d", (int)s);
    }

    {

        assert_int_equal(lmmc_mat_create(2, 2, &resources->pinv), LMMC_STATUS_OK);
        s = lmmc_pinv(NULL, 0.0, &resources->pinv);
        if (!(s == LMMC_STATUS_INVALID_ARGUMENT)) {
            fail_msg("pinv(NULL) should return INVALID_ARGUMENT, got %d", (int)s);
        }
        lmmc_mat_destroy(&resources->pinv);
    }

    {
        lmmc_real_t cond_val;
        s = lmmc_cond(NULL, &cond_val);
        if (!(s == LMMC_STATUS_INVALID_ARGUMENT)) {
            fail_msg("cond(NULL) should return INVALID_ARGUMENT, got %d", (int)s);
        }
    }
}

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_destroy(&fixture->test_scalar_symmetric.mat);
    lmmc_eigen_sym_result_destroy(&fixture->test_scalar_symmetric.result);
    lmmc_mat_destroy(&fixture->test_scalar_general.mat);
    lmmc_eigen_gen_result_destroy(&fixture->test_scalar_general.result);
    lmmc_mat_destroy(&fixture->test_scalar_svd.mat);
    lmmc_svd_result_destroy(&fixture->test_scalar_svd.result);
    lmmc_mat_destroy(&fixture->test_scalar_pinv.mat);
    lmmc_mat_destroy(&fixture->test_scalar_pinv.pinv);
    lmmc_mat_destroy(&fixture->test_scalar_condition.mat);
    lmmc_eigen_sym_result_destroy(&fixture->test_error_handling.sym_result);
    lmmc_eigen_gen_result_destroy(&fixture->test_error_handling.gen_result);
    lmmc_svd_result_destroy(&fixture->test_error_handling.svd_result);
    lmmc_mat_destroy(&fixture->test_error_handling.mat);
    lmmc_mat_destroy(&fixture->test_error_handling.pinv);
    free(fixture);
    *state = NULL;
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_scalar_symmetric, setup, teardown),
        cmocka_unit_test_setup_teardown(test_scalar_general, setup, teardown),
        cmocka_unit_test_setup_teardown(test_scalar_svd, setup, teardown),
        cmocka_unit_test_setup_teardown(test_scalar_pinv, setup, teardown),
        cmocka_unit_test_setup_teardown(test_scalar_condition, setup, teardown),
        cmocka_unit_test_setup_teardown(test_error_handling, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
