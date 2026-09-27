/**
 * @file test_eigen_extended_condition.c
 * @brief 矩阵条件数测试。
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
    struct test_cond_well_conditioned_resources {
        lmmc_mat_t mat;
    } test_cond_well_conditioned;
    struct test_cond_near_singular_resources {
        lmmc_mat_t mat;
    } test_cond_near_singular;
};

static void test_cond_well_conditioned(void **state) {
    struct test_fixture *fixture = *state;
    struct test_cond_well_conditioned_resources *resources = &fixture->test_cond_well_conditioned;

    assert_int_equal(lmmc_mat_create(3, 3, &resources->mat), LMMC_STATUS_OK);
    assert_int_equal(lmmc_mat_fill(&resources->mat, 0.0),
                     LMMC_STATUS_OK);
    resources->mat.data[0] = 1.0;
    resources->mat.data[4] = 1.0;
    resources->mat.data[8] = 1.0;

    lmmc_real_t cond_val;
    lmmc_status_t s = lmmc_cond(&resources->mat, &cond_val);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("cond(I) should succeed, got %d", (int)s);
    }
    if (!(fabs(cond_val - 1.0) < TOL)) {
        fail_msg("cond(I) should be 1.0, got %f", cond_val);
    }

    lmmc_mat_destroy(&resources->mat);
}

static void test_cond_near_singular(void **state) {
    struct test_fixture *fixture = *state;
    struct test_cond_near_singular_resources *resources = &fixture->test_cond_near_singular;

    assert_int_equal(lmmc_mat_create(2, 2, &resources->mat), LMMC_STATUS_OK);
    resources->mat.data[0] = 1.0;
    resources->mat.data[1] = 0.0;
    resources->mat.data[2] = 0.0;
    resources->mat.data[3] = 1e-12;

    lmmc_real_t cond_val;
    lmmc_status_t s = lmmc_cond(&resources->mat, &cond_val);
    if (!(s == LMMC_STATUS_OK)) {
        fail_msg("cond near-singular should succeed, got %d", (int)s);
    }
    if (!(isfinite(cond_val) && cond_val > 1e10)) {
        fail_msg("cond of near-singular should be large, got %f", cond_val);
    }

    lmmc_mat_destroy(&resources->mat);
}

static int setup(void **state) {
    struct test_fixture *fixture = calloc(1, sizeof(*fixture));
    assert_non_null(fixture);
    *state = fixture;
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_destroy(&fixture->test_cond_well_conditioned.mat);
    lmmc_mat_destroy(&fixture->test_cond_near_singular.mat);
    free(fixture);
    *state = NULL;
    return 0;
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_cond_well_conditioned, setup, teardown),
        cmocka_unit_test_setup_teardown(test_cond_near_singular, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
