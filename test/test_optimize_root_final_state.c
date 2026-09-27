/**
 * @file test_optimize_root_final_state.c
 * @brief 求根器最终状态测试。
 */
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "lmmc/optimize.h"
#include "test_common.h"
#include "test_optimize_common.h"

typedef struct {
    size_t n;
    lmmc_vec_t x;
    int broyden;
} vector_fixture_t;

static int setup_vector(void **state) {
    vector_fixture_t *fixture = *state;
    assert_int_equal(lmmc_vec_create(fixture->n, &fixture->x), LMMC_STATUS_OK);
    return 0;
}

static int teardown_vector(void **state) {
    vector_fixture_t *fixture = *state;
    lmmc_vec_destroy(&fixture->x);
    return 0;
}

static lmmc_status_t sqrt_domain_residual(
    const lmmc_vec_t *x, lmmc_vec_t *r, void *user_data) {
    (void)user_data;
    if (x->data[0] < 0.0)
        return LMMC_STATUS_NUMERICAL_FAILURE;
    r->data[0] = sqrt(x->data[0]) + 1.0;
    return LMMC_STATUS_OK;
}
static void test_root_converged(void **state) {
    vector_fixture_t *fixture = *state;
    int broyden = fixture->broyden;
    lmmc_vec_t x = fixture->x;
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};

    lmmc_optimize_default_config(&cfg);
    cfg.max_iter = 1;
    lmmc_status_t st;
    x.data[0] = 1.0;
    st = broyden
             ? lmmc_nleq_broyden(lmmc_test_identity_system_f, NULL, &x, &cfg, &result)
             : lmmc_nleq_newton(lmmc_test_identity_system_f, lmmc_test_identity_system_j,
                                NULL, &x, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK || !result.converged ||
                 result.failure_reason != LMMC_OPT_FAILURE_NONE ||
                 result.num_iter != 1 || x.data[0] != 0.0 ||
                 result.final_residual != 0.0);
}
static void test_root_exhausted(void **state) {
    vector_fixture_t *fixture = *state;
    int broyden = fixture->broyden;
    lmmc_vec_t x = fixture->x;
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};

    lmmc_optimize_default_config(&cfg);
    cfg.max_iter = 1;
    lmmc_status_t st;
    x.data[0] = 1.0;
    st = broyden
             ? lmmc_nleq_broyden(lmmc_test_scalar_square_system_f, NULL, &x, &cfg, &result)
             : lmmc_nleq_newton(lmmc_test_scalar_square_system_f, NULL,
                                NULL, &x, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK || result.converged ||
                 result.failure_reason != LMMC_OPT_FAILURE_MAX_ITER ||
                 result.num_iter != 1 || fabs(x.data[0] - 1.5) > 1e-6 ||
                 !isfinite(result.final_residual) ||
                 fabs(result.final_residual - fabs(x.data[0] * x.data[0] - 2.0)) > 1e-12);
}
static void test_root_domain_failure(void **state) {
    vector_fixture_t *fixture = *state;
    int broyden = fixture->broyden;
    lmmc_vec_t x = fixture->x;
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};

    lmmc_optimize_default_config(&cfg);
    cfg.max_iter = 1;
    lmmc_status_t st;
    x.data[0] = 1.0;
    st = broyden
             ? lmmc_nleq_broyden(sqrt_domain_residual, NULL, &x, &cfg, &result)
             : lmmc_nleq_newton(sqrt_domain_residual, NULL,
                                NULL, &x, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK || result.converged ||
                 result.failure_reason != LMMC_OPT_FAILURE_NUMERICAL_ISSUE ||
                 result.num_iter != 1 || !(x.data[0] < 0.0) ||
                 !isnan(result.final_residual));
}

static void check_newton_initial_state(lmmc_vec_t *vector,
                                       const lmmc_real_t values[2], int converged, lmmc_optimize_result_t *result) {
    lmmc_vec_t x = *vector;
    lmmc_optimize_config_t cfg = {0};
    memcpy(x.data, values, 2 * sizeof(lmmc_real_t));
    lmmc_optimize_default_config(&cfg);
    cfg.abs_tol = DBL_MAX;
    lmmc_status_t st = lmmc_nleq_newton(
        lmmc_test_identity_system_f, lmmc_test_identity_system_j, NULL, &x, &cfg, result);
    const lmmc_optimize_failure_t expected_failure = converged
                                                         ? LMMC_OPT_FAILURE_NONE
                                                         : LMMC_OPT_FAILURE_NUMERICAL_ISSUE;
    const int failed = st != LMMC_STATUS_OK ||
                       result->converged != converged ||
                       result->failure_reason != expected_failure ||
                       result->num_iter != 0 ||
                       memcmp(x.data, values, 2 * sizeof(lmmc_real_t)) != 0;
    assert_false(failed);
}

static void test_root_nonfinite_norm(void **state) {
    vector_fixture_t *fixture = *state;
    const lmmc_real_t values[][2] = {
        {INFINITY, NAN}, {-INFINITY, NAN}, {NAN, INFINITY}, {DBL_MAX, DBL_MAX}};
    const lmmc_real_t expected[] = {INFINITY, INFINITY, NAN, INFINITY};
    for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i) {
        lmmc_optimize_result_t result = {0};
        check_newton_initial_state(&fixture->x, values[i], 0, &result);
        const int matches = isnan(expected[i])
                                ? isnan(result.final_residual)
                                : result.final_residual == expected[i];
        assert_true(matches);
    }
}

static void test_root_scaled_norm(void **state) {
    vector_fixture_t *fixture = *state;
    const lmmc_real_t values[][2] = {{3e200, 4e200}, {3e-200, 4e-200}};
    const lmmc_real_t scales[] = {1e200, 1e-200};
    for (size_t i = 0; i < sizeof(scales) / sizeof(scales[0]); ++i) {
        lmmc_optimize_result_t result = {0};
        check_newton_initial_state(&fixture->x, values[i], 1, &result);
        const lmmc_real_t relative_error =
            fabs(result.final_residual / scales[i] - 5.0) / 5.0;
        assert_true((relative_error <= 8.0 * DBL_EPSILON));
    }
}

static void test_root_zero_norm(void **state) {
    vector_fixture_t *fixture = *state;
    const lmmc_real_t values[2] = {0.0, -0.0};
    lmmc_optimize_result_t result = {0};
    check_newton_initial_state(&fixture->x, values, 1, &result);
    assert_false(result.final_residual != 0.0 || signbit(result.final_residual));
}

int main(void) {
    vector_fixture_t newton_converged = {.n = 1, .broyden = 0};
    vector_fixture_t newton_exhausted = {.n = 1, .broyden = 0};
    vector_fixture_t newton_domain_failure = {.n = 1, .broyden = 0};
    vector_fixture_t broyden_converged = {.n = 1, .broyden = 1};
    vector_fixture_t broyden_exhausted = {.n = 1, .broyden = 1};
    vector_fixture_t broyden_domain_failure = {.n = 1, .broyden = 1};
    vector_fixture_t nonfinite_norm = {.n = 2};
    vector_fixture_t scaled_norm = {.n = 2};
    vector_fixture_t zero_norm = {.n = 2};
    const struct CMUnitTest tests[] = {
        {"newton_converged", test_root_converged, setup_vector, teardown_vector, &newton_converged},
        {"newton_exhausted", test_root_exhausted, setup_vector, teardown_vector, &newton_exhausted},
        {"newton_domain_failure", test_root_domain_failure, setup_vector, teardown_vector, &newton_domain_failure},
        {"broyden_converged", test_root_converged, setup_vector, teardown_vector, &broyden_converged},
        {"broyden_exhausted", test_root_exhausted, setup_vector, teardown_vector, &broyden_exhausted},
        {"broyden_domain_failure", test_root_domain_failure, setup_vector, teardown_vector, &broyden_domain_failure},
        cmocka_unit_test_prestate_setup_teardown(test_root_nonfinite_norm, setup_vector, teardown_vector, &nonfinite_norm),
        cmocka_unit_test_prestate_setup_teardown(test_root_scaled_norm, setup_vector, teardown_vector, &scaled_norm),
        cmocka_unit_test_prestate_setup_teardown(test_root_zero_norm, setup_vector, teardown_vector, &zero_norm),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
