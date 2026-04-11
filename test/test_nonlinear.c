#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static double lmmc_test_fn_sqrt2(double x, void* user_data) {
    (void)user_data;
    return x * x - 2.0;
}

static double lmmc_test_df_sqrt2(double x, void* user_data) {
    (void)user_data;
    return 2.0 * x;
}

static double lmmc_test_fn_linear(double x, void* user_data) {
    (void)user_data;
    return x - 1.0;
}

static double lmmc_test_df_linear(double x, void* user_data) {
    (void)user_data;
    (void)x;
    return 1.0;
}

static double lmmc_test_fn_nan(double x, void* user_data) {
    (void)user_data;
    if (x > 1.5) {
        return NAN;
    }
    return x - 1.0;
}

static double lmmc_test_fn_flat(double x, void* user_data) {
    (void)user_data;
    return x * x * x + 1.0;
}

static double lmmc_test_df_flat(double x, void* user_data) {
    (void)user_data;
    return 3.0 * x * x;
}

static double lmmc_test_fn_const(double x, void* user_data) {
    (void)x;
    (void)user_data;
    return 1.0;
}

static double lmmc_test_fn_exp_pos(double x, void* user_data) {
    (void)user_data;
    return exp(x);
}

static double lmmc_test_df_exp_pos(double x, void* user_data) {
    (void)user_data;
    return exp(x);
}

int main(void) {
    lmmc_nonlinear_config_t cfg = {0};
    lmmc_nonlinear_config_t cfg_hard = {0};
    lmmc_nonlinear_result_t result = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    int rc = 0;

    st = lmmc_nonlinear_default_config(NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto done;
    }

    st = lmmc_nonlinear_default_config(&cfg);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto done;
    }

    st = lmmc_bisection_solve(lmmc_test_fn_sqrt2, NULL, 0.0, 2.0, &cfg, &result);
    if (st != LMMC_STATUS_OK || result.converged != 1 ||
        !lmmc_test_nearly_equal(result.root, 1.4142135623730951, 2e-10) ||
        result.failure_reason != LMMC_NONLINEAR_FAILURE_NONE) {
        rc = 1;
        goto done;
    }

    st = lmmc_bisection_solve(lmmc_test_fn_linear, NULL, 1.0, 3.0, &cfg, &result);
    if (st != LMMC_STATUS_OK || result.converged != 1 || result.num_iter != 0 ||
        !lmmc_test_nearly_equal(result.root, 1.0, 1e-12)) {
        rc = 1;
        goto done;
    }

    st = lmmc_bisection_solve(lmmc_test_fn_linear, NULL, 2.0, 3.0, &cfg, &result);
    if (st != LMMC_STATUS_INVALID_ARGUMENT || result.failure_reason != LMMC_NONLINEAR_FAILURE_INVALID_BRACKET) {
        rc = 1;
        goto done;
    }

    cfg_hard = cfg;
    cfg_hard.abs_tol = 0.0;
    cfg_hard.rel_tol = 0.0;
    cfg_hard.max_iter = 1;

    st = lmmc_bisection_solve(lmmc_test_fn_sqrt2, NULL, 0.0, 2.0, &cfg_hard, &result);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE || result.converged != 0 || result.num_iter != 1 ||
        result.failure_reason != LMMC_NONLINEAR_FAILURE_MAX_ITER) {
        rc = 1;
        goto done;
    }

    st = lmmc_bisection_solve(lmmc_test_fn_nan, NULL, 0.0, 2.0, &cfg, &result);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE || result.failure_reason != LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE) {
        rc = 1;
        goto done;
    }

    st = lmmc_bisection_solve(NULL, NULL, 0.0, 2.0, &cfg, &result);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto done;
    }

    st = lmmc_bisection_solve(lmmc_test_fn_sqrt2, NULL, 2.0, 0.0, &cfg, &result);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto done;
    }

    st = lmmc_bisection_solve(lmmc_test_fn_sqrt2, NULL, 0.0, 2.0, &cfg, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto done;
    }

    cfg_hard = cfg;
    cfg_hard.max_iter = 0;
    st = lmmc_bisection_solve(lmmc_test_fn_sqrt2, NULL, 0.0, 2.0, &cfg_hard, &result);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto done;
    }

    st = lmmc_newton_solve(lmmc_test_fn_sqrt2, lmmc_test_df_sqrt2, NULL, 1.5, &cfg, &result);
    if (st != LMMC_STATUS_OK || result.converged != 1 ||
        !lmmc_test_nearly_equal(result.root, 1.4142135623730951, 1e-10) ||
        result.failure_reason != LMMC_NONLINEAR_FAILURE_NONE) {
        rc = 1;
        goto done;
    }

    st = lmmc_newton_solve(lmmc_test_fn_sqrt2, NULL, NULL, 1.5, &cfg, &result);
    if (st != LMMC_STATUS_OK || result.converged != 1 ||
        !lmmc_test_nearly_equal(result.root, 1.4142135623730951, 1e-8) ||
        result.failure_reason != LMMC_NONLINEAR_FAILURE_NONE) {
        rc = 1;
        goto done;
    }

    st = lmmc_newton_solve(lmmc_test_fn_flat, lmmc_test_df_flat, NULL, 0.0, &cfg, &result);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE || result.converged != 0 || result.num_iter != 1 ||
        result.failure_reason != LMMC_NONLINEAR_FAILURE_ZERO_DERIVATIVE) {
        rc = 1;
        goto done;
    }

    cfg_hard = cfg;
    cfg_hard.max_iter = 2;
    cfg_hard.abs_tol = 0.0;
    cfg_hard.rel_tol = 0.0;
    st = lmmc_newton_solve(lmmc_test_fn_exp_pos, lmmc_test_df_exp_pos, NULL, 0.0, &cfg_hard, &result);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE || result.converged != 0 || result.num_iter != 2 ||
        result.failure_reason != LMMC_NONLINEAR_FAILURE_MAX_ITER) {
        rc = 1;
        goto done;
    }

    cfg_hard = cfg;
    cfg_hard.min_step = 0.6;
    st = lmmc_newton_solve(lmmc_test_fn_sqrt2, lmmc_test_df_sqrt2, NULL, 1.0, &cfg_hard, &result);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE || result.converged != 0 ||
        result.failure_reason != LMMC_NONLINEAR_FAILURE_SINGULAR_STEP) {
        rc = 1;
        goto done;
    }

    st = lmmc_newton_solve(lmmc_test_fn_nan, NULL, NULL, 2.0, &cfg, &result);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE || result.failure_reason != LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE) {
        rc = 1;
        goto done;
    }

    st = lmmc_newton_solve(NULL, lmmc_test_df_sqrt2, NULL, 1.0, &cfg, &result);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto done;
    }

    st = lmmc_newton_solve(lmmc_test_fn_sqrt2, lmmc_test_df_sqrt2, NULL, INFINITY, &cfg, &result);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto done;
    }

    st = lmmc_newton_solve(lmmc_test_fn_sqrt2, lmmc_test_df_sqrt2, NULL, 1.0, &cfg, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto done;
    }

    cfg_hard = cfg;
    cfg_hard.max_iter = 0;
    st = lmmc_newton_solve(lmmc_test_fn_sqrt2, lmmc_test_df_sqrt2, NULL, 1.0, &cfg_hard, &result);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto done;
    }

    st = lmmc_secant_solve(lmmc_test_fn_sqrt2, NULL, 1.0, 2.0, &cfg, &result);
    if (st != LMMC_STATUS_OK || result.converged != 1 ||
        !lmmc_test_nearly_equal(result.root, 1.4142135623730951, 1e-10) ||
        result.failure_reason != LMMC_NONLINEAR_FAILURE_NONE) {
        rc = 1;
        goto done;
    }

    st = lmmc_secant_solve(lmmc_test_fn_linear, NULL, 0.0, 2.0, &cfg, &result);
    if (st != LMMC_STATUS_OK || result.converged != 1 ||
        !lmmc_test_nearly_equal(result.root, 1.0, 1e-12) ||
        result.failure_reason != LMMC_NONLINEAR_FAILURE_NONE) {
        rc = 1;
        goto done;
    }

    st = lmmc_secant_solve(lmmc_test_fn_const, NULL, 0.0, 2.0, &cfg, &result);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE || result.converged != 0 || result.num_iter != 1 ||
        result.failure_reason != LMMC_NONLINEAR_FAILURE_SINGULAR_STEP) {
        rc = 1;
        goto done;
    }

    cfg_hard = cfg;
    cfg_hard.max_iter = 2;
    cfg_hard.abs_tol = 0.0;
    cfg_hard.rel_tol = 0.0;
    st = lmmc_secant_solve(lmmc_test_fn_exp_pos, NULL, 0.0, 1.0, &cfg_hard, &result);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE || result.converged != 0 || result.num_iter != 2 ||
        result.failure_reason != LMMC_NONLINEAR_FAILURE_MAX_ITER) {
        rc = 1;
        goto done;
    }

    cfg_hard = cfg;
    cfg_hard.min_step = 1.0;
    st = lmmc_secant_solve(lmmc_test_fn_sqrt2, NULL, 1.0, 2.0, &cfg_hard, &result);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE || result.converged != 0 ||
        result.failure_reason != LMMC_NONLINEAR_FAILURE_SINGULAR_STEP) {
        rc = 1;
        goto done;
    }

    st = lmmc_secant_solve(lmmc_test_fn_nan, NULL, 0.0, 2.0, &cfg, &result);
    if (st != LMMC_STATUS_NUMERICAL_FAILURE || result.failure_reason != LMMC_NONLINEAR_FAILURE_NUMERICAL_ISSUE) {
        rc = 1;
        goto done;
    }

    st = lmmc_secant_solve(NULL, NULL, 1.0, 2.0, &cfg, &result);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto done;
    }

    st = lmmc_secant_solve(lmmc_test_fn_sqrt2, NULL, 1.0, 1.0, &cfg, &result);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto done;
    }

    st = lmmc_secant_solve(lmmc_test_fn_sqrt2, NULL, 1.0, 2.0, &cfg, NULL);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto done;
    }

    cfg_hard = cfg;
    cfg_hard.max_iter = 0;
    st = lmmc_secant_solve(lmmc_test_fn_sqrt2, NULL, 1.0, 2.0, &cfg_hard, &result);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto done;
    }

    if (lmmc_nonlinear_failure_string(LMMC_NONLINEAR_FAILURE_MAX_ITER) == NULL) {
        rc = 1;
        goto done;
    }

    if (lmmc_nonlinear_failure_string((lmmc_nonlinear_failure_t)9999) == NULL) {
        rc = 1;
        goto done;
    }

    st = lmmc_newton_solve(lmmc_test_fn_linear, lmmc_test_df_linear, NULL, 2.0, &cfg, &result);
    if (st != LMMC_STATUS_OK || result.converged != 1 || !lmmc_test_nearly_equal(result.root, 1.0, 1e-12)) {
        rc = 1;
        goto done;
    }

done:
    if (rc != 0) {
        printf("nonlinear test failed\n");
    }
    return rc;
}
