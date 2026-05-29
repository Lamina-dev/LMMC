/**
 * @file test_nonlinear_extended.c
 * 针对 LMMC 中 nonlinear extended 相关接口的单元测试。
 */
#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"


static double fn_cos_minus_x(double x, void* user_data) {
    (void)user_data;
    return cos(x) - x;
}


static double fn_cubic(double x, void* user_data) {
    (void)user_data;
    return x * x * x - 2.0 * x - 5.0;
}

static double df_cubic(double x, void* user_data) {
    (void)user_data;
    return 3.0 * x * x - 2.0;
}


static double fn_exp_minus_3(double x, void* user_data) {
    (void)user_data;
    return exp(x) - 3.0;
}


static double fn_sin(double x, void* user_data) {
    (void)user_data;
    return sin(x);
}


static double fn_endpoint_zero(double x, void* user_data) {
    (void)user_data;
    return x - 1.0;
}


static double fn_cbrt(double x, void* user_data) {
    (void)user_data;
    return cbrt(x) - 1.0;
}

static double df_cbrt(double x, void* user_data) {
    (void)user_data;
    if (x == 0.0) return 1e300;
    return 1.0 / (3.0 * cbrt(x * x));
}

int main(void) {
    lmmc_nonlinear_config_t cfg = {0};
    lmmc_nonlinear_result_t result = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    int rc = 0;

    st = lmmc_nonlinear_default_config(&cfg);
    if (st != LMMC_STATUS_OK) { rc = 1; goto done; }


    {
        double expected_root = 0.7390851332151607;
        st = lmmc_bisection_solve(fn_cos_minus_x, NULL, 0.0, 1.0, &cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(result.root, expected_root, 1e-10)) { rc = 1; goto done; }
    }


    {
        double expected_root = 2.0945514815423265;
        st = lmmc_newton_solve(fn_cubic, df_cubic, NULL, 2.0, &cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(result.root, expected_root, 1e-10)) { rc = 1; goto done; }
    }


    {
        double expected_root = log(3.0);
        st = lmmc_secant_solve(fn_exp_minus_3, NULL, 0.0, 2.0, &cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(result.root, expected_root, 1e-10)) { rc = 1; goto done; }
    }


    {
        double pi_val = 3.14159265358979323846;


        st = lmmc_bisection_solve(fn_sin, NULL, 2.5, 3.8, &cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(result.root, pi_val, 1e-9)) { rc = 1; goto done; }


        st = lmmc_bisection_solve(fn_sin, NULL, 5.5, 6.8, &cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(result.root, 2.0 * pi_val, 1e-9)) { rc = 1; goto done; }
    }


    {

        st = lmmc_bisection_solve(fn_endpoint_zero, NULL, 1.0, 3.0, &cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(result.root, 1.0, 1e-14)) { rc = 1; goto done; }
        if (result.num_iter != 0) { rc = 1; goto done; }
    }


    {

        st = lmmc_newton_solve(fn_cbrt, df_cbrt, NULL, 0.5, &cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(result.root, 1.0, 1e-10)) { rc = 1; goto done; }


        st = lmmc_newton_solve(fn_cbrt, df_cbrt, NULL, 1e-20, &cfg, &result);
        if (st != LMMC_STATUS_OK && st != LMMC_STATUS_NUMERICAL_FAILURE) {
            rc = 1; goto done;
        }
    }


    {

        st = lmmc_bisection_solve(fn_sin, NULL, 0.5, 2.5, &cfg, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT ||
            result.failure_reason != LMMC_NONLINEAR_FAILURE_INVALID_BRACKET) {
            rc = 1; goto done;
        }
    }


    {
        lmmc_nonlinear_config_t strict_cfg = cfg;
        strict_cfg.abs_tol = 1e-15;
        strict_cfg.rel_tol = 0.0;
        strict_cfg.max_iter = 200;


        double expected_root = 0.7390851332151607;
        st = lmmc_bisection_solve(fn_cos_minus_x, NULL, 0.0, 1.0, &strict_cfg, &result);
        if (st == LMMC_STATUS_OK) {
            if (result.converged != 1) { rc = 1; goto done; }
            if (!lmmc_test_nearly_equal(result.root, expected_root, 1e-14)) { rc = 1; goto done; }
        } else if (st == LMMC_STATUS_NUMERICAL_FAILURE) {

            if (result.failure_reason != LMMC_NONLINEAR_FAILURE_MAX_ITER) { rc = 1; goto done; }

            if (!lmmc_test_nearly_equal(result.root, expected_root, 1e-12)) { rc = 1; goto done; }
        } else {
            rc = 1; goto done;
        }


        strict_cfg.max_iter = 100;
        st = lmmc_newton_solve(fn_cubic, df_cubic, NULL, 2.0, &strict_cfg, &result);
        if (st == LMMC_STATUS_OK) {
            if (result.converged != 1) { rc = 1; goto done; }
            if (!lmmc_test_nearly_equal(result.root, 2.0945514815423265, 1e-14)) { rc = 1; goto done; }
        } else if (st == LMMC_STATUS_NUMERICAL_FAILURE) {
            if (result.failure_reason != LMMC_NONLINEAR_FAILURE_MAX_ITER) { rc = 1; goto done; }
        } else {
            rc = 1; goto done;
        }
    }

done:
    if (rc != 0) {
        printf("nonlinear extended test failed\n");
    }
    return rc;
}
