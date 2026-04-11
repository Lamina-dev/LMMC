#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

static double lmmc_pi_value(void) {
    return 3.14159265358979323846;
}

static lmmc_status_t rhs_exp(double t, const double* y, double* y_prime, size_t dim, void* user_data) {
    (void)t;
    (void)user_data;
    if (dim != 1) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    y_prime[0] = y[0];
    return LMMC_STATUS_OK;
}

static lmmc_status_t rhs_decay(double t, const double* y, double* y_prime, size_t dim, void* user_data) {
    (void)t;
    (void)user_data;
    if (dim != 1) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    y_prime[0] = -2.0 * y[0];
    return LMMC_STATUS_OK;
}

static lmmc_status_t rhs_nan(double t, const double* y, double* y_prime, size_t dim, void* user_data) {
    (void)t;
    (void)y;
    (void)user_data;
    if (dim == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    y_prime[0] = NAN;
    return LMMC_STATUS_OK;
}

static lmmc_status_t rhs_fail(double t, const double* y, double* y_prime, size_t dim, void* user_data) {
    (void)t;
    (void)y;
    (void)y_prime;
    (void)dim;
    (void)user_data;
    return LMMC_STATUS_INVALID_ARGUMENT;
}

static lmmc_status_t rhs_harmonic(double t, const double* y, double* y_prime, size_t dim, void* user_data) {
    (void)t;
    (void)user_data;
    if (dim != 2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    y_prime[0] = y[1];
    y_prime[1] = -y[0];
    return LMMC_STATUS_OK;
}

int main(void) {
    lmmc_ode_config_t cfg = {0};
    lmmc_ode_result_t result = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    int rc = 0;

    st = lmmc_ode_default_config(0.0, 1.0, 0, &cfg);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto done;
    }

    st = lmmc_ode_default_config(1.0, 1.0, 1, &cfg);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        rc = 1;
        goto done;
    }

    st = lmmc_ode_default_config(0.0, 1.0, 1, &cfg);
    if (st != LMMC_STATUS_OK) {
        rc = 1;
        goto done;
    }

    {
        double y[1] = {1.0};
        st = lmmc_ode_euler_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1 ||
            !lmmc_test_nearly_equal(y[0], exp(1.0), 2e-2) ||
            result.failure_reason != LMMC_ODE_FAILURE_NONE) {
            rc = 1;
            goto done;
        }
    }

    {
        double y[1] = {1.0};
        lmmc_ode_config_t rk4_cfg = cfg;
        rk4_cfg.initial_step = 0.05;
        rk4_cfg.min_step = 0.05;
        rk4_cfg.max_step = 0.05;
        rk4_cfg.max_steps = 100;

        st = lmmc_ode_rk4_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &rk4_cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1 ||
            !lmmc_test_nearly_equal(y[0], exp(1.0), 1e-5) ||
            result.failure_reason != LMMC_ODE_FAILURE_NONE) {
            rc = 1;
            goto done;
        }
    }

    {
        double y[1] = {1.0};
        lmmc_ode_config_t adapt_cfg = cfg;
        adapt_cfg.initial_step = 0.2;
        adapt_cfg.min_step = 1e-8;
        adapt_cfg.max_step = 0.2;
        adapt_cfg.abs_tol = 1e-12;
        adapt_cfg.rel_tol = 1e-10;
        adapt_cfg.max_steps = 10000;

        st = lmmc_ode_rk45_solve(rhs_decay, NULL, 1, 0.0, 1.0, y, &adapt_cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1 ||
            !lmmc_test_nearly_equal(y[0], exp(-2.0), 1e-7) ||
            result.failure_reason != LMMC_ODE_FAILURE_NONE) {
            rc = 1;
            goto done;
        }
    }

    {
        double y[2] = {1.0, 0.0};
        lmmc_ode_config_t adapt_cfg = cfg;
        adapt_cfg.initial_step = 0.1;
        adapt_cfg.min_step = 1e-8;
        adapt_cfg.max_step = 0.2;
        adapt_cfg.abs_tol = 1e-10;
        adapt_cfg.rel_tol = 1e-9;
        adapt_cfg.max_steps = 200000;

        st = lmmc_ode_rk45_solve(rhs_harmonic, NULL, 2, 0.0, 2.0 * lmmc_pi_value(), y, &adapt_cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1 ||
            !lmmc_test_nearly_equal(y[0], 1.0, 1e-5) ||
            !lmmc_test_nearly_equal(y[1], 0.0, 1e-5)) {
            rc = 1;
            goto done;
        }
    }

    {
        double y[1] = {1.0};
        st = lmmc_ode_euler_solve(rhs_exp, NULL, 0, 0.0, 1.0, y, &cfg, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT || result.failure_reason != LMMC_ODE_FAILURE_INVALID_DIMENSION) {
            rc = 1;
            goto done;
        }
    }

    {
        double y[1] = {1.0};
        st = lmmc_ode_euler_solve(rhs_exp, NULL, 1, 1.0, 0.0, y, &cfg, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT || result.failure_reason != LMMC_ODE_FAILURE_INVALID_STEP) {
            rc = 1;
            goto done;
        }
    }

    {
        double y[1] = {1.0};
        st = lmmc_ode_euler_solve(NULL, NULL, 1, 0.0, 1.0, y, &cfg, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            rc = 1;
            goto done;
        }
    }

    {
        st = lmmc_ode_euler_solve(rhs_exp, NULL, 1, 0.0, 1.0, NULL, &cfg, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            rc = 1;
            goto done;
        }
    }

    {
        double y[1] = {1.0};
        st = lmmc_ode_euler_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &cfg, NULL);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) {
            rc = 1;
            goto done;
        }
    }

    {
        double y[1] = {1.0};
        lmmc_ode_config_t bad_cfg = cfg;
        bad_cfg.abs_tol = -1.0;
        st = lmmc_ode_euler_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &bad_cfg, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT || result.failure_reason != LMMC_ODE_FAILURE_TOLERANCE_INCONSISTENT) {
            rc = 1;
            goto done;
        }
    }

    {
        double y[1] = {1.0};
        lmmc_ode_config_t bad_cfg = cfg;
        bad_cfg.min_step = 0.5;
        bad_cfg.max_step = 0.1;
        st = lmmc_ode_rk4_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &bad_cfg, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT || result.failure_reason != LMMC_ODE_FAILURE_INVALID_STEP) {
            rc = 1;
            goto done;
        }
    }

    {
        double y[1] = {1.0};
        lmmc_ode_config_t bad_cfg = cfg;
        bad_cfg.initial_step = 0.0;
        st = lmmc_ode_rk4_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &bad_cfg, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT || result.failure_reason != LMMC_ODE_FAILURE_INVALID_STEP) {
            rc = 1;
            goto done;
        }
    }

    {
        double y[1] = {1.0};
        lmmc_ode_config_t bad_cfg = cfg;
        bad_cfg.max_steps = 0;
        st = lmmc_ode_euler_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &bad_cfg, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT || result.failure_reason != LMMC_ODE_FAILURE_INVALID_STEP) {
            rc = 1;
            goto done;
        }
    }

    {
        double y[1] = {1.0};
        st = lmmc_ode_euler_solve(rhs_fail, NULL, 1, 0.0, 1.0, y, &cfg, &result);
        if (st != LMMC_STATUS_NUMERICAL_FAILURE || result.failure_reason != LMMC_ODE_FAILURE_RHS_EVAL_FAILED) {
            rc = 1;
            goto done;
        }
    }

    {
        double y[1] = {1.0};
        st = lmmc_ode_rk4_solve(rhs_nan, NULL, 1, 0.0, 1.0, y, &cfg, &result);
        if (st != LMMC_STATUS_NUMERICAL_FAILURE || result.failure_reason != LMMC_ODE_FAILURE_NUMERICAL_ISSUE) {
            rc = 1;
            goto done;
        }
    }

    {
        double y[1] = {NAN};
        st = lmmc_ode_euler_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &cfg, &result);
        if (st != LMMC_STATUS_NUMERICAL_FAILURE || result.failure_reason != LMMC_ODE_FAILURE_NUMERICAL_ISSUE) {
            rc = 1;
            goto done;
        }
    }

    {
        double y[1] = {1.0};
        lmmc_ode_config_t hard_cfg = cfg;
        hard_cfg.initial_step = 1e-4;
        hard_cfg.min_step = 1e-4;
        hard_cfg.max_step = 1e-4;
        hard_cfg.max_steps = 1;
        st = lmmc_ode_euler_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &hard_cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 0 || result.failure_reason != LMMC_ODE_FAILURE_MAX_STEPS) {
            rc = 1;
            goto done;
        }
    }

    {
        double y[1] = {1.0};
        lmmc_ode_config_t hard_cfg = cfg;
        hard_cfg.initial_step = 1.0;
        hard_cfg.min_step = 1.0;
        hard_cfg.max_step = 1.0;
        hard_cfg.abs_tol = 0.0;
        hard_cfg.rel_tol = 0.0;
        hard_cfg.max_steps = 100;
        st = lmmc_ode_rk45_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &hard_cfg, &result);
        if (st != LMMC_STATUS_NUMERICAL_FAILURE || result.failure_reason != LMMC_ODE_FAILURE_INVALID_STEP) {
            rc = 1;
            goto done;
        }
    }

    if (lmmc_ode_failure_string(LMMC_ODE_FAILURE_MAX_STEPS) == NULL) {
        rc = 1;
        goto done;
    }

done:
    if (rc != 0) {
        printf("ode test failed\n");
    }
    return rc;
}
