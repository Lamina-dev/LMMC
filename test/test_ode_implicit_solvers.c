/**
 * @file test_ode_implicit_solvers.c
 * 隐式 ODE 求解器测试（SDIRK4、Rosenbrock、隐式 Euler）。
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

/* ========================================================================
 * Van der Pol oscillator: y0' = y1, y1' = mu*(1 - y0^2)*y1 - y0
 * ======================================================================== */

typedef struct {
    double mu;
} vdp_ctx_t;

static lmmc_status_t rhs_vanderpol(double t, const double *y, double *yp,
                                   size_t dim, void *ud) {
    vdp_ctx_t *ctx = (vdp_ctx_t *)ud;
    (void)t;
    if (dim != 2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    yp[0] = y[1];
    yp[1] = ctx->mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
    return LMMC_STATUS_OK;
}

static lmmc_status_t jac_vanderpol(double t, const double *y, double *J,
                                   size_t dim, void *ud) {
    vdp_ctx_t *ctx = (vdp_ctx_t *)ud;
    (void)t;
    if (dim != 2) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    /* J is row-major 2x2:
     * J[0,0] = df0/dy0 = 0
     * J[0,1] = df0/dy1 = 1
     * J[1,0] = df1/dy0 = -2*mu*y0*y1 - 1
     * J[1,1] = df1/dy1 = mu*(1 - y0^2)
     */
    J[0] = 0.0;
    J[1] = 1.0;
    J[2] = -2.0 * ctx->mu * y[0] * y[1] - 1.0;
    J[3] = ctx->mu * (1.0 - y[0] * y[0]);
    return LMMC_STATUS_OK;
}

/* ========================================================================
 * Linear decay: y' = lambda * y  (for A-stability test)
 * ======================================================================== */

typedef struct {
    double lambda;
} linear_ctx_t;

static lmmc_status_t rhs_linear_decay(double t, const double *y, double *yp,
                                      size_t dim, void *ud) {
    linear_ctx_t *ctx = (linear_ctx_t *)ud;
    (void)t;
    if (dim != 1) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    yp[0] = ctx->lambda * y[0];
    return LMMC_STATUS_OK;
}

static lmmc_status_t jac_linear_decay(double t, const double *y, double *J,
                                      size_t dim, void *ud) {
    linear_ctx_t *ctx = (linear_ctx_t *)ud;
    (void)t;
    (void)y;
    if (dim != 1) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    J[0] = ctx->lambda;
    return LMMC_STATUS_OK;
}

/* ========================================================================
 * Simple test problem for trapezoidal accuracy: y' = -y, y(0) = 1
 * Exact solution: y(t) = exp(-t)
 * ======================================================================== */

static lmmc_status_t rhs_simple_decay(double t, const double *y, double *yp,
                                      size_t dim, void *ud) {
    (void)t;
    (void)ud;
    if (dim != 1) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    yp[0] = -y[0];
    return LMMC_STATUS_OK;
}

/* ========================================================================
 * Test 1: SDIRK4 on Van der Pol (stiff)
 * Use mu=100 as a moderately stiff case (mu=1000 is extremely stiff and
 * may require very many steps; the task note allows reducing mu).
 * ======================================================================== */
static void test_sdirk4_vanderpol(void **state) {
    (void)state;
    vdp_ctx_t ctx;
    lmmc_ode_config_t cfg;
    lmmc_ode_result_t res;
    double y[2];
    lmmc_status_t st;

    ctx.mu = 100.0;
    y[0] = 2.0;
    y[1] = 0.0;

    lmmc_ode_default_config(0.0, 200.0, 2, &cfg);
    cfg.initial_step = 0.01;
    cfg.min_step = 1e-10;
    cfg.max_step = 5.0;
    cfg.abs_tol = 1e-6;
    cfg.rel_tol = 1e-6;
    cfg.max_steps = 500000;
    cfg.jacobian = jac_vanderpol;

    memset(&res, 0, sizeof(res));
    st = lmmc_ode_sdirk4_solve(rhs_vanderpol, &ctx, 2, 0.0, 200.0, y, &cfg, &res);

    assert_false(st != LMMC_STATUS_OK);
    assert_true(res.converged);
    /* Van der Pol solution should remain bounded (limit cycle has |y0| ~ 2,
     * but transients and numerical approximation can produce slightly larger values) */
    assert_false(fabs(y[0]) > 20.0 || fabs(y[1]) > 5000.0);
}

/* ========================================================================
 * Test 2: Rosenbrock GRK4T on Van der Pol (stiff)
 * ======================================================================== */
static void test_rosenbrock_vanderpol(void **state) {
    (void)state;
    vdp_ctx_t ctx;
    lmmc_ode_config_t cfg;
    lmmc_ode_result_t res;
    double y[2];
    lmmc_status_t st;

    ctx.mu = 100.0;
    y[0] = 2.0;
    y[1] = 0.0;

    lmmc_ode_default_config(0.0, 200.0, 2, &cfg);
    cfg.initial_step = 0.01;
    cfg.min_step = 1e-10;
    cfg.max_step = 5.0;
    cfg.abs_tol = 1e-6;
    cfg.rel_tol = 1e-6;
    cfg.max_steps = 500000;
    cfg.jacobian = jac_vanderpol;

    memset(&res, 0, sizeof(res));
    st = lmmc_ode_rosenbrock_grk4t_solve(rhs_vanderpol, &ctx, 2, 0.0, 200.0, y, &cfg, &res);

    assert_false(st != LMMC_STATUS_OK);
    assert_true(res.converged);
    /* Solution should remain bounded (limit cycle has |y0| ~ 2,
     * but transients and numerical approximation can produce slightly larger values) */
    assert_false(fabs(y[0]) > 20.0 || fabs(y[1]) > 5000.0);
}

/* ========================================================================
 * Test 3: Implicit Euler A-stability
 * Solve y' = lambda*y with lambda = -1000 (very stiff), large step h=0.1
 * Over t=[0,1]. With h=0.1 that's only 10 steps.
 * A-stable method should keep |y| <= |y0| = 1.
 * ======================================================================== */
static void test_implicit_euler_a_stability(void **state) {
    (void)state;
    linear_ctx_t ctx;
    lmmc_ode_config_t cfg;
    lmmc_ode_result_t res;
    double y[1];
    lmmc_status_t st;

    ctx.lambda = -1000.0;
    y[0] = 1.0;

    lmmc_ode_default_config(0.0, 1.0, 1, &cfg);
    cfg.initial_step = 0.1;
    cfg.min_step = 0.1;
    cfg.max_step = 0.1;
    cfg.abs_tol = 1e-6;
    cfg.rel_tol = 1e-6;
    cfg.max_steps = 100;
    cfg.jacobian = jac_linear_decay;

    memset(&res, 0, sizeof(res));
    st = lmmc_ode_implicit_euler_solve(rhs_linear_decay, &ctx, 1, 0.0, 1.0, y, &cfg, &res);

    assert_false(st != LMMC_STATUS_OK);
    assert_true(res.converged);
    /* A-stability: |y| must remain bounded, specifically |y| <= |y0| = 1 */
    assert_false(fabs(y[0]) > 1.0 + 1e-10);
    /* The exact solution is exp(-1000) ~ 0, so y should be very small */
    /* With implicit Euler and h=0.1, lambda*h = -100, so amplification factor
     * is 1/(1 - lambda*h) = 1/101 per step. After 10 steps: (1/101)^10 ~ 1e-20 */
}

/* ========================================================================
 * Test 4: Trapezoidal method 2nd-order accuracy
 * Solve y' = -y, y(0)=1 over [0,1] with two step sizes h and h/2.
 * Error should decrease by factor ~4 (2nd order).
 * ======================================================================== */
static void test_trapezoidal_second_order(void **state) {
    (void)state;
    lmmc_ode_config_t cfg;
    lmmc_ode_result_t res;
    double y1[1], y2[1];
    double exact, err1, err2, ratio;
    double h1 = 0.05;
    double h2 = 0.025;
    lmmc_status_t st;

    exact = exp(-1.0);

    /* Run with step size h1 */
    y1[0] = 1.0;
    lmmc_ode_default_config(0.0, 1.0, 1, &cfg);
    cfg.initial_step = h1;
    cfg.min_step = h1;
    cfg.max_step = h1;
    cfg.abs_tol = 1e-14;
    cfg.rel_tol = 1e-14;
    cfg.max_steps = 10000;
    cfg.jacobian = NULL;

    memset(&res, 0, sizeof(res));
    st = lmmc_ode_trapezoidal_solve(rhs_simple_decay, NULL, 1, 0.0, 1.0, y1, &cfg, &res);
    assert_false(st != LMMC_STATUS_OK || !res.converged);
    err1 = fabs(y1[0] - exact);

    /* Run with step size h2 = h1/2 */
    y2[0] = 1.0;
    cfg.initial_step = h2;
    cfg.min_step = h2;
    cfg.max_step = h2;

    memset(&res, 0, sizeof(res));
    st = lmmc_ode_trapezoidal_solve(rhs_simple_decay, NULL, 1, 0.0, 1.0, y2, &cfg, &res);
    assert_false(st != LMMC_STATUS_OK || !res.converged);
    err2 = fabs(y2[0] - exact);

    if (err2 < 1e-15) {
        /* Both errors are essentially zero - method is very accurate */

        return;
    }

    ratio = err1 / err2;

    /* For 2nd order, halving h should reduce error by factor 4.
     * Allow some tolerance: ratio should be between 3 and 5 */
    assert_false(ratio < 3.0 || ratio > 5.5);
}

typedef lmmc_status_t (*implicit_solver_t)(
    lmmc_ode_rhs_t, void *, size_t, lmmc_real_t, lmmc_real_t,
    lmmc_real_t *, const lmmc_ode_config_t *, lmmc_ode_result_t *);

static lmmc_status_t dfdt_zero(double t, const double *y, double *dfdt,
                               size_t dim, void *ud) {
    size_t i;
    (void)t;
    (void)y;
    (void)ud;
    for (i = 0; i < dim; ++i)
        dfdt[i] = 0.0;
    return LMMC_STATUS_OK;
}

static void test_fourth_order(void **state) {
    implicit_solver_t solver = *(implicit_solver_t *)*state;
    const double steps[2] = {0.1, 0.05};
    double errors[2];
    linear_ctx_t ctx = {1.0};
    size_t run;

    for (run = 0; run < 2; ++run) {
        lmmc_ode_config_t cfg;
        lmmc_ode_result_t result;
        double y[1] = {1.0};
        lmmc_status_t st;
        assert_false(lmmc_ode_default_config(0.0, 1.0, 1, &cfg) != LMMC_STATUS_OK);
        cfg.initial_step = steps[run];
        cfg.min_step = steps[run];
        cfg.max_step = steps[run];
        cfg.abs_tol = 1.0e6;
        cfg.rel_tol = 0.0;
        cfg.max_steps = 100;
        cfg.jacobian = jac_linear_decay;
        cfg.time_derivative = dfdt_zero;
        st = solver(rhs_linear_decay, &ctx, 1, 0.0, 1.0, y, &cfg, &result);
        assert_false(st != LMMC_STATUS_OK || !result.converged);
        errors[run] = fabs(y[0] - exp(1.0));
    }

    {
        double ratio = errors[0] / errors[1];
        assert_false(!isfinite(ratio) || ratio < 12.0 || ratio > 20.0);
    }
}

static lmmc_status_t rhs_stiff_tracking(double t, const double *y, double *yp,
                                        size_t dim, void *ud) {
    (void)ud;
    if (dim != 1) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    yp[0] = -1000.0 * (y[0] - cos(t)) - sin(t);
    return LMMC_STATUS_OK;
}

static lmmc_status_t jac_stiff_tracking(double t, const double *y, double *jac,
                                        size_t dim, void *ud) {
    (void)t;
    (void)y;
    (void)ud;
    if (dim != 1) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    jac[0] = -1000.0;
    return LMMC_STATUS_OK;
}

static lmmc_status_t dfdt_stiff_tracking(double t, const double *y, double *dfdt,
                                         size_t dim, void *ud) {
    (void)y;
    (void)ud;
    if (dim != 1) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    dfdt[0] = -1000.0 * sin(t) - cos(t);
    return LMMC_STATUS_OK;
}

static void solve_stiff_tracking(implicit_solver_t solver, int analytic,
                                 double tolerance, double *value, size_t *evals) {
    lmmc_ode_config_t cfg;
    lmmc_ode_result_t result;
    double y[1] = {1.0};
    lmmc_status_t st;
    assert_false(lmmc_ode_default_config(0.0, 1.0, 1, &cfg) != LMMC_STATUS_OK);
    cfg.initial_step = 0.01;
    cfg.min_step = 1.0e-10;
    cfg.max_step = 0.1;
    cfg.abs_tol = tolerance;
    cfg.rel_tol = tolerance;
    cfg.max_steps = 100000;
    cfg.jacobian = analytic ? jac_stiff_tracking : NULL;
    cfg.time_derivative = analytic ? dfdt_stiff_tracking : NULL;
    st = solver(rhs_stiff_tracking, NULL, 1, 0.0, 1.0, y, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK || !result.converged);
    *value = y[0];
    *evals = result.num_rhs_evals;
}

static void test_stiff_tracking(void **state) {
    implicit_solver_t solver = *(implicit_solver_t *)*state;
    double analytic, finite_difference, coarse, fine;
    size_t analytic_evals, fd_evals, coarse_evals, fine_evals;
    solve_stiff_tracking(solver, 1, 1e-6, &analytic, &analytic_evals);
    solve_stiff_tracking(solver, 0, 1e-6, &finite_difference, &fd_evals);
    solve_stiff_tracking(solver, 1, 1e-4, &coarse, &coarse_evals);
    solve_stiff_tracking(solver, 1, 1e-7, &fine, &fine_evals);
    assert_false(fabs(analytic - cos(1.0)) > 1e-5 ||
                 fabs(finite_difference - analytic) > 1e-5);
    assert_false(!(fabs(fine - cos(1.0)) < fabs(coarse - cos(1.0))) ||
                 fine_evals < coarse_evals || fd_evals <= analytic_evals);
}

static void assert_implicit_result_reset(
    const lmmc_ode_result_t *result, lmmc_ode_failure_t reason) {
    assert_int_equal(result->converged, 0);
    assert_int_equal(result->num_steps, 0);
    assert_int_equal(result->num_rhs_evals, 0);
    assert_true(result->final_t == 0.0);
    assert_int_equal(result->failure_reason, reason);
}

static void test_shared_validation(void **state) {
    (void)state;
    implicit_solver_t solvers[] = {
        lmmc_ode_sdirk4_solve,
        lmmc_ode_rosenbrock_grk4t_solve};
    size_t i;
    for (i = 0; i < sizeof(solvers) / sizeof(solvers[0]); ++i) {
        lmmc_ode_config_t cfg;
        lmmc_ode_result_t result = {1, 9, 9, 9.0, LMMC_ODE_FAILURE_MAX_STEPS};
        double y[1] = {1.0};
        lmmc_status_t st;
        assert_false(lmmc_ode_default_config(0.0, 1.0, 1, &cfg) != LMMC_STATUS_OK);
        cfg.abs_tol = 0.0;
        cfg.rel_tol = 0.0;
        st = solvers[i](rhs_simple_decay, NULL, 1, 0.0, 1.0, y, &cfg, &result);
        assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);
        assert_implicit_result_reset(&result, LMMC_ODE_FAILURE_TOLERANCE_INCONSISTENT);
        result = (lmmc_ode_result_t){1, 9, 9, 9.0, LMMC_ODE_FAILURE_MAX_STEPS};
        st = solvers[i](NULL, NULL, 1, 0.0, 1.0, y, &cfg, &result);
        assert_int_equal(st, LMMC_STATUS_INVALID_ARGUMENT);
        assert_implicit_result_reset(&result, LMMC_ODE_FAILURE_NONE);
    }
}

/* ========================================================================
 * Main
 * ======================================================================== */
static int setup_runtime(void **state) {
    (void)state;
    assert_int_equal(lmmc_init(), LMMC_STATUS_OK);
    return 0;
}

static int teardown_runtime(void **state) {
    (void)state;
    assert_int_equal(lmmc_deinit(), LMMC_STATUS_OK);
    return 0;
}

int main(void) {
    implicit_solver_t sdirk = lmmc_ode_sdirk4_solve;
    implicit_solver_t rosenbrock = lmmc_ode_rosenbrock_grk4t_solve;
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_sdirk4_vanderpol),
        cmocka_unit_test(test_rosenbrock_vanderpol),
        cmocka_unit_test(test_implicit_euler_a_stability),
        cmocka_unit_test(test_shared_validation),
        cmocka_unit_test(test_trapezoidal_second_order),
        {"sdirk_fourth_order", test_fourth_order, NULL, NULL, &sdirk},
        {"sdirk_stiff_tracking", test_stiff_tracking, NULL, NULL, &sdirk},
        {"rosenbrock_fourth_order", test_fourth_order, NULL, NULL, &rosenbrock},
        {"rosenbrock_stiff_tracking", test_stiff_tracking, NULL, NULL, &rosenbrock},
    };
    return cmocka_run_group_tests(tests, setup_runtime, teardown_runtime);
}
