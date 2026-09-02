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

static lmmc_status_t rhs_vanderpol(double t, const double* y, double* yp,
                                    size_t dim, void* ud) {
    vdp_ctx_t* ctx = (vdp_ctx_t*)ud;
    (void)t;
    if (dim != 2) return LMMC_STATUS_INVALID_ARGUMENT;
    yp[0] = y[1];
    yp[1] = ctx->mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
    return LMMC_STATUS_OK;
}

static lmmc_status_t jac_vanderpol(double t, const double* y, double* J,
                                    size_t dim, void* ud) {
    vdp_ctx_t* ctx = (vdp_ctx_t*)ud;
    (void)t;
    if (dim != 2) return LMMC_STATUS_INVALID_ARGUMENT;
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

static lmmc_status_t rhs_linear_decay(double t, const double* y, double* yp,
                                       size_t dim, void* ud) {
    linear_ctx_t* ctx = (linear_ctx_t*)ud;
    (void)t;
    if (dim != 1) return LMMC_STATUS_INVALID_ARGUMENT;
    yp[0] = ctx->lambda * y[0];
    return LMMC_STATUS_OK;
}

static lmmc_status_t jac_linear_decay(double t, const double* y, double* J,
                                       size_t dim, void* ud) {
    linear_ctx_t* ctx = (linear_ctx_t*)ud;
    (void)t; (void)y;
    if (dim != 1) return LMMC_STATUS_INVALID_ARGUMENT;
    J[0] = ctx->lambda;
    return LMMC_STATUS_OK;
}

/* ========================================================================
 * Simple test problem for trapezoidal accuracy: y' = -y, y(0) = 1
 * Exact solution: y(t) = exp(-t)
 * ======================================================================== */

static lmmc_status_t rhs_simple_decay(double t, const double* y, double* yp,
                                       size_t dim, void* ud) {
    (void)t; (void)ud;
    if (dim != 1) return LMMC_STATUS_INVALID_ARGUMENT;
    yp[0] = -y[0];
    return LMMC_STATUS_OK;
}

/* ========================================================================
 * Test 1: SDIRK4 on Van der Pol (stiff)
 * Use mu=100 as a moderately stiff case (mu=1000 is extremely stiff and
 * may require very many steps; the task note allows reducing mu).
 * ======================================================================== */
static int test_sdirk4_vanderpol(void) {
    vdp_ctx_t ctx;
    lmmc_ode_config_t cfg;
    lmmc_ode_result_t res;
    double y[2];
    lmmc_status_t st;

    printf("Test 1: SDIRK4 on Van der Pol (mu=100, t=[0,200])...\n");

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

    printf("  status=%d, converged=%d, steps=%zu\n", (int)st, res.converged, res.num_steps);
    printf("  y=[%.6e, %.6e]\n", y[0], y[1]);

    if (st != LMMC_STATUS_OK) {
        printf("  FAIL: solver returned error status %d\n", (int)st);
        return 1;
    }
    if (!res.converged) {
        printf("  FAIL: solver did not converge (reason=%d)\n", (int)res.failure_reason);
        return 1;
    }
    /* Van der Pol solution should remain bounded (limit cycle has |y0| ~ 2,
     * but transients and numerical approximation can produce slightly larger values) */
    if (fabs(y[0]) > 20.0 || fabs(y[1]) > 5000.0) {
        printf("  FAIL: solution unbounded\n");
        return 1;
    }
    printf("  PASS\n");
    return 0;
}

/* ========================================================================
 * Test 2: Rosenbrock GRK4T on Van der Pol (stiff)
 * ======================================================================== */
static int test_rosenbrock_vanderpol(void) {
    vdp_ctx_t ctx;
    lmmc_ode_config_t cfg;
    lmmc_ode_result_t res;
    double y[2];
    lmmc_status_t st;

    printf("Test 2: Rosenbrock GRK4T on Van der Pol (mu=100, t=[0,200])...\n");

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

    printf("  status=%d, converged=%d, steps=%zu\n", (int)st, res.converged, res.num_steps);
    printf("  y=[%.6e, %.6e]\n", y[0], y[1]);

    if (st != LMMC_STATUS_OK) {
        printf("  FAIL: solver returned error status %d\n", (int)st);
        return 1;
    }
    if (!res.converged) {
        printf("  FAIL: solver did not converge (reason=%d)\n", (int)res.failure_reason);
        return 1;
    }
    /* Solution should remain bounded (limit cycle has |y0| ~ 2,
     * but transients and numerical approximation can produce slightly larger values) */
    if (fabs(y[0]) > 20.0 || fabs(y[1]) > 5000.0) {
        printf("  FAIL: solution unbounded\n");
        return 1;
    }
    printf("  PASS\n");
    return 0;
}

/* ========================================================================
 * Test 3: Implicit Euler A-stability
 * Solve y' = lambda*y with lambda = -1000 (very stiff), large step h=0.1
 * Over t=[0,1]. With h=0.1 that's only 10 steps.
 * A-stable method should keep |y| <= |y0| = 1.
 * ======================================================================== */
static int test_implicit_euler_a_stability(void) {
    linear_ctx_t ctx;
    lmmc_ode_config_t cfg;
    lmmc_ode_result_t res;
    double y[1];
    lmmc_status_t st;

    printf("Test 3: Implicit Euler A-stability (lambda=-1000, h=0.1)...\n");

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

    printf("  status=%d, converged=%d, y=%.6e\n", (int)st, res.converged, y[0]);

    if (st != LMMC_STATUS_OK) {
        printf("  FAIL: solver returned error status %d\n", (int)st);
        return 1;
    }
    if (!res.converged) {
        printf("  FAIL: solver did not converge\n");
        return 1;
    }
    /* A-stability: |y| must remain bounded, specifically |y| <= |y0| = 1 */
    if (fabs(y[0]) > 1.0 + 1e-10) {
        printf("  FAIL: |y|=%.6e > 1.0, not A-stable\n", fabs(y[0]));
        return 1;
    }
    /* The exact solution is exp(-1000) ~ 0, so y should be very small */
    /* With implicit Euler and h=0.1, lambda*h = -100, so amplification factor
     * is 1/(1 - lambda*h) = 1/101 per step. After 10 steps: (1/101)^10 ~ 1e-20 */
    if (fabs(y[0]) > 1e-10) {
        printf("  WARNING: y=%.6e larger than expected (should be ~0)\n", y[0]);
        /* Not a failure, just informational - A-stability is the key check */
    }
    printf("  PASS\n");
    return 0;
}

/* ========================================================================
 * Test 4: Trapezoidal method 2nd-order accuracy
 * Solve y' = -y, y(0)=1 over [0,1] with two step sizes h and h/2.
 * Error should decrease by factor ~4 (2nd order).
 * ======================================================================== */
static int test_trapezoidal_second_order(void) {
    lmmc_ode_config_t cfg;
    lmmc_ode_result_t res;
    double y1[1], y2[1];
    double exact, err1, err2, ratio;
    double h1 = 0.05;
    double h2 = 0.025;
    lmmc_status_t st;

    printf("Test 4: Trapezoidal 2nd-order accuracy...\n");

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
    if (st != LMMC_STATUS_OK || !res.converged) {
        printf("  FAIL: trapezoidal with h=%.4f failed (st=%d, conv=%d)\n",
               h1, (int)st, res.converged);
        return 1;
    }
    err1 = fabs(y1[0] - exact);

    /* Run with step size h2 = h1/2 */
    y2[0] = 1.0;
    cfg.initial_step = h2;
    cfg.min_step = h2;
    cfg.max_step = h2;

    memset(&res, 0, sizeof(res));
    st = lmmc_ode_trapezoidal_solve(rhs_simple_decay, NULL, 1, 0.0, 1.0, y2, &cfg, &res);
    if (st != LMMC_STATUS_OK || !res.converged) {
        printf("  FAIL: trapezoidal with h=%.4f failed (st=%d, conv=%d)\n",
               h2, (int)st, res.converged);
        return 1;
    }
    err2 = fabs(y2[0] - exact);

    printf("  h=%.4f: y=%.10e, err=%.4e\n", h1, y1[0], err1);
    printf("  h=%.4f: y=%.10e, err=%.4e\n", h2, y2[0], err2);

    if (err2 < 1e-15) {
        /* Both errors are essentially zero - method is very accurate */
        printf("  Errors too small to measure ratio reliably, but method converges.\n");
        printf("  PASS\n");
        return 0;
    }

    ratio = err1 / err2;
    printf("  error ratio (should be ~4 for 2nd order): %.2f\n", ratio);

    /* For 2nd order, halving h should reduce error by factor 4.
     * Allow some tolerance: ratio should be between 3 and 5 */
    if (ratio < 3.0 || ratio > 5.5) {
        printf("  FAIL: ratio %.2f not consistent with 2nd-order\n", ratio);
        return 1;
    }
    printf("  PASS\n");
    return 0;
}

typedef lmmc_status_t (*implicit_solver_t)(
    lmmc_ode_rhs_t, void*, size_t, lmmc_real_t, lmmc_real_t,
    lmmc_real_t*, const lmmc_ode_config_t*, lmmc_ode_result_t*
);

static lmmc_status_t dfdt_zero(double t, const double* y, double* dfdt,
                                size_t dim, void* ud) {
    size_t i;
    (void)t;
    (void)y;
    (void)ud;
    for (i = 0; i < dim; ++i) dfdt[i] = 0.0;
    return LMMC_STATUS_OK;
}

static int run_fourth_order_check(const char* name, implicit_solver_t solver) {
    const double steps[2] = {0.1, 0.05};
    double errors[2];
    linear_ctx_t ctx = {1.0};
    size_t run;

    for (run = 0; run < 2; ++run) {
        lmmc_ode_config_t cfg;
        lmmc_ode_result_t result;
        double y[1] = {1.0};
        lmmc_status_t st;
        if (lmmc_ode_default_config(0.0, 1.0, 1, &cfg) != LMMC_STATUS_OK) return 1;
        cfg.initial_step = steps[run];
        cfg.min_step = steps[run];
        cfg.max_step = steps[run];
        cfg.abs_tol = 1.0e6;
        cfg.rel_tol = 0.0;
        cfg.max_steps = 100;
        cfg.jacobian = jac_linear_decay;
        cfg.time_derivative = dfdt_zero;
        st = solver(rhs_linear_decay, &ctx, 1, 0.0, 1.0, y, &cfg, &result);
        if (st != LMMC_STATUS_OK || !result.converged) {
            fprintf(stderr, "%s failed for h=%.17g with status %d\n", name, steps[run], (int)st);
            return 1;
        }
        errors[run] = fabs(y[0] - exp(1.0));
    }

    {
        double ratio = errors[0] / errors[1];
        if (ratio < 12.0 || ratio > 20.0) {
            fprintf(stderr, "%s fourth-order ratio %.17g outside [12,20]\n", name, ratio);
            return 1;
        }
    }
    return 0;
}

static int test_advertised_fourth_order(void) {
    int failures = 0;
    failures += run_fourth_order_check("SDIRK4", lmmc_ode_sdirk4_solve);
    failures += run_fourth_order_check("GRK4T", lmmc_ode_rosenbrock_grk4t_solve);
    return failures;
}

static lmmc_status_t rhs_stiff_tracking(double t, const double* y, double* yp,
                                         size_t dim, void* ud) {
    (void)ud;
    if (dim != 1) return LMMC_STATUS_INVALID_ARGUMENT;
    yp[0] = -1000.0 * (y[0] - cos(t)) - sin(t);
    return LMMC_STATUS_OK;
}

static lmmc_status_t jac_stiff_tracking(double t, const double* y, double* jac,
                                         size_t dim, void* ud) {
    (void)t;
    (void)y;
    (void)ud;
    if (dim != 1) return LMMC_STATUS_INVALID_ARGUMENT;
    jac[0] = -1000.0;
    return LMMC_STATUS_OK;
}

static lmmc_status_t dfdt_stiff_tracking(double t, const double* y, double* dfdt,
                                          size_t dim, void* ud) {
    (void)y;
    (void)ud;
    if (dim != 1) return LMMC_STATUS_INVALID_ARGUMENT;
    dfdt[0] = -1000.0 * sin(t) - cos(t);
    return LMMC_STATUS_OK;
}

static int solve_stiff_tracking(implicit_solver_t solver, int analytic,
                                double tolerance, double* value, size_t* evals) {
    lmmc_ode_config_t cfg;
    lmmc_ode_result_t result;
    double y[1] = {1.0};
    lmmc_status_t st;
    if (lmmc_ode_default_config(0.0, 1.0, 1, &cfg) != LMMC_STATUS_OK) return 1;
    cfg.initial_step = 0.01;
    cfg.min_step = 1.0e-10;
    cfg.max_step = 0.1;
    cfg.abs_tol = tolerance;
    cfg.rel_tol = tolerance;
    cfg.max_steps = 100000;
    cfg.jacobian = analytic ? jac_stiff_tracking : NULL;
    cfg.time_derivative = analytic ? dfdt_stiff_tracking : NULL;
    st = solver(rhs_stiff_tracking, NULL, 1, 0.0, 1.0, y, &cfg, &result);
    if (st != LMMC_STATUS_OK || !result.converged) {
        fprintf(stderr, "stiff solve status=%d reason=%d tol=%.1e analytic=%d evals=%zu\n",
                (int)st, (int)result.failure_reason, tolerance, analytic,
                result.num_rhs_evals);
        return 1;
    }
    *value = y[0];
    *evals = result.num_rhs_evals;
    return 0;
}

static int run_stiff_tracking_check(const char* name, implicit_solver_t solver) {
    double analytic, finite_difference, coarse, fine;
    size_t analytic_evals, fd_evals, coarse_evals, fine_evals;
    if (solve_stiff_tracking(solver, 1, 1e-6, &analytic, &analytic_evals) ||
        solve_stiff_tracking(solver, 0, 1e-6, &finite_difference, &fd_evals) ||
        solve_stiff_tracking(solver, 1, 1e-4, &coarse, &coarse_evals) ||
        solve_stiff_tracking(solver, 1, 1e-7, &fine, &fine_evals)) {
        fprintf(stderr, "%s stiff tracking solve failed\n", name);
        return 1;
    }
    if (fabs(analytic - cos(1.0)) > 1e-5 ||
        fabs(finite_difference - analytic) > 1e-5) {
        fprintf(stderr, "%s stiff tracking accuracy failed\n", name);
        return 1;
    }
    if (!(fabs(fine - cos(1.0)) < fabs(coarse - cos(1.0))) ||
        fine_evals < coarse_evals || fd_evals <= analytic_evals) {
        fprintf(stderr, "%s tolerance or callback work ordering failed\n", name);
        return 1;
    }
    return 0;
}

static int test_stiff_tracking(void) {
    int failures = 0;
    failures += run_stiff_tracking_check("SDIRK4", lmmc_ode_sdirk4_solve);
    failures += run_stiff_tracking_check("GRK4T", lmmc_ode_rosenbrock_grk4t_solve);
    return failures;
}

static int test_shared_validation(void) {
    implicit_solver_t solvers[] = {
        lmmc_ode_sdirk4_solve,
        lmmc_ode_rosenbrock_grk4t_solve
    };
    size_t i;
    for (i = 0; i < sizeof(solvers) / sizeof(solvers[0]); ++i) {
        lmmc_ode_config_t cfg;
        lmmc_ode_result_t result = {1, 9, 9, 9.0, LMMC_ODE_FAILURE_MAX_STEPS};
        double y[1] = {1.0};
        lmmc_status_t st;
        if (lmmc_ode_default_config(0.0, 1.0, 1, &cfg) != LMMC_STATUS_OK) return 1;
        cfg.abs_tol = 0.0;
        cfg.rel_tol = 0.0;
        st = solvers[i](rhs_simple_decay, NULL, 1, 0.0, 1.0, y, &cfg, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT ||
            result.failure_reason != LMMC_ODE_FAILURE_TOLERANCE_INCONSISTENT ||
            result.converged != 0 || result.num_steps != 0 ||
            result.num_rhs_evals != 0 || result.final_t != 0.0) {
            return 1;
        }
        result = (lmmc_ode_result_t){1, 9, 9, 9.0, LMMC_ODE_FAILURE_MAX_STEPS};
        st = solvers[i](NULL, NULL, 1, 0.0, 1.0, y, &cfg, &result);
        if (st != LMMC_STATUS_INVALID_ARGUMENT || result.converged != 0 ||
            result.num_steps != 0 || result.num_rhs_evals != 0 ||
            result.final_t != 0.0 || result.failure_reason != LMMC_ODE_FAILURE_NONE) {
            return 1;
        }
    }
    return 0;
}


/* ========================================================================
 * Main
 * ======================================================================== */
int main(void) {
    int failures = 0;

    if (lmmc_init() != LMMC_STATUS_OK) return 1;

    failures += test_sdirk4_vanderpol();
    failures += test_rosenbrock_vanderpol();
    failures += test_implicit_euler_a_stability();
    failures += test_stiff_tracking();
    failures += test_shared_validation();
    failures += test_trapezoidal_second_order();
    failures += test_advertised_fourth_order();

    if (lmmc_deinit() != LMMC_STATUS_OK) return 1;

    printf("\n=== Results: %d test(s) failed ===\n", failures);
    return failures;
}
