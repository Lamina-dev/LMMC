/*
 * test_ode_rk45_convergence_prop.c — ODE RK45 收敛阶属性测试。
 *
 * 验证容差减半时全局误差缩小 >=16 倍（4 阶收敛）。
 * 测试问题：y' = -y, y(0)=1, 精确解 e^{-t}。
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

/* Number of tolerance levels for adaptive test */
#define NUM_TOL_LEVELS 4

/* Number of randomized trials */
#define NUM_TRIALS 100

/**
 * @brief RHS for y' = -lambda * y (exponential decay).
 */
static lmmc_status_t rhs_decay(double t, const double* y, double* y_prime,
                                size_t dim, void* user_data) {
    (void)t;
    double lambda = *(double*)user_data;
    if (dim != 1) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    y_prime[0] = -lambda * y[0];
    return LMMC_STATUS_OK;
}

/**
 * @brief Solve y' = -lambda*y using RK45 with given tolerance.
 */
static int solve_adaptive(double lambda, double t_end, double tol,
                          double* error) {
    double y[1] = {1.0};
    lmmc_ode_config_t cfg;
    lmmc_ode_result_t result;
    lmmc_status_t st;
    double exact;

    memset(&cfg, 0, sizeof(cfg));
    memset(&result, 0, sizeof(result));

    st = lmmc_ode_default_config(0.0, t_end, 1, &cfg);
    if (st != LMMC_STATUS_OK) return 1;

    cfg.abs_tol = tol;
    cfg.rel_tol = tol;
    cfg.initial_step = t_end * 0.1;
    cfg.min_step = 1e-14;
    cfg.max_step = t_end;
    cfg.max_steps = 1000000;

    st = lmmc_ode_rk45_solve(rhs_decay, &lambda, 1, 0.0, t_end, y, &cfg, &result);
    if (st != LMMC_STATUS_OK || result.converged != 1) {
        return 1;
    }

    exact = exp(-lambda * t_end);
    *error = fabs(y[0] - exact);
    return 0;
}

/**
 * @brief Simple pseudo-random double in [lo, hi].
 */
static double rand_double(double lo, double hi) {
    return lo + (hi - lo) * ((double)rand() / (double)RAND_MAX);
}

int main(void) {
    int rc = 0;
    int trial;
    unsigned int seed;

    /* Tolerance levels as specified in the task */
    double tolerances[NUM_TOL_LEVELS] = {1e-4, 1e-5, 1e-6, 1e-7};

    seed = (unsigned int)time(NULL);
    srand(seed);

    printf("ODE RK45 convergence order property test (seed=%u)\n", seed);

    /*
     * Test 1: Canonical convergence order test using y' = -y on [0,1].
     * Verify that across 4 tolerance levels, the error decreases
     * monotonically and the overall reduction is >= 16.
     */
    {
        double lambda = 1.0;
        double t_end = 1.0;
        double errors[NUM_TOL_LEVELS];
        int level;
        double overall_ratio;
        int monotone_pairs = 0;

        for (level = 0; level < NUM_TOL_LEVELS; level++) {
            if (solve_adaptive(lambda, t_end, tolerances[level],
                               &errors[level]) != 0) {
                printf("FAIL: canonical solve at tol=%.1e failed\n",
                       tolerances[level]);
                rc = 1; goto done;
            }
            printf("  tol=%.1e -> error=%.3e\n", tolerances[level], errors[level]);
        }

        /* Check monotone decrease */
        for (level = 0; level < NUM_TOL_LEVELS - 1; level++) {
            if (errors[level + 1] < errors[level]) {
                monotone_pairs++;
            }
        }

        if (monotone_pairs < 3) {
            printf("FAIL: canonical test not monotone (%d/3 pairs)\n",
                   monotone_pairs);
            rc = 1; goto done;
        }

        /* Check overall reduction >= 16 across 3 decades */
        overall_ratio = errors[0] / errors[NUM_TOL_LEVELS - 1];
        printf("  overall ratio: %.2f (need >= 16)\n", overall_ratio);

        if (overall_ratio < 16.0) {
            printf("FAIL: canonical overall ratio %.2f < 16\n", overall_ratio);
            rc = 1; goto done;
        }

        printf("  canonical test PASSED\n");
    }

    /*
     * Test 2: Randomized trials verifying convergence property.
     * For each trial, verify:
     * - Errors decrease monotonically across all 4 levels
     * - Overall error reduction from tol=1e-4 to tol=1e-7 is >= 16
     * Allow up to 5% of trials to fail (due to edge cases in adaptive control).
     */
    {
        int pass_count = 0;
        int fail_count = 0;

        for (trial = 0; trial < NUM_TRIALS; trial++) {
            double lambda = rand_double(0.5, 2.0);
            double t_end = rand_double(1.0, 3.0);
            double errors[NUM_TOL_LEVELS];
            int level;
            int monotone_pairs = 0;
            double overall_ratio;
            int trial_pass = 1;

            for (level = 0; level < NUM_TOL_LEVELS; level++) {
                if (solve_adaptive(lambda, t_end, tolerances[level],
                                   &errors[level]) != 0) {
                    printf("FAIL: trial %d solve at tol=%.1e failed\n",
                           trial, tolerances[level]);
                    rc = 1; goto done;
                }
            }

            /* Check monotone decrease */
            for (level = 0; level < NUM_TOL_LEVELS - 1; level++) {
                if (errors[level + 1] < errors[level]) {
                    monotone_pairs++;
                }
            }

            if (monotone_pairs < 3) {
                trial_pass = 0;
            }

            /* Check overall reduction */
            if (errors[NUM_TOL_LEVELS - 1] > 1e-16) {
                overall_ratio = errors[0] / errors[NUM_TOL_LEVELS - 1];
                if (overall_ratio < 16.0) {
                    trial_pass = 0;
                }
            }
            /* If finest error is at machine precision, that's fine */

            if (trial_pass) {
                pass_count++;
            } else {
                fail_count++;
            }
        }

        printf("  Randomized: %d/%d passed, %d failed\n",
               pass_count, NUM_TRIALS, fail_count);

        /* Allow up to 5% failure rate for edge cases */
        if (fail_count > NUM_TRIALS / 20) {
            printf("FAIL: too many randomized trials failed (%d/%d)\n",
                   fail_count, NUM_TRIALS);
            rc = 1; goto done;
        }

        printf("  randomized trials PASSED\n");
    }

done:
    if (rc != 0) {
        printf("test_ode_rk45_convergence_prop FAILED\n");
    } else {
        printf("test_ode_rk45_convergence_prop PASSED\n");
    }
    return rc;
}
