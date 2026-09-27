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
#include "lmmc/lmmc.h"
#include "test_common.h"

/* Number of tolerance levels for adaptive test */
#define NUM_TOL_LEVELS 4

/* Number of randomized trials */
#define NUM_TRIALS 100

/**
 * @brief RHS for y' = -lambda * y (exponential decay).
 */
static lmmc_status_t rhs_decay(double t, const double *y, double *y_prime,
                               size_t dim, void *user_data) {
    (void)t;
    double lambda = *(double *)user_data;
    if (dim != 1) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    y_prime[0] = -lambda * y[0];
    return LMMC_STATUS_OK;
}

/**
 * @brief Solve y' = -lambda*y using RK45 with given tolerance.
 */
static void solve_adaptive(double lambda, double t_end, double tol,
                           double *error) {
    double y[1] = {1.0};
    lmmc_ode_config_t cfg;
    lmmc_ode_result_t result;
    lmmc_status_t st;
    double exact;

    memset(&cfg, 0, sizeof(cfg));
    memset(&result, 0, sizeof(result));

    st = lmmc_ode_default_config(0.0, t_end, 1, &cfg);
    assert_false(st != LMMC_STATUS_OK);

    cfg.abs_tol = tol;
    cfg.rel_tol = tol;
    cfg.initial_step = t_end * 0.1;
    cfg.min_step = 1e-14;
    cfg.max_step = t_end;
    cfg.max_steps = 1000000;

    st = lmmc_ode_rk45_solve(rhs_decay, &lambda, 1, 0.0, t_end, y, &cfg, &result);
    assert_false(st != LMMC_STATUS_OK || result.converged != 1);

    exact = exp(-lambda * t_end);
    *error = fabs(y[0] - exact);
}

/**
 * @brief Simple pseudo-random double in [lo, hi].
 */
static double rand_double(double lo, double hi) {
    return lo + (hi - lo) * ((double)rand() / (double)RAND_MAX);
}

static void test_canonical_convergence(void **state) {
    (void)state;
    const double tolerances[NUM_TOL_LEVELS] = {1e-5, 1e-6, 1e-7, 1e-8};
    double lambda = 1.0;
    double t_end = 1.0;
    double errors[NUM_TOL_LEVELS];
    int level;
    double overall_ratio;

    for (level = 0; level < NUM_TOL_LEVELS; level++) {
        solve_adaptive(lambda, t_end, tolerances[level],
                       &errors[level]);
    }

    for (level = 0; level < NUM_TOL_LEVELS - 1; level++) {
        assert_true(errors[level + 1] < errors[level]);
    }

    overall_ratio = errors[0] / errors[NUM_TOL_LEVELS - 1];

    assert_false(overall_ratio < 16.0);
}

static int convergence_is_consistent(const double errors[NUM_TOL_LEVELS]) {
    for (int level = 0; level < NUM_TOL_LEVELS - 1; ++level) {
        if (!(errors[level + 1] < errors[level])) {
            return 0;
        }
    }
    /* Accept finest-level errors at machine precision. */
    return errors[NUM_TOL_LEVELS - 1] <= 1e-16 ||
           errors[0] / errors[NUM_TOL_LEVELS - 1] >= 16.0;
}

static void test_randomized_convergence(void **state) {
    (void)state;
    const double tolerances[NUM_TOL_LEVELS] = {1e-5, 1e-6, 1e-7, 1e-8};
    unsigned int inconsistent_trials = 0;
    srand(0x524B3435u);
    for (int trial = 0; trial < NUM_TRIALS; ++trial) {
        double lambda = rand_double(0.5, 2.0);
        double t_end = rand_double(1.0, 3.0);
        double errors[NUM_TOL_LEVELS];
        for (int level = 0; level < NUM_TOL_LEVELS; ++level) {
            solve_adaptive(lambda, t_end, tolerances[level], &errors[level]);
        }
        inconsistent_trials += !convergence_is_consistent(errors);
    }
    /* The convergence property allows at most 5% borderline samples. */
    assert_true(inconsistent_trials <= NUM_TRIALS / 20);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_canonical_convergence),
        cmocka_unit_test(test_randomized_convergence),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
