#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"

static double lmmc_pi_value(void) {
    return 3.14159265358979323846;
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
    double y[2] = {1.0, 0.0};
    double t_end = 6.0 * lmmc_pi_value();

    st = lmmc_ode_default_config(0.0, t_end, 2, &cfg);
    if (st != LMMC_STATUS_OK) {
        printf("default_config failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    cfg.initial_step = 0.2;
    cfg.min_step = 1e-8;
    cfg.max_step = 0.5;
    cfg.abs_tol = 1e-10;
    cfg.rel_tol = 1e-9;
    cfg.max_steps = 200000;

    st = lmmc_ode_rk45_solve(rhs_harmonic, NULL, 2, 0.0, t_end, y, &cfg, &result);
    if (st != LMMC_STATUS_OK) {
        printf("rk45 failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    printf("converged: %d\n", result.converged);
    printf("steps: %zu\n", result.num_steps);
    printf("rhs_evals: %zu\n", result.num_rhs_evals);
    printf("final_t: %.10f\n", result.final_t);
    printf("y0: %.12f\n", y[0]);
    printf("y1: %.12f\n", y[1]);
    printf("energy: %.12f\n", 0.5 * (y[0] * y[0] + y[1] * y[1]));
    printf("reason: %s\n", lmmc_ode_failure_string(result.failure_reason));

    return result.converged ? 0 : 1;
}
