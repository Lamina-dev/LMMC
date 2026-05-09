#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"

static lmmc_status_t rhs_exp(lmmc_real_t t, const lmmc_real_t* y, lmmc_real_t* y_prime, size_t dim, void* user_data) {
    (void)t;
    (void)user_data;
    if (dim != 1) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    y_prime[0] = y[0];
    return LMMC_STATUS_OK;
}

int main(void) {
    lmmc_ode_config_t cfg = {0};
    lmmc_ode_result_t result = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    lmmc_real_t y[1] = {1.0};
    double exact = exp(1.0);

    st = lmmc_ode_default_config(0.0, 1.0, 1, &cfg);
    if (st != LMMC_STATUS_OK) {
        printf("default_config failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    cfg.initial_step = 0.01;
    cfg.min_step = 0.01;
    cfg.max_step = 0.01;
    cfg.max_steps = 1000;

    st = lmmc_ode_euler_solve(rhs_exp, NULL, 1, 0.0, 1.0, y, &cfg, &result);
    if (st != LMMC_STATUS_OK) {
        printf("euler failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    printf("converged: %d\n", result.converged);
    printf("steps: %zu\n", result.num_steps);
    printf("rhs_evals: %zu\n", result.num_rhs_evals);
    printf("final_t: %.10f\n", result.final_t);
    printf("y(1): %.12f\n", y[0]);
    printf("exact: %.12f\n", exact);
    printf("abs_error: %.12e\n", fabs(y[0] - exact));
    printf("reason: %s\n", lmmc_ode_failure_string(result.failure_reason));

    return result.converged ? 0 : 1;
}
