#include <stdio.h>
#include "lmmc/lmmc.h"

// Custom ODE callback
static void my_ode_logger(size_t step, lmmc_real_t t, const lmmc_real_t* y, size_t dim, void* user_data) {
    (void)user_data;
    printf("[Step %zu] Time = %.3f, State = [", step, t);
    for (size_t i = 0; i < dim; ++i) {
        printf("%.4f%s", y[i], (i == dim - 1) ? "" : ", ");
    }
    printf("]\n");
}

// Simple RHS: y' = -y
static lmmc_status_t decay_rhs(lmmc_real_t t, const lmmc_real_t* y, lmmc_real_t* y_prime, size_t dim, void* user_data) {
    (void)t; (void)user_data;
    for (size_t i = 0; i < dim; ++i) {
        y_prime[i] = -y[i];
    }
    return LMMC_STATUS_OK;
}

int main(void) {
    lmmc_ode_config_t cfg = {0};
    lmmc_ode_result_t res = {0};
    lmmc_real_t y[1];
    LMMC_REAL_INIT(&y[0]); LMMC_REAL_SET_D(&y[0], 1.0);
    lmmc_status_t st = LMMC_STATUS_OK;

    st = lmmc_ode_default_config(0.0, 1.0, 1, &cfg);
    if (st != LMMC_STATUS_OK) return 1;

    printf("--- Part 1: ODE RK4 with Verbose Logging ---\n");
    cfg.verbose = 1;
    cfg.initial_step = 0.2;
    st = lmmc_ode_rk4_solve(decay_rhs, NULL, 1, 0.0, 1.0, y, &cfg, &res);
    printf("Status: %s", lmmc_status_string(st));
    if (st == LMMC_STATUS_OK) {
        printf(", Steps: %zu, Final y: %.6f\n\n", res.num_steps, y[0]);
    } else {
        printf("\n\n");
    }

    printf("--- Part 2: ODE RK45 with Custom Callback ---\n");
    LMMC_REAL_SET_D(&y[0], 1.0); // Reset
    cfg.verbose = 0;
    cfg.log_cb = my_ode_logger;
    st = lmmc_ode_rk45_solve(decay_rhs, NULL, 1, 0.0, 1.0, y, &cfg, &res);
    printf("Status: %s", lmmc_status_string(st));
    if (st == LMMC_STATUS_OK) {
        printf(", Steps: %zu, Final y: %.6f\n", res.num_steps, y[0]);
    } else {
        printf("\n");
    }

    return 0;
}
