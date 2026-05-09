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
    lmmc_ode_config_t cfg;
    lmmc_ode_result_t res;
    lmmc_real_t y[1] = {1.0}; // Initial condition y(0) = 1
    lmmc_status_t st;

    lmmc_ode_default_config(0.0, 1.0, 1, &cfg);
    
    printf("--- Part 1: ODE RK4 with Verbose Logging ---\n");
    cfg.verbose = 1;
    cfg.initial_step = 0.2;
    st = lmmc_ode_rk4_solve(decay_rhs, NULL, 1, 0.0, 1.0, y, &cfg, &res);
    printf("Status: %s, Steps: %zu, Final y: %.6f\n\n", lmmc_status_string(st), res.num_steps, y[0]);

    printf("--- Part 2: ODE RK45 with Custom Callback ---\n");
    y[0] = 1.0; // Reset
    cfg.verbose = 0;
    cfg.log_cb = my_ode_logger;
    st = lmmc_ode_rk45_solve(decay_rhs, NULL, 1, 0.0, 1.0, y, &cfg, &res);
    printf("Status: %s, Steps: %zu, Final y: %.6f\n", lmmc_status_string(st), res.num_steps, y[0]);

    return 0;
}
