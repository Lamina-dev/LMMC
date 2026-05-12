#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"

// 1. Define a custom callback for nonlinear solver
static void my_nonlinear_logger(size_t iter, lmmc_real_t x, lmmc_real_t f_x, void* user_data) {
    const char* prefix = (const char*)user_data;
    printf("[%s] Step %zu: root_guess = %.6f, error = %.3e\n", prefix, iter, x, f_x);
}

static lmmc_real_t my_function(lmmc_real_t x, void* user_data) {
    (void)user_data;
    return x * x - 2.0; // Finding sqrt(2)
}

static lmmc_real_t my_derivative(lmmc_real_t x, void* user_data) {
    (void)user_data;
    return 2.0 * x;
}

int main(void) {
    lmmc_nonlinear_config_t cfg = {0};
    lmmc_nonlinear_result_t res = {0};
    lmmc_status_t st = LMMC_STATUS_OK;

    st = lmmc_nonlinear_default_config(&cfg);
    if (st != LMMC_STATUS_OK) {
        printf("Failed to init config\n");
        return 1;
    }

    printf("--- Part 1: Default Verbose Logging ---\n");
    cfg.verbose = 1;
    st = lmmc_newton_solve(my_function, my_derivative, NULL, 1.0, &cfg, &res);
    printf("Final Status: %s", lmmc_status_string(st));
    if (st == LMMC_STATUS_OK) {
        printf(", Iterations: %zu\n\n", res.num_iter);
    } else {
        printf("\n\n");
    }

    printf("--- Part 2: Custom Callback Logging ---\n");
    cfg.verbose = 0;
    cfg.log_cb = my_nonlinear_logger;
    cfg.log_user_data = "CustomLog";
    
    st = lmmc_newton_solve(my_function, my_derivative, NULL, 2.0, &cfg, &res);
    printf("Final Status: %s", lmmc_status_string(st));
    if (st == LMMC_STATUS_OK) {
        printf(", Iterations: %zu\n\n", res.num_iter);
    } else {
        printf("\n\n");
    }

    printf("--- Part 3: Custom Callback Logging ---\n");
    cfg.verbose = 0; // Disable built-in printf
    cfg.log_cb = my_nonlinear_logger;
    cfg.log_user_data = (void*)"CustomSolver";
    
    st = lmmc_newton_solve(my_function, my_derivative, NULL, 2.0, &cfg, &res);
    printf("Final Status: %s", lmmc_status_string(st));
    if (st == LMMC_STATUS_OK) {
        printf(", Iterations: %zu, Root: %.10f\n", res.num_iter, res.root);
    } else {
        printf("\n");
    }

    return 0;
}
