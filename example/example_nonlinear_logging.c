#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"

// 1. Define a custom callback for nonlinear solver
static void my_nonlinear_logger(size_t iter, double x, double f_x, void* user_data) {
    const char* prefix = (const char*)user_data;
    printf("[%s] Step %zu: root_guess = %.6f, error = %.3e\n", prefix, iter, x, f_x);
}

static double my_function(double x, void* user_data) {
    (void)user_data;
    return x * x - 2.0; // Finding sqrt(2)
}

static double my_derivative(double x, void* user_data) {
    (void)user_data;
    return 2.0 * x;
}

int main(void) {
    lmmc_nonlinear_config_t cfg;
    lmmc_nonlinear_result_t res;
    lmmc_status_t st;

    lmmc_nonlinear_default_config(&cfg);
    
    printf("--- Part 1: Default Verbose Logging ---\n");
    cfg.verbose = 1;
    st = lmmc_newton_solve(my_function, my_derivative, NULL, 1.0, &cfg, &res);
    printf("Final Status: %s, Iterations: %zu\n\n", lmmc_status_string(st), res.num_iter);

    printf("--- Part 2: Custom Callback Logging ---\n");
    cfg.verbose = 0; // Disable built-in printf
    cfg.log_cb = my_nonlinear_logger;
    cfg.log_user_data = (void*)"CustomSolver";
    
    st = lmmc_newton_solve(my_function, my_derivative, NULL, 2.0, &cfg, &res);
    printf("Final Status: %s, Iterations: %zu, Root: %.10f\n", lmmc_status_string(st), res.num_iter, res.root);

    return 0;
}
