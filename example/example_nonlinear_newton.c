#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"

static double lmmc_example_fn(double x, void* user_data) {
    (void)user_data;
    return cos(x) - x;
}

static double lmmc_example_dfn(double x, void* user_data) {
    (void)user_data;
    return -sin(x) - 1.0;
}

int main(void) {
    lmmc_nonlinear_config_t cfg = {0};
    lmmc_nonlinear_result_t result = {0};
    lmmc_status_t st = LMMC_STATUS_OK;

    st = lmmc_nonlinear_default_config(&cfg);
    if (st != LMMC_STATUS_OK) {
        printf("default_config failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    cfg.abs_tol = 1e-12;
    cfg.rel_tol = 1e-10;
    cfg.max_iter = 50;
    cfg.min_derivative = 1e-14;

    st = lmmc_newton_solve(lmmc_example_fn, lmmc_example_dfn, NULL, 1.0, &cfg, &result);
    if (st != LMMC_STATUS_OK) {
        printf("newton failed: %s\n", lmmc_status_string(st));
        return 1;
    }

    printf("converged: %d\n", result.converged);
    printf("iterations: %zu\n", result.num_iter);
    printf("root: %.15f\n", result.root);
    printf("f(root): %.15e\n", result.function_value);
    printf("reason: %s\n", lmmc_nonlinear_failure_string(result.failure_reason));

    return result.converged ? 0 : 1;
}
