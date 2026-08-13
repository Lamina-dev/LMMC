/**
 * @file example_nonlinear_logging.c
 * @brief 演示 LMMC 中 nonlinear logging 相关接口的使用。
 */
#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"


static void my_diagnostic_logger(const lmmc_diagnostic_t* diagnostic, void* user_data) {
    const char* prefix = (const char*)user_data;
    printf("[%s] %s step %zu", prefix, diagnostic->operation, diagnostic->iteration);
    for (size_t i = 0; i < diagnostic->value_count; ++i) {
        printf(" value[%zu]=%.6g", i, diagnostic->values[i]);
    }
    printf("\n");
}

static lmmc_real_t my_function(lmmc_real_t x, void* user_data) {
    (void)user_data;
    return x * x - 2.0;
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

    printf("--- Part 1: Diagnostic Sink ---\n");
    cfg.diagnostics = (lmmc_diagnostic_sink_t){my_diagnostic_logger, "Newton", LMMC_DIAGNOSTIC_TRACE};
    st = lmmc_newton_solve(my_function, my_derivative, NULL, 1.0, &cfg, &res);
    printf("Final Status: %s", lmmc_status_string(st));
    if (st == LMMC_STATUS_OK) {
        printf(", Iterations: %zu\n\n", res.num_iter);
    } else {
        printf("\n\n");
    }

    printf("--- Part 2: Reconfigured Diagnostic Sink ---\n");
    cfg.diagnostics.user_data = "CustomLog";

    st = lmmc_newton_solve(my_function, my_derivative, NULL, 2.0, &cfg, &res);
    printf("Final Status: %s", lmmc_status_string(st));
    if (st == LMMC_STATUS_OK) {
        printf(", Iterations: %zu\n\n", res.num_iter);
    } else {
        printf("\n\n");
    }

    printf("--- Part 3: Reconfigured Diagnostic Sink ---\n");
    cfg.diagnostics.user_data = (void*)"CustomSolver";

    st = lmmc_newton_solve(my_function, my_derivative, NULL, 2.0, &cfg, &res);
    printf("Final Status: %s", lmmc_status_string(st));
    if (st == LMMC_STATUS_OK) {
        printf(", Iterations: %zu, Root: %.10f\n", res.num_iter, res.root);
    } else {
        printf("\n");
    }

    return 0;
}
