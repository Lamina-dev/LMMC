/**
 * @file test_logging_ext.c
 * 针对 LMMC 中 logging ext 相关接口的单元测试。
 */
#include <stdio.h>
#include <assert.h>
#include "lmmc/lmmc.h"

typedef struct {
    size_t count;
    lmmc_real_t last_val;
} test_ctx_t;

static void test_diagnostic_cb(const lmmc_diagnostic_t* diagnostic, void* user_data) {
    test_ctx_t* ctx = (test_ctx_t*)user_data;
    ctx->count++;
    if (diagnostic->value_count > 0) {
        ctx->last_val = diagnostic->values[0];
    }
}

static lmmc_real_t test_fn(lmmc_real_t x, void* user_data) {
    (void)user_data;
    lmmc_real_t result; LMMC_REAL_INIT(&result);
    lmmc_real_t five; LMMC_REAL_INIT(&five); LMMC_REAL_SET_D(&five, 5.0);
    LMMC_REAL_SUB(&result, &x, &five);
    LMMC_REAL_CLEAR(&five);
    return result;
}

static lmmc_status_t test_rhs(lmmc_real_t t, const lmmc_real_t* y, lmmc_real_t* y_prime, size_t dim, void* user_data) {
    (void)t; (void)user_data;
    for (size_t i = 0; i < dim; ++i) LMMC_REAL_SET_D(&y_prime[i], 1.0);
    return LMMC_STATUS_OK;
}

int main(void) {
    {
        printf("Testing Nonlinear Logging Callback...\n");
        lmmc_nonlinear_config_t cfg = {0};
        lmmc_nonlinear_result_t res = {0};
        test_ctx_t ctx;
        ctx.count = 0;
        LMMC_REAL_INIT(&ctx.last_val);
        LMMC_REAL_SET_D(&ctx.last_val, 0.0);

        lmmc_status_t st_cfg = lmmc_nonlinear_default_config(&cfg);
        assert(st_cfg == LMMC_STATUS_OK);
        cfg.diagnostics = (lmmc_diagnostic_sink_t){test_diagnostic_cb, &ctx, LMMC_DIAGNOSTIC_TRACE};

        lmmc_status_t st = lmmc_bisection_solve(test_fn, NULL, 0.0, 10.0, &cfg, &res);
        assert(st == LMMC_STATUS_OK);
        assert(ctx.count > 0);
        assert(ctx.count >= res.num_iter);
        printf("Nonlinear Logging Callback Test Passed (%zu calls)\n", ctx.count);
    }

    {
        printf("Testing ODE Logging Callback...\n");
        lmmc_ode_config_t cfg = {0};
        lmmc_ode_result_t res = {0};
        test_ctx_t ctx;
        ctx.count = 0;
        LMMC_REAL_INIT(&ctx.last_val);
        LMMC_REAL_SET_D(&ctx.last_val, 0.0);
        lmmc_real_t y[1];
        LMMC_REAL_INIT(&y[0]);
        LMMC_REAL_SET_D(&y[0], 0.0);

        lmmc_status_t st_cfg = lmmc_ode_default_config(0.0, 1.0, 1, &cfg);
        assert(st_cfg == LMMC_STATUS_OK);
        cfg.diagnostics = (lmmc_diagnostic_sink_t){test_diagnostic_cb, &ctx, LMMC_DIAGNOSTIC_TRACE};
        cfg.initial_step = 0.1;

        lmmc_status_t st = lmmc_ode_euler_solve(test_rhs, NULL, 1, 0.0, 1.0, y, &cfg, &res);
        assert(st == LMMC_STATUS_OK);

        assert(ctx.count > 0);
        printf("ODE Logging Callback Test Passed (%zu calls)\n", ctx.count);
    }

    printf("\nAll logging extension tests passed!\n");
    return 0;
}
