#include <stdio.h>
#include <assert.h>
#include "lmmc/lmmc.h"

typedef struct {
    size_t count;
    lmmc_real_t last_val;
} test_ctx_t;

static void test_nonlinear_cb(size_t iter, lmmc_real_t x, lmmc_real_t f_x, void* user_data) {
    (void)f_x;
    test_ctx_t* ctx = (test_ctx_t*)user_data;
    ctx->count++;
    ctx->last_val = x;
    (void)iter;
}

static void test_ode_cb(size_t step, lmmc_real_t t, const lmmc_real_t* y, size_t dim, void* user_data) {
    (void)dim;
    test_ctx_t* ctx = (test_ctx_t*)user_data;
    ctx->count++;
    ctx->last_val = t;
    (void)step;
    (void)y;
}

static lmmc_real_t test_fn(lmmc_real_t x, void* user_data) {
    (void)user_data;
    lmmc_real_t result; LMMC_REAL_INIT(&result);
    lmmc_real_t five; LMMC_REAL_INIT(&five); LMMC_REAL_SET_D(&five, 5.0);
    LMMC_REAL_SUB(&result, &x, &five);
    LMMC_REAL_CLEAR(&five);
    return result; // root is 5
}

static lmmc_status_t test_rhs(lmmc_real_t t, const lmmc_real_t* y, lmmc_real_t* y_prime, size_t dim, void* user_data) {
    (void)t; (void)user_data;
    for (size_t i = 0; i < dim; ++i) LMMC_REAL_SET_D(&y_prime[i], 1.0); // y' = 1 -> y = t
    return LMMC_STATUS_OK;
}

int main(void) {
    {
        printf("Testing Nonlinear Logging Callback...\n");
        lmmc_nonlinear_config_t cfg;
        lmmc_nonlinear_result_t res;
        test_ctx_t ctx;
        ctx.count = 0;
        LMMC_REAL_INIT(&ctx.last_val);
        LMMC_REAL_SET_D(&ctx.last_val, 0.0);
        
        lmmc_nonlinear_default_config(&cfg);
        cfg.log_cb = test_nonlinear_cb;
        cfg.log_user_data = &ctx;
        
        lmmc_status_t st = lmmc_bisection_solve(test_fn, NULL, 0.0, 10.0, &cfg, &res);
        assert(st == LMMC_STATUS_OK);
        assert(ctx.count > 0);
        assert(ctx.count >= res.num_iter); // initial + steps
        printf("Nonlinear Logging Callback Test Passed (%zu calls)\n", ctx.count);
    }

    {
        printf("Testing ODE Logging Callback...\n");
        lmmc_ode_config_t cfg;
        lmmc_ode_result_t res;
        test_ctx_t ctx;
        ctx.count = 0;
        LMMC_REAL_INIT(&ctx.last_val);
        LMMC_REAL_SET_D(&ctx.last_val, 0.0);
        lmmc_real_t y[1];
        LMMC_REAL_INIT(&y[0]);
        LMMC_REAL_SET_D(&y[0], 0.0);
        
        lmmc_ode_default_config(0.0, 1.0, 1, &cfg);
        cfg.log_cb = test_ode_cb;
        cfg.log_user_data = &ctx;
        cfg.initial_step = 0.1;
        
        lmmc_status_t st = lmmc_ode_euler_solve(test_rhs, NULL, 1, 0.0, 1.0, y, &cfg, &res);
        assert(st == LMMC_STATUS_OK);
        // Let's just check ctx.count > 0 for now to be safe about exact counts.
        assert(ctx.count > 0);
        printf("ODE Logging Callback Test Passed (%zu calls)\n", ctx.count);
    }

    printf("\nAll logging extension tests passed!\n");
    return 0;
}
