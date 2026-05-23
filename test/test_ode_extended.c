/**
 * @file test_ode_extended.c
 * @brief 针对 LMMC 中 ode extended 相关接口的单元测试。
 *
 * @internal
 */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#define TEST_EPS_TIGHT   1e-12
#define TEST_EPS_NORMAL  1e-7
#define TEST_EPS_LOOSE   1e-5

static double pi_val(void) {
    return 3.14159265358979323846;
}


static lmmc_status_t rhs_lotka_volterra(double t, const double* y, double* yp,
                                         size_t dim, void* ud) {
    (void)t; (void)ud;
    if (dim != 2) return LMMC_STATUS_INVALID_ARGUMENT;
    yp[0] = y[0] * (1.0 - y[1]);
    yp[1] = y[1] * (y[0] - 1.0);
    return LMMC_STATUS_OK;
}


static lmmc_status_t rhs_stiff(double t, const double* y, double* yp,
                                size_t dim, void* ud) {
    (void)t; (void)ud;
    if (dim != 1) return LMMC_STATUS_INVALID_ARGUMENT;
    yp[0] = -1000.0 * y[0];
    return LMMC_STATUS_OK;
}


static lmmc_status_t rhs_exp_growth(double t, const double* y, double* yp,
                                     size_t dim, void* ud) {
    (void)t; (void)ud;
    if (dim != 1) return LMMC_STATUS_INVALID_ARGUMENT;
    yp[0] = y[0];
    return LMMC_STATUS_OK;
}


static lmmc_status_t rhs_high_dim(double t, const double* y, double* yp,
                                   size_t dim, void* ud) {
    (void)t; (void)ud;
    size_t i;
    for (i = 0; i < dim; i++) {
        yp[i] = -(double)(i + 1) * y[i];
    }
    return LMMC_STATUS_OK;
}


static lmmc_status_t rhs_harmonic(double t, const double* y, double* yp,
                                   size_t dim, void* ud) {
    (void)t; (void)ud;
    if (dim != 2) return LMMC_STATUS_INVALID_ARGUMENT;
    yp[0] = y[1];
    yp[1] = -y[0];
    return LMMC_STATUS_OK;
}


static lmmc_status_t rhs_zero(double t, const double* y, double* yp,
                               size_t dim, void* ud) {
    (void)t; (void)y; (void)ud;
    size_t i;
    for (i = 0; i < dim; i++) {
        yp[i] = 0.0;
    }
    return LMMC_STATUS_OK;
}

int main(void) {
    int rc = 0;
    lmmc_ode_config_t cfg;
    lmmc_ode_result_t result;
    lmmc_status_t st;

    memset(&cfg, 0, sizeof(cfg));
    memset(&result, 0, sizeof(result));


    {
        double y[2] = {1.5, 1.0};
        double H_initial, H_final;
        lmmc_ode_config_t lv_cfg;


        H_initial = y[0] - log(y[0]) + y[1] - log(y[1]);

        st = lmmc_ode_default_config(0.0, 6.0, 2, &lv_cfg);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lv_cfg.initial_step = 0.001;
        lv_cfg.min_step = 0.001;
        lv_cfg.max_step = 0.001;
        lv_cfg.max_steps = 10000;

        st = lmmc_ode_rk4_solve(rhs_lotka_volterra, NULL, 2, 0.0, 6.0, y, &lv_cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }

        H_final = y[0] - log(y[0]) + y[1] - log(y[1]);
        if (!lmmc_test_nearly_equal(H_initial, H_final, 1e-6)) { rc = 1; goto done; }
    }


    {
        double y[1] = {1.0};
        lmmc_ode_config_t stiff_cfg;

        st = lmmc_ode_default_config(0.0, 0.01, 1, &stiff_cfg);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        stiff_cfg.initial_step = 0.0005;
        stiff_cfg.min_step = 0.0005;
        stiff_cfg.max_step = 0.0005;
        stiff_cfg.max_steps = 100;

        st = lmmc_ode_euler_solve(rhs_stiff, NULL, 1, 0.0, 0.01, y, &stiff_cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }


        if (y[0] < 0.0 || y[0] > 1.0) { rc = 1; goto done; }
    }


    {
        double y[1] = {1.0};
        double exact;
        lmmc_ode_config_t rk4_cfg;

        st = lmmc_ode_default_config(0.0, 5.0, 1, &rk4_cfg);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        rk4_cfg.initial_step = 0.01;
        rk4_cfg.min_step = 0.01;
        rk4_cfg.max_step = 0.01;
        rk4_cfg.max_steps = 1000;

        st = lmmc_ode_rk4_solve(rhs_exp_growth, NULL, 1, 0.0, 5.0, y, &rk4_cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }

        exact = exp(5.0);

        if (!lmmc_test_nearly_equal(y[0], exact, TEST_EPS_NORMAL)) { rc = 1; goto done; }
    }


    {
        double y[10];
        double exact_val;
        lmmc_ode_config_t hd_cfg;
        size_t i;
        double t_end = 1.0;

        for (i = 0; i < 10; i++) {
            y[i] = 1.0;
        }

        st = lmmc_ode_default_config(0.0, t_end, 10, &hd_cfg);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        hd_cfg.initial_step = 0.005;
        hd_cfg.min_step = 0.005;
        hd_cfg.max_step = 0.005;
        hd_cfg.max_steps = 1000;

        st = lmmc_ode_rk4_solve(rhs_high_dim, NULL, 10, 0.0, t_end, y, &hd_cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }

        for (i = 0; i < 10; i++) {
            exact_val = exp(-(double)(i + 1) * t_end);
            if (!lmmc_test_nearly_equal(y[i], exact_val, TEST_EPS_NORMAL)) { rc = 1; goto done; }
        }
    }


    {
        double y_h[1], y_h2[1];
        double exact;
        double err_h, err_h2;
        lmmc_ode_config_t step_cfg;

        exact = exp(1.0);


        y_h[0] = 1.0;
        st = lmmc_ode_default_config(0.0, 1.0, 1, &step_cfg);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        step_cfg.initial_step = 0.1;
        step_cfg.min_step = 0.1;
        step_cfg.max_step = 0.1;
        step_cfg.max_steps = 100;

        st = lmmc_ode_rk4_solve(rhs_exp_growth, NULL, 1, 0.0, 1.0, y_h, &step_cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }
        err_h = fabs(y_h[0] - exact);


        y_h2[0] = 1.0;
        step_cfg.initial_step = 0.05;
        step_cfg.min_step = 0.05;
        step_cfg.max_step = 0.05;
        step_cfg.max_steps = 100;

        st = lmmc_ode_rk4_solve(rhs_exp_growth, NULL, 1, 0.0, 1.0, y_h2, &step_cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }
        err_h2 = fabs(y_h2[0] - exact);


        if (err_h2 == 0.0 || (err_h / err_h2) < 8.0) { rc = 1; goto done; }
    }


    {
        double y[2] = {1.0, 0.0};
        lmmc_ode_config_t ho_cfg;
        double period = 2.0 * pi_val();

        st = lmmc_ode_default_config(0.0, period, 2, &ho_cfg);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        ho_cfg.initial_step = 0.005;
        ho_cfg.min_step = 0.005;
        ho_cfg.max_step = 0.005;
        ho_cfg.max_steps = 2000;

        st = lmmc_ode_rk4_solve(rhs_harmonic, NULL, 2, 0.0, period, y, &ho_cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }


        if (!lmmc_test_nearly_equal(y[0], 1.0, TEST_EPS_LOOSE)) { rc = 1; goto done; }
        if (!lmmc_test_nearly_equal(y[1], 0.0, TEST_EPS_LOOSE)) { rc = 1; goto done; }
    }


    {
        double y[3] = {1.5, -2.3, 4.7};
        double y_init[3] = {1.5, -2.3, 4.7};
        lmmc_ode_config_t zero_cfg;
        size_t i;


        st = lmmc_ode_default_config(0.0, 10.0, 3, &zero_cfg);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        zero_cfg.initial_step = 0.1;
        zero_cfg.min_step = 0.1;
        zero_cfg.max_step = 0.1;
        zero_cfg.max_steps = 200;

        st = lmmc_ode_euler_solve(rhs_zero, NULL, 3, 0.0, 10.0, y, &zero_cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }
        for (i = 0; i < 3; i++) {
            if (!lmmc_test_nearly_equal(y[i], y_init[i], TEST_EPS_TIGHT)) {
                rc = 1; goto done;
            }
        }


        y[0] = 1.5; y[1] = -2.3; y[2] = 4.7;
        zero_cfg.initial_step = 1.0;
        zero_cfg.min_step = 1.0;
        zero_cfg.max_step = 1.0;
        zero_cfg.max_steps = 100;
        st = lmmc_ode_rk4_solve(rhs_zero, NULL, 3, 0.0, 10.0, y, &zero_cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) { rc = 1; goto done; }
        for (i = 0; i < 3; i++) {
            if (!lmmc_test_nearly_equal(y[i], y_init[i], TEST_EPS_TIGHT)) {
                rc = 1; goto done;
            }
        }
    }


    {
        double y[1] = {1.0};
        lmmc_ode_config_t big_step_cfg;


        st = lmmc_ode_default_config(0.0, 0.5, 1, &big_step_cfg);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        big_step_cfg.initial_step = 2.0;
        big_step_cfg.min_step = 2.0;
        big_step_cfg.max_step = 2.0;
        big_step_cfg.max_steps = 100;


        st = lmmc_ode_rk4_solve(rhs_exp_growth, NULL, 1, 0.0, 0.5, y, &big_step_cfg, &result);


        if (st == LMMC_STATUS_OK && result.converged == 1) {

            if (!lmmc_test_nearly_equal(y[0], exp(0.5), 0.1)) { rc = 1; goto done; }
        }


        y[0] = 1.0;
        st = lmmc_ode_euler_solve(rhs_exp_growth, NULL, 1, 0.0, 0.5, y, &big_step_cfg, &result);
        if (st == LMMC_STATUS_OK && result.converged == 1) {


            if (fabs(y[0] - exp(0.5)) > 0.5) { rc = 1; goto done; }
        }
    }

done:
    if (rc != 0) {
        printf("ode_extended test failed\n");
    }
    return rc;
}
