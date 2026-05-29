/**
 * @file test_convergence.c
 * 针对 LMMC 中 convergence 相关接口的单元测试。
 */
#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"
#include "test_common.h"


static lmmc_status_t rhs_exp_growth(double t, const double* y, double* y_prime,
                                     size_t dim, void* user_data) {
    (void)t;
    (void)user_data;
    if (dim != 1) return LMMC_STATUS_INVALID_ARGUMENT;
    y_prime[0] = y[0];
    return LMMC_STATUS_OK;
}


static lmmc_real_t quad_sin(lmmc_real_t x, void* user_data) {
    (void)user_data;
    return sin(x);
}


static lmmc_real_t newton_fn(lmmc_real_t x, void* user_data) {
    (void)user_data;
    return x * x - 2.0;
}

static lmmc_real_t newton_dfn(lmmc_real_t x, void* user_data) {
    (void)user_data;
    return 2.0 * x;
}


static lmmc_real_t bisect_fn(lmmc_real_t x, void* user_data) {
    (void)user_data;
    return x * x - 2.0;
}


#define MAX_NEWTON_ITERS 64
typedef struct {
    double x_values[MAX_NEWTON_ITERS];
    size_t count;
} newton_log_data_t;

static void newton_log_cb(size_t iter, lmmc_real_t x, lmmc_real_t f_x,
                           void* user_data) {
    (void)iter;
    (void)f_x;
    newton_log_data_t* data = (newton_log_data_t*)user_data;
    if (data->count < MAX_NEWTON_ITERS) {
        data->x_values[data->count] = x;
        data->count++;
    }
}


#define MAX_BISECT_ITERS 128
typedef struct {
    double x_values[MAX_BISECT_ITERS];
    size_t count;
} bisect_log_data_t;

static void bisect_log_cb(size_t iter, lmmc_real_t x, lmmc_real_t f_x,
                           void* user_data) {
    (void)iter;
    (void)f_x;
    bisect_log_data_t* data = (bisect_log_data_t*)user_data;
    if (data->count < MAX_BISECT_ITERS) {
        data->x_values[data->count] = x;
        data->count++;
    }
}

int main(void) {
    int rc = 0;


    {
        double h = 0.01;
        double exact = exp(1.0);


        double y_h[1] = {1.0};
        lmmc_ode_config_t cfg_h = {0};
        lmmc_ode_result_t res_h = {0};
        lmmc_ode_default_config(0.0, 1.0, 1, &cfg_h);
        cfg_h.initial_step = h;
        cfg_h.min_step = h;
        cfg_h.max_step = h;
        cfg_h.max_steps = 200;

        lmmc_status_t st = lmmc_ode_euler_solve(rhs_exp_growth, NULL, 1, 0.0, 1.0,
                                                 y_h, &cfg_h, &res_h);
        if (st != LMMC_STATUS_OK || res_h.converged != 1) {
            printf("convergence test failed: Euler h solve\n");
            rc = 1; goto done;
        }


        double y_h2[1] = {1.0};
        lmmc_ode_config_t cfg_h2 = {0};
        lmmc_ode_result_t res_h2 = {0};
        lmmc_ode_default_config(0.0, 1.0, 1, &cfg_h2);
        cfg_h2.initial_step = h / 2.0;
        cfg_h2.min_step = h / 2.0;
        cfg_h2.max_step = h / 2.0;
        cfg_h2.max_steps = 400;

        st = lmmc_ode_euler_solve(rhs_exp_growth, NULL, 1, 0.0, 1.0,
                                  y_h2, &cfg_h2, &res_h2);
        if (st != LMMC_STATUS_OK || res_h2.converged != 1) {
            printf("convergence test failed: Euler h/2 solve\n");
            rc = 1; goto done;
        }

        double error_h = fabs(y_h[0] - exact);
        double error_h2 = fabs(y_h2[0] - exact);
        double ratio = error_h / error_h2;
        double expected_ratio = 2.0;

        if (ratio < expected_ratio * 0.7 || ratio > expected_ratio * 1.3) {
            printf("convergence test failed: Euler order 1, ratio=%.4f expected~%.1f\n",
                   ratio, expected_ratio);
            rc = 1; goto done;
        }
    }


    {
        double h = 0.1;
        double exact = exp(1.0);


        double y_h[1] = {1.0};
        lmmc_ode_config_t cfg_h = {0};
        lmmc_ode_result_t res_h = {0};
        lmmc_ode_default_config(0.0, 1.0, 1, &cfg_h);
        cfg_h.initial_step = h;
        cfg_h.min_step = h;
        cfg_h.max_step = h;
        cfg_h.max_steps = 100;

        lmmc_status_t st = lmmc_ode_rk4_solve(rhs_exp_growth, NULL, 1, 0.0, 1.0,
                                               y_h, &cfg_h, &res_h);
        if (st != LMMC_STATUS_OK || res_h.converged != 1) {
            printf("convergence test failed: RK4 h solve\n");
            rc = 1; goto done;
        }


        double y_h2[1] = {1.0};
        lmmc_ode_config_t cfg_h2 = {0};
        lmmc_ode_result_t res_h2 = {0};
        lmmc_ode_default_config(0.0, 1.0, 1, &cfg_h2);
        cfg_h2.initial_step = h / 2.0;
        cfg_h2.min_step = h / 2.0;
        cfg_h2.max_step = h / 2.0;
        cfg_h2.max_steps = 200;

        st = lmmc_ode_rk4_solve(rhs_exp_growth, NULL, 1, 0.0, 1.0,
                                y_h2, &cfg_h2, &res_h2);
        if (st != LMMC_STATUS_OK || res_h2.converged != 1) {
            printf("convergence test failed: RK4 h/2 solve\n");
            rc = 1; goto done;
        }

        double error_h = fabs(y_h[0] - exact);
        double error_h2 = fabs(y_h2[0] - exact);
        double ratio = error_h / error_h2;
        double expected_ratio = 16.0;

        if (ratio < expected_ratio * 0.7 || ratio > expected_ratio * 1.3) {
            printf("convergence test failed: RK4 order 4, ratio=%.4f expected~%.1f\n",
                   ratio, expected_ratio);
            rc = 1; goto done;
        }
    }


    {
        double exact = 2.0;
        size_t n = 32;

        double result_n = 0.0;
        lmmc_status_t st = lmmc_quad_trapezoid(quad_sin, NULL, 0.0, LMMC_CONST_PI,
                                                n, &result_n);
        if (st != LMMC_STATUS_OK) {
            printf("convergence test failed: trapezoid n solve\n");
            rc = 1; goto done;
        }

        double result_2n = 0.0;
        st = lmmc_quad_trapezoid(quad_sin, NULL, 0.0, LMMC_CONST_PI,
                                 2 * n, &result_2n);
        if (st != LMMC_STATUS_OK) {
            printf("convergence test failed: trapezoid 2n solve\n");
            rc = 1; goto done;
        }

        double error_n = fabs(result_n - exact);
        double error_2n = fabs(result_2n - exact);
        double ratio = error_n / error_2n;
        double expected_ratio = 4.0;

        if (ratio < expected_ratio * 0.7 || ratio > expected_ratio * 1.3) {
            printf("convergence test failed: trapezoid order 2, ratio=%.4f expected~%.1f\n",
                   ratio, expected_ratio);
            rc = 1; goto done;
        }
    }


    {
        double exact = 2.0;
        size_t n = 8;

        double result_n = 0.0;
        lmmc_status_t st = lmmc_quad_simpson(quad_sin, NULL, 0.0, LMMC_CONST_PI,
                                             n, &result_n);
        if (st != LMMC_STATUS_OK) {
            printf("convergence test failed: simpson n solve\n");
            rc = 1; goto done;
        }

        double result_2n = 0.0;
        st = lmmc_quad_simpson(quad_sin, NULL, 0.0, LMMC_CONST_PI,
                               2 * n, &result_2n);
        if (st != LMMC_STATUS_OK) {
            printf("convergence test failed: simpson 2n solve\n");
            rc = 1; goto done;
        }

        double error_n = fabs(result_n - exact);
        double error_2n = fabs(result_2n - exact);
        double ratio = error_n / error_2n;
        double expected_ratio = 16.0;

        if (ratio < expected_ratio * 0.7 || ratio > expected_ratio * 1.3) {
            printf("convergence test failed: simpson order 4, ratio=%.4f expected~%.1f\n",
                   ratio, expected_ratio);
            rc = 1; goto done;
        }
    }


    {
        newton_log_data_t log_data = {0};
        log_data.count = 0;

        lmmc_nonlinear_config_t cfg = {0};
        lmmc_nonlinear_result_t result = {0};
        lmmc_nonlinear_default_config(&cfg);
        cfg.max_iter = 50;
        cfg.abs_tol = 1e-15;
        cfg.rel_tol = 1e-15;
        cfg.log_cb = newton_log_cb;
        cfg.log_user_data = &log_data;

        lmmc_status_t st = lmmc_newton_solve(newton_fn, newton_dfn, NULL, 2.0,
                                             &cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) {
            printf("convergence test failed: Newton solve\n");
            rc = 1; goto done;
        }

        double root = sqrt(2.0);


        int quadratic_pairs = 0;
        for (size_t i = 1; i + 1 < log_data.count; i++) {
            double e_k = fabs(log_data.x_values[i] - root);
            double e_k1 = fabs(log_data.x_values[i + 1] - root);


            if (e_k < 1e-14 || e_k > 1.0) continue;
            if (e_k1 < 1e-15) continue;


            double c_est = e_k1 / (e_k * e_k);

            if (c_est > 0.0 && c_est < 10.0) {
                quadratic_pairs++;
            }
        }


        if (quadratic_pairs < 2) {
            printf("convergence test failed: Newton quadratic, only %d valid pairs\n",
                   quadratic_pairs);
            rc = 1; goto done;
        }
    }


    {
        bisect_log_data_t log_data = {0};
        log_data.count = 0;

        lmmc_nonlinear_config_t cfg = {0};
        lmmc_nonlinear_result_t result = {0};
        lmmc_nonlinear_default_config(&cfg);
        cfg.max_iter = 50;
        cfg.abs_tol = 1e-15;
        cfg.rel_tol = 1e-15;
        cfg.log_cb = bisect_log_cb;
        cfg.log_user_data = &log_data;

        lmmc_status_t st = lmmc_bisection_solve(bisect_fn, NULL, 0.0, 2.0,
                                                &cfg, &result);
        if (st != LMMC_STATUS_OK || result.converged != 1) {
            printf("convergence test failed: bisection solve\n");
            rc = 1; goto done;
        }

        double root = sqrt(2.0);


        int linear_pairs = 0;
        for (size_t i = 0; i + 1 < log_data.count; i++) {
            double e_k = fabs(log_data.x_values[i] - root);
            double e_k1 = fabs(log_data.x_values[i + 1] - root);


            if (e_k1 < 1e-14) continue;
            if (e_k < 1e-14) continue;

            double ratio = e_k / e_k1;

            if (ratio > 0.5 && ratio < 8.0) {
                linear_pairs++;
            }
        }


        if (log_data.count >= 3) {

            double e_first = fabs(log_data.x_values[0] - root);
            double e_mid = fabs(log_data.x_values[log_data.count / 2] - root);
            size_t half_iters = log_data.count / 2;

            if (e_mid > 1e-14 && e_first > 1e-14) {

                double expected_reduction = pow(2.0, (double)half_iters);
                double actual_reduction = e_first / e_mid;


                if (actual_reduction < expected_reduction * 0.3 ||
                    actual_reduction > expected_reduction * 3.0) {
                    printf("convergence test failed: bisection linear, "
                           "reduction=%.2f expected~%.2f\n",
                           actual_reduction, expected_reduction);
                    rc = 1; goto done;
                }
            }
        }


        if (linear_pairs < 3) {
            printf("convergence test failed: bisection linear, only %d valid pairs\n",
                   linear_pairs);
            rc = 1; goto done;
        }
    }

done:
    if (rc != 0) {
        printf("convergence test failed\n");
    }
    return rc;
}
