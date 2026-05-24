#include <math.h>
#include <stdio.h>
#include "lmmc/lmmc.h"

static lmmc_status_t rhs_decay(double t, const double* y, double* yp,
                                size_t dim, void* ud) {
    (void)t; (void)ud; (void)dim;
    yp[0] = -10.0 * y[0];
    return LMMC_STATUS_OK;
}

int main(void) {
    lmmc_ode_config_t cfg = {0};
    lmmc_ode_result_t res = {0};
    double y[1];
    double exact;

    lmmc_init();
    lmmc_ode_default_config(0.0, 1.0, 1, &cfg);
    cfg.jacobian = NULL;
    exact = exp(-10.0);

    /* Test implicit Euler */
    y[0] = 1.0;
    lmmc_ode_implicit_euler_solve(rhs_decay, NULL, 1, 0.0, 1.0, y, &cfg, &res);
    printf("Implicit Euler: y=%.6e, exact=%.6e, conv=%d\n",
           y[0], exact, res.converged);
    if (!res.converged) { printf("FAIL\n"); return 1; }

    /* Test trapezoidal */
    y[0] = 1.0;
    lmmc_ode_trapezoidal_solve(rhs_decay, NULL, 1, 0.0, 1.0, y, &cfg, &res);
    printf("Trapezoidal: y=%.6e, exact=%.6e, conv=%d\n",
           y[0], exact, res.converged);
    if (!res.converged) { printf("FAIL\n"); return 1; }

    /* Test SDIRK4 */
    y[0] = 1.0;
    lmmc_ode_sdirk4_solve(rhs_decay, NULL, 1, 0.0, 1.0, y, &cfg, &res);
    printf("SDIRK4: y=%.6e, exact=%.6e, conv=%d\n",
           y[0], exact, res.converged);
    if (!res.converged) { printf("FAIL\n"); return 1; }

    /* Test Rosenbrock GRK4T */
    y[0] = 1.0;
    lmmc_ode_rosenbrock_grk4t_solve(rhs_decay, NULL, 1, 0.0, 1.0, y, &cfg, &res);
    printf("Rosenbrock: y=%.6e, exact=%.6e, conv=%d\n",
           y[0], exact, res.converged);
    if (!res.converged) { printf("FAIL\n"); return 1; }

    /* Test A-stability: implicit Euler with large step */
    cfg.initial_step = 0.5;
    cfg.min_step = 0.1;
    cfg.max_step = 1.0;
    y[0] = 1.0;
    lmmc_ode_implicit_euler_solve(rhs_decay, NULL, 1, 0.0, 1.0, y, &cfg, &res);
    printf("A-stability: y=%.6e, bounded=%d\n", y[0], (fabs(y[0]) < 2.0));
    if (fabs(y[0]) > 2.0) { printf("FAIL: not A-stable\n"); return 1; }

    printf("ALL PASSED\n");
    lmmc_deinit();
    return 0;
}
