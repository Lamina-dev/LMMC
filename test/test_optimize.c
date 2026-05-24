/**
 * @file test_optimize.c
 * @brief Unit tests for the LMMC optimization module.
 *
 * Tests:
 * 1. L-BFGS on Rosenbrock function for n=2,5,10,20 (gradient norm < 1e-6)
 * 2. Newton solver quadratic convergence on SPD linear system
 * 3. Broyden on a simple nonlinear system
 * 4. Gradient descent converges on a simple quadratic
 * 5. Levenberg-Marquardt on a simple least-squares problem
 *
 * Validates: Requirements 13.8, 13.9
 * @internal
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "lmmc/lmmc.h"
#include "lmmc/optimize.h"
#include "test_common.h"

/* ===================== Rosenbrock helpers ===================== */

/**
 * Rosenbrock objective: f(x) = sum_{i=0}^{n-2} [100*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]
 */
static lmmc_real_t rosenbrock_obj(const lmmc_vec_t* x, void* user_data) {
    (void)user_data;
    size_t n = x->size;
    lmmc_real_t f = 0.0;
    for (size_t i = 0; i < n - 1; i++) {
        lmmc_real_t xi = x->data[i];
        lmmc_real_t xi1 = x->data[i + 1];
        lmmc_real_t t1 = xi1 - xi * xi;
        lmmc_real_t t2 = 1.0 - xi;
        f += 100.0 * t1 * t1 + t2 * t2;
    }
    return f;
}

/**
 * Rosenbrock gradient.
 */
static lmmc_status_t rosenbrock_grad(const lmmc_vec_t* x, lmmc_vec_t* grad, void* user_data) {
    (void)user_data;
    size_t n = x->size;
    memset(grad->data, 0, n * sizeof(lmmc_real_t));
    for (size_t i = 0; i < n - 1; i++) {
        lmmc_real_t xi = x->data[i];
        lmmc_real_t xi1 = x->data[i + 1];
        lmmc_real_t t1 = xi1 - xi * xi;
        /* df/dx_i = -400*x_i*(x_{i+1} - x_i^2) - 2*(1 - x_i) */
        grad->data[i] += -400.0 * xi * t1 + 2.0 * (xi - 1.0);
        /* df/dx_{i+1} = 200*(x_{i+1} - x_i^2) */
        grad->data[i + 1] += 200.0 * t1;
    }
    return LMMC_STATUS_OK;
}

/* ===================== Newton / Broyden helpers ===================== */

/**
 * Linear system F(x) = Ax - b where A is SPD.
 * user_data points to a struct holding A and b.
 */
typedef struct {
    size_t n;
    const lmmc_real_t* A_data; /* n x n row-major SPD matrix */
    const lmmc_real_t* b_data; /* n-vector */
} linear_system_data_t;

static lmmc_status_t linear_system_F(const lmmc_vec_t* x, lmmc_vec_t* F, void* user_data) {
    linear_system_data_t* sys = (linear_system_data_t*)user_data;
    size_t n = sys->n;
    /* F = A*x - b */
    for (size_t i = 0; i < n; i++) {
        lmmc_real_t sum = 0.0;
        for (size_t j = 0; j < n; j++) {
            sum += sys->A_data[i * n + j] * x->data[j];
        }
        F->data[i] = sum - sys->b_data[i];
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t linear_system_J(const lmmc_vec_t* x, lmmc_mat_t* J, void* user_data) {
    (void)x;
    linear_system_data_t* sys = (linear_system_data_t*)user_data;
    size_t n = sys->n;
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            J->data[i * J->stride + j] = sys->A_data[i * n + j];
        }
    }
    return LMMC_STATUS_OK;
}

/* ===================== Broyden nonlinear system ===================== */

/**
 * Simple nonlinear system:
 *   F_0(x) = x_0^2 + x_1 - 3
 *   F_1(x) = x_0 + x_1^2 - 5
 * Solution near (1, 2): 1 + 2 - 3 = 0, 1 + 4 - 5 = 0.
 * Also (2.79..., -4.79...) but we start near (1,2).
 */
static lmmc_status_t broyden_nonlinear_F(const lmmc_vec_t* x, lmmc_vec_t* F, void* user_data) {
    (void)user_data;
    F->data[0] = x->data[0] * x->data[0] + x->data[1] - 3.0;
    F->data[1] = x->data[0] + x->data[1] * x->data[1] - 5.0;
    return LMMC_STATUS_OK;
}

/* ===================== Gradient descent helpers ===================== */

/**
 * Simple quadratic: f(x) = 0.5 * x^T * A * x - b^T * x
 * where A = diag(1, 2, ..., n), b = (1, 1, ..., 1).
 * Minimum at x_i = 1/i.
 */
typedef struct {
    size_t n;
} quadratic_data_t;

static lmmc_real_t quadratic_obj(const lmmc_vec_t* x, void* user_data) {
    quadratic_data_t* qd = (quadratic_data_t*)user_data;
    size_t n = qd->n;
    lmmc_real_t f = 0.0;
    for (size_t i = 0; i < n; i++) {
        lmmc_real_t ai = (lmmc_real_t)(i + 1);
        f += 0.5 * ai * x->data[i] * x->data[i] - x->data[i];
    }
    return f;
}

static lmmc_status_t quadratic_grad(const lmmc_vec_t* x, lmmc_vec_t* grad, void* user_data) {
    quadratic_data_t* qd = (quadratic_data_t*)user_data;
    size_t n = qd->n;
    for (size_t i = 0; i < n; i++) {
        lmmc_real_t ai = (lmmc_real_t)(i + 1);
        grad->data[i] = ai * x->data[i] - 1.0;
    }
    return LMMC_STATUS_OK;
}

/* ===================== Levenberg-Marquardt helpers ===================== */

/**
 * Least-squares problem: minimize ||r(x)||^2 where
 *   r_i(x) = x_i - c_i for a known target c.
 * This is trivial: solution is x = c.
 * We use c = (1, 2, 3).
 */
static lmmc_real_t lm_target[] = {1.0, 2.0, 3.0};

static lmmc_status_t lm_residual(const lmmc_vec_t* x, lmmc_vec_t* F, void* user_data) {
    (void)user_data;
    for (size_t i = 0; i < x->size; i++) {
        F->data[i] = x->data[i] - lm_target[i];
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lm_jacobian(const lmmc_vec_t* x, lmmc_mat_t* J, void* user_data) {
    (void)x;
    (void)user_data;
    size_t n = J->rows;
    /* J = I */
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            J->data[i * J->stride + j] = (i == j) ? 1.0 : 0.0;
        }
    }
    return LMMC_STATUS_OK;
}

/* ===================== Test functions ===================== */

/**
 * Test 1: L-BFGS on Rosenbrock for n=2,5,10,20.
 * Verify gradient norm < 1e-6 at convergence and solution near (1,...,1).
 */
static int test_lbfgs_rosenbrock(void) {
    size_t dims[] = {2, 5, 10, 20};
    size_t num_dims = sizeof(dims) / sizeof(dims[0]);

    for (size_t di = 0; di < num_dims; di++) {
        size_t n = dims[di];
        lmmc_vec_t x = {0};
        lmmc_optimize_config_t cfg = {0};
        lmmc_optimize_result_t result = {0};
        lmmc_status_t st;

        st = lmmc_vec_create(n, &x);
        if (st != LMMC_STATUS_OK) {
            printf("FAIL test_lbfgs_rosenbrock: vec_create failed for n=%zu\n", n);
            return 1;
        }

        /* Starting point: x = (-1, -1, ..., -1) — away from minimum */
        for (size_t i = 0; i < n; i++) {
            x.data[i] = -1.0;
        }

        st = lmmc_optimize_default_config(&cfg);
        if (st != LMMC_STATUS_OK) {
            lmmc_vec_destroy(&x);
            printf("FAIL test_lbfgs_rosenbrock: default_config failed\n");
            return 1;
        }
        cfg.max_iter = 50000;
        cfg.abs_tol = 1e-12;
        cfg.rel_tol = 1e-10;
        cfg.lbfgs_memory = 20;

        st = lmmc_minimize_lbfgs(rosenbrock_obj, rosenbrock_grad, NULL, &x, &cfg, &result);
        if (st != LMMC_STATUS_OK) {
            lmmc_vec_destroy(&x);
            printf("FAIL test_lbfgs_rosenbrock: lbfgs returned error for n=%zu\n", n);
            return 1;
        }

        if (!result.converged) {
            lmmc_vec_destroy(&x);
            printf("FAIL test_lbfgs_rosenbrock: did not converge for n=%zu (iter=%zu, residual=%.2e, reason=%d)\n",
                   n, result.num_iter, result.final_residual, (int)result.failure_reason);
            return 1;
        }

        /* Verify gradient norm < 1e-6 */
        lmmc_vec_t grad = {0};
        st = lmmc_vec_create(n, &grad);
        if (st != LMMC_STATUS_OK) {
            lmmc_vec_destroy(&x);
            printf("FAIL test_lbfgs_rosenbrock: grad vec_create failed\n");
            return 1;
        }
        rosenbrock_grad(&x, &grad, NULL);
        lmmc_real_t grad_norm = 0.0;
        lmmc_vec_norm2(&grad, &grad_norm);

        if (grad_norm > 1e-6) {
            printf("FAIL test_lbfgs_rosenbrock: gradient norm %.2e > 1e-6 for n=%zu\n", grad_norm, n);
            lmmc_vec_destroy(&x);
            lmmc_vec_destroy(&grad);
            return 1;
        }

        /* Verify solution is near (1,1,...,1) */
        for (size_t i = 0; i < n; i++) {
            if (fabs(x.data[i] - 1.0) > 1e-3) {
                printf("FAIL test_lbfgs_rosenbrock: x[%zu]=%.6f != 1.0 for n=%zu\n", i, x.data[i], n);
                lmmc_vec_destroy(&x);
                lmmc_vec_destroy(&grad);
                return 1;
            }
        }

        lmmc_vec_destroy(&x);
        lmmc_vec_destroy(&grad);
    }

    printf("PASS test_lbfgs_rosenbrock\n");
    return 0;
}

/**
 * Test 2: Newton solver on quadratic system F(x) = Ax - b where A is SPD.
 * Verify quadratic convergence (should converge in 1 iteration for linear system).
 */
static int test_newton_quadratic(void) {
    /* 3x3 SPD matrix: A = [[4,1,0],[1,3,1],[0,1,2]] */
    size_t n = 3;
    lmmc_real_t A_data[] = {4.0, 1.0, 0.0,
                            1.0, 3.0, 1.0,
                            0.0, 1.0, 2.0};
    lmmc_real_t b_data[] = {1.0, 2.0, 3.0};

    linear_system_data_t sys = {n, A_data, b_data};

    lmmc_vec_t x = {0};
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};
    lmmc_status_t st;

    st = lmmc_vec_create(n, &x);
    if (st != LMMC_STATUS_OK) {
        printf("FAIL test_newton_quadratic: vec_create failed\n");
        return 1;
    }

    /* Start at origin */
    for (size_t i = 0; i < n; i++) x.data[i] = 0.0;

    st = lmmc_optimize_default_config(&cfg);
    if (st != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&x);
        printf("FAIL test_newton_quadratic: default_config failed\n");
        return 1;
    }
    cfg.abs_tol = 1e-12;

    st = lmmc_nleq_newton(linear_system_F, linear_system_J, &sys, &x, &cfg, &result);
    if (st != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&x);
        printf("FAIL test_newton_quadratic: newton returned error\n");
        return 1;
    }

    if (!result.converged) {
        lmmc_vec_destroy(&x);
        printf("FAIL test_newton_quadratic: did not converge (iter=%zu, residual=%.2e)\n",
               result.num_iter, result.final_residual);
        return 1;
    }

    /* For a linear system, Newton should converge in exactly 1 iteration */
    if (result.num_iter > 2) {
        printf("FAIL test_newton_quadratic: expected <=2 iterations, got %zu\n", result.num_iter);
        lmmc_vec_destroy(&x);
        return 1;
    }

    /* Verify residual: compute Ax - b */
    lmmc_real_t max_residual = 0.0;
    for (size_t i = 0; i < n; i++) {
        lmmc_real_t sum = 0.0;
        for (size_t j = 0; j < n; j++) {
            sum += A_data[i * n + j] * x.data[j];
        }
        lmmc_real_t ri = fabs(sum - b_data[i]);
        if (ri > max_residual) max_residual = ri;
    }

    if (max_residual > 1e-10) {
        printf("FAIL test_newton_quadratic: residual %.2e > 1e-10\n", max_residual);
        lmmc_vec_destroy(&x);
        return 1;
    }

    lmmc_vec_destroy(&x);
    printf("PASS test_newton_quadratic\n");
    return 0;
}

/**
 * Test 3: Broyden on a simple nonlinear system.
 *   F_0(x) = x_0^2 + x_1 - 3
 *   F_1(x) = x_0 + x_1^2 - 5
 * Solution: (1, 2).
 */
static int test_broyden_nonlinear(void) {
    size_t n = 2;
    lmmc_vec_t x = {0};
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};
    lmmc_status_t st;

    st = lmmc_vec_create(n, &x);
    if (st != LMMC_STATUS_OK) {
        printf("FAIL test_broyden_nonlinear: vec_create failed\n");
        return 1;
    }

    /* Start near the solution */
    x.data[0] = 0.5;
    x.data[1] = 1.5;

    st = lmmc_optimize_default_config(&cfg);
    if (st != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&x);
        printf("FAIL test_broyden_nonlinear: default_config failed\n");
        return 1;
    }
    cfg.abs_tol = 1e-10;
    cfg.max_iter = 1000;

    st = lmmc_nleq_broyden(broyden_nonlinear_F, NULL, &x, &cfg, &result);
    if (st != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&x);
        printf("FAIL test_broyden_nonlinear: broyden returned error (st=%d)\n", (int)st);
        return 1;
    }

    if (!result.converged) {
        lmmc_vec_destroy(&x);
        printf("FAIL test_broyden_nonlinear: did not converge (iter=%zu, residual=%.2e, reason=%d)\n",
               result.num_iter, result.final_residual, (int)result.failure_reason);
        return 1;
    }

    /* Verify solution near (1, 2) */
    if (fabs(x.data[0] - 1.0) > 1e-6 || fabs(x.data[1] - 2.0) > 1e-6) {
        printf("FAIL test_broyden_nonlinear: solution (%.6f, %.6f) != (1, 2)\n",
               x.data[0], x.data[1]);
        lmmc_vec_destroy(&x);
        return 1;
    }

    lmmc_vec_destroy(&x);
    printf("PASS test_broyden_nonlinear\n");
    return 0;
}

/**
 * Test 4: Gradient descent on a simple quadratic.
 * f(x) = 0.5 * sum_i (i+1)*x_i^2 - sum_i x_i
 * Minimum at x_i = 1/(i+1).
 */
static int test_gradient_descent_quadratic(void) {
    size_t n = 5;
    quadratic_data_t qd = {n};
    lmmc_vec_t x = {0};
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};
    lmmc_status_t st;

    st = lmmc_vec_create(n, &x);
    if (st != LMMC_STATUS_OK) {
        printf("FAIL test_gradient_descent_quadratic: vec_create failed\n");
        return 1;
    }

    /* Start at x = (2, 2, 2, 2, 2) - closer to solution */
    for (size_t i = 0; i < n; i++) x.data[i] = 2.0;

    st = lmmc_optimize_default_config(&cfg);
    if (st != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&x);
        printf("FAIL test_gradient_descent_quadratic: default_config failed\n");
        return 1;
    }
    cfg.abs_tol = 1e-6;
    cfg.max_iter = 200000;

    st = lmmc_minimize_gradient_descent(quadratic_obj, quadratic_grad, &qd, &x, &cfg, &result);
    if (st != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&x);
        printf("FAIL test_gradient_descent_quadratic: gd returned error (st=%d)\n", (int)st);
        return 1;
    }

    if (!result.converged) {
        lmmc_vec_destroy(&x);
        printf("FAIL test_gradient_descent_quadratic: did not converge (iter=%zu, residual=%.2e, reason=%d)\n",
               result.num_iter, result.final_residual, (int)result.failure_reason);
        return 1;
    }

    /* Verify solution: x_i = 1/(i+1) */
    for (size_t i = 0; i < n; i++) {
        lmmc_real_t expected = 1.0 / (lmmc_real_t)(i + 1);
        if (fabs(x.data[i] - expected) > 1e-3) {
            printf("FAIL test_gradient_descent_quadratic: x[%zu]=%.6f, expected %.6f\n",
                   i, x.data[i], expected);
            lmmc_vec_destroy(&x);
            return 1;
        }
    }

    lmmc_vec_destroy(&x);
    printf("PASS test_gradient_descent_quadratic\n");
    return 0;
}

/**
 * Test 5: Levenberg-Marquardt on a simple least-squares problem.
 * r_i(x) = x_i - c_i, solution is x = c = (1, 2, 3).
 */
static int test_levenberg_marquardt(void) {
    size_t n = 3;
    lmmc_vec_t x = {0};
    lmmc_optimize_config_t cfg = {0};
    lmmc_optimize_result_t result = {0};
    lmmc_status_t st;

    st = lmmc_vec_create(n, &x);
    if (st != LMMC_STATUS_OK) {
        printf("FAIL test_levenberg_marquardt: vec_create failed\n");
        return 1;
    }

    /* Start at origin */
    for (size_t i = 0; i < n; i++) x.data[i] = 0.0;

    st = lmmc_optimize_default_config(&cfg);
    if (st != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&x);
        printf("FAIL test_levenberg_marquardt: default_config failed\n");
        return 1;
    }
    cfg.abs_tol = 1e-12;
    cfg.max_iter = 1000;

    st = lmmc_minimize_levenberg_marquardt(lm_residual, lm_jacobian, NULL, &x, &cfg, &result);
    if (st != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&x);
        printf("FAIL test_levenberg_marquardt: LM returned error (st=%d)\n", (int)st);
        return 1;
    }

    if (!result.converged) {
        lmmc_vec_destroy(&x);
        printf("FAIL test_levenberg_marquardt: did not converge (iter=%zu, residual=%.2e, reason=%d)\n",
               result.num_iter, result.final_residual, (int)result.failure_reason);
        return 1;
    }

    /* Verify solution near (1, 2, 3) */
    for (size_t i = 0; i < n; i++) {
        if (fabs(x.data[i] - lm_target[i]) > 1e-8) {
            printf("FAIL test_levenberg_marquardt: x[%zu]=%.10f, expected %.1f\n",
                   i, x.data[i], lm_target[i]);
            lmmc_vec_destroy(&x);
            return 1;
        }
    }

    lmmc_vec_destroy(&x);
    printf("PASS test_levenberg_marquardt\n");
    return 0;
}

int main(void) {
    int rc = 0;

    rc |= test_lbfgs_rosenbrock();
    rc |= test_newton_quadratic();
    rc |= test_broyden_nonlinear();
    rc |= test_gradient_descent_quadratic();
    rc |= test_levenberg_marquardt();

    if (rc == 0) {
        printf("\nAll optimization tests PASSED.\n");
    } else {
        printf("\nSome optimization tests FAILED.\n");
    }

    return rc;
}
