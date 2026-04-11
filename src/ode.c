#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "lmmc/ode.h"

static int lmmc_is_finite_number(double v) {
    return isfinite(v) ? 1 : 0;
}

static int lmmc_mul_overflow_size(size_t a, size_t b, size_t* out) {
    if (a == 0 || b == 0) {
        *out = 0;
        return 0;
    }
    if (a > ((size_t)-1) / b) {
        return 1;
    }
    *out = a * b;
    return 0;
}

static double lmmc_absd(double x) {
    return x < 0.0 ? -x : x;
}

static double lmmc_maxd(double a, double b) {
    return a > b ? a : b;
}

static double lmmc_clampd(double v, double lo, double hi) {
    if (v < lo) {
        return lo;
    }
    if (v > hi) {
        return hi;
    }
    return v;
}

static void lmmc_ode_reset_result(lmmc_ode_result_t* out_result, double t_start) {
    if (out_result == NULL) {
        return;
    }

    out_result->converged = 0;
    out_result->num_steps = 0;
    out_result->num_rhs_evals = 0;
    out_result->final_t = t_start;
    out_result->failure_reason = LMMC_ODE_FAILURE_NONE;
}

const char* lmmc_ode_failure_string(lmmc_ode_failure_t reason) {
    switch (reason) {
        case LMMC_ODE_FAILURE_NONE:
            return "none";
        case LMMC_ODE_FAILURE_INVALID_DIMENSION:
            return "invalid dimension";
        case LMMC_ODE_FAILURE_INVALID_STEP:
            return "invalid step";
        case LMMC_ODE_FAILURE_MAX_STEPS:
            return "max steps reached";
        case LMMC_ODE_FAILURE_NUMERICAL_ISSUE:
            return "numerical issue";
        case LMMC_ODE_FAILURE_RHS_EVAL_FAILED:
            return "rhs evaluation failed";
        case LMMC_ODE_FAILURE_TOLERANCE_INCONSISTENT:
            return "tolerance inconsistent";
        default:
            return "unknown";
    }
}

lmmc_status_t lmmc_ode_default_config(
    double t_start,
    double t_end,
    size_t problem_dim,
    lmmc_ode_config_t* out_cfg
) {
    double span = 0.0;
    double initial_step = 0.0;
    double min_step = 0.0;
    double max_step = 0.0;
    size_t max_steps = 0;

    if (out_cfg == NULL || problem_dim == 0 || !lmmc_is_finite_number(t_start) ||
        !lmmc_is_finite_number(t_end) || !(t_end > t_start)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    span = t_end - t_start;
    initial_step = span / 100.0;
    if (!lmmc_is_finite_number(initial_step) || initial_step <= 0.0) {
        initial_step = 1e-3;
    }

    min_step = span * 1e-10;
    if (!lmmc_is_finite_number(min_step) || min_step <= 0.0) {
        min_step = initial_step * 1e-6;
    }
    if (min_step > initial_step) {
        min_step = initial_step * 0.1;
    }

    max_step = span / 10.0;
    if (!lmmc_is_finite_number(max_step) || max_step <= 0.0) {
        max_step = initial_step;
    }
    if (max_step < initial_step) {
        max_step = initial_step;
    }

    if (problem_dim > ((size_t)-1) / 10000) {
        max_steps = 100000;
    } else {
        max_steps = problem_dim * 10000;
        if (max_steps < 1000) {
            max_steps = 1000;
        }
    }

    out_cfg->initial_step = initial_step;
    out_cfg->min_step = min_step;
    out_cfg->max_step = max_step;
    out_cfg->abs_tol = LMMC_DEFAULT_ABS_TOL;
    out_cfg->rel_tol = LMMC_DEFAULT_REL_TOL;
    out_cfg->max_steps = max_steps;
    out_cfg->adaptive_step_beta = 0.9;
    out_cfg->verbose = 0;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_ode_load_and_validate_config(
    size_t dim,
    double t_start,
    double t_end,
    const lmmc_ode_config_t* cfg,
    lmmc_ode_config_t* out_cfg,
    lmmc_ode_result_t* out_result
) {
    if (dim == 0) {
        if (out_result != NULL) {
            out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_DIMENSION;
        }
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (!lmmc_is_finite_number(t_start) || !lmmc_is_finite_number(t_end) || !(t_end > t_start)) {
        if (out_result != NULL) {
            out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_STEP;
        }
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (out_cfg == NULL) {
        if (out_result != NULL) {
            out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_STEP;
        }
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (cfg != NULL) {
        *out_cfg = *cfg;
    } else {
        lmmc_status_t st = lmmc_ode_default_config(t_start, t_end, dim, out_cfg);
        if (st != LMMC_STATUS_OK) {
            if (out_result != NULL) {
                out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_STEP;
            }
            return st;
        }
    }

    if (!lmmc_is_finite_number(out_cfg->abs_tol) || !lmmc_is_finite_number(out_cfg->rel_tol) ||
        out_cfg->abs_tol < 0.0 || out_cfg->rel_tol < 0.0) {
        if (out_result != NULL) {
            out_result->failure_reason = LMMC_ODE_FAILURE_TOLERANCE_INCONSISTENT;
        }
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (!lmmc_is_finite_number(out_cfg->initial_step) || !lmmc_is_finite_number(out_cfg->min_step) ||
        !lmmc_is_finite_number(out_cfg->max_step) || out_cfg->initial_step <= 0.0 ||
        out_cfg->min_step <= 0.0 || out_cfg->max_step <= 0.0 || out_cfg->min_step > out_cfg->max_step ||
        out_cfg->max_steps == 0 || !lmmc_is_finite_number(out_cfg->adaptive_step_beta) ||
        out_cfg->adaptive_step_beta <= 0.0 || out_cfg->adaptive_step_beta > 1.0) {
        if (out_result != NULL) {
            out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_STEP;
        }
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_ode_rhs_eval(
    lmmc_ode_rhs_t rhs,
    double t,
    const double* y,
    double* y_prime,
    size_t dim,
    void* user_data,
    size_t* io_eval_count,
    int* out_callback_failed
) {
    size_t i = 0;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (rhs == NULL || y == NULL || y_prime == NULL || io_eval_count == NULL || out_callback_failed == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    *out_callback_failed = 0;
    st = rhs(t, y, y_prime, dim, user_data);
    if (st != LMMC_STATUS_OK) {
        *out_callback_failed = 1;
        return st;
    }

    *io_eval_count += 1;
    for (i = 0; i < dim; ++i) {
        if (!lmmc_is_finite_number(y_prime[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }

    return LMMC_STATUS_OK;
}

static int lmmc_ode_state_is_finite(const double* y, size_t dim) {
    size_t i = 0;
    if (y == NULL || dim == 0) {
        return 0;
    }

    for (i = 0; i < dim; ++i) {
        if (!lmmc_is_finite_number(y[i])) {
            return 0;
        }
    }

    return 1;
}

lmmc_status_t lmmc_ode_euler_solve(
    lmmc_ode_rhs_t rhs,
    void* user_data,
    size_t dim,
    double t_start,
    double t_end,
    double* y,
    const lmmc_ode_config_t* cfg,
    lmmc_ode_result_t* out_result
) {
    lmmc_ode_config_t local_cfg = {0};
    size_t work_bytes = 0;
    double* y_prime = NULL;
    double t = t_start;
    double h = 0.0;

    if (rhs == NULL || y == NULL || out_result == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_ode_reset_result(out_result, t_start);

    if (lmmc_ode_load_and_validate_config(dim, t_start, t_end, cfg, &local_cfg, out_result) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (!lmmc_ode_state_is_finite(y, dim)) {
        out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    if (lmmc_mul_overflow_size(dim, sizeof(double), &work_bytes)) {
        out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_DIMENSION;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    y_prime = (double*)lmmc_alloc(work_bytes);
    if (y_prime == NULL) {
        out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    h = lmmc_clampd(local_cfg.initial_step, local_cfg.min_step, local_cfg.max_step);
    if (h > (t_end - t_start)) {
        h = t_end - t_start;
    }

    while (t < t_end && out_result->num_steps < local_cfg.max_steps) {
        size_t i = 0;
        double rem = t_end - t;
        int callback_failed = 0;
        lmmc_status_t st = LMMC_STATUS_OK;

        if (rem <= 0.0 || !lmmc_is_finite_number(rem)) {
            break;
        }

        if (h > rem) {
            h = rem;
        }

        st = lmmc_ode_rhs_eval(rhs, t, y, y_prime, dim, user_data, &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            lmmc_free(y_prime);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        for (i = 0; i < dim; ++i) {
            y[i] += h * y_prime[i];
        }

        if (!lmmc_ode_state_is_finite(y, dim)) {
            out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            lmmc_free(y_prime);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        t += h;
        if (!lmmc_is_finite_number(t)) {
            out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            lmmc_free(y_prime);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }

        out_result->num_steps += 1;
        out_result->final_t = t;
    }

    if (t >= t_end) {
        out_result->converged = 1;
        out_result->failure_reason = LMMC_ODE_FAILURE_NONE;
    } else {
        out_result->converged = 0;
        out_result->failure_reason = LMMC_ODE_FAILURE_MAX_STEPS;
    }

    lmmc_free(y_prime);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_ode_rk4_solve(
    lmmc_ode_rhs_t rhs,
    void* user_data,
    size_t dim,
    double t_start,
    double t_end,
    double* y,
    const lmmc_ode_config_t* cfg,
    lmmc_ode_result_t* out_result
) {
    lmmc_ode_config_t local_cfg = {0};
    size_t work_bytes = 0;
    double* k1 = NULL;
    double* k2 = NULL;
    double* k3 = NULL;
    double* k4 = NULL;
    double* y_tmp = NULL;
    double t = t_start;
    double h = 0.0;

    if (rhs == NULL || y == NULL || out_result == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_ode_reset_result(out_result, t_start);

    if (lmmc_ode_load_and_validate_config(dim, t_start, t_end, cfg, &local_cfg, out_result) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (!lmmc_ode_state_is_finite(y, dim)) {
        out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    if (lmmc_mul_overflow_size(dim, sizeof(double), &work_bytes)) {
        out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_DIMENSION;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    k1 = (double*)lmmc_alloc(work_bytes);
    k2 = (double*)lmmc_alloc(work_bytes);
    k3 = (double*)lmmc_alloc(work_bytes);
    k4 = (double*)lmmc_alloc(work_bytes);
    y_tmp = (double*)lmmc_alloc(work_bytes);

    if (k1 == NULL || k2 == NULL || k3 == NULL || k4 == NULL || y_tmp == NULL) {
        lmmc_free(y_tmp);
        lmmc_free(k4);
        lmmc_free(k3);
        lmmc_free(k2);
        lmmc_free(k1);
        out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    h = lmmc_clampd(local_cfg.initial_step, local_cfg.min_step, local_cfg.max_step);
    if (h > (t_end - t_start)) {
        h = t_end - t_start;
    }

    while (t < t_end && out_result->num_steps < local_cfg.max_steps) {
        size_t i = 0;
        double rem = t_end - t;
        int callback_failed = 0;
        lmmc_status_t st = LMMC_STATUS_OK;

        if (rem <= 0.0 || !lmmc_is_finite_number(rem)) {
            break;
        }

        if (h > rem) {
            h = rem;
        }

        st = lmmc_ode_rhs_eval(rhs, t, y, k1, dim, user_data, &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            goto fail;
        }

        for (i = 0; i < dim; ++i) {
            y_tmp[i] = y[i] + 0.5 * h * k1[i];
        }

        st = lmmc_ode_rhs_eval(rhs, t + 0.5 * h, y_tmp, k2, dim, user_data, &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            goto fail;
        }

        for (i = 0; i < dim; ++i) {
            y_tmp[i] = y[i] + 0.5 * h * k2[i];
        }

        st = lmmc_ode_rhs_eval(rhs, t + 0.5 * h, y_tmp, k3, dim, user_data, &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            goto fail;
        }

        for (i = 0; i < dim; ++i) {
            y_tmp[i] = y[i] + h * k3[i];
        }

        st = lmmc_ode_rhs_eval(rhs, t + h, y_tmp, k4, dim, user_data, &out_result->num_rhs_evals, &callback_failed);
        if (st != LMMC_STATUS_OK) {
            out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            goto fail;
        }

        for (i = 0; i < dim; ++i) {
            y[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }

        if (!lmmc_ode_state_is_finite(y, dim)) {
            out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            goto fail;
        }

        t += h;
        if (!lmmc_is_finite_number(t)) {
            out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
            goto fail;
        }

        out_result->num_steps += 1;
        out_result->final_t = t;
    }

    if (t >= t_end) {
        out_result->converged = 1;
        out_result->failure_reason = LMMC_ODE_FAILURE_NONE;
    } else {
        out_result->converged = 0;
        out_result->failure_reason = LMMC_ODE_FAILURE_MAX_STEPS;
    }

    lmmc_free(y_tmp);
    lmmc_free(k4);
    lmmc_free(k3);
    lmmc_free(k2);
    lmmc_free(k1);
    return LMMC_STATUS_OK;

fail:
    lmmc_free(y_tmp);
    lmmc_free(k4);
    lmmc_free(k3);
    lmmc_free(k2);
    lmmc_free(k1);
    return LMMC_STATUS_NUMERICAL_FAILURE;
}

lmmc_status_t lmmc_ode_rk45_solve(
    lmmc_ode_rhs_t rhs,
    void* user_data,
    size_t dim,
    double t_start,
    double t_end,
    double* y,
    const lmmc_ode_config_t* cfg,
    lmmc_ode_result_t* out_result
) {
    lmmc_ode_config_t local_cfg = {0};
    size_t work_bytes = 0;
    double* k1 = NULL;
    double* k2 = NULL;
    double* k3 = NULL;
    double* k4 = NULL;
    double* k5 = NULL;
    double* k6 = NULL;
    double* k7 = NULL;
    double* y_tmp = NULL;
    double* y5 = NULL;
    double t = t_start;
    double h = 0.0;

    if (rhs == NULL || y == NULL || out_result == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_ode_reset_result(out_result, t_start);

    if (lmmc_ode_load_and_validate_config(dim, t_start, t_end, cfg, &local_cfg, out_result) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (!lmmc_ode_state_is_finite(y, dim)) {
        out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }

    if (lmmc_mul_overflow_size(dim, sizeof(double), &work_bytes)) {
        out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_DIMENSION;
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    k1 = (double*)lmmc_alloc(work_bytes);
    k2 = (double*)lmmc_alloc(work_bytes);
    k3 = (double*)lmmc_alloc(work_bytes);
    k4 = (double*)lmmc_alloc(work_bytes);
    k5 = (double*)lmmc_alloc(work_bytes);
    k6 = (double*)lmmc_alloc(work_bytes);
    k7 = (double*)lmmc_alloc(work_bytes);
    y_tmp = (double*)lmmc_alloc(work_bytes);
    y5 = (double*)lmmc_alloc(work_bytes);

    if (k1 == NULL || k2 == NULL || k3 == NULL || k4 == NULL || k5 == NULL || k6 == NULL ||
        k7 == NULL || y_tmp == NULL || y5 == NULL) {
        lmmc_free(y5);
        lmmc_free(y_tmp);
        lmmc_free(k7);
        lmmc_free(k6);
        lmmc_free(k5);
        lmmc_free(k4);
        lmmc_free(k3);
        lmmc_free(k2);
        lmmc_free(k1);
        out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    h = lmmc_clampd(local_cfg.initial_step, local_cfg.min_step, local_cfg.max_step);
    if (h > (t_end - t_start)) {
        h = t_end - t_start;
    }

    while (t < t_end && out_result->num_steps < local_cfg.max_steps) {
        double rem = t_end - t;
        int accepted = 0;
        size_t reject_guard = 0;

        if (rem <= 0.0 || !lmmc_is_finite_number(rem)) {
            break;
        }

        if (h > rem) {
            h = rem;
        }

        while (!accepted) {
            size_t i = 0;
            int callback_failed = 0;
            double err_norm = 0.0;
            double factor = 1.0;
            double h_new = 0.0;
            lmmc_status_t st = LMMC_STATUS_OK;

            if (++reject_guard > 1000) {
                out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
                goto fail;
            }

            if (h < local_cfg.min_step) {
                out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_STEP;
                goto fail;
            }

            st = lmmc_ode_rhs_eval(rhs, t, y, k1, dim, user_data, &out_result->num_rhs_evals, &callback_failed);
            if (st != LMMC_STATUS_OK) {
                out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
                goto fail;
            }

            for (i = 0; i < dim; ++i) {
                y_tmp[i] = y[i] + h * (1.0 / 5.0) * k1[i];
            }
            st = lmmc_ode_rhs_eval(rhs, t + h * (1.0 / 5.0), y_tmp, k2, dim, user_data, &out_result->num_rhs_evals, &callback_failed);
            if (st != LMMC_STATUS_OK) {
                out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
                goto fail;
            }

            for (i = 0; i < dim; ++i) {
                y_tmp[i] = y[i] + h * ((3.0 / 40.0) * k1[i] + (9.0 / 40.0) * k2[i]);
            }
            st = lmmc_ode_rhs_eval(rhs, t + h * (3.0 / 10.0), y_tmp, k3, dim, user_data, &out_result->num_rhs_evals, &callback_failed);
            if (st != LMMC_STATUS_OK) {
                out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
                goto fail;
            }

            for (i = 0; i < dim; ++i) {
                y_tmp[i] = y[i] + h * ((44.0 / 45.0) * k1[i] + (-56.0 / 15.0) * k2[i] + (32.0 / 9.0) * k3[i]);
            }
            st = lmmc_ode_rhs_eval(rhs, t + h * (4.0 / 5.0), y_tmp, k4, dim, user_data, &out_result->num_rhs_evals, &callback_failed);
            if (st != LMMC_STATUS_OK) {
                out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
                goto fail;
            }

            for (i = 0; i < dim; ++i) {
                y_tmp[i] = y[i] + h * ((19372.0 / 6561.0) * k1[i] + (-25360.0 / 2187.0) * k2[i] +
                                       (64448.0 / 6561.0) * k3[i] + (-212.0 / 729.0) * k4[i]);
            }
            st = lmmc_ode_rhs_eval(rhs, t + h * (8.0 / 9.0), y_tmp, k5, dim, user_data, &out_result->num_rhs_evals, &callback_failed);
            if (st != LMMC_STATUS_OK) {
                out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
                goto fail;
            }

            for (i = 0; i < dim; ++i) {
                y_tmp[i] = y[i] + h * ((9017.0 / 3168.0) * k1[i] + (-355.0 / 33.0) * k2[i] +
                                       (46732.0 / 5247.0) * k3[i] + (49.0 / 176.0) * k4[i] +
                                       (-5103.0 / 18656.0) * k5[i]);
            }
            st = lmmc_ode_rhs_eval(rhs, t + h, y_tmp, k6, dim, user_data, &out_result->num_rhs_evals, &callback_failed);
            if (st != LMMC_STATUS_OK) {
                out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
                goto fail;
            }

            for (i = 0; i < dim; ++i) {
                y_tmp[i] = y[i] + h * ((35.0 / 384.0) * k1[i] + (500.0 / 1113.0) * k3[i] +
                                       (125.0 / 192.0) * k4[i] + (-2187.0 / 6784.0) * k5[i] +
                                       (11.0 / 84.0) * k6[i]);
            }
            st = lmmc_ode_rhs_eval(rhs, t + h, y_tmp, k7, dim, user_data, &out_result->num_rhs_evals, &callback_failed);
            if (st != LMMC_STATUS_OK) {
                out_result->failure_reason = callback_failed ? LMMC_ODE_FAILURE_RHS_EVAL_FAILED : LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
                goto fail;
            }

            for (i = 0; i < dim; ++i) {
                double y4 = 0.0;
                double diff = 0.0;
                double tol = 0.0;
                double ratio = 0.0;

                y5[i] = y[i] + h * ((35.0 / 384.0) * k1[i] + (500.0 / 1113.0) * k3[i] +
                                    (125.0 / 192.0) * k4[i] + (-2187.0 / 6784.0) * k5[i] +
                                    (11.0 / 84.0) * k6[i]);

                y4 = y[i] + h * ((5179.0 / 57600.0) * k1[i] + (7571.0 / 16695.0) * k3[i] +
                                 (393.0 / 640.0) * k4[i] + (-92097.0 / 339200.0) * k5[i] +
                                 (187.0 / 2100.0) * k6[i] + (1.0 / 40.0) * k7[i]);

                diff = lmmc_absd(y5[i] - y4);
                tol = local_cfg.abs_tol + local_cfg.rel_tol * lmmc_maxd(lmmc_absd(y[i]), lmmc_absd(y5[i]));
                if (tol <= 0.0) {
                    tol = 1e-30;
                }

                ratio = diff / tol;
                if (!lmmc_is_finite_number(ratio)) {
                    out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
                    goto fail;
                }
                if (ratio > err_norm) {
                    err_norm = ratio;
                }
            }

            if (err_norm <= 1.0) {
                for (i = 0; i < dim; ++i) {
                    y[i] = y5[i];
                }

                if (!lmmc_ode_state_is_finite(y, dim)) {
                    out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
                    goto fail;
                }

                t += h;
                if (!lmmc_is_finite_number(t)) {
                    out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
                    goto fail;
                }

                out_result->num_steps += 1;
                out_result->final_t = t;
                accepted = 1;
            }

            if (err_norm <= 1e-16) {
                factor = 2.0;
            } else {
                factor = local_cfg.adaptive_step_beta * pow(1.0 / err_norm, 0.2);
            }

            if (!lmmc_is_finite_number(factor)) {
                out_result->failure_reason = LMMC_ODE_FAILURE_NUMERICAL_ISSUE;
                goto fail;
            }

            factor = lmmc_clampd(factor, 0.2, 5.0);
            h_new = lmmc_clampd(h * factor, local_cfg.min_step, local_cfg.max_step);
            if (!lmmc_is_finite_number(h_new) || h_new <= 0.0) {
                out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_STEP;
                goto fail;
            }

            h = h_new;

            if (!accepted && h <= local_cfg.min_step) {
                out_result->failure_reason = LMMC_ODE_FAILURE_INVALID_STEP;
                goto fail;
            }

            if (accepted) {
                break;
            }
        }
    }

    if (t >= t_end) {
        out_result->converged = 1;
        out_result->failure_reason = LMMC_ODE_FAILURE_NONE;
    } else {
        out_result->converged = 0;
        out_result->failure_reason = LMMC_ODE_FAILURE_MAX_STEPS;
    }

    lmmc_free(y5);
    lmmc_free(y_tmp);
    lmmc_free(k7);
    lmmc_free(k6);
    lmmc_free(k5);
    lmmc_free(k4);
    lmmc_free(k3);
    lmmc_free(k2);
    lmmc_free(k1);
    return LMMC_STATUS_OK;

fail:
    lmmc_free(y5);
    lmmc_free(y_tmp);
    lmmc_free(k7);
    lmmc_free(k6);
    lmmc_free(k5);
    lmmc_free(k4);
    lmmc_free(k3);
    lmmc_free(k2);
    lmmc_free(k1);
    return LMMC_STATUS_NUMERICAL_FAILURE;
}
