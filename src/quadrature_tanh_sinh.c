#include "internal/quadrature_internal.h"

/** @brief 双指数映射：x(t) = center + half_length*tanh(pi/2*sinh(t))。 */
typedef struct {
    lmmc_quad_func_t func;
    void* user_data;
    lmmc_real_t a;
    lmmc_real_t b;
    lmmc_real_t center;
    lmmc_real_t half_length;
    size_t max_nodes;
    size_t num_evals;
} lmmc_quad_tanh_sinh_state_t;

static int lmmc_quad_tanh_sinh_arguments(
    lmmc_real_t a, lmmc_real_t b, lmmc_real_t abs_tol, size_t max_nodes)
{
    if (!isfinite(a) || !isfinite(b) || a >= b) {
        return 0;
    }
    if (!isfinite(abs_tol)) {
        return 0;
    }
    if (abs_tol < 1e-15 || abs_tol > 1e-1) {
        return 0;
    }
    return max_nodes >= 1 && max_nodes <= 1000000;
}

static lmmc_real_t lmmc_quad_tanh_sinh_weight(lmmc_real_t t, lmmc_real_t u)
{
    const lmmc_real_t pi_half = LMMC_CONST_PI / 2.0;
    if (u > 6.0) {
        return pi_half * cosh(t) * 4.0 * exp(-2.0 * u);
    }
    const lmmc_real_t cosh_u = cosh(u);
    return pi_half * cosh(t) / (cosh_u * cosh_u);
}

static lmmc_status_t lmmc_quad_tanh_sinh_pair(
    lmmc_quad_tanh_sinh_state_t* state, lmmc_real_t u,
    lmmc_real_t weight, lmmc_real_t* sum)
{
    const lmmc_real_t tanh_u = u > 6.0 ? 1.0 - 2.0 * exp(-2.0 * u) : tanh(u);
    const lmmc_real_t xp = fma(state->half_length, tanh_u, state->center);
    const lmmc_real_t xn = fma(-state->half_length, tanh_u, state->center);
    const size_t needed = (size_t)(xp != state->b) + (size_t)(xn != state->a);
    if (state->max_nodes - state->num_evals < needed) {
        return LMMC_STATUS_CONVERGENCE_FAILED;
    }
    lmmc_real_t fp = 0.0, fn = 0.0;
    /** @brief 舍入到端点的节点取零极限贡献，跳过采样。 */
    if (xp != state->b) {
        fp = state->func(xp, state->user_data);
        ++state->num_evals;
    }
    if (xn != state->a) {
        fn = state->func(xn, state->user_data);
        ++state->num_evals;
    }
    if (!isfinite(fp) || !isfinite(fn)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    const lmmc_real_t next = *sum + (fp + fn) * weight;
    if (!isfinite(next)) return LMMC_STATUS_NUMERICAL_FAILURE;
    *sum = next;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_quad_tanh_sinh_level(
    lmmc_quad_tanh_sinh_state_t* state, lmmc_real_t h, lmmc_real_t* integral)
{
    const lmmc_real_t pi_half = LMMC_CONST_PI / 2.0;
    if (state->num_evals == state->max_nodes) {
        return LMMC_STATUS_CONVERGENCE_FAILED;
    }
    const lmmc_real_t fmid = state->func(state->center, state->user_data);
    lmmc_real_t sum = fmid * pi_half;
    ++state->num_evals;
    if (!isfinite(fmid) || !isfinite(sum)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    for (size_t j = 1; ; ++j) {
        const lmmc_real_t t = (lmmc_real_t)j * h;
        const lmmc_real_t u = pi_half * sinh(t);
        lmmc_real_t weight;
        if (u > 20.0) break;
        weight = lmmc_quad_tanh_sinh_weight(t, u);
        if (!isfinite(weight)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        if (weight < 1e-50) break;
        const lmmc_status_t status = lmmc_quad_tanh_sinh_pair(state, u, weight, &sum);
        if (status != LMMC_STATUS_OK) return status;
    }
    const lmmc_real_t completed = state->half_length * h * sum;
    if (!isfinite(completed)) return LMMC_STATUS_NUMERICAL_FAILURE;
    *integral = completed;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_quad_tanh_sinh_integrate(
    lmmc_quad_tanh_sinh_state_t* state, lmmc_real_t abs_tol,
    lmmc_quad_result_t* out)
{
    int have_complete = 0;
    lmmc_real_t last_complete = 0.0;
    lmmc_real_t last_error = INFINITY;
    lmmc_real_t h = 0.03125;
    for (size_t level = 0; level < 10; ++level) {
        lmmc_real_t integral;
        const lmmc_status_t status = lmmc_quad_tanh_sinh_level(state, h, &integral);
        out->num_evals = state->num_evals;
        if (status != LMMC_STATUS_OK) {
            out->value = last_complete;
            out->error = INFINITY;
            return status;
        }
        if (have_complete) {
            last_error = lmmc_abs(integral - last_complete);
            if (!isfinite(last_error)) {
                out->error = INFINITY;
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
        }
        last_complete = integral;
        out->value = last_complete;
        out->error = last_error;
        if (have_complete && last_error <= abs_tol) return LMMC_STATUS_OK;
        have_complete = 1;
        h /= 2.0;
    }
    return LMMC_STATUS_CONVERGENCE_FAILED;
}

lmmc_status_t lmmc_quad_tanh_sinh(
    lmmc_quad_func_t f, void* ud, lmmc_real_t a, lmmc_real_t b,
    lmmc_real_t abs_tol, size_t max_nodes, lmmc_quad_result_t* out)
{
    lmmc_quad_tanh_sinh_state_t state = {0};
    if (f == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    out->value = 0.0;
    out->error = INFINITY;
    out->num_evals = 0;
    if (!lmmc_quad_tanh_sinh_arguments(a, b, abs_tol, max_nodes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    state.func = f;
    state.user_data = ud;
    state.a = a;
    state.b = b;
    state.max_nodes = max_nodes;
    lmmc_quad_center_half_length(a, b, &state.center, &state.half_length);
    if (!isfinite(state.center) || !isfinite(state.half_length)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    return lmmc_quad_tanh_sinh_integrate(&state, abs_tol, out);
}
