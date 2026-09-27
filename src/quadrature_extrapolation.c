#include "internal/quadrature_internal.h"

typedef struct {
    lmmc_quad_func_t func;
    void* user_data;
    lmmc_real_t center;
    lmmc_real_t half_length;
    lmmc_real_t prev[30];
    lmmc_real_t curr[30];
    lmmc_quad_result_t* out;
} lmmc_quad_romberg_state_t;

static int lmmc_quad_romberg_arguments(
    lmmc_real_t a, lmmc_real_t b, lmmc_real_t abs_tol, size_t max_iter)
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
    return max_iter >= 1 && max_iter <= 30;
}

static int lmmc_quad_romberg_initial(
    lmmc_quad_romberg_state_t* state, lmmc_real_t a, lmmc_real_t b)
{
    lmmc_real_t fa, fb;
    lmmc_quad_center_half_length(a, b, &state->center, &state->half_length);
    if (!isfinite(state->center) || !isfinite(state->half_length)) {
        return 0;
    }
    fa = state->func(a, state->user_data);
    fb = state->func(b, state->user_data);
    state->out->num_evals = 2;
    if (!isfinite(fa) || !isfinite(fb)) {
        return 0;
    }
    state->prev[0] = state->half_length * (fa + fb);
    if (!isfinite(state->prev[0])) {
        return 0;
    }
    state->out->value = state->prev[0];
    return 1;
}

static int lmmc_quad_romberg_sample_row(
    lmmc_quad_romberg_state_t* state, size_t row)
{
    const size_t n = (size_t)1 << row;
    const lmmc_real_t normalized_h = 2.0 / (lmmc_real_t)n;
    lmmc_real_t sum_new = 0.0;
    for (size_t k = 1; k <= n / 2; ++k) {
        const lmmc_real_t normalized_x =
            -1.0 + (2.0 * (lmmc_real_t)k - 1.0) * normalized_h;
        const lmmc_real_t x = fma(state->half_length, normalized_x, state->center);
        const lmmc_real_t value = state->func(x, state->user_data);
        ++state->out->num_evals;
        if (!isfinite(value)) {
            return 0;
        }
        sum_new += value;
        if (!isfinite(sum_new)) {
            return 0;
        }
    }
    state->curr[0] = state->prev[0] / 2.0 +
        state->half_length * normalized_h * sum_new;
    return isfinite(state->curr[0]);
}

static int lmmc_quad_romberg_extrapolate(
    lmmc_quad_romberg_state_t* state, size_t row)
{
    lmmc_real_t pow4 = 1.0;
    for (size_t j = 1; j <= row; ++j) {
        pow4 *= 4.0;
        state->curr[j] = state->curr[j - 1] +
            (state->curr[j - 1] - state->prev[j - 1]) / (pow4 - 1.0);
        if (!isfinite(state->curr[j])) {
            return 0;
        }
    }
    state->out->value = state->curr[row];
    state->out->error = lmmc_abs(state->curr[row] - state->prev[row - 1]);
    return isfinite(state->out->error);
}

lmmc_status_t lmmc_quad_romberg(
    lmmc_quad_func_t f, void* ud, lmmc_real_t a, lmmc_real_t b,
    lmmc_real_t abs_tol, size_t max_iter, lmmc_quad_result_t* out)
{
    lmmc_quad_romberg_state_t state = {0};
    if (f == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    out->value = 0.0;
    out->error = INFINITY;
    out->num_evals = 0;
    if (!lmmc_quad_romberg_arguments(a, b, abs_tol, max_iter)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    state.func = f;
    state.user_data = ud;
    state.out = out;
    if (!lmmc_quad_romberg_initial(&state, a, b)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    for (size_t i = 1; i < max_iter; ++i) {
        if (!lmmc_quad_romberg_sample_row(&state, i) ||
            !lmmc_quad_romberg_extrapolate(&state, i)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        if (out->error <= abs_tol) {
            return LMMC_STATUS_OK;
        }
        for (size_t j = 0; j <= i; ++j) state.prev[j] = state.curr[j];
    }
    return LMMC_STATUS_CONVERGENCE_FAILED;
}
