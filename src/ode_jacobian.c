#include "ode_internal.h"

static int lmmc_ode_jacobian_buffers_valid(const lmmc_ode_jacobian_request_t* s) {
    return s->jacobian_data != NULL && s->base_rhs_work != NULL &&
           s->y_perturbed != NULL && s->rhs_perturbed != NULL;
}

static int lmmc_ode_jacobian_request_valid(
    const lmmc_ode_jacobian_request_t* s, size_t* count
) {
    if (s->rhs == NULL || s->y == NULL || !lmmc_ode_jacobian_buffers_valid(s)) { return 0; }
    if (s->io_rhs_evals == NULL || s->out_callback_failed == NULL || s->dim == 0) { return 0; }
    return lmmc_safe_mul_size(s->dim, s->dim, count);
}

lmmc_status_t lmmc_ode_jacobian_eval(const lmmc_ode_jacobian_request_t* s) {
    const lmmc_real_t* base = s->base_rhs;
    size_t i, j;
    size_t jacobian_count;
    lmmc_status_t st;

    if (!lmmc_ode_jacobian_request_valid(s, &jacobian_count)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *s->out_callback_failed = 0;
    if (s->jacobian != NULL) {
        st = s->jacobian(s->t, s->y, s->jacobian_data, s->dim, s->user_data);
        if (st != LMMC_STATUS_OK) { return st; }
        return lmmc_ode_values_are_finite(s->jacobian_data, jacobian_count);
    }

    if (base == NULL) {
        st = lmmc_ode_rhs_eval(s->rhs, s->t, s->y, s->base_rhs_work, s->dim, s->user_data,
                               s->io_rhs_evals, s->out_callback_failed);
        if (st != LMMC_STATUS_OK) { return st; }
        base = s->base_rhs_work;
    }

    for (j = 0; j < s->dim; ++j) {
        const lmmc_real_t delta =
            sqrt(LMMC_REAL_EPSILON) * fmax(1.0, fabs(s->y[j]));
        memcpy(s->y_perturbed, s->y, s->dim * sizeof(*s->y));
        s->y_perturbed[j] += delta;
        st = lmmc_ode_rhs_eval(s->rhs, s->t, s->y_perturbed, s->rhs_perturbed, s->dim,
                               s->user_data, s->io_rhs_evals, s->out_callback_failed);
        if (st != LMMC_STATUS_OK) { return st; }
        for (i = 0; i < s->dim; ++i) {
            s->jacobian_data[i * s->dim + j] =
                (s->rhs_perturbed[i] - base[i]) / delta;
        }
    }
    return lmmc_ode_values_are_finite(s->jacobian_data, jacobian_count);
}
