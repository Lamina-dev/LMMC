#include "internal/quadrature_internal.h"

static int lmmc_quad_fixed_interior(
    lmmc_quad_func_t func, void* user_data,
    lmmc_real_t center, lmmc_real_t half_length, lmmc_real_t normalized_h,
    size_t n, int simpson_rule, lmmc_real_t* sum)
{
    for (size_t i = 1; i < n; ++i) {
        const lmmc_real_t normalized_x = -1.0 + (lmmc_real_t)i * normalized_h;
        const lmmc_real_t x = fma(half_length, normalized_x, center);
        const lmmc_real_t value = func(x, user_data);
        const lmmc_real_t weight = simpson_rule && i % 2 == 1 ? 4.0 : 2.0;
        if (!lmmc_is_finite(&x) || !lmmc_is_finite(&value)) {
            return 0;
        }
        *sum += weight * value;
        if (!lmmc_is_finite(sum)) {
            return 0;
        }
    }
    return 1;
}

lmmc_status_t lmmc_quad_trapezoid(
    lmmc_quad_func_t func,
    void* user_data,
    lmmc_real_t a,
    lmmc_real_t b,
    size_t n,
    lmmc_real_t* out_result)
{
    lmmc_real_t center, half_length, normalized_h, left, right, sum;
    if (func == NULL || out_result == NULL || n == 0 || a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_is_finite(&a) || !lmmc_is_finite(&b)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_quad_center_half_length(a, b, &center, &half_length);
    normalized_h = 2.0 / (lmmc_real_t)n;
    left = func(a, user_data);
    right = func(b, user_data);
    if (!lmmc_is_finite(&center) || !lmmc_is_finite(&half_length) ||
        !lmmc_is_finite(&left) || !lmmc_is_finite(&right)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    sum = left + right;
    if (!lmmc_quad_fixed_interior(
            func, user_data, center, half_length, normalized_h, n, 0, &sum)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out_result = half_length * (normalized_h * 0.5 * sum);
    return lmmc_is_finite(out_result) ? LMMC_STATUS_OK
                                      : LMMC_STATUS_NUMERICAL_FAILURE;
}


lmmc_status_t lmmc_quad_simpson(
    lmmc_quad_func_t func,
    void* user_data,
    lmmc_real_t a,
    lmmc_real_t b,
    size_t n,
    lmmc_real_t* out_result)
{
    lmmc_real_t center, half_length, normalized_h, left, right, sum;
    if (func == NULL || out_result == NULL || n == 0 || n % 2 != 0 ||
        a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_quad_center_half_length(a, b, &center, &half_length);
    normalized_h = 2.0 / (lmmc_real_t)n;
    left = func(a, user_data);
    right = func(b, user_data);
    if (!lmmc_is_finite(&center) || !lmmc_is_finite(&half_length) ||
        !lmmc_is_finite(&left) || !lmmc_is_finite(&right)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    sum = left + right;
    if (!lmmc_quad_fixed_interior(
            func, user_data, center, half_length, normalized_h, n, 1, &sum)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    *out_result = half_length * (normalized_h / 3.0 * sum);
    return lmmc_is_finite(out_result) ? LMMC_STATUS_OK
                                      : LMMC_STATUS_NUMERICAL_FAILURE;
}
