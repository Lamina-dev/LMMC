#include "lmmc/lsr_stdlib.h"

#include <math.h>

#include "lmmc/dense.h"
#include "lmmc/numeric.h"
#include "lmmc/random.h"
#include "lmmc/stats.h"

static lmmc_status_t lmmc_lsr_store_real(lmmc_real_t value, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = value;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_pi(lmmc_real_t* out)
{
    return lmmc_lsr_store_real(LMMC_CONST_PI, out);
}

lmmc_status_t lmmc_lsr_math_e(lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)2.71828182845904523536, out);
}

lmmc_status_t lmmc_lsr_math_phi(lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)1.61803398874989484820, out);
}

lmmc_status_t lmmc_lsr_math_i(lmmc_complex_t* out)
{
    return lmmc_complex_create((lmmc_real_t)0, (lmmc_real_t)1, out);
}

lmmc_status_t lmmc_lsr_math_I(lmmc_complex_t* out)
{
    return lmmc_lsr_math_i(out);
}

lmmc_status_t lmmc_lsr_math_complex(lmmc_real_t real,
                                    lmmc_real_t imag,
                                    lmmc_complex_t* out)
{
    return lmmc_complex_create(real, imag, out);
}

lmmc_status_t lmmc_lsr_math_real(const lmmc_complex_t* z, lmmc_real_t* out)
{
    if (!z || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = z->real;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_imag(const lmmc_complex_t* z, lmmc_real_t* out)
{
    if (!z || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    *out = z->imag;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_conj(const lmmc_complex_t* z, lmmc_complex_t* out)
{
    return lmmc_complex_conj(z, out);
}

lmmc_status_t lmmc_lsr_math_complex_abs(const lmmc_complex_t* z,
                                        lmmc_real_t* out)
{
    return lmmc_complex_modulus(z, out);
}

lmmc_status_t lmmc_lsr_math_sin(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)sin((double)x), out);
}

lmmc_status_t lmmc_lsr_math_cos(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)cos((double)x), out);
}

lmmc_status_t lmmc_lsr_math_tan(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)tan((double)x), out);
}

lmmc_status_t lmmc_lsr_math_asin(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_asin(x, out);
}

lmmc_status_t lmmc_lsr_math_acos(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_acos(x, out);
}

lmmc_status_t lmmc_lsr_math_atan(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_atan(x, out);
}

lmmc_status_t lmmc_lsr_math_sqrt(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < (lmmc_real_t)0) return LMMC_STATUS_OUT_OF_RANGE;
    *out = (lmmc_real_t)sqrt((double)x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_exp(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)exp((double)x), out);
}

lmmc_status_t lmmc_lsr_math_ln(lmmc_real_t x, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= (lmmc_real_t)0) return LMMC_STATUS_OUT_OF_RANGE;
    *out = (lmmc_real_t)log((double)x);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_log(lmmc_real_t x, lmmc_real_t base,
                                lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x <= (lmmc_real_t)0 || base <= (lmmc_real_t)0 ||
        base == (lmmc_real_t)1) {
        return LMMC_STATUS_OUT_OF_RANGE;
    }
    *out = (lmmc_real_t)(log((double)x) / log((double)base));
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_math_abs(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_lsr_store_real((lmmc_real_t)fabs((double)x), out);
}

lmmc_status_t lmmc_lsr_math_floor(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_floor(x, out);
}

lmmc_status_t lmmc_lsr_math_ceil(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_ceil(x, out);
}

lmmc_status_t lmmc_lsr_math_round(lmmc_real_t x, lmmc_real_t* out)
{
    return lmmc_round(x, out);
}

lmmc_status_t lmmc_lsr_math_clamp(lmmc_real_t x, lmmc_real_t lo,
                                  lmmc_real_t hi, lmmc_real_t* out)
{
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    if (lo > hi) return LMMC_STATUS_INVALID_ARGUMENT;
    if (x < lo) *out = lo;
    else if (x > hi) *out = hi;
    else *out = x;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_lsr_wrap_const_vec(const lmmc_real_t* values,
                                             size_t count,
                                             lmmc_vec_t* out)
{
    if (!values || !out || count == 0) return LMMC_STATUS_INVALID_ARGUMENT;
    out->size = count;
    out->data = (lmmc_real_t*)values;
    out->owns_data = 0;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_stats_mean(const lmmc_real_t* values, size_t count,
                                  lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_status_t status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_vec_mean(&view, out);
}

lmmc_status_t lmmc_lsr_stats_median(const lmmc_real_t* values, size_t count,
                                    lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_status_t status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_vec_median(&view, out);
}

lmmc_status_t lmmc_lsr_stats_var(const lmmc_real_t* values, size_t count,
                                 lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_status_t status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_vec_variance_sample(&view, out);
}

lmmc_status_t lmmc_lsr_stats_std(const lmmc_real_t* values, size_t count,
                                 lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_status_t status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_vec_stddev_sample(&view, out);
}

lmmc_status_t lmmc_lsr_stats_quantile(const lmmc_real_t* values, size_t count,
                                      lmmc_real_t q, lmmc_real_t* out)
{
    lmmc_vec_t view;
    lmmc_status_t status = lmmc_lsr_wrap_const_vec(values, count, &view);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_vec_quantile(&view, q, out);
}

lmmc_status_t lmmc_lsr_random_seed(lmmc_rng_t* rng, uint64_t seed)
{
    return lmmc_rng_seed(rng, seed);
}

lmmc_status_t lmmc_lsr_random_rand(lmmc_rng_t* rng, lmmc_real_t* out)
{
    return lmmc_rng_uniform(rng, (lmmc_real_t)0, (lmmc_real_t)1, out);
}

lmmc_status_t lmmc_lsr_random_randint(lmmc_rng_t* rng, int64_t lo,
                                      int64_t hi, int64_t* out)
{
    return lmmc_rng_int_uniform(rng, lo, hi, out);
}

lmmc_status_t lmmc_lsr_random_normal(lmmc_rng_t* rng, lmmc_real_t mean,
                                     lmmc_real_t stddev, lmmc_real_t* out)
{
    return lmmc_rng_normal(rng, mean, stddev, out);
}

lmmc_status_t lmmc_lsr_random_choice(lmmc_rng_t* rng,
                                     const lmmc_real_t* values,
                                     size_t count,
                                     lmmc_real_t* out)
{
    int64_t index = 0;
    lmmc_status_t status;
    if (!values || !out || count == 0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (count > (size_t)INT64_MAX + 1u) return LMMC_STATUS_OUT_OF_RANGE;
    status = lmmc_rng_int_uniform(rng, 0, (int64_t)count - 1, &index);
    if (status != LMMC_STATUS_OK) return status;
    *out = values[index];
    return LMMC_STATUS_OK;
}
