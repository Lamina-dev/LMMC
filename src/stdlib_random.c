#include "lmmc/stdlib.h"

#include "lmmc/random.h"

#include "stdlib_internal.h"

lmmc_status_t lmmc_std_random_seed(lmmc_rng_t* rng, uint64_t seed)
{
    if (!rng) { return LMMC_STATUS_INVALID_ARGUMENT; }
    return lmmc_rng_seed(rng, seed);
}

lmmc_status_t lmmc_std_random_rand(lmmc_rng_t* rng, lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!rng || !out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    status = lmmc_rng_uniform(rng, (lmmc_real_t)0, (lmmc_real_t)1, &value);
    if (status != LMMC_STATUS_OK) { return status; }
    return lmmc_std_store_finite_real(value, out);
}

lmmc_status_t lmmc_std_random_randint(lmmc_rng_t* rng, int64_t lo,
                                      int64_t hi, int64_t* out)
{
    if (!rng || !out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    return lmmc_rng_int_uniform(rng, lo, hi, out);
}

lmmc_status_t lmmc_std_random_normal(lmmc_rng_t* rng, lmmc_real_t mean,
                                     lmmc_real_t stddev, lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!rng || !out) { return LMMC_STATUS_INVALID_ARGUMENT; }
    if (!lmmc_std_real_is_finite(mean) ||
        !lmmc_std_real_is_finite(stddev)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_rng_normal(rng, mean, stddev, &value);
    if (status != LMMC_STATUS_OK) { return status; }
    return lmmc_std_store_finite_real(value, out);
}

lmmc_status_t lmmc_std_random_choice(lmmc_rng_t* rng,
                                     const lmmc_real_t* values,
                                     size_t count,
                                     lmmc_real_t* out)
{
    int64_t index = 0;
    lmmc_status_t status;
    if (!rng || !values || !out) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (count == 0) { return LMMC_STATUS_EMPTY_INPUT; }
    if (count > (size_t)INT64_MAX) { return LMMC_STATUS_OUT_OF_RANGE; }
    if (!lmmc_std_real_array_is_finite(values, count)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    status = lmmc_rng_int_uniform(rng, 0, (int64_t)count - 1, &index);
    if (status != LMMC_STATUS_OK) { return status; }
    return lmmc_std_store_finite_real(values[index], out);
}


void lmmc_std_random_default_deinit(void)
{
    lmmc_rng_default_reset();
}

lmmc_status_t lmmc_std_random_default_seed(uint64_t seed)
{
    return lmmc_std_random_seed(lmmc_rng_default_get(), seed);
}

lmmc_status_t lmmc_std_random_default_rand(lmmc_real_t* out)
{
    return lmmc_std_random_rand(lmmc_rng_default_get(), out);
}

lmmc_status_t lmmc_std_random_default_randint(int64_t lo,
                                              int64_t hi,
                                              int64_t* out)
{
    return lmmc_std_random_randint(
        lmmc_rng_default_get(), lo, hi, out);
}

lmmc_status_t lmmc_std_random_default_normal(lmmc_real_t mean,
                                             lmmc_real_t stddev,
                                             lmmc_real_t* out)
{
    return lmmc_std_random_normal(
        lmmc_rng_default_get(), mean, stddev, out);
}

lmmc_status_t lmmc_std_random_default_choice(const lmmc_real_t* values,
                                             size_t count,
                                             lmmc_real_t* out)
{
    return lmmc_std_random_choice(
        lmmc_rng_default_get(), values, count, out);
}
