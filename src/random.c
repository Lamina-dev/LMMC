/**
 * @file random.c
 * @brief 伪随机数发生器与常用分布实现。
 */
#include <string.h>
#include <math.h>
#include "memory_bridge.h"
#include "lmmc/config.h"
#include "lmmc/random.h"
#include "lmmc/status.h"


struct lmmc_rng_t {
    uint64_t state[4];
};


static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}


static inline uint64_t splitmix64_next(uint64_t* state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}


#define LMMC_RNG_DEFAULT_SEED UINT64_C(0x12345678DEADBEEF)


static inline uint64_t xoshiro256ss_next(uint64_t* s) {
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = rotl(s[3], 45);

    return result;
}


lmmc_status_t lmmc_rng_create(lmmc_rng_t** out_rng) {
    lmmc_rng_t* rng;

    if (out_rng == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    rng = (lmmc_rng_t*)lmmc_alloc(sizeof(lmmc_rng_t));
    if (rng == NULL) {
        *out_rng = NULL;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }


    lmmc_rng_seed(rng, LMMC_RNG_DEFAULT_SEED);

    *out_rng = rng;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_rng_seed(lmmc_rng_t* rng, uint64_t seed) {
    uint64_t sm_state;

    if (rng == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }


    sm_state = seed;
    rng->state[0] = splitmix64_next(&sm_state);
    rng->state[1] = splitmix64_next(&sm_state);
    rng->state[2] = splitmix64_next(&sm_state);
    rng->state[3] = splitmix64_next(&sm_state);

    return LMMC_STATUS_OK;
}

void lmmc_rng_destroy(lmmc_rng_t* rng) {
    if (rng != NULL) {

        memset(rng->state, 0, sizeof(rng->state));
        lmmc_free(rng);
    }
}


uint64_t lmmc_rng_next_u64(lmmc_rng_t* rng) {
    if (rng == NULL) {
        return 0;
    }
    return xoshiro256ss_next(rng->state);
}


static inline double u64_to_double01(uint64_t x) {
    return (double)(x >> 11) * (1.0 / 9007199254740992.0);
}


lmmc_status_t lmmc_rng_uniform(
    lmmc_rng_t* rng,
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t* out_value)
{
    double u;

    if (rng == NULL || out_value == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    u = u64_to_double01(xoshiro256ss_next(rng->state));
    *out_value = a + (b - a) * u;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_rng_normal(
    lmmc_rng_t* rng,
    lmmc_real_t mean,
    lmmc_real_t stddev,
    lmmc_real_t* out_value)
{
    double u1, u2, r, theta, z;

    if (rng == NULL || out_value == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (stddev <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }


    do {
        u1 = u64_to_double01(xoshiro256ss_next(rng->state));
    } while (u1 == 0.0);

    u2 = u64_to_double01(xoshiro256ss_next(rng->state));

    r = sqrt(-2.0 * log(u1));
    theta = 2.0 * LMMC_CONST_PI * u2;
    z = r * cos(theta);

    *out_value = mean + stddev * z;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_rng_exponential(
    lmmc_rng_t* rng,
    lmmc_real_t rate,
    lmmc_real_t* out_value)
{
    double u;

    if (rng == NULL || out_value == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (rate <= 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }


    do {
        u = u64_to_double01(xoshiro256ss_next(rng->state));
    } while (u == 1.0);

    *out_value = -log(1.0 - u) / rate;
    return LMMC_STATUS_OK;
}


lmmc_status_t lmmc_rng_fill_uniform(
    lmmc_rng_t* rng,
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t* array,
    size_t count)
{
    size_t i;
    double u;

    if (rng == NULL || array == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < count; i++) {
        u = u64_to_double01(xoshiro256ss_next(rng->state));
        array[i] = a + (b - a) * u;
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_rng_shuffle(
    lmmc_rng_t* rng,
    void* array,
    size_t count,
    size_t elem_size)
{
    size_t i, j;
    unsigned char* arr;
    unsigned char* tmp;

    if (rng == NULL || array == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (elem_size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }


    if (count <= 1) {
        return LMMC_STATUS_OK;
    }

    arr = (unsigned char*)array;
    tmp = (unsigned char*)lmmc_alloc(elem_size);
    if (tmp == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }


    for (i = count - 1; i > 0; i--) {

        uint64_t r = xoshiro256ss_next(rng->state);
        j = (size_t)(r % (i + 1));


        if (i != j) {
            memcpy(tmp, arr + i * elem_size, elem_size);
            memcpy(arr + i * elem_size, arr + j * elem_size, elem_size);
            memcpy(arr + j * elem_size, tmp, elem_size);
        }
    }

    lmmc_free(tmp);
    return LMMC_STATUS_OK;
}
