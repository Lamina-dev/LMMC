/**
 * @file random.c
 * @brief 伪随机数发生器与常用分布实现.
 */
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <stdatomic.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/config.h"
#include "lmmc/random.h"
#include "lmmc/status.h"


#include "random_internal.h"

static _Thread_local struct lmmc_rng_t lmmc_default_rng;
static _Thread_local int lmmc_default_rng_initialized = 0;




static inline uint64_t splitmix64_next(uint64_t* state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}


/**
 * @brief Generate a robust default seed by mixing time, stack address (ASLR),
 *        and a monotonically increasing atomic counter through SplitMix64.
 */
static uint64_t generate_default_seed(void) {
    static _Atomic uint64_t seed_counter = 0;

    uint64_t t = (uint64_t)time(NULL);
    /* Use address of a local variable for ASLR entropy */
    volatile int stack_var = 0;
    uint64_t addr = (uint64_t)(uintptr_t)&stack_var;
    uint64_t counter = atomic_fetch_add(&seed_counter, 1);

    /* Mix all sources via SplitMix64 finalizer */
    uint64_t mixed = t ^ addr ^ counter;
    return splitmix64_next(&mixed);
}
lmmc_rng_t* lmmc_rng_default_get(void) {
    if (!lmmc_default_rng_initialized) {
        lmmc_rng_seed(&lmmc_default_rng, generate_default_seed());
        lmmc_default_rng_initialized = 1;
    }
    return &lmmc_default_rng;
}

void lmmc_rng_default_reset(void) {
    memset(&lmmc_default_rng, 0, sizeof(lmmc_default_rng));
    lmmc_default_rng_initialized = 0;
}



static uint64_t rng_bounded_u64(uint64_t* state, uint64_t bound) {
    const uint64_t threshold = (uint64_t)(-bound) % bound;
    uint64_t value;
    do {
        value = xoshiro256ss_next(state);
    } while (value < threshold);
    return value % bound;
}



lmmc_status_t lmmc_rng_create(lmmc_rng_t** out_rng) {
    lmmc_rng_t* rng;

    if (out_rng == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    rng = (lmmc_rng_t*)lmmc_memory_alloc(sizeof(lmmc_rng_t));
    if (rng == NULL) {
        *out_rng = NULL;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /** 将平台熵混入默认种子,为各随机数发生器建立独立数据流. */
    lmmc_rng_seed(rng, generate_default_seed());

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
        lmmc_memory_free(rng);
    }
}


lmmc_status_t lmmc_rng_clone(const lmmc_rng_t* src, lmmc_rng_t** out_rng) {
    lmmc_rng_t* copy;

    if (src == NULL || out_rng == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    copy = (lmmc_rng_t*)lmmc_memory_alloc(sizeof(lmmc_rng_t));
    if (copy == NULL) {
        *out_rng = NULL;
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    memcpy(copy->state, src->state, sizeof(src->state));
    *out_rng = copy;
    return LMMC_STATUS_OK;
}


/*
 * xoshiro256** jump polynomial constants.
 * Advances the state by 2^128 steps.
 * Reference: https://prng.di.unimi.it/xoshiro256starstar.c
 */
static const uint64_t JUMP_POLY[4] = {
    UINT64_C(0x180ec6d33cfd0aba),
    UINT64_C(0xd5a61266f0c9392c),
    UINT64_C(0xa9582618e03fc9aa),
    UINT64_C(0x39abdc4529b1661c)
};

/*
 * xoshiro256** long_jump polynomial constants.
 * Advances the state by 2^192 steps.
 * Reference: https://prng.di.unimi.it/xoshiro256starstar.c
 */
static const uint64_t LONG_JUMP_POLY[4] = {
    UINT64_C(0x76e15d3efefdcbbf),
    UINT64_C(0xc5004e441c522fb3),
    UINT64_C(0x77710069854ee241),
    UINT64_C(0x39109bb02acbe635)
};


/**
 * @brief Internal helper: apply a jump polynomial to the RNG state.
 */
static void rng_apply_jump_poly(uint64_t* s, const uint64_t* poly) {
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i, b;

    for (i = 0; i < 4; i++) {
        for (b = 0; b < 64; b++) {
            if (poly[i] & (UINT64_C(1) << b)) {
                s0 ^= s[0];
                s1 ^= s[1];
                s2 ^= s[2];
                s3 ^= s[3];
            }
            xoshiro256ss_next(s);
        }
    }

    s[0] = s0;
    s[1] = s1;
    s[2] = s2;
    s[3] = s3;
}


lmmc_status_t lmmc_rng_jump(lmmc_rng_t* rng) {
    if (rng == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    rng_apply_jump_poly(rng->state, JUMP_POLY);
    return LMMC_STATUS_OK;
}


lmmc_status_t lmmc_rng_long_jump(lmmc_rng_t* rng) {
    if (rng == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    rng_apply_jump_poly(rng->state, LONG_JUMP_POLY);
    return LMMC_STATUS_OK;
}


uint64_t lmmc_rng_next_u64(lmmc_rng_t* rng) {
    if (rng == NULL) {
        return 0;
    }
    return xoshiro256ss_next(rng->state);
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
    if (!isfinite(a) || !isfinite(b) || a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    u = u64_to_double01(xoshiro256ss_next(rng->state));
    *out_value = fma(b, u, a * (1.0 - u));
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
    if (!isfinite(rate) || rate <= 0.0) {
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
    if (!isfinite(a) || !isfinite(b) || a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < count; i++) {
        u = u64_to_double01(xoshiro256ss_next(rng->state));
        array[i] = fma(b, u, a * (1.0 - u));
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
    tmp = (unsigned char*)lmmc_memory_alloc(elem_size);
    if (tmp == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    for (i = count - 1; i > 0; i--) {
        j = (size_t)rng_bounded_u64(rng->state, (uint64_t)i + UINT64_C(1));

        if (i != j) {
            memcpy(tmp, arr + i * elem_size, elem_size);
            memcpy(arr + i * elem_size, arr + j * elem_size, elem_size);
            memcpy(arr + j * elem_size, tmp, elem_size);
        }
    }

    lmmc_memory_free(tmp);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_rng_int_uniform(
    lmmc_rng_t* rng,
    int64_t lo,
    int64_t hi,
    int64_t* out)
{
    uint64_t span, offset, bits;

    if (rng == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lo > hi) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lo == hi) {
        *out = lo;
        return LMMC_STATUS_OK;
    }

    span = (uint64_t)hi - (uint64_t)lo + UINT64_C(1);
    offset = span == 0 ? xoshiro256ss_next(rng->state)
                       : rng_bounded_u64(rng->state, span);
    bits = (uint64_t)lo + offset;
    if (bits <= (uint64_t)INT64_MAX) {
        *out = (int64_t)bits;
    } else {
        *out = INT64_MIN + (int64_t)(bits - (UINT64_C(1) << 63));
    }
    return LMMC_STATUS_OK;
}
