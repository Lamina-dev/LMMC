#ifndef LMMC_RANDOM_INTERNAL_H
#define LMMC_RANDOM_INTERNAL_H

#include <stdint.h>
#include "lmmc/random.h"

struct lmmc_rng_t {
    uint64_t state[4];
};
static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

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

static inline double u64_to_double01(uint64_t x) {
    return (double)(x >> 11) * (1.0 / 9007199254740992.0);
}

double lmmc_rng_standard_normal(uint64_t* state);

#endif
