#include <stdint.h>
#include <stdio.h>

#include "lammp/lmmp.h"
#include "lammp/mprand.h"
#include "lammp/numth.h"
#include "lammp/secret.h"
#include "lammp/version.h"

int main(void) {
    mp_limb_t rnd = 0;
    mp_limb_t strong_values[4] = {0, 0, 0, 0};
    uint64_t sip_key[2] = {0x0123456789abcdefull, 0xfedcba9876543210ull};
    uint64_t words[4] = {11, 22, 33, 44};
    uint64_t h_xx = 0;
    uint64_t h_sip = 0;
    lmmp_strong_rng_t* strong_rng = NULL;
    mp_size_t strong_actual = 0;

    printf("LAMMP version: %s\n", LAMMP_VERSION);
    printf("LAMMP compiler: %s (%d)\n", LAMMP_COMPILER, LAMMP_COMPILER_VERSION);
    printf("gcd(48,18) = %llu\n", (unsigned long long)lmmp_gcd_11_(48, 18));
    printf("powmod(7,128,13) = %llu\n", (unsigned long long)lmmp_powmod_ulong_(7, 128, 13));
    printf("is_prime(97) = %d\n", lmmp_is_prime_ulong_(97) ? 1 : 0);

    lmmp_global_rng_init_(20260411, 1);
    (void)lmmp_random_(&rnd, 1);
    printf("random_u64 = %llu\n", (unsigned long long)rnd);

    (void)lmmp_seed_random_(&rnd, 1, 123456789u, 0);
    printf("seeded_random_u64 = %llu\n", (unsigned long long)rnd);

    h_xx = lmmp_xxhash_(words, 4, (const uint64_t[1]){0x9e3779b97f4a7c15ull});
    h_sip = lmmp_siphash24_(words, 4, sip_key);
    printf("xxhash(words) = %llu\n", (unsigned long long)h_xx);
    printf("siphash(words) = %llu\n", (unsigned long long)h_sip);

    strong_rng = lmmp_strong_rng_init_(4, 2026);
    if (strong_rng == NULL) {
        return 1;
    }
    strong_actual = lmmp_strong_random_(strong_values, 4, strong_rng);
    printf("strong_rng sample actual=%llu first=%llu\n",
           (unsigned long long)strong_actual,
           (unsigned long long)strong_values[0]);
    lmmp_strong_rng_free_(strong_rng);

    return 0;
}