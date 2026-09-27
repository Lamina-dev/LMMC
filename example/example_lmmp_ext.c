/**
 * @file example_lmmp_ext.c
 * @brief 演示 LMMC 中 LMMP 扩展接口的使用。
 */
#include <lmmp.h>
#include <mprand.h>
#include <numth.h>
#include <secret.h>
#include <version.h>

#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>

int main(void) {
    mp_limb_t rnd = 0;
    mp_limb_t strong_values[4] = {0, 0, 0, 0};
    uint64_t sip_key[2] = {UINT64_C(0x0123456789abcdef), UINT64_C(0xfedcba9876543210)};
    uint64_t words[4] = {11, 22, 33, 44};
    uint64_t h_xx = 0;
    uint64_t h_sip = 0;
    lmmp_strong_rng_t* strong_rng = NULL;
    mp_size_t strong_actual = 0;

    printf("LMMP version: %s\n", lmmp_get_version());
    printf("LMMP build: %s\n", lmmp_get_build_type());
    printf("gcd(48,18) = %" PRIu64 "\n", (uint64_t)lmmp_gcd_11_(48, 18));
    printf("powmod(7,128,13) = %" PRIu64 "\n", (uint64_t)lmmp_powmod_ulong_odd_(7, 128, 13));
    printf("is_prime(97) = %d\n", lmmp_is_prime_ulong_(97) ? 1 : 0);

    lmmp_global_rng_init_(20260411, 1);
    (void)lmmp_random_(&rnd, 1);
    printf("random_u64 = %" PRIu64 "\n", (uint64_t)rnd);

    (void)lmmp_seed_random_(&rnd, 1, 123456789u, 0);
    printf("seeded_random_u64 = %" PRIu64 "\n", (uint64_t)rnd);

    h_xx = lmmp_xxhash_(words, 4, (const uint64_t[1]){UINT64_C(0x9e3779b97f4a7c15)});
    h_sip = lmmp_siphash24_(words, 4, sip_key);
    printf("xxhash(words) = %" PRIu64 "\n", h_xx);
    printf("siphash(words) = %" PRIu64 "\n", h_sip);

    strong_rng = lmmp_strong_rng_init_(4, 2026);
    if (strong_rng == NULL) {
        return 1;
    }
    strong_actual = lmmp_strong_random_(strong_values, 4, strong_rng);
    printf("strong_rng sample actual=%zu first=%" PRIu64 "\n",
           (size_t)strong_actual,
           (uint64_t)strong_values[0]);
    lmmp_strong_rng_free_(strong_rng);

    return 0;
}
