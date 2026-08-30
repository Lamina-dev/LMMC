/**
 * @file test_random_rng.c
 * 针对 LMMC 中 random rng 相关接口的单元测试。
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "lmmc/random.h"
#include "lmmc/status.h"


#define SEQUENCE_LENGTH 200


#define NUM_ITERATIONS 100

static int test_failures = 0;

#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("FAIL: " msg "\n", ##__VA_ARGS__); \
        test_failures++; \
        return 1; \
    } \
} while (0)


static uint64_t rand_u64(void)
{
    uint64_t val = 0;
    size_t i;
    for (i = 0; i < sizeof(uint64_t); i++) {
        val = (val << 8) | (uint64_t)(rand() & 0xFF);
    }
    return val;
}


static int test_same_seed_same_sequence(uint64_t seed)
{
    lmmc_rng_t* rng1 = NULL;
    lmmc_rng_t* rng2 = NULL;
    lmmc_status_t st;
    int i;

    st = lmmc_rng_create(&rng1);
    CHECK(st == LMMC_STATUS_OK, "Failed to create rng1");

    st = lmmc_rng_create(&rng2);
    CHECK(st == LMMC_STATUS_OK, "Failed to create rng2");

    st = lmmc_rng_seed(rng1, seed);
    CHECK(st == LMMC_STATUS_OK, "Failed to seed rng1 with %llu", (unsigned long long)seed);

    st = lmmc_rng_seed(rng2, seed);
    CHECK(st == LMMC_STATUS_OK, "Failed to seed rng2 with %llu", (unsigned long long)seed);


    for (i = 0; i < SEQUENCE_LENGTH; i++) {
        uint64_t v1 = lmmc_rng_next_u64(rng1);
        uint64_t v2 = lmmc_rng_next_u64(rng2);
        CHECK(v1 == v2,
              "seed=%llu, index=%d: rng1 produced %llu but rng2 produced %llu",
              (unsigned long long)seed, i,
              (unsigned long long)v1, (unsigned long long)v2);
    }

    lmmc_rng_destroy(rng1);
    lmmc_rng_destroy(rng2);
    return 0;
}


static int test_different_seeds_different_sequences(void)
{
    lmmc_rng_t* rng1 = NULL;
    lmmc_rng_t* rng2 = NULL;
    lmmc_status_t st;
    int i;
    int found_diff;
    uint64_t seed1 = 12345;
    uint64_t seed2 = 67890;

    st = lmmc_rng_create(&rng1);
    CHECK(st == LMMC_STATUS_OK, "Failed to create rng1");

    st = lmmc_rng_create(&rng2);
    CHECK(st == LMMC_STATUS_OK, "Failed to create rng2");

    st = lmmc_rng_seed(rng1, seed1);
    CHECK(st == LMMC_STATUS_OK, "Failed to seed rng1");

    st = lmmc_rng_seed(rng2, seed2);
    CHECK(st == LMMC_STATUS_OK, "Failed to seed rng2");


    found_diff = 0;
    for (i = 0; i < SEQUENCE_LENGTH; i++) {
        uint64_t v1 = lmmc_rng_next_u64(rng1);
        uint64_t v2 = lmmc_rng_next_u64(rng2);
        if (v1 != v2) {
            found_diff = 1;
            break;
        }
    }

    CHECK(found_diff,
          "seeds %llu and %llu produced identical sequences over %d values",
          (unsigned long long)seed1, (unsigned long long)seed2, SEQUENCE_LENGTH);

    lmmc_rng_destroy(rng1);
    lmmc_rng_destroy(rng2);
    return 0;
}


static int test_different_seeds_random_pairs(void)
{
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        lmmc_rng_t* rng1 = NULL;
        lmmc_rng_t* rng2 = NULL;
        lmmc_status_t st;
        uint64_t seed1 = rand_u64();
        uint64_t seed2 = rand_u64();
        int i;
        int found_diff;


        if (seed1 == seed2) continue;

        st = lmmc_rng_create(&rng1);
        CHECK(st == LMMC_STATUS_OK, "Failed to create rng1");

        st = lmmc_rng_create(&rng2);
        CHECK(st == LMMC_STATUS_OK, "Failed to create rng2");

        st = lmmc_rng_seed(rng1, seed1);
        CHECK(st == LMMC_STATUS_OK, "Failed to seed rng1");

        st = lmmc_rng_seed(rng2, seed2);
        CHECK(st == LMMC_STATUS_OK, "Failed to seed rng2");

        found_diff = 0;
        for (i = 0; i < SEQUENCE_LENGTH; i++) {
            uint64_t v1 = lmmc_rng_next_u64(rng1);
            uint64_t v2 = lmmc_rng_next_u64(rng2);
            if (v1 != v2) {
                found_diff = 1;
                break;
            }
        }

        CHECK(found_diff,
              "different seeds %llu and %llu produced identical sequences",
              (unsigned long long)seed1, (unsigned long long)seed2);

        lmmc_rng_destroy(rng1);
        lmmc_rng_destroy(rng2);
    }

    return 0;
}


static int test_reseed_resets_state(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t st;
    uint64_t first_run[SEQUENCE_LENGTH];
    uint64_t second_run[SEQUENCE_LENGTH];
    int i;
    uint64_t seed = 0xABCDEF0123456789ULL;

    st = lmmc_rng_create(&rng);
    CHECK(st == LMMC_STATUS_OK, "Failed to create rng");


    st = lmmc_rng_seed(rng, seed);
    CHECK(st == LMMC_STATUS_OK, "Failed to seed rng (first time)");

    for (i = 0; i < SEQUENCE_LENGTH; i++) {
        first_run[i] = lmmc_rng_next_u64(rng);
    }


    for (i = 0; i < 1000; i++) {
        (void)lmmc_rng_next_u64(rng);
    }


    st = lmmc_rng_seed(rng, seed);
    CHECK(st == LMMC_STATUS_OK, "Failed to re-seed rng");


    for (i = 0; i < SEQUENCE_LENGTH; i++) {
        second_run[i] = lmmc_rng_next_u64(rng);
    }


    for (i = 0; i < SEQUENCE_LENGTH; i++) {
        CHECK(first_run[i] == second_run[i],
              "reseed: index=%d, first=%llu, second=%llu",
              i, (unsigned long long)first_run[i], (unsigned long long)second_run[i]);
    }

    lmmc_rng_destroy(rng);
    return 0;
}


static int test_multiple_specific_seeds(void)
{
    uint64_t seeds[] = {
        0,
        1,
        UINT64_MAX,
        0x0000000000000002ULL,
        0x7FFFFFFFFFFFFFFFULL,
        0x8000000000000000ULL,
        0xDEADBEEFCAFEBABEULL,
        0x0123456789ABCDEFULL
    };
    size_t num_seeds = sizeof(seeds) / sizeof(seeds[0]);
    size_t s;

    for (s = 0; s < num_seeds; s++) {
        if (test_same_seed_same_sequence(seeds[s])) {
            printf("  [FAIL] specific seed %llu\n", (unsigned long long)seeds[s]);
            return 1;
        }
    }

    return 0;
}


static int test_random_seeds_reproducibility(void)
{
    int iter;

    for (iter = 0; iter < NUM_ITERATIONS; iter++) {
        uint64_t seed = rand_u64();
        if (test_same_seed_same_sequence(seed)) {
            printf("  [FAIL] random seed %llu at iteration %d\n",
                   (unsigned long long)seed, iter);
            return 1;
        }
    }

    return 0;
}


int main(void)
{
    int rc = 0;

    const unsigned int seed = 0x52524E47u;
    srand(seed);

    printf("=== RNG Reproducibility ===\n");

    printf("--- Same seed produces identical sequences ---\n");
    if (test_multiple_specific_seeds()) { rc = 1; printf("  [FAIL] specific seeds\n"); }
    else { printf("  [PASS] specific seeds (0, 1, UINT64_MAX, etc.)\n"); }

    if (test_random_seeds_reproducibility()) { rc = 1; printf("  [FAIL] random seeds\n"); }
    else { printf("  [PASS] random seeds (%d iterations, %d values each)\n", NUM_ITERATIONS, SEQUENCE_LENGTH); }

    printf("\n--- Different seeds produce different sequences ---\n");
    if (test_different_seeds_different_sequences()) { rc = 1; printf("  [FAIL] different seeds (fixed)\n"); }
    else { printf("  [PASS] different seeds (fixed pair)\n"); }

    if (test_different_seeds_random_pairs()) { rc = 1; printf("  [FAIL] different seeds (random pairs)\n"); }
    else { printf("  [PASS] different seeds (random pairs, %d iterations)\n", NUM_ITERATIONS); }

    printf("\n--- Re-seeding resets state ---\n");
    if (test_reseed_resets_state()) { rc = 1; printf("  [FAIL] reseed resets state\n"); }
    else { printf("  [PASS] reseed resets state\n"); }

    printf("\n");
    if (rc == 0) {
        printf("All RNG reproducibility property tests PASSED.\n");
    } else {
        printf("Some RNG reproducibility property tests FAILED.\n");
    }

    return rc;
}
