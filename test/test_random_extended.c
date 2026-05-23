/**
 * @file test_random_extended.c
 * @brief Extended tests for the random module.
 *
 * Covers:
 * - Same seed reproducibility (Req 13.1)
 * - uniform(0,1) statistical properties (Req 13.2)
 * - normal(0,1) statistical properties (Req 13.3)
 * - exponential(rate=2) statistical properties (Req 13.4)
 * - shuffle permutation property (Req 13.5)
 * - shuffle single-element invariance (Req 13.6)
 * - Error handling: a>=b, rate<=0, stddev<0, NULL (Req 13.7-13.10)
 *
 * Validates: Requirements 13.1-13.10
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "lmmc/lmmc.h"
#include "test_common.h"

#define NUM_SAMPLES 10000

static int cmp_int(const void* a, const void* b)
{
    int va = *(const int*)a;
    int vb = *(const int*)b;
    return (va > vb) - (va < vb);
}

int main(void)
{
    int rc = 0;

    /* ================================================================
     * Test 1: Same seed reproducibility (Req 13.1)
     * Two RNG instances with the same seed produce identical sequences.
     * ================================================================ */
    {
        lmmc_rng_t* rng1 = NULL;
        lmmc_rng_t* rng2 = NULL;
        lmmc_status_t st;
        int i;

        st = lmmc_rng_create(&rng1);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        st = lmmc_rng_create(&rng2);
        if (st != LMMC_STATUS_OK) { lmmc_rng_destroy(rng1); rc = 1; goto done; }

        lmmc_rng_seed(rng1, 12345);
        lmmc_rng_seed(rng2, 12345);

        for (i = 0; i < 100; i++) {
            uint64_t v1 = lmmc_rng_next_u64(rng1);
            uint64_t v2 = lmmc_rng_next_u64(rng2);
            if (v1 != v2) { rc = 1; lmmc_rng_destroy(rng1); lmmc_rng_destroy(rng2); goto done; }
        }

        /* Also verify uniform produces same values */
        lmmc_rng_seed(rng1, 99999);
        lmmc_rng_seed(rng2, 99999);
        for (i = 0; i < 50; i++) {
            lmmc_real_t u1, u2;
            lmmc_rng_uniform(rng1, 0.0, 1.0, &u1);
            lmmc_rng_uniform(rng2, 0.0, 1.0, &u2);
            if (u1 != u2) { rc = 1; lmmc_rng_destroy(rng1); lmmc_rng_destroy(rng2); goto done; }
        }

        lmmc_rng_destroy(rng1);
        lmmc_rng_destroy(rng2);
    }

    /* ================================================================
     * Test 2: uniform(0,1) statistical properties (Req 13.2)
     * Mean ≈ 0.5 (error < 0.02), all values in [0,1].
     * ================================================================ */
    {
        lmmc_rng_t* rng = NULL;
        lmmc_status_t st;
        int i;
        double sum = 0.0;

        st = lmmc_rng_create(&rng);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_rng_seed(rng, 42);

        for (i = 0; i < NUM_SAMPLES; i++) {
            lmmc_real_t val;
            st = lmmc_rng_uniform(rng, 0.0, 1.0, &val);
            if (st != LMMC_STATUS_OK) { rc = 1; lmmc_rng_destroy(rng); goto done; }
            if (val < 0.0 || val > 1.0) { rc = 1; lmmc_rng_destroy(rng); goto done; }
            sum += val;
        }

        double mean = sum / NUM_SAMPLES;
        if (fabs(mean - 0.5) > 0.02) { rc = 1; lmmc_rng_destroy(rng); goto done; }

        lmmc_rng_destroy(rng);
    }

    /* ================================================================
     * Test 3: normal(0,1) statistical properties (Req 13.3)
     * Mean ≈ 0 (error < 0.05), stddev ≈ 1 (error < 0.05).
     * ================================================================ */
    {
        lmmc_rng_t* rng = NULL;
        lmmc_status_t st;
        int i;
        double sum = 0.0;
        double sum_sq = 0.0;

        st = lmmc_rng_create(&rng);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_rng_seed(rng, 77);

        for (i = 0; i < NUM_SAMPLES; i++) {
            lmmc_real_t val;
            st = lmmc_rng_normal(rng, 0.0, 1.0, &val);
            if (st != LMMC_STATUS_OK) { rc = 1; lmmc_rng_destroy(rng); goto done; }
            sum += val;
            sum_sq += val * val;
        }

        double mean = sum / NUM_SAMPLES;
        double variance = (sum_sq / NUM_SAMPLES) - (mean * mean);
        double stddev = sqrt(variance);

        if (fabs(mean) > 0.05) { rc = 1; lmmc_rng_destroy(rng); goto done; }
        if (fabs(stddev - 1.0) > 0.05) { rc = 1; lmmc_rng_destroy(rng); goto done; }

        lmmc_rng_destroy(rng);
    }

    /* ================================================================
     * Test 4: exponential(rate=2) statistical properties (Req 13.4)
     * Mean ≈ 0.5 (error < 0.05), all values >= 0.
     * ================================================================ */
    {
        lmmc_rng_t* rng = NULL;
        lmmc_status_t st;
        int i;
        double sum = 0.0;

        st = lmmc_rng_create(&rng);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_rng_seed(rng, 123);

        for (i = 0; i < NUM_SAMPLES; i++) {
            lmmc_real_t val;
            st = lmmc_rng_exponential(rng, 2.0, &val);
            if (st != LMMC_STATUS_OK) { rc = 1; lmmc_rng_destroy(rng); goto done; }
            if (val < 0.0) { rc = 1; lmmc_rng_destroy(rng); goto done; }
            sum += val;
        }

        double mean = sum / NUM_SAMPLES;
        if (fabs(mean - 0.5) > 0.05) { rc = 1; lmmc_rng_destroy(rng); goto done; }

        lmmc_rng_destroy(rng);
    }

    /* ================================================================
     * Test 5: shuffle permutation property (Req 13.5)
     * All elements still present after shuffle.
     * ================================================================ */
    {
        lmmc_rng_t* rng = NULL;
        lmmc_status_t st;
        int arr[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        int sorted_orig[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        int i;

        st = lmmc_rng_create(&rng);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_rng_seed(rng, 555);

        st = lmmc_rng_shuffle(rng, arr, 10, sizeof(int));
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_rng_destroy(rng); goto done; }

        /* Sort the shuffled array and compare with original sorted */
        qsort(arr, 10, sizeof(int), cmp_int);
        for (i = 0; i < 10; i++) {
            if (arr[i] != sorted_orig[i]) { rc = 1; lmmc_rng_destroy(rng); goto done; }
        }

        lmmc_rng_destroy(rng);
    }

    /* ================================================================
     * Test 6: shuffle single-element invariance (Req 13.6)
     * Array of length 1 remains unchanged.
     * ================================================================ */
    {
        lmmc_rng_t* rng = NULL;
        lmmc_status_t st;
        int val = 42;

        st = lmmc_rng_create(&rng);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_rng_seed(rng, 777);

        st = lmmc_rng_shuffle(rng, &val, 1, sizeof(int));
        if (st != LMMC_STATUS_OK) { rc = 1; lmmc_rng_destroy(rng); goto done; }
        if (val != 42) { rc = 1; lmmc_rng_destroy(rng); goto done; }

        lmmc_rng_destroy(rng);
    }

    /* ================================================================
     * Test 7: Error handling - uniform a >= b (Req 13.7)
     * ================================================================ */
    {
        lmmc_rng_t* rng = NULL;
        lmmc_status_t st;
        lmmc_real_t val;

        st = lmmc_rng_create(&rng);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_rng_seed(rng, 1);

        /* a == b */
        st = lmmc_rng_uniform(rng, 5.0, 5.0, &val);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; lmmc_rng_destroy(rng); goto done; }

        /* a > b */
        st = lmmc_rng_uniform(rng, 10.0, 3.0, &val);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; lmmc_rng_destroy(rng); goto done; }

        lmmc_rng_destroy(rng);
    }

    /* ================================================================
     * Test 8: Error handling - exponential rate <= 0 (Req 13.8)
     * ================================================================ */
    {
        lmmc_rng_t* rng = NULL;
        lmmc_status_t st;
        lmmc_real_t val;

        st = lmmc_rng_create(&rng);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_rng_seed(rng, 2);

        /* rate = 0 */
        st = lmmc_rng_exponential(rng, 0.0, &val);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; lmmc_rng_destroy(rng); goto done; }

        /* rate < 0 */
        st = lmmc_rng_exponential(rng, -1.0, &val);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; lmmc_rng_destroy(rng); goto done; }

        lmmc_rng_destroy(rng);
    }

    /* ================================================================
     * Test 9: Error handling - normal stddev < 0 (Req 13.9)
     * ================================================================ */
    {
        lmmc_rng_t* rng = NULL;
        lmmc_status_t st;
        lmmc_real_t val;

        st = lmmc_rng_create(&rng);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_rng_seed(rng, 3);

        st = lmmc_rng_normal(rng, 0.0, -1.0, &val);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; lmmc_rng_destroy(rng); goto done; }

        lmmc_rng_destroy(rng);
    }

    /* ================================================================
     * Test 10: Error handling - NULL pointer (Req 13.10)
     * ================================================================ */
    {
        lmmc_status_t st;

        st = lmmc_rng_seed(NULL, 42);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; goto done; }
    }

done:
    if (rc != 0) {
        printf("random extended test failed\n");
    }
    return rc;
}
