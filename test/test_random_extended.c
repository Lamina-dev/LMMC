/**
 * @file test_random_extended.c
 * 针对 LMMC 中 random extended 相关接口的单元测试。
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


        qsort(arr, 10, sizeof(int), cmp_int);
        for (i = 0; i < 10; i++) {
            if (arr[i] != sorted_orig[i]) { rc = 1; lmmc_rng_destroy(rng); goto done; }
        }

        lmmc_rng_destroy(rng);
    }


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


    {
        lmmc_rng_t* rng = NULL;
        lmmc_status_t st;
        lmmc_real_t val;

        st = lmmc_rng_create(&rng);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_rng_seed(rng, 1);


        st = lmmc_rng_uniform(rng, 5.0, 5.0, &val);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; lmmc_rng_destroy(rng); goto done; }


        st = lmmc_rng_uniform(rng, 10.0, 3.0, &val);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; lmmc_rng_destroy(rng); goto done; }

        lmmc_rng_destroy(rng);
    }


    {
        lmmc_rng_t* rng = NULL;
        lmmc_status_t st;
        lmmc_real_t val;

        st = lmmc_rng_create(&rng);
        if (st != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_rng_seed(rng, 2);


        st = lmmc_rng_exponential(rng, 0.0, &val);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; lmmc_rng_destroy(rng); goto done; }


        st = lmmc_rng_exponential(rng, -1.0, &val);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; lmmc_rng_destroy(rng); goto done; }

        lmmc_rng_destroy(rng);
    }


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
