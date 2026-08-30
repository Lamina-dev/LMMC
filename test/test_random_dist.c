/**
 * @file test_random_dist.c
 * 针对 LMMC 中 random dist 相关接口的单元测试。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "lmmc/config.h"
#include "lmmc/random.h"


#define NUM_SAMPLES 1200


#define NUM_CONFIGS 50

static int test_failures = 0;

#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("FAIL: " msg "\n", ##__VA_ARGS__); \
        test_failures++; \
        return 1; \
    } \
} while (0)


static double rand_in_range(double lo, double hi)
{
    double u = (double)rand() / (double)RAND_MAX;
    return lo + (hi - lo) * u;
}


static int cmp_real(const void* a, const void* b)
{
    lmmc_real_t va = *(const lmmc_real_t*)a;
    lmmc_real_t vb = *(const lmmc_real_t*)b;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}


static int cmp_int(const void* a, const void* b)
{
    int va = *(const int*)a;
    int vb = *(const int*)b;
    return va - vb;
}


static int test_uniform_range_single(void)
{
    int cfg;
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status;

    status = lmmc_rng_create(&rng);
    CHECK(status == LMMC_STATUS_OK, "rng_create failed: %d", (int)status);
    lmmc_rng_seed(rng, (uint64_t)42);

    for (cfg = 0; cfg < NUM_CONFIGS; cfg++) {

        double a = rand_in_range(-1000.0, 1000.0);
        double b = a + rand_in_range(0.001, 2000.0);
        int i;

        for (i = 0; i < NUM_SAMPLES; i++) {
            lmmc_real_t value;
            status = lmmc_rng_uniform(rng, (lmmc_real_t)a, (lmmc_real_t)b, &value);
            CHECK(status == LMMC_STATUS_OK,
                  "uniform returned error %d (cfg=%d, i=%d)", (int)status, cfg, i);
            CHECK(value >= a,
                  "uniform value %g < a=%g (cfg=%d, i=%d)", value, a, cfg, i);
            CHECK(value < b,
                  "uniform value %g >= b=%g (cfg=%d, i=%d)", value, b, cfg, i);
        }
    }

    lmmc_rng_destroy(rng);
    return 0;
}


static int test_uniform_range_fill(void)
{
    int cfg;
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status;

    status = lmmc_rng_create(&rng);
    CHECK(status == LMMC_STATUS_OK, "rng_create failed: %d", (int)status);
    lmmc_rng_seed(rng, (uint64_t)123);

    for (cfg = 0; cfg < NUM_CONFIGS; cfg++) {
        double a = rand_in_range(-500.0, 500.0);
        double b = a + rand_in_range(0.01, 1000.0);
        lmmc_real_t array[NUM_SAMPLES];
        size_t i;

        status = lmmc_rng_fill_uniform(rng, (lmmc_real_t)a, (lmmc_real_t)b,
                                       array, NUM_SAMPLES);
        CHECK(status == LMMC_STATUS_OK,
              "fill_uniform returned error %d (cfg=%d)", (int)status, cfg);

        for (i = 0; i < NUM_SAMPLES; i++) {
            CHECK(array[i] >= a,
                  "fill_uniform value %g < a=%g (cfg=%d, i=%zu)",
                  array[i], a, cfg, i);
            CHECK(array[i] < b,
                  "fill_uniform value %g >= b=%g (cfg=%d, i=%zu)",
                  array[i], b, cfg, i);
        }
    }

    lmmc_rng_destroy(rng);
    return 0;
}


static int test_uniform_range_tiny_interval(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status;
    int i;

    status = lmmc_rng_create(&rng);
    CHECK(status == LMMC_STATUS_OK, "rng_create failed");
    lmmc_rng_seed(rng, (uint64_t)999);

    double a = 1.0;
    double b = 1.0 + 1e-10;

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t value;
        status = lmmc_rng_uniform(rng, (lmmc_real_t)a, (lmmc_real_t)b, &value);
        CHECK(status == LMMC_STATUS_OK,
              "uniform tiny interval returned error %d (i=%d)", (int)status, i);
        CHECK(value >= a,
              "uniform tiny: value %g < a=%g (i=%d)", value, a, i);
        CHECK(value < b,
              "uniform tiny: value %g >= b=%g (i=%d)", value, b, i);
    }

    lmmc_rng_destroy(rng);
    return 0;
}


static int test_uniform_range_large_interval(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status;
    int i;

    status = lmmc_rng_create(&rng);
    CHECK(status == LMMC_STATUS_OK, "rng_create failed");
    lmmc_rng_seed(rng, (uint64_t)7777);

    double a = -1e15;
    double b = 1e15;

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t value;
        status = lmmc_rng_uniform(rng, (lmmc_real_t)a, (lmmc_real_t)b, &value);
        CHECK(status == LMMC_STATUS_OK,
              "uniform large interval returned error %d (i=%d)", (int)status, i);
        CHECK(value >= a,
              "uniform large: value %g < a=%g (i=%d)", value, a, i);
        CHECK(value < b,
              "uniform large: value %g >= b=%g (i=%d)", value, b, i);
    }

    lmmc_rng_destroy(rng);
    return 0;
}


static int test_exponential_nonneg(void)
{
    int cfg;
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status;

    status = lmmc_rng_create(&rng);
    CHECK(status == LMMC_STATUS_OK, "rng_create failed: %d", (int)status);
    lmmc_rng_seed(rng, (uint64_t)314159);

    for (cfg = 0; cfg < NUM_CONFIGS; cfg++) {

        double rate = rand_in_range(0.001, 1000.0);
        int i;

        for (i = 0; i < NUM_SAMPLES; i++) {
            lmmc_real_t value;
            status = lmmc_rng_exponential(rng, (lmmc_real_t)rate, &value);
            CHECK(status == LMMC_STATUS_OK,
                  "exponential returned error %d (cfg=%d, rate=%g, i=%d)",
                  (int)status, cfg, rate, i);
            CHECK(value >= 0.0,
                  "exponential value %g < 0 (cfg=%d, rate=%g, i=%d)",
                  value, cfg, rate, i);
        }
    }

    lmmc_rng_destroy(rng);
    return 0;
}


static int test_exponential_nonneg_small_rate(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status;
    int i;

    status = lmmc_rng_create(&rng);
    CHECK(status == LMMC_STATUS_OK, "rng_create failed");
    lmmc_rng_seed(rng, (uint64_t)271828);

    double rate = 1e-10;

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t value;
        status = lmmc_rng_exponential(rng, (lmmc_real_t)rate, &value);
        CHECK(status == LMMC_STATUS_OK,
              "exponential small rate returned error %d (i=%d)", (int)status, i);
        CHECK(value >= 0.0,
              "exponential small rate: value %g < 0 (i=%d)", value, i);
    }

    lmmc_rng_destroy(rng);
    return 0;
}


static int test_exponential_nonneg_large_rate(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status;
    int i;

    status = lmmc_rng_create(&rng);
    CHECK(status == LMMC_STATUS_OK, "rng_create failed");
    lmmc_rng_seed(rng, (uint64_t)161803);

    double rate = 1e10;

    for (i = 0; i < NUM_SAMPLES; i++) {
        lmmc_real_t value;
        status = lmmc_rng_exponential(rng, (lmmc_real_t)rate, &value);
        CHECK(status == LMMC_STATUS_OK,
              "exponential large rate returned error %d (i=%d)", (int)status, i);
        CHECK(value >= 0.0,
              "exponential large rate: value %g < 0 (i=%d)", value, i);
    }

    lmmc_rng_destroy(rng);
    return 0;
}


static int test_shuffle_preserves_elements_real(void)
{
    int cfg;
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status;

    status = lmmc_rng_create(&rng);
    CHECK(status == LMMC_STATUS_OK, "rng_create failed: %d", (int)status);
    lmmc_rng_seed(rng, (uint64_t)55555);

    for (cfg = 0; cfg < NUM_CONFIGS; cfg++) {

        size_t n = 2 + (size_t)(rand() % 199);
        lmmc_real_t* array = (lmmc_real_t*)malloc(n * sizeof(lmmc_real_t));
        lmmc_real_t* sorted_before = (lmmc_real_t*)malloc(n * sizeof(lmmc_real_t));
        size_t i;

        CHECK(array != NULL && sorted_before != NULL,
              "malloc failed (cfg=%d)", cfg);


        for (i = 0; i < n; i++) {
            array[i] = rand_in_range(-1000.0, 1000.0);
        }


        memcpy(sorted_before, array, n * sizeof(lmmc_real_t));
        qsort(sorted_before, n, sizeof(lmmc_real_t), cmp_real);


        status = lmmc_rng_shuffle(rng, array, n, sizeof(lmmc_real_t));
        CHECK(status == LMMC_STATUS_OK,
              "shuffle returned error %d (cfg=%d, n=%zu)", (int)status, cfg, n);


        qsort(array, n, sizeof(lmmc_real_t), cmp_real);


        for (i = 0; i < n; i++) {
            CHECK(array[i] == sorted_before[i],
                  "shuffle changed elements: sorted[%zu]=%g vs original sorted[%zu]=%g (cfg=%d)",
                  i, array[i], i, sorted_before[i], cfg);
        }

        free(array);
        free(sorted_before);
    }

    lmmc_rng_destroy(rng);
    return 0;
}


static int test_shuffle_preserves_elements_int(void)
{
    int cfg;
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status;

    status = lmmc_rng_create(&rng);
    CHECK(status == LMMC_STATUS_OK, "rng_create failed: %d", (int)status);
    lmmc_rng_seed(rng, (uint64_t)88888);

    for (cfg = 0; cfg < NUM_CONFIGS; cfg++) {
        size_t n = 2 + (size_t)(rand() % 199);
        int* array = (int*)malloc(n * sizeof(int));
        int* sorted_before = (int*)malloc(n * sizeof(int));
        size_t i;

        CHECK(array != NULL && sorted_before != NULL,
              "malloc failed (cfg=%d)", cfg);


        for (i = 0; i < n; i++) {
            array[i] = rand() % 1000 - 500;
        }


        memcpy(sorted_before, array, n * sizeof(int));
        qsort(sorted_before, n, sizeof(int), cmp_int);


        status = lmmc_rng_shuffle(rng, array, n, sizeof(int));
        CHECK(status == LMMC_STATUS_OK,
              "shuffle int returned error %d (cfg=%d, n=%zu)", (int)status, cfg, n);


        qsort(array, n, sizeof(int), cmp_int);


        for (i = 0; i < n; i++) {
            CHECK(array[i] == sorted_before[i],
                  "shuffle int changed elements: sorted[%zu]=%d vs original sorted[%zu]=%d (cfg=%d)",
                  i, array[i], i, sorted_before[i], cfg);
        }

        free(array);
        free(sorted_before);
    }

    lmmc_rng_destroy(rng);
    return 0;
}


static int test_shuffle_single_element(void)
{
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status;

    status = lmmc_rng_create(&rng);
    CHECK(status == LMMC_STATUS_OK, "rng_create failed");
    lmmc_rng_seed(rng, (uint64_t)11111);

    lmmc_real_t val = 42.0;
    status = lmmc_rng_shuffle(rng, &val, 1, sizeof(lmmc_real_t));
    CHECK(status == LMMC_STATUS_OK, "shuffle single returned error %d", (int)status);
    CHECK(val == 42.0, "shuffle single changed value: got %g", val);

    lmmc_rng_destroy(rng);
    return 0;
}


static int test_shuffle_with_duplicates(void)
{
    int cfg;
    lmmc_rng_t* rng = NULL;
    lmmc_status_t status;

    status = lmmc_rng_create(&rng);
    CHECK(status == LMMC_STATUS_OK, "rng_create failed");
    lmmc_rng_seed(rng, (uint64_t)22222);

    for (cfg = 0; cfg < NUM_CONFIGS; cfg++) {

        size_t n = 50 + (size_t)(rand() % 151);
        int* array = (int*)malloc(n * sizeof(int));
        int* sorted_before = (int*)malloc(n * sizeof(int));
        size_t i;

        CHECK(array != NULL && sorted_before != NULL,
              "malloc failed (cfg=%d)", cfg);


        for (i = 0; i < n; i++) {
            array[i] = rand() % 10;
        }

        memcpy(sorted_before, array, n * sizeof(int));
        qsort(sorted_before, n, sizeof(int), cmp_int);

        status = lmmc_rng_shuffle(rng, array, n, sizeof(int));
        CHECK(status == LMMC_STATUS_OK,
              "shuffle duplicates returned error %d (cfg=%d)", (int)status, cfg);

        qsort(array, n, sizeof(int), cmp_int);

        for (i = 0; i < n; i++) {
            CHECK(array[i] == sorted_before[i],
                  "shuffle duplicates: mismatch at [%zu] (cfg=%d)", i, cfg);
        }

        free(array);
        free(sorted_before);
    }

    lmmc_rng_destroy(rng);
    return 0;
}


int main(void)
{
    int rc = 0;

    const unsigned int seed = 0x52444953u;
    srand(seed);

    printf("=== 均匀分布范围约束 ===\n");

    if (test_uniform_range_single()) { rc = 1; printf("  [FAIL] uniform range (single)\n"); }
    else { printf("  [PASS] uniform range single (%d configs x %d samples)\n", NUM_CONFIGS, NUM_SAMPLES); }

    if (test_uniform_range_fill()) { rc = 1; printf("  [FAIL] uniform range (fill)\n"); }
    else { printf("  [PASS] uniform range fill (%d configs x %d samples)\n", NUM_CONFIGS, NUM_SAMPLES); }

    if (test_uniform_range_tiny_interval()) { rc = 1; printf("  [FAIL] uniform tiny interval\n"); }
    else { printf("  [PASS] uniform tiny interval (%d samples)\n", NUM_SAMPLES); }

    if (test_uniform_range_large_interval()) { rc = 1; printf("  [FAIL] uniform large interval\n"); }
    else { printf("  [PASS] uniform large interval (%d samples)\n", NUM_SAMPLES); }

    printf("\n=== 指数分布非负性 ===\n");

    if (test_exponential_nonneg()) { rc = 1; printf("  [FAIL] exponential non-negative\n"); }
    else { printf("  [PASS] exponential non-negative (%d configs x %d samples)\n", NUM_CONFIGS, NUM_SAMPLES); }

    if (test_exponential_nonneg_small_rate()) { rc = 1; printf("  [FAIL] exponential small rate\n"); }
    else { printf("  [PASS] exponential small rate (%d samples)\n", NUM_SAMPLES); }

    if (test_exponential_nonneg_large_rate()) { rc = 1; printf("  [FAIL] exponential large rate\n"); }
    else { printf("  [PASS] exponential large rate (%d samples)\n", NUM_SAMPLES); }

    printf("\n=== Shuffle 保持元素集合不变 ===\n");

    if (test_shuffle_preserves_elements_real()) { rc = 1; printf("  [FAIL] shuffle preserves (real)\n"); }
    else { printf("  [PASS] shuffle preserves elements real (%d configs)\n", NUM_CONFIGS); }

    if (test_shuffle_preserves_elements_int()) { rc = 1; printf("  [FAIL] shuffle preserves (int)\n"); }
    else { printf("  [PASS] shuffle preserves elements int (%d configs)\n", NUM_CONFIGS); }

    if (test_shuffle_single_element()) { rc = 1; printf("  [FAIL] shuffle single element\n"); }
    else { printf("  [PASS] shuffle single element\n"); }

    if (test_shuffle_with_duplicates()) { rc = 1; printf("  [FAIL] shuffle with duplicates\n"); }
    else { printf("  [PASS] shuffle with duplicates (%d configs)\n", NUM_CONFIGS); }

    printf("\n");
    if (rc == 0) {
        printf("All distribution sampling property tests PASSED.\n");
    } else {
        printf("Some distribution sampling property tests FAILED.\n");
    }

    return rc;
}
