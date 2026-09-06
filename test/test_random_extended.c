/**
 * @file test_random_extended.c
 * 针对 LMMC 中 random extended 相关接口的单元测试。
 */
#include <math.h>
#include <float.h>
#include <limits.h>
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

static uint64_t reference_bounded(lmmc_rng_t* rng, uint64_t bound)
{
    const uint64_t threshold = (uint64_t)(-bound) % bound;
    uint64_t value;
    do {
        value = lmmc_rng_next_u64(rng);
    } while (value < threshold);
    return value % bound;
}

static int64_t signed_from_bits(uint64_t bits)
{
    if (bits <= (uint64_t)INT64_MAX) return (int64_t)bits;
    return INT64_MIN + (int64_t)(bits - (UINT64_C(1) << 63));
}

static int64_t reference_int_uniform(lmmc_rng_t* rng, int64_t lo, int64_t hi)
{
    const uint64_t span = (uint64_t)hi - (uint64_t)lo + UINT64_C(1);
    const uint64_t offset = span == 0 ? lmmc_rng_next_u64(rng)
                                      : reference_bounded(rng, span);
    return signed_from_bits((uint64_t)lo + offset);
}

static void reference_shuffle(lmmc_rng_t* rng, int* values, size_t count)
{
    for (size_t i = count - 1; i > 0; --i) {
        const size_t j = (size_t)reference_bounded(rng, (uint64_t)i + 1);
        const int tmp = values[i];
        values[i] = values[j];
        values[j] = tmp;
    }
}

int main(void)
{
    int rc = 0;
    lmmc_status_t st;


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

        st = lmmc_rng_uniform(rng, -DBL_MAX, DBL_MAX, &val);
        if (st != LMMC_STATUS_OK || !isfinite(val) ||
            val < -DBL_MAX || val >= DBL_MAX) {
            rc = 1;
            lmmc_rng_destroy(rng);
            goto done;
        }

        st = lmmc_rng_uniform(rng, NAN, 1.0, &val);
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

        st = lmmc_rng_exponential(rng, NAN, &val);
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

        st = lmmc_rng_normal(rng, NAN, 1.0, &val);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; lmmc_rng_destroy(rng); goto done; }

        st = lmmc_rng_normal(rng, 0.0, NAN, &val);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; lmmc_rng_destroy(rng); goto done; }

        st = lmmc_rng_gamma(rng, 1.0, NAN, &val);
        if (st != LMMC_STATUS_INVALID_ARGUMENT) { rc = 1; lmmc_rng_destroy(rng); goto done; }

        lmmc_rng_destroy(rng);
    }


    {
        lmmc_rng_t* rng1 = NULL;
        lmmc_rng_t* rng2 = NULL;
        lmmc_status_t st;
        uint64_t raw, bits;
        int64_t actual, expected;
        size_t binomial = 23;

        if (lmmc_rng_create(&rng1) != LMMC_STATUS_OK ||
            lmmc_rng_create(&rng2) != LMMC_STATUS_OK) {
            lmmc_rng_destroy(rng1);
            lmmc_rng_destroy(rng2);
            rc = 1;
            goto done;
        }
        lmmc_rng_seed(rng1, UINT64_C(0x1122334455667788));
        lmmc_rng_seed(rng2, UINT64_C(0x1122334455667788));

        raw = lmmc_rng_next_u64(rng1);
        bits = (uint64_t)INT64_MIN + raw;
        expected = bits <= (uint64_t)INT64_MAX
                 ? (int64_t)bits
                 : INT64_MIN + (int64_t)(bits - (UINT64_C(1) << 63));
        st = lmmc_rng_int_uniform(rng2, INT64_MIN, INT64_MAX, &actual);
        if (st != LMMC_STATUS_OK || actual != expected) {
            rc = 1; lmmc_rng_destroy(rng1); lmmc_rng_destroy(rng2); goto done;
        }

        lmmc_rng_seed(rng1, 991);
        lmmc_rng_seed(rng2, 991);
        st = lmmc_rng_int_uniform(rng1, -7, -7, &actual);
        if (st != LMMC_STATUS_OK || actual != -7 ||
            lmmc_rng_next_u64(rng1) != lmmc_rng_next_u64(rng2)) {
            rc = 1; lmmc_rng_destroy(rng1); lmmc_rng_destroy(rng2); goto done;
        }

        lmmc_rng_seed(rng1, 992);
        lmmc_rng_seed(rng2, 992);
        st = lmmc_rng_binomial(rng1, 10, NAN, &binomial);
        if (st != LMMC_STATUS_INVALID_ARGUMENT || binomial != 23 ||
            lmmc_rng_next_u64(rng1) != lmmc_rng_next_u64(rng2)) {
            rc = 1; lmmc_rng_destroy(rng1); lmmc_rng_destroy(rng2); goto done;
        }

        st = lmmc_rng_binomial(rng1, SIZE_MAX, DBL_MIN, &binomial);
        if (st != LMMC_STATUS_OK || binomial != 0) {
            rc = 1; lmmc_rng_destroy(rng1); lmmc_rng_destroy(rng2); goto done;
        }

        lmmc_rng_destroy(rng1);
        lmmc_rng_destroy(rng2);
    }

    {
        lmmc_rng_t* rng = NULL;
        size_t counts[3] = {0, 0, 0};
        size_t trial;

        if (lmmc_rng_create(&rng) != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_rng_seed(rng, UINT64_C(0x53485546464C45));
        for (trial = 0; trial < 60000; ++trial) {
            int values[3] = {0, 1, 2};
            if (lmmc_rng_shuffle(rng, values, 3, sizeof(values[0])) != LMMC_STATUS_OK) {
                rc = 1; lmmc_rng_destroy(rng); goto done;
            }
            ++counts[(size_t)values[0]];
        }
        for (trial = 0; trial < 3; ++trial) {
            if (counts[trial] < 19000 || counts[trial] > 21000) {
                rc = 1; lmmc_rng_destroy(rng); goto done;
            }
        }
        lmmc_rng_destroy(rng);
    }



    {
        const struct {
            int64_t lo;
            int64_t hi;
        } intervals[] = {
            {INT64_MIN, INT64_MAX},
            {-17, 23},
            {INT64_MIN, INT64_MIN + 100},
            {INT64_MAX - 100, INT64_MAX}
        };
        lmmc_rng_t* actual_rng = NULL;
        lmmc_rng_t* reference_rng = NULL;
        int actual_shuffle[17];
        int expected_shuffle[17];

        if (lmmc_rng_create(&actual_rng) != LMMC_STATUS_OK ||
            lmmc_rng_create(&reference_rng) != LMMC_STATUS_OK) {
            lmmc_rng_destroy(actual_rng);
            lmmc_rng_destroy(reference_rng);
            rc = 1; goto done;
        }
        lmmc_rng_seed(actual_rng, UINT64_C(0x424F554E444544));
        lmmc_rng_seed(reference_rng, UINT64_C(0x424F554E444544));
        for (size_t interval = 0; interval < sizeof(intervals) / sizeof(intervals[0]); ++interval) {
            for (size_t draw = 0; draw < 1000; ++draw) {
                int64_t actual;
                const int64_t expected = reference_int_uniform(
                    reference_rng, intervals[interval].lo, intervals[interval].hi);
                st = lmmc_rng_int_uniform(actual_rng, intervals[interval].lo,
                                          intervals[interval].hi, &actual);
                if (st != LMMC_STATUS_OK || actual != expected) {
                    rc = 1; lmmc_rng_destroy(actual_rng);
                    lmmc_rng_destroy(reference_rng); goto done;
                }
            }
        }

        for (size_t i = 0; i < 17; ++i) {
            actual_shuffle[i] = (int)i;
            expected_shuffle[i] = (int)i;
        }
        reference_shuffle(reference_rng, expected_shuffle, 17);
        st = lmmc_rng_shuffle(actual_rng, actual_shuffle, 17, sizeof(actual_shuffle[0]));
        if (st != LMMC_STATUS_OK ||
            memcmp(actual_shuffle, expected_shuffle, sizeof(actual_shuffle)) != 0) {
            rc = 1; lmmc_rng_destroy(actual_rng);
            lmmc_rng_destroy(reference_rng); goto done;
        }
        lmmc_rng_destroy(actual_rng);
        lmmc_rng_destroy(reference_rng);
    }

    {
        const size_t samples = 10000;
        const size_t n = 1000000;
        const double p = 0.37;
        const double expected_mean = (double)n * p;
        const double expected_variance = (double)n * p * (1.0 - p);
        double sum = 0.0;
        double sum_sq = 0.0;
        lmmc_rng_t* rng = NULL;
        size_t value = 0;

        if (lmmc_rng_create(&rng) != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_rng_seed(rng, UINT64_C(0x42545045313036));
        for (size_t i = 0; i < samples; ++i) {
            st = lmmc_rng_binomial(rng, n, p, &value);
            if (st != LMMC_STATUS_OK || value > n) {
                rc = 1; lmmc_rng_destroy(rng); goto done;
            }
            sum += (double)value;
            sum_sq += (double)value * (double)value;
        }
        {
            const double mean = sum / (double)samples;
            const double variance = sum_sq / (double)samples - mean * mean;
            const double mean_se = sqrt(expected_variance / (double)samples);
            const double variance_se = expected_variance *
                                       sqrt(2.0 / (double)(samples - 1));
            if (fabs(mean - expected_mean) > 6.0 * mean_se ||
                fabs(variance - expected_variance) > 6.0 * variance_se) {
                rc = 1; lmmc_rng_destroy(rng); goto done;
            }
        }

        st = lmmc_rng_binomial(rng, SIZE_MAX, 0.5, &value);
        if (st != LMMC_STATUS_OK || value == 0 || value == SIZE_MAX) {
            rc = 1; lmmc_rng_destroy(rng); goto done;
        }
        lmmc_rng_destroy(rng);
    }

    {
        const size_t samples = 100000;
        const size_t n = 8;
        const double p = 0.3;
        size_t observed[9] = {0};
        double probability[9];
        double chi_square = 0.0;
        const double q = 1.0 - p;
        const double threshold_df = 7.0;
        lmmc_rng_t* rng = NULL;

        probability[0] = pow(q, (double)n);
        for (size_t k = 1; k <= n; ++k) {
            probability[k] = probability[k - 1] *
                             ((double)(n - k + 1) / (double)k) * (p / q);
        }
        if (lmmc_rng_create(&rng) != LMMC_STATUS_OK) { rc = 1; goto done; }
        lmmc_rng_seed(rng, UINT64_C(0x504D4643484932));
        for (size_t i = 0; i < samples; ++i) {
            size_t value;
            st = lmmc_rng_binomial(rng, n, p, &value);
            if (st != LMMC_STATUS_OK || value > n) {
                rc = 1; lmmc_rng_destroy(rng); goto done;
            }
            ++observed[value];
        }
        for (size_t k = 0; k < 7; ++k) {
            const double expected = (double)samples * probability[k];
            const double delta = (double)observed[k] - expected;
            chi_square += delta * delta / expected;
        }
        {
            const double expected = (double)samples * (probability[7] + probability[8]);
            const double delta = (double)(observed[7] + observed[8]) - expected;
            chi_square += delta * delta / expected;
        }
        if (chi_square >= threshold_df + 6.0 * sqrt(2.0 * threshold_df)) {
            rc = 1; lmmc_rng_destroy(rng); goto done;
        }
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
