/**
 * @file test_vectorized.c
 * @brief Property-based and unit tests for vectorized apply pattern.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stddef.h>

#include "lmmc/lmmc.h"
#include "test_common.h"

#define PBT_ITERATIONS 100

static int test_failures = 0;
static int test_count = 0;

#define REPORT(name, result) do { \
    test_count++; \
    if (result) { \
        printf("  FAIL: %s\n", name); \
        test_failures++; \
    } else { \
        printf("  PASS: %s\n", name); \
    } \
} while (0)

static double rand_double(double lo, double hi)
{
    return ((double)rand() / RAND_MAX) * (hi - lo) + lo;
}

static int test_property16_vec_apply_exp_log_roundtrip(void)
{
    int i;
    double eps = 1e-10;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        size_t size = (size_t)(rand() % 20) + 1;
        size_t j;
        lmmc_vec_t v, after_exp, recovered;
        lmmc_status_t st;

        st = lmmc_vec_create(size, &v);
        if (st != LMMC_STATUS_OK) { printf("    vec_create(v) failed at iter %d\n", i); return 1; }
        st = lmmc_vec_create(size, &after_exp);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&v); printf("    vec_create(after_exp) failed at iter %d\n", i); return 1; }
        st = lmmc_vec_create(size, &recovered);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&v); lmmc_vec_destroy(&after_exp); printf("    vec_create(recovered) failed at iter %d\n", i); return 1; }

        /* Fill v with positive values in [0.1, 10.0] */
        for (j = 0; j < size; j++) {
            v.data[j] = rand_double(0.1, 10.0);
        }

        /* after_exp = exp(v) */
        st = lmmc_vec_apply_exp(&v, &after_exp);
        if (st != LMMC_STATUS_OK) {
            printf("    vec_apply_exp failed at iter %d, status=%d\n", i, (int)st);
            lmmc_vec_destroy(&v); lmmc_vec_destroy(&after_exp); lmmc_vec_destroy(&recovered);
            return 1;
        }

        /* recovered = log(after_exp) */
        st = lmmc_vec_apply_log(&after_exp, &recovered);
        if (st != LMMC_STATUS_OK) {
            printf("    vec_apply_log failed at iter %d, status=%d\n", i, (int)st);
            lmmc_vec_destroy(&v); lmmc_vec_destroy(&after_exp); lmmc_vec_destroy(&recovered);
            return 1;
        }

        /* Check recovered ~ v */
        for (j = 0; j < size; j++) {
            if (!lmmc_test_nearly_equal(recovered.data[j], v.data[j], eps)) {
                printf("    Exp-log round-trip failed at iter %d, elem %zu: got %g, expected %g\n",
                       i, j, recovered.data[j], v.data[j]);
                lmmc_vec_destroy(&v); lmmc_vec_destroy(&after_exp); lmmc_vec_destroy(&recovered);
                return 1;
            }
        }

        lmmc_vec_destroy(&v);
        lmmc_vec_destroy(&after_exp);
        lmmc_vec_destroy(&recovered);
    }
    return 0;
}

static int test_property17_vec_apply_inplace_equivalence(void)
{
    int i;
    double eps = 1e-15;

    for (i = 0; i < PBT_ITERATIONS; i++) {
        size_t size = (size_t)(rand() % 20) + 1;
        size_t j;
        lmmc_vec_t v_inplace, v_copy, out_of_place;
        lmmc_status_t st;

        st = lmmc_vec_create(size, &v_inplace);
        if (st != LMMC_STATUS_OK) { printf("    vec_create(v_inplace) failed at iter %d\n", i); return 1; }
        st = lmmc_vec_create(size, &v_copy);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&v_inplace); printf("    vec_create(v_copy) failed at iter %d\n", i); return 1; }
        st = lmmc_vec_create(size, &out_of_place);
        if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&v_inplace); lmmc_vec_destroy(&v_copy); printf("    vec_create(out_of_place) failed at iter %d\n", i); return 1; }

        /* Fill both copies with the same random values */
        for (j = 0; j < size; j++) {
            double val = rand_double(-10.0, 10.0);
            v_inplace.data[j] = val;
            v_copy.data[j] = val;
        }

        /* Apply sin in-place: v_inplace = sin(v_inplace) */
        st = lmmc_vec_apply(&v_inplace, sin, &v_inplace);
        if (st != LMMC_STATUS_OK) {
            printf("    vec_apply in-place failed at iter %d, status=%d\n", i, (int)st);
            lmmc_vec_destroy(&v_inplace); lmmc_vec_destroy(&v_copy); lmmc_vec_destroy(&out_of_place);
            return 1;
        }

        /* Apply sin out-of-place: out_of_place = sin(v_copy) */
        st = lmmc_vec_apply(&v_copy, sin, &out_of_place);
        if (st != LMMC_STATUS_OK) {
            printf("    vec_apply out-of-place failed at iter %d, status=%d\n", i, (int)st);
            lmmc_vec_destroy(&v_inplace); lmmc_vec_destroy(&v_copy); lmmc_vec_destroy(&out_of_place);
            return 1;
        }

        /* Check in-place result matches out-of-place result */
        for (j = 0; j < size; j++) {
            if (!lmmc_test_nearly_equal(v_inplace.data[j], out_of_place.data[j], eps)) {
                printf("    In-place vs out-of-place mismatch at iter %d, elem %zu: in-place=%g, out-of-place=%g\n",
                       i, j, v_inplace.data[j], out_of_place.data[j]);
                lmmc_vec_destroy(&v_inplace); lmmc_vec_destroy(&v_copy); lmmc_vec_destroy(&out_of_place);
                return 1;
            }
        }

        lmmc_vec_destroy(&v_inplace);
        lmmc_vec_destroy(&v_copy);
        lmmc_vec_destroy(&out_of_place);
    }
    return 0;
}

/* ---- Unit Tests ---- */

/**
 * Unit test: NULL func should return LMMC_STATUS_INVALID_ARGUMENT.
 */
static int test_unit_null_func(void)
{
    lmmc_vec_t v, out;
    lmmc_status_t st;

    st = lmmc_vec_create(3, &v);
    if (st != LMMC_STATUS_OK) return 1;
    st = lmmc_vec_create(3, &out);
    if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&v); return 1; }

    v.data[0] = 1.0; v.data[1] = 2.0; v.data[2] = 3.0;

    st = lmmc_vec_apply(&v, NULL, &out);
    if (st != LMMC_STATUS_INVALID_ARGUMENT) {
        printf("    Expected LMMC_STATUS_INVALID_ARGUMENT for NULL func, got %d\n", (int)st);
        lmmc_vec_destroy(&v); lmmc_vec_destroy(&out);
        return 1;
    }

    lmmc_vec_destroy(&v);
    lmmc_vec_destroy(&out);
    return 0;
}

/**
 * Unit test: Dimension mismatch between input and output vectors.
 */
static int test_unit_dimension_mismatch(void)
{
    lmmc_vec_t v, out;
    lmmc_status_t st;

    st = lmmc_vec_create(3, &v);
    if (st != LMMC_STATUS_OK) return 1;
    st = lmmc_vec_create(5, &out);
    if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&v); return 1; }

    v.data[0] = 1.0; v.data[1] = 2.0; v.data[2] = 3.0;

    st = lmmc_vec_apply(&v, sin, &out);
    if (st != LMMC_STATUS_DIMENSION_MISMATCH) {
        printf("    Expected LMMC_STATUS_DIMENSION_MISMATCH, got %d\n", (int)st);
        lmmc_vec_destroy(&v); lmmc_vec_destroy(&out);
        return 1;
    }

    lmmc_vec_destroy(&v);
    lmmc_vec_destroy(&out);
    return 0;
}

/**
 * Unit test: Known values — apply sin to [0, pi/2, pi] should give [0, 1, 0].
 */
static int test_unit_known_values_sin(void)
{
    lmmc_vec_t v, out;
    lmmc_status_t st;
    double eps = 1e-12;

    st = lmmc_vec_create(3, &v);
    if (st != LMMC_STATUS_OK) return 1;
    st = lmmc_vec_create(3, &out);
    if (st != LMMC_STATUS_OK) { lmmc_vec_destroy(&v); return 1; }

    v.data[0] = 0.0;
    v.data[1] = LMMC_CONST_PI / 2.0;
    v.data[2] = LMMC_CONST_PI;

    st = lmmc_vec_apply_sin(&v, &out);
    if (st != LMMC_STATUS_OK) {
        printf("    vec_apply_sin failed, status=%d\n", (int)st);
        lmmc_vec_destroy(&v); lmmc_vec_destroy(&out);
        return 1;
    }

    /* sin(0) = 0 */
    if (!lmmc_test_nearly_equal(out.data[0], 0.0, eps)) {
        printf("    sin(0) = %g, expected 0\n", out.data[0]);
        lmmc_vec_destroy(&v); lmmc_vec_destroy(&out);
        return 1;
    }

    /* sin(pi/2) = 1 */
    if (!lmmc_test_nearly_equal(out.data[1], 1.0, eps)) {
        printf("    sin(pi/2) = %g, expected 1\n", out.data[1]);
        lmmc_vec_destroy(&v); lmmc_vec_destroy(&out);
        return 1;
    }

    /* sin(pi) = 0 */
    if (!lmmc_test_nearly_equal(out.data[2], 0.0, eps)) {
        printf("    sin(pi) = %g, expected 0\n", out.data[2]);
        lmmc_vec_destroy(&v); lmmc_vec_destroy(&out);
        return 1;
    }

    lmmc_vec_destroy(&v);
    lmmc_vec_destroy(&out);
    return 0;
}

/* ---- Main ---- */

int main(void)
{
    srand(12345);
    lmmc_init();

    printf("=== Vectorized Apply Property Tests ===\n");
    REPORT("Property 16: Vectorized apply exp-log round-trip",
           test_property16_vec_apply_exp_log_roundtrip());
    REPORT("Property 17: Vectorized apply in-place equivalence",
           test_property17_vec_apply_inplace_equivalence());

    printf("\n=== Vectorized Apply Unit Tests ===\n");
    REPORT("Unit: NULL func returns INVALID_ARGUMENT",
           test_unit_null_func());
    REPORT("Unit: Dimension mismatch",
           test_unit_dimension_mismatch());
    REPORT("Unit: Known values (sin of [0, pi/2, pi])",
           test_unit_known_values_sin());

    printf("\n=== Results: %d/%d passed ===\n",
           test_count - test_failures, test_count);

    lmmc_deinit();
    return test_failures > 0 ? 1 : 0;
}
