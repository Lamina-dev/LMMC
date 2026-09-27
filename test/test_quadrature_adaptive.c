#include <float.h>
#include <math.h>
#include <stdio.h>
#include "lmmc/quadrature.h"
#include "lmmc/init.h"
#include "internal_test_hooks.h"
#include "test_common.h"

static lmmc_real_t fn_cancellation(lmmc_real_t x, void *user_data) {
    (void)user_data;
    const lmmc_real_t square = x * x;
    return square * square * square + 10000.0 * x;
}

static lmmc_real_t fn_sixth(lmmc_real_t x, void *user_data) {
    (void)user_data;
    const lmmc_real_t square = x * x;
    return square * square * square;
}

static int result_unchanged(const lmmc_quad_result_t *result) {
    return result->value == 123.0 && result->error == 456.0 && result->num_evals == 789;
}

static void test_global_cancellation(void **state) {
    (void)state;
    lmmc_quad_result_t result = {123.0, 456.0, 789};
    const lmmc_status_t status = lmmc_quad_adaptive(
        fn_cancellation, NULL, -1.0, 1.0, 1e-12, 1e-6, 8, &result);
    const lmmc_real_t tolerance = fmax(1e-12, 1e-6 * fabs(result.value));
    assert_false(status != LMMC_STATUS_OK);
    assert_false(!isfinite(result.error) || result.error < 0.0 || result.error > tolerance);
    assert_true(lmmc_test_nearly_equal(result.value, 2.0 / 7.0, tolerance));
    assert_true(lmmc_test_nearly_equal(result.value, 2.0 / 7.0, result.error + 1e-11));
}

static void test_depth_warning(void **state) {
    (void)state;
    lmmc_quad_result_t result = {123.0, 456.0, 789};
    const lmmc_status_t status = lmmc_quad_adaptive(
        fn_cancellation, NULL, -1.0, 1.0, 1e-12, 1e-6, 0, &result);
    assert_false(status != LMMC_STATUS_WARNING_MAX_DEPTH);
    assert_false(!isfinite(result.value) || !isfinite(result.error));
    assert_false(result.error <= fmax(1e-12, 1e-6 * fabs(result.value)));
}

typedef struct {
    lmmc_real_t samples[1025];
    size_t count;
    int duplicate;
} sample_probe_t;

static lmmc_real_t fn_unique_exp(lmmc_real_t x, void *user_data) {
    sample_probe_t *probe = (sample_probe_t *)user_data;
    for (size_t i = 0; i < probe->count; ++i) {
        if (probe->samples[i] == x) {
            probe->duplicate = 1;
            return NAN;
        }
    }
    if (probe->count == 1025)
        return NAN;
    probe->samples[probe->count++] = x;
    return exp(10.0 * x);
}

static void test_blocked_leaves_and_cached_samples(void **state) {
    (void)state;
    sample_probe_t probe = {{0}, 0, 0};
    lmmc_quad_result_t result;
    const lmmc_status_t status = lmmc_quad_adaptive(
        fn_unique_exp, &probe, 0.0, 1.0, 1e-12, 0.0, 2, &result);
    assert_false(status != LMMC_STATUS_WARNING_MAX_DEPTH || probe.duplicate);
    assert_false(probe.count != 17 || result.num_evals != probe.count);
    assert_false(!isfinite(result.value) || !isfinite(result.error) || result.error <= 1e-12);
}

static lmmc_real_t fn_late_nan(lmmc_real_t x, void *user_data) {
    size_t *count = (size_t *)user_data;
    ++*count;
    return *count == 8 ? NAN : exp(x);
}

static void test_callback_failure(void **state) {
    (void)state;
    size_t calls = 0;
    lmmc_quad_result_t result = {123.0, 456.0, 789};
    const lmmc_status_t status = lmmc_quad_adaptive(
        fn_late_nan, &calls, 0.0, 1.0, 1e-14, 0.0, 20, &result);
    assert_false(status != LMMC_STATUS_NUMERICAL_FAILURE || !result_unchanged(&result));
}

static lmmc_real_t fn_nonfinite(lmmc_real_t x, void *user_data) {
    (void)x;
    return *(const lmmc_real_t *)user_data;
}

static void test_initial_nonfinite(void **state) {
    (void)state;
    const lmmc_real_t values[] = {NAN, INFINITY, -INFINITY, DBL_MAX};
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        lmmc_quad_result_t result = {123.0, 456.0, 789};
        const lmmc_status_t status = lmmc_quad_adaptive(
            fn_nonfinite, (void *)&values[i], 0.0, 1.0, 1e-12, 0.0, 8, &result);
        assert_false(status != LMMC_STATUS_NUMERICAL_FAILURE || !result_unchanged(&result));
    }
}

static void test_unrepresentable_midpoint(void **state) {
    (void)state;
    lmmc_quad_result_t result = {123.0, 456.0, 789};
    const lmmc_status_t status = lmmc_quad_adaptive(
        fn_sixth, NULL, 1.0, nextafter(1.0, 2.0), 1e-12, 0.0, SIZE_MAX, &result);
    assert_false(status != LMMC_STATUS_NUMERICAL_FAILURE || !result_unchanged(&result));
}

static void test_large_depth_bound(void **state) {
    (void)state;
    lmmc_quad_result_t result;
    const lmmc_status_t status = lmmc_quad_adaptive(
        fn_sixth, NULL, 0.0, 1.0, 1e-10, 0.0, SIZE_MAX, &result);
    assert_false(status != LMMC_STATUS_OK ||
                 !lmmc_test_nearly_equal(result.value, 1.0 / 7.0, 1e-10));
}

static void test_allocation_failure(void **state) {
    (void)state;
    const size_t successful_allocations = *(const size_t *)*state;
    lmmc_quad_result_t result = {123.0, 456.0, 789};
    lmmc_status_t status;
#ifdef LMMC_DEBUG_LEAKS
    const size_t live_before = lmmc_debug_leaks_get_count();
#endif
    lmmc_memory_fail_after_for_test(successful_allocations);
    status = lmmc_quad_adaptive(
        fn_sixth, NULL, 0.0, 1.0, 1e-14, 0.0, 10, &result);
    lmmc_memory_fail_reset_for_test();
    assert_false(status != LMMC_STATUS_ALLOCATION_FAILED || !result_unchanged(&result));
#ifdef LMMC_DEBUG_LEAKS
    assert_false(lmmc_debug_leaks_get_count() != live_before);
#endif
    status = lmmc_quad_adaptive(fn_sixth, NULL, 0.0, 1.0, 1e-10, 0.0, 10, &result);
    assert_false(status != LMMC_STATUS_OK ||
                 !lmmc_test_nearly_equal(result.value, 1.0 / 7.0, 1e-10));
}

int main(void) {
    static const size_t allocation_counts[] = {0, 1, 2};
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_global_cancellation),
        cmocka_unit_test(test_depth_warning),
        cmocka_unit_test(test_blocked_leaves_and_cached_samples),
        cmocka_unit_test(test_callback_failure),
        cmocka_unit_test(test_initial_nonfinite),
        cmocka_unit_test(test_unrepresentable_midpoint),
        cmocka_unit_test(test_large_depth_bound),
        cmocka_unit_test_prestate(test_allocation_failure, (void *)&allocation_counts[0]),
        cmocka_unit_test_prestate(test_allocation_failure, (void *)&allocation_counts[1]),
        cmocka_unit_test_prestate(test_allocation_failure, (void *)&allocation_counts[2]),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
