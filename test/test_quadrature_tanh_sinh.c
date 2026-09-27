#include <math.h>
#include <stdio.h>

#include "lmmc/quadrature.h"
#include "test_common.h"

static lmmc_real_t fn_exp(lmmc_real_t x, void *data) {
    (void)data;
    return exp(x);
}

static lmmc_real_t fn_inv_sqrt(lmmc_real_t x, void *data) {
    (void)data;
    return x <= 0.0 ? INFINITY : 1.0 / sqrt(x);
}

static lmmc_real_t fn_nan_midpoint(lmmc_real_t x, void *data) {
    (void)data;
    return fabs(x - 0.5) < 1e-15 ? NAN : 1.0;
}

typedef struct {
    lmmc_real_t a;
    lmmc_real_t b;
    size_t endpoint_calls;
} endpoint_probe_t;

static lmmc_real_t fn_endpoint_probe(lmmc_real_t x, void *data) {
    endpoint_probe_t *probe = (endpoint_probe_t *)data;
    if (x == probe->a || x == probe->b) {
        ++probe->endpoint_calls;
        return INFINITY;
    }
    return 1.0;
}

static lmmc_real_t fn_counted_exp(lmmc_real_t x, void *data) {
    size_t *calls = (size_t *)data;
    ++*calls;
    return exp(x);
}

typedef struct {
    size_t calls;
    size_t centers;
    size_t first_layer_calls;
    size_t endpoint_calls;
    int fail_second_center;
} tanh_sinh_budget_probe_t;

static lmmc_real_t fn_counted_one(lmmc_real_t x, void *data) {
    tanh_sinh_budget_probe_t *probe = (tanh_sinh_budget_probe_t *)data;
    if (x == 0.5) {
        if (probe->centers == 1) {
            probe->first_layer_calls = probe->calls;
        }
        ++probe->centers;
    }
    ++probe->calls;
    if (x == 0.0 || x == 1.0) {
        ++probe->endpoint_calls;
    }
    if (probe->fail_second_center && x == 0.5 && probe->centers == 2) {
        return NAN;
    }
    return 1.0;
}

static int result_is_uninitialized(const lmmc_quad_result_t *result) {
    return result->value == 0.0 && isinf(result->error) &&
           result->error >= 0.0;
}

static void test_tanh_sinh_singular(void **state) {
    (void)state;
    lmmc_quad_result_t result;
    const lmmc_real_t exact = 2.0;
    const lmmc_status_t status = lmmc_quad_tanh_sinh(
        fn_inv_sqrt, NULL, 0.0, 1.0, 1e-10, 100000, &result);
    printf("Tanh-Sinh 1/sqrt(x): status=%d, value=%.15f, exact=%.15f, error=%.2e\n",
           status, result.value, exact, fabs(result.value - exact));
    assert_true(status == LMMC_STATUS_OK ||
                status == LMMC_STATUS_CONVERGENCE_FAILED);
    assert_true(lmmc_test_nearly_equal(result.value, exact, 1e-8));
}

static void test_tanh_sinh_endpoints(void **state) {
    (void)state;
    lmmc_quad_result_t result;
    endpoint_probe_t probe = {0.0, 1.0, 0};
    const lmmc_status_t status = lmmc_quad_tanh_sinh(
        fn_endpoint_probe, &probe, probe.a, probe.b,
        1e-12, 100000, &result);
    assert_int_equal(status, LMMC_STATUS_OK);
    assert_int_equal(probe.endpoint_calls, 0);
    assert_true(lmmc_test_nearly_equal(result.value, 1.0, 1e-9));
}

static void test_tanh_sinh_exp(void **state) {
    (void)state;
    lmmc_quad_result_t result;
    const lmmc_real_t exact = exp(1.0) - 1.0;
    const lmmc_status_t status = lmmc_quad_tanh_sinh(
        fn_exp, NULL, 0.0, 1.0, 1e-12, 100000, &result);
    printf("Tanh-Sinh exp: status=%d, value=%.15f, exact=%.15f, error=%.2e\n",
           status, result.value, exact, fabs(result.value - exact));
    assert_int_equal(status, LMMC_STATUS_OK);
    assert_true(lmmc_test_nearly_equal(result.value, exact, 1e-9));
}

static void test_tanh_sinh_small_budgets(void **state) {
    (void)state;
    lmmc_quad_result_t result;
    size_t calls = 0;
    lmmc_status_t status = lmmc_quad_tanh_sinh(
        fn_counted_exp, &calls, 0.0, 1.0, 1e-12, 2, &result);
    assert_int_equal(status, LMMC_STATUS_CONVERGENCE_FAILED);
    assert_true(result.num_evals <= 2);
    assert_int_equal(calls, result.num_evals);
    assert_true(result_is_uninitialized(&result));

    status = lmmc_quad_tanh_sinh(
        fn_exp, NULL, 0.0, 1.0, 1e-12, 1, &result);
    assert_int_equal(status, LMMC_STATUS_CONVERGENCE_FAILED);
    assert_true(result.num_evals <= 1);
    assert_true(result_is_uninitialized(&result));

    status = lmmc_quad_tanh_sinh(
        fn_nan_midpoint, NULL, 0.0, 1.0, 1e-12, 100, &result);
    assert_int_equal(status, LMMC_STATUS_NUMERICAL_FAILURE);
    assert_int_equal(result.num_evals, 1);
    assert_true(result_is_uninitialized(&result));
}

static void test_tanh_sinh_counted_budgets(void **state) {
    (void)state;
    lmmc_quad_result_t result;
    for (size_t budget = 1; budget <= 2; ++budget) {
        tanh_sinh_budget_probe_t probe = {0};
        const lmmc_status_t status = lmmc_quad_tanh_sinh(
            fn_counted_one, &probe, 0.0, 1.0, 0.02, budget, &result);
        assert_int_equal(status, LMMC_STATUS_CONVERGENCE_FAILED);
        assert_true(result.value == 0.0 && isinf(result.error) &&
                    result.error > 0.0);
        assert_int_equal(result.num_evals, probe.calls);
        assert_true(probe.calls <= budget);
    }
}

static void check_completed_layer(size_t first_budget,
                                  size_t second_budget,
                                  const lmmc_quad_result_t *first_result) {
    lmmc_quad_result_t result;
    tanh_sinh_budget_probe_t complete = {0};
    lmmc_status_t status = lmmc_quad_tanh_sinh(
        fn_counted_one, &complete, 0.0, 1.0, 0.02,
        second_budget, &result);
    assert_int_equal(status, LMMC_STATUS_OK);
    assert_int_equal(complete.calls, second_budget);
    assert_int_equal(result.num_evals, complete.calls);
    assert_int_equal(complete.endpoint_calls, 0);
    assert_true(fabs(result.value - 1.0) <= 0.02);
    assert_true(isfinite(result.error) && result.error <= 0.02);

    tanh_sinh_budget_probe_t failing = {0};
    failing.fail_second_center = 1;
    status = lmmc_quad_tanh_sinh(
        fn_counted_one, &failing, 0.0, 1.0, 0.02,
        second_budget, &result);
    assert_int_equal(status, LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(result.value == first_result->value);
    assert_int_equal(result.num_evals, failing.calls);
    assert_int_equal(failing.calls, first_budget + 1);
    assert_true(isinf(result.error) && result.error > 0.0);
}

static void test_tanh_sinh_layer_budgets(void **state) {
    (void)state;
    lmmc_quad_result_t result;
    tanh_sinh_budget_probe_t reference = {0};
    lmmc_status_t status = lmmc_quad_tanh_sinh(
        fn_counted_one, &reference, 0.0, 1.0, 0.02, 100000, &result);
    assert_int_equal(status, LMMC_STATUS_OK);
    assert_int_equal(reference.centers, 2);
    assert_int_equal(reference.endpoint_calls, 0);
    assert_int_equal(result.num_evals, reference.calls);
    assert_true(fabs(result.value - 1.0) <= 0.02);

    const size_t first_budget = reference.first_layer_calls;
    const size_t second_budget = reference.calls;
    assert_true(first_budget > 0 && second_budget > first_budget);

    tanh_sinh_budget_probe_t first = {0};
    lmmc_quad_result_t first_result;
    status = lmmc_quad_tanh_sinh(
        fn_counted_one, &first, 0.0, 1.0, 0.02,
        first_budget, &first_result);
    assert_int_equal(status, LMMC_STATUS_CONVERGENCE_FAILED);
    assert_int_equal(first.calls, first_budget);
    assert_int_equal(first_result.num_evals, first.calls);
    assert_true(fabs(first_result.value - 1.0) <= 0.02);
    assert_true(isinf(first_result.error) && first_result.error > 0.0);

    tanh_sinh_budget_probe_t partial = {0};
    status = lmmc_quad_tanh_sinh(
        fn_counted_one, &partial, 0.0, 1.0, 0.02,
        second_budget - 1, &result);
    assert_int_equal(status, LMMC_STATUS_CONVERGENCE_FAILED);
    assert_true(result.value == first_result.value);
    assert_int_equal(result.num_evals, partial.calls);
    assert_true(partial.calls < second_budget);
    assert_true(isinf(result.error) && result.error > 0.0);

    check_completed_layer(first_budget, second_budget, &first_result);
}

static void test_tanh_sinh_invalid_inputs(void **state) {
    (void)state;
    lmmc_quad_result_t result;
    const lmmc_real_t invalid_tolerances[] = {
        0.0, 1e-16, 0.2, NAN, INFINITY};
    for (size_t index = 0;
         index < sizeof(invalid_tolerances) / sizeof(invalid_tolerances[0]);
         ++index) {
        assert_int_equal(
            lmmc_quad_tanh_sinh(fn_exp, NULL, 0.0, 1.0,
                                invalid_tolerances[index], 1000, &result),
            LMMC_STATUS_INVALID_ARGUMENT);
    }
    assert_int_equal(
        lmmc_quad_tanh_sinh(fn_exp, NULL, 0.0, 1.0, 0.02, 0, &result),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(
        lmmc_quad_tanh_sinh(fn_exp, NULL, 0.0, 1.0, 0.02, 1000001, &result),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(
        lmmc_quad_tanh_sinh(fn_exp, NULL, 1.0, 0.0, 0.02, 1000, &result),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(
        lmmc_quad_tanh_sinh(fn_exp, NULL, 1.0, 1.0, 0.02, 1000, &result),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(
        lmmc_quad_tanh_sinh(fn_exp, NULL, 0.0, INFINITY, 0.02, 1000, &result),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(
        lmmc_quad_tanh_sinh(NULL, NULL, 0.0, 1.0, 0.02, 1000, &result),
        LMMC_STATUS_INVALID_ARGUMENT);
    assert_int_equal(
        lmmc_quad_tanh_sinh(fn_exp, NULL, 0.0, 1.0, 0.02, 1000, NULL),
        LMMC_STATUS_INVALID_ARGUMENT);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_tanh_sinh_singular),
        cmocka_unit_test(test_tanh_sinh_endpoints),
        cmocka_unit_test(test_tanh_sinh_exp),
        cmocka_unit_test(test_tanh_sinh_small_budgets),
        cmocka_unit_test(test_tanh_sinh_counted_budgets),
        cmocka_unit_test(test_tanh_sinh_layer_budgets),
        cmocka_unit_test(test_tanh_sinh_invalid_inputs),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
