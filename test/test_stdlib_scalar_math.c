#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "test_stdlib_common.h"

static void test_error_names(void **state) {
    (void)state;
    static const struct {
        lmmc_status_t status;
        const char *name;
    } cases[] = {
        {LMMC_STATUS_OK, "Ok"},
        {LMMC_STATUS_INVALID_ARGUMENT, "InvalidArgument"},
        {LMMC_STATUS_DIMENSION_MISMATCH, "DimensionMismatch"},
        {LMMC_STATUS_ALLOCATION_FAILED, "ResourceLimit"},
        {LMMC_STATUS_SINGULAR_MATRIX, "SingularMatrix"},
        {LMMC_STATUS_NOT_IMPLEMENTED, "UnsupportedExpression"},
        {LMMC_STATUS_NUMERICAL_FAILURE, "NumericFailure"},
        {LMMC_STATUS_NOT_POSITIVE_DEFINITE, "DomainError"},
        {LMMC_STATUS_CONVERGENCE_FAILED, "NumericFailure"},
        {LMMC_STATUS_OUT_OF_RANGE, "DomainError"},
        {LMMC_STATUS_INDEX_OUT_OF_BOUNDS, "InvalidArgument"},
        {LMMC_STATUS_WARNING_MAX_DEPTH, "ResourceLimit"},
        {LMMC_STATUS_EMPTY_INPUT, "EmptyInput"},
        {LMMC_STATUS_UNIT_STRIP_TYPE_MISMATCH, "UnitStripTypeMismatch"},
        {LMMC_STATUS_UNIT_STRIP_OVERFLOW, "UnitStripOverflow"},
        {LMMC_STATUS_UNIT_STRIP_INVALID, "UnitStripInvalid"},
        {LMMC_STATUS_UNIT_STRIP_LEGACY_SYNTAX, "UnitStripLegacySyntax"},
        {LMMC_STATUS_NOT_INITIALIZED, "NotInitialized"},
        {LMMC_STATUS_BUSY, "Busy"},
        {LMMC_STATUS_REFERENCE_LIMIT, "ResourceLimit"},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        assert_string_equal(lmmc_std_error_name(cases[i].status),
                            cases[i].name);
    }
}

static void test_domain_errors(void **state) {
    (void)state;
    lmmc_real_t out = 0;

    assert_true(lmmc_std_math_sqrt(-1, &out) == LMMC_STATUS_OUT_OF_RANGE);
    assert_true(lmmc_std_math_log(-1, &out) == LMMC_STATUS_OUT_OF_RANGE);
    assert_true(lmmc_std_math_log10(0, &out) == LMMC_STATUS_OUT_OF_RANGE);
    assert_true(lmmc_std_math_log_base(8, 1, &out) == LMMC_STATUS_OUT_OF_RANGE);
    assert_true(lmmc_std_math_asin(2, &out) == LMMC_STATUS_OUT_OF_RANGE);
    assert_true(lmmc_std_math_acos(-2, &out) == LMMC_STATUS_OUT_OF_RANGE);
    assert_true(lmmc_std_math_pow(0, -1, &out) == LMMC_STATUS_OUT_OF_RANGE);
    assert_true(lmmc_std_math_pow(-1, 0.5, &out) == LMMC_STATUS_OUT_OF_RANGE);
}

static void test_logarithms_powers_overflow(void **state) {
    (void)state;
    lmmc_real_t out = 0;

    assert_true(lmmc_std_math_log(exp(1.0), &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 1));

    assert_true(lmmc_std_math_log_base(8, 2, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 3));

    assert_true(lmmc_std_math_log10(1000, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 3));

    assert_true(lmmc_std_math_pow(2, 3, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 8));

    assert_true(lmmc_std_math_exp(1000, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
}

static void test_invalid_arguments(void **state) {
    (void)state;
    lmmc_real_t out = 0;

    assert_true(lmmc_std_math_sin(0, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_cos(0, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_tan(0, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_pow(2, 3, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_asin(0, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_acos(0, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_atan(0, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_sqrt(4, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_exp(1, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_ln(1, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_log(1, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_log_base(8, 2, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_log10(10, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_abs(-1, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_floor(1, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_ceil(1, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_round(1, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_clamp(1, 0, 2, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_clamp(2, 3, 1, &out) == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_clamping(void **state) {
    (void)state;
    lmmc_real_t out = 0;

    assert_true(lmmc_std_math_clamp(5, 1, 3, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 3));
}

static void test_nonfinite_inputs(void **state) {
    (void)state;
    lmmc_real_t out = 0;

    assert_true(lmmc_std_math_asin(NAN, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_acos(NAN, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_atan(INFINITY, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_sin(INFINITY, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_cos(INFINITY, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_tan(INFINITY, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_pow(INFINITY, 0, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_sqrt(NAN, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_exp(INFINITY, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_abs(INFINITY, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_ln(INFINITY, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_log10(INFINITY, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_log_base(INFINITY, 2, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_log_base(8, NAN, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_floor(INFINITY, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_ceil(INFINITY, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_round(INFINITY, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_clamp(NAN, 1, 3, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_clamp(2, NAN, 3, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_math_clamp(2, 1, INFINITY, &out) == LMMC_STATUS_NUMERICAL_FAILURE);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_error_names),
        cmocka_unit_test(test_domain_errors),
        cmocka_unit_test(test_logarithms_powers_overflow),
        cmocka_unit_test(test_invalid_arguments),
        cmocka_unit_test(test_clamping),
        cmocka_unit_test(test_nonfinite_inputs),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
