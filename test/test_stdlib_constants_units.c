#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "test_stdlib_common.h"

typedef struct {
    const char *name;
    lmmc_real_t value;
    const char *unit;
} required_constant_t;

static void test_math_constants(void **state) {
    (void)state;
    lmmc_real_t out = 0;
    lmmc_real_t out2 = 0;
    lmmc_complex_t z;

    assert_true(lmmc_std_math_pi(&out) == LMMC_STATUS_OK);
    assert_true(close_real(out, LMMC_CONST_PI));
    assert_true(lmmc_std_math_e(&out) == LMMC_STATUS_OK);
    assert_true(close_real(out, (lmmc_real_t)2.71828182845904523536));
    assert_true(lmmc_std_math_phi(&out2) == LMMC_STATUS_OK);
    assert_true(close_real(out2, (lmmc_real_t)1.61803398874989484820));

    assert_true(lmmc_std_math_i(&z) == LMMC_STATUS_OK);
    assert_true(close_real(z.real, 0));
    assert_true(close_real(z.imag, 1));
    assert_true(lmmc_std_math_pi(NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_e(NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_phi(NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_math_i(NULL) == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_named_constant_values(void **state) {
    (void)state;
    lmmc_real_t out = 0;

    assert_true(lmmc_std_constants_get("EARTH_GRAVITY", &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 9.80665));
    assert_true(strcmp(lmmc_std_constants_unit("EARTH_GRAVITY"), "m*s^-2") == 0);

    assert_true(lmmc_std_constants_get("C", &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 2.99792458e8));
    assert_true(strcmp(lmmc_std_constants_unit("C"), "m*s^-1") == 0);

    assert_true(lmmc_std_constants_get("AVOGADRO", &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 6.02214076e23));
    assert_true(strcmp(lmmc_std_constants_unit("AVOGADRO"), "mol^-1") == 0);

    assert_true(lmmc_std_constants_get("NO_SUCH_CONSTANT", &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_constants_unit("NO_SUCH_CONSTANT") == NULL);
}

static void test_constant_lookup_errors(void **state) {
    (void)state;
    lmmc_real_t out = 0;
    const char *constant_name = NULL;
    const char *constant_unit = NULL;
    lmmc_real_t constant_value = 0;

    assert_true(lmmc_std_constants_get(NULL, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_constants_get("C", NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_constants_unit(NULL) == NULL);
    assert_true(lmmc_std_constants_entry(
                    lmmc_std_constants_count(),
                    &constant_name,
                    &constant_value,
                    &constant_unit) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_constants_entry(0, NULL, &constant_value, &constant_unit) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_constants_entry(0, &constant_name, NULL, &constant_unit) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_constants_entry(0, &constant_name, &constant_value, NULL) ==
                LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_constant_enumeration(void **state) {
    (void)state;
    lmmc_real_t out = 0;
    lmmc_real_t out2 = 0;
    const char *constant_name = NULL;
    const char *constant_unit = NULL;
    lmmc_real_t constant_value = 0;

    assert_true(lmmc_std_constants_name(0) != NULL);
    assert_true(lmmc_std_constants_name(lmmc_std_constants_count()) == NULL);
    for (size_t i = 0; i < lmmc_std_constants_count(); ++i) {
        const char *name = lmmc_std_constants_name(i);
        const char *unit = name ? lmmc_std_constants_unit(name) : NULL;
        assert_true(name);
        assert_true(unit);
        assert_true(lmmc_std_constants_entry(i, &constant_name, &constant_value, &constant_unit) ==
                    LMMC_STATUS_OK);
        assert_true(strcmp(constant_name, name) == 0);
        assert_true(strcmp(constant_unit, unit) == 0);
        assert_true(lmmc_std_constants_get(name, &out) == LMMC_STATUS_OK);
        assert_true(close_real(constant_value, out));
        assert_true(lmmc_std_units_convert(1, unit, unit, &out2) == LMMC_STATUS_OK);
        assert_true(close_real(out2, 1));
    }
}

static void test_required_constants(void **state) {
    (void)state;
    lmmc_real_t out = 0;

    const required_constant_t required_constants[] = {
        {"EARTH_GRAVITY", (lmmc_real_t)9.80665, "m*s^-2"},
        {"MOON_GRAVITY", (lmmc_real_t)1.625, "m*s^-2"},
        {"MARS_GRAVITY", (lmmc_real_t)3.72076, "m*s^-2"},
        {"WATER_DENSITY", (lmmc_real_t)1000.0, "kg*m^-3"},
        {"STANDARD_PRESSURE", (lmmc_real_t)101325.0, "Pa"},
        {"STANDARD_TEMPERATURE", (lmmc_real_t)273.15, "K"},
        {"AIR_DENSITY", (lmmc_real_t)1.225, "kg*m^-3"},
        {"C", (lmmc_real_t)2.99792458e8, "m*s^-1"},
        {"G", (lmmc_real_t)6.67430e-11, "m^3*kg^-1*s^-2"},
        {"H", (lmmc_real_t)6.62607015e-34, "J*s"},
        {"KB", (lmmc_real_t)1.380649e-23, "J*K^-1"},
        {"EPSILON_0", (lmmc_real_t)8.8541878128e-12, "F*m^-1"},
        {"MU_0", (lmmc_real_t)1.25663706212e-6, "H*m^-1"},
        {"AVOGADRO", (lmmc_real_t)6.02214076e23, "mol^-1"},
        {"R", (lmmc_real_t)8.314462618, "J*mol^-1*K^-1"},
        {"FARADAY", (lmmc_real_t)9.648533212e4, "C*mol^-1"},
        {"AMU", (lmmc_real_t)1.66053906660e-27, "kg"},
        {"MOLAR_VOLUME_IDEAL", (lmmc_real_t)0.024465, "m^3*mol^-1"},
        {"ROOM_PRESSURE", (lmmc_real_t)1.0e5, "Pa"},
        {"ROOM_TEMPERATURE", (lmmc_real_t)297.15, "K"},
    };
    assert_true(lmmc_std_constants_count() ==
                sizeof(required_constants) / sizeof(required_constants[0]));
    for (size_t i = 0; i < sizeof(required_constants) /
                               sizeof(required_constants[0]);
         ++i) {
        const char *unit =
            lmmc_std_constants_unit(required_constants[i].name);
        assert_true(lmmc_std_constants_get(required_constants[i].name, &out) == LMMC_STATUS_OK);
        assert_true(close_real(out, required_constants[i].value));
        assert_true(unit);
        assert_true(strcmp(unit, required_constants[i].unit) == 0);
    }
}

static void test_unit_conversions(void **state) {
    (void)state;
    lmmc_real_t out = 0;

    assert_true(lmmc_std_units_convert(36, "km/h", "m/s", &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 10));
    assert_true(lmmc_std_units_convert(10, "km", "m", &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 10000));
    assert_true(lmmc_std_units_convert(10, "km", "km", &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 10));
    assert_true(lmmc_std_units_convert(2, "kg", "g", &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 2000));
    assert_true(lmmc_std_units_convert(3600, "s", "h", &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 1));
    assert_true(lmmc_std_units_convert(1, "N", "kg*m*s^-2", &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 1));
    assert_true(lmmc_std_units_convert(1, "J", "N*m", &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 1));
    assert_true(lmmc_std_units_convert_from_si(10000, "km", &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 10));
    assert_true(lmmc_std_units_convert_num(10000, "km", &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 10));
    assert_true(lmmc_std_units_convert_from_si(10, "km", &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 0.01));
}

static void test_unit_dimensions_and_stripping(void **state) {
    (void)state;
    lmmc_real_t out = 0;
    int dimensionless = 0;

    assert_true(lmmc_std_units_is_dimensionless("1", &dimensionless) == LMMC_STATUS_OK);
    assert_true(dimensionless);
    assert_true(lmmc_std_units_is_dimensionless_num(12.5, &dimensionless) == LMMC_STATUS_OK);
    assert_true(dimensionless);
    assert_true(lmmc_std_units_convert(10, "m", "s", &out) == LMMC_STATUS_DIMENSION_MISMATCH);
    assert_true(lmmc_std_units_strip(12.5, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 12.5));
    assert_true(lmmc_std_units_strip_num(10, "km", &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 10000));
    assert_true(lmmc_std_units_strip_num(10, "score", &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 10));
    assert_true(lmmc_std_units_strip_scalar(10, &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 10));
    assert_true(lmmc_std_units_convert(10, "score", "score", &out) == LMMC_STATUS_OK);
    assert_true(close_real(out, 10));
    assert_true(lmmc_std_units_convert(10, "score", "token", &out) == LMMC_STATUS_DIMENSION_MISMATCH);
    assert_true(lmmc_std_units_is_dimensionless("score", &dimensionless) == LMMC_STATUS_OK);
    assert_true(!dimensionless);
    assert_true(lmmc_std_units_is_dimensionless("score/score", &dimensionless) == LMMC_STATUS_OK);
    assert_true(dimensionless);
    assert_true(lmmc_std_units_is_dimensionless("m/m", &dimensionless) == LMMC_STATUS_OK);
    assert_true(dimensionless);
    assert_true(lmmc_std_units_is_dimensionless("m*s^-1", &dimensionless) == LMMC_STATUS_OK);
    assert_true(!dimensionless);
}

static void test_unit_grammar(void **state) {
    (void)state;
    lmmc_real_t out = 0;

    assert_true(lmmc_std_units_convert(1, "m^32", "m^32", &out) == LMMC_STATUS_OK);
    assert_true(lmmc_std_units_convert(1, "m^32*m", "m", &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_convert(1, "m^33", "m^33", &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_convert(1, "m^999999999999", "m", &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_convert(1, "m/", "m", &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_convert(1, "m*", "m", &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_convert(1, "unknown", "m", &out) == LMMC_STATUS_INVALID_ARGUMENT);
}

static void test_unit_nonfinite_inputs(void **state) {
    (void)state;
    lmmc_real_t out = 0;
    int dimensionless = 0;

    assert_true(lmmc_std_units_convert(INFINITY, "m", "m", &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_units_convert_from_si(INFINITY, "m", &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_units_convert_num(INFINITY, "m", &out) == LMMC_STATUS_NUMERICAL_FAILURE);
    assert_true(lmmc_std_units_strip(INFINITY, &out) == LMMC_STATUS_UNIT_STRIP_OVERFLOW);
    assert_true(lmmc_std_units_strip_num(INFINITY, "m", &out) == LMMC_STATUS_UNIT_STRIP_OVERFLOW);
    assert_true(lmmc_std_units_strip_scalar(INFINITY, &out) == LMMC_STATUS_UNIT_STRIP_OVERFLOW);
    assert_true(lmmc_std_units_is_dimensionless_num(INFINITY, &dimensionless) ==
                LMMC_STATUS_NUMERICAL_FAILURE);
}

static void test_unit_invalid_arguments(void **state) {
    (void)state;
    lmmc_real_t out = 0;
    int dimensionless = 0;

    assert_true(lmmc_std_units_convert(1, NULL, "m", &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_convert(1, "m", NULL, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_convert(1, "m", "m", NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_convert_from_si(1, NULL, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_convert_from_si(1, "unknown", &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_convert_from_si(1, "m", NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_convert_num(1, NULL, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_convert_num(1, "unknown", &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_convert_num(1, "m", NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_strip(1, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_strip_num(1, NULL, &out) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_strip_num(1, "m", NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_strip_num(1, "unknown", &out) == LMMC_STATUS_UNIT_STRIP_INVALID);
    assert_true(lmmc_std_units_strip_num(1, "num<m>", &out) == LMMC_STATUS_UNIT_STRIP_LEGACY_SYNTAX);
    assert_true(lmmc_std_units_strip_scalar(1, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_is_dimensionless_num(1, NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_is_dimensionless(NULL, &dimensionless) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_is_dimensionless("m", NULL) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_units_is_dimensionless("unknown", &dimensionless) ==
                LMMC_STATUS_INVALID_ARGUMENT);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_math_constants),
        cmocka_unit_test(test_named_constant_values),
        cmocka_unit_test(test_constant_lookup_errors),
        cmocka_unit_test(test_constant_enumeration),
        cmocka_unit_test(test_required_constants),
        cmocka_unit_test(test_unit_conversions),
        cmocka_unit_test(test_unit_dimensions_and_stripping),
        cmocka_unit_test(test_unit_grammar),
        cmocka_unit_test(test_unit_nonfinite_inputs),
        cmocka_unit_test(test_unit_invalid_arguments),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
