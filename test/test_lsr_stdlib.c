#include "lmmc/lmmc.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int close_real(lmmc_real_t a, lmmc_real_t b)
{
    return fabs((double)(a - b)) <= 1e-12;
}

static void set_mat_values(lmmc_mat_t* mat, const lmmc_real_t* values)
{
    size_t k = 0;
    for (size_t i = 0; i < mat->rows; ++i) {
        for (size_t j = 0; j < mat->cols; ++j) {
            mat->data[i * mat->stride + j] = values[k++];
        }
    }
}

static int mat_close_at(const lmmc_mat_t* mat, size_t row, size_t col,
                        lmmc_real_t expected)
{
    return close_real(mat->data[row * mat->stride + col], expected);
}

static int num_set_has(const lmmc_lsr_num_set_t* set, lmmc_real_t value)
{
    int contains = 0;
    return lmmc_lsr_num_set_contains(set, value, &contains) ==
               LMMC_STATUS_OK &&
           contains;
}

static int complex_set_has(const lmmc_lsr_complex_set_t* set,
                           lmmc_real_t real,
                           lmmc_real_t imag)
{
    int contains = 0;
    lmmc_complex_t value = {real, imag};
    return lmmc_lsr_complex_set_contains(set, &value, &contains) ==
               LMMC_STATUS_OK &&
           contains;
}

static int bool_set_has(const lmmc_lsr_bool_set_t* set, int value)
{
    int contains = 0;
    return lmmc_lsr_bool_set_contains(set, value, &contains) ==
               LMMC_STATUS_OK &&
           contains;
}

static int text_set_has(const lmmc_lsr_text_set_t* set, const char* value)
{
    int contains = 0;
    return lmmc_lsr_text_set_contains(set, value, &contains) ==
               LMMC_STATUS_OK &&
           contains;
}

typedef struct {
    const char* name;
    lmmc_real_t value;
    const char* unit;
} required_constant_t;

int main(void)
{
    lmmc_real_t out = 0;
    lmmc_real_t out2 = 0;
    lmmc_real_t values[] = {1, 2, 3, 4};
    lmmc_real_t scaled_values[] = {2, 4, 6, 8};
    lmmc_real_t constant_values[] = {1, 1, 1, 1};
    lmmc_real_t single_value[] = {1};
    lmmc_real_t single_scaled_value[] = {2};
    lmmc_real_t nonfinite_values[] = {1, NAN, 3};
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t singular_values[] = {1, 2, 2, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_real_t vec_b_values[] = {4, 5, 6};
    lmmc_real_t vec_short_values[] = {1, 2};
    lmmc_real_t vec_nonfinite_values[] = {1, NAN, 3};
    lmmc_real_t vec_zero_values[] = {1, 0, 3};
    lmmc_real_t vec_pow_values[] = {2, 3, 4};
    lmmc_real_t vec_pow_exp_values[] = {3, 2, 0.5};
    lmmc_real_t vec_negative_values[] = {-1, 2, 3};
    lmmc_real_t vec_negative_exp_values[] = {0.5, 2, 3};
    lmmc_real_t vec_negative_integer_exp_values[] = {3, 2, 1};
    lmmc_real_t vec_zero_negative_exp_values[] = {2, -1, 1};
    lmmc_real_t set_a_values[] = {1, 2, 2, -0.0, 0.0};
    lmmc_real_t set_b_values[] = {2, 3, 0};
    lmmc_real_t set_bad_values[] = {1, NAN};
    lmmc_complex_t complex_set_a_values[] = {
        {1, 2}, {1, 2}, {-0.0, 0.0}, {3, 4}};
    lmmc_complex_t complex_set_b_values[] = {{3, 4}, {5, 6}, {0.0, -0.0}};
    lmmc_complex_t complex_set_bad_values[] = {{1, 2}, {NAN, 0}};
    int bool_set_a_values[] = {1, 1, 0};
    int bool_set_b_values[] = {0};
    int bool_set_bad_values[] = {1, 2};
    const char* text_set_a_values[] = {"alpha", "beta", "alpha"};
    const char* text_set_b_values[] = {"beta", "gamma"};
    const char* text_set_bad_values[] = {"alpha", NULL};
    lmmc_complex_t z;
    lmmc_complex_t w;
    lmmc_rng_t* rng = NULL;
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    lmmc_vec_t vec_b = {3, vec_b_values, 0};
    lmmc_vec_t vec_short = {2, vec_short_values, 0};
    lmmc_vec_t vec_nonfinite = {3, vec_nonfinite_values, 0};
    lmmc_vec_t vec_zero = {3, vec_zero_values, 0};
    lmmc_vec_t vec_pow = {3, vec_pow_values, 0};
    lmmc_vec_t vec_pow_exp = {3, vec_pow_exp_values, 0};
    lmmc_vec_t vec_negative = {3, vec_negative_values, 0};
    lmmc_vec_t vec_negative_exp = {3, vec_negative_exp_values, 0};
    lmmc_vec_t vec_negative_integer_exp = {3, vec_negative_integer_exp_values, 0};
    lmmc_vec_t vec_zero_negative_exp = {3, vec_zero_negative_exp_values, 0};
    lmmc_vec_t cross_result = {0};
    lmmc_vec_t matvec_result = {0};
    lmmc_vec_t shape_result = {0};
    lmmc_lsr_bool_vec_t bool_vec_result = {0};
    lmmc_lsr_bool_mat_t bool_mat_result = {0};
    lmmc_mat_t mat = {0};
    lmmc_mat_t singular = {0};
    lmmc_mat_t rhs = {0};
    lmmc_mat_t mismatched_rhs = {0};
    lmmc_mat_t rectangular = {0};
    lmmc_mat_t invalid_stride = {0};
    lmmc_mat_t result = {0};
    lmmc_eigen_gen_full_result_t eig_result = {0};
    lmmc_svd_result_t svd_result = {0};
    lmmc_lsr_eig_table_t eig_table = {0};
    lmmc_lsr_svd_table_t svd_table = {0};
    const lmmc_mat_t* named = NULL;
    size_t rows = 0;
    size_t cols = 0;
    size_t rank = 0;
    int64_t randint_out = 0;
    int64_t randint_out2 = 0;
    uint64_t complex_hash = 0;
    uint64_t complex_hash_same = 0;
    uint64_t complex_hash_other = 0;
    uint64_t num_hash = 0;
    uint64_t num_hash_same = 0;
    uint64_t num_hash_other = 0;
    uint64_t bool_hash = 0;
    uint64_t bool_hash_same = 0;
    uint64_t bool_hash_other = 0;
    uint64_t text_hash = 0;
    uint64_t text_hash_same = 0;
    uint64_t text_hash_other = 0;
    int complex_equal = 0;
    int num_equal = 0;
    int bool_equal = 0;
    int text_equal = 0;
    int set_contains = 0;
    int set_subset = 0;
    lmmc_lsr_num_set_t set_a = {0};
    lmmc_lsr_num_set_t set_b = {0};
    lmmc_lsr_num_set_t set_empty = {0};
    lmmc_lsr_num_set_t set_union = {0};
    lmmc_lsr_num_set_t set_intersection = {0};
    lmmc_lsr_num_set_t set_difference = {0};
    lmmc_lsr_num_set_t set_xor = {0};
    lmmc_lsr_complex_set_t complex_set_a = {0};
    lmmc_lsr_complex_set_t complex_set_b = {0};
    lmmc_lsr_complex_set_t complex_set_empty = {0};
    lmmc_lsr_complex_set_t complex_set_union = {0};
    lmmc_lsr_complex_set_t complex_set_intersection = {0};
    lmmc_lsr_complex_set_t complex_set_difference = {0};
    lmmc_lsr_complex_set_t complex_set_xor = {0};
    lmmc_lsr_bool_set_t bool_set_a = {0};
    lmmc_lsr_bool_set_t bool_set_b = {0};
    lmmc_lsr_bool_set_t bool_set_empty = {0};
    lmmc_lsr_bool_set_t bool_set_union = {0};
    lmmc_lsr_bool_set_t bool_set_intersection = {0};
    lmmc_lsr_bool_set_t bool_set_difference = {0};
    lmmc_lsr_bool_set_t bool_set_xor = {0};
    lmmc_lsr_text_set_t text_set_a = {0};
    lmmc_lsr_text_set_t text_set_b = {0};
    lmmc_lsr_text_set_t text_set_empty = {0};
    lmmc_lsr_text_set_t text_set_union = {0};
    lmmc_lsr_text_set_t text_set_intersection = {0};
    lmmc_lsr_text_set_t text_set_difference = {0};
    lmmc_lsr_text_set_t text_set_xor = {0};
    const char* constant_name = NULL;
    const char* constant_unit = NULL;
    lmmc_real_t constant_value = 0;
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

    if (lmmc_lsr_math_pi(&out) != LMMC_STATUS_OK ||
        !close_real(out, LMMC_CONST_PI)) {
        fprintf(stderr, "std.math.pi mismatch\n");
        return 1;
    }
    if (lmmc_lsr_math_e(&out) != LMMC_STATUS_OK ||
        !close_real(out, (lmmc_real_t)2.71828182845904523536) ||
        lmmc_lsr_math_phi(&out2) != LMMC_STATUS_OK ||
        !close_real(out2, (lmmc_real_t)1.61803398874989484820)) {
        fprintf(stderr, "std.math e/phi constants mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_i(&z) != LMMC_STATUS_OK ||
        !close_real(z.real, 0) || !close_real(z.imag, 1)) {
        fprintf(stderr, "std.math.i mismatch\n");
        return 1;
    }
    if (lmmc_lsr_math_pi(NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_e(NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_phi(NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_i(NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_I(NULL) != LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.math constant null outputs not rejected\n");
        return 1;
    }

    if (lmmc_lsr_constants_count() !=
            sizeof(required_constants) / sizeof(required_constants[0]) ||
        lmmc_lsr_constants_name(0) == NULL ||
        lmmc_lsr_constants_name(lmmc_lsr_constants_count()) != NULL) {
        fprintf(stderr, "std.constants enumeration mismatch\n");
        return 1;
    }

    if (lmmc_lsr_constants_get("EARTH_GRAVITY", &out) != LMMC_STATUS_OK ||
        !close_real(out, 9.80665) ||
        strcmp(lmmc_lsr_constants_unit("EARTH_GRAVITY"), "m*s^-2") != 0) {
        fprintf(stderr, "std.constants.EARTH_GRAVITY mismatch\n");
        return 1;
    }

    if (lmmc_lsr_constants_get("C", &out) != LMMC_STATUS_OK ||
        !close_real(out, 2.99792458e8) ||
        strcmp(lmmc_lsr_constants_unit("C"), "m*s^-1") != 0) {
        fprintf(stderr, "std.constants.C mismatch\n");
        return 1;
    }

    if (lmmc_lsr_constants_get("AVOGADRO", &out) != LMMC_STATUS_OK ||
        !close_real(out, 6.02214076e23) ||
        strcmp(lmmc_lsr_constants_unit("AVOGADRO"), "mol^-1") != 0) {
        fprintf(stderr, "std.constants.AVOGADRO mismatch\n");
        return 1;
    }

    if (lmmc_lsr_constants_get("NO_SUCH_CONSTANT", &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_constants_unit("NO_SUCH_CONSTANT") != NULL) {
        fprintf(stderr, "std.constants unknown name mismatch\n");
        return 1;
    }
    if (lmmc_lsr_constants_get(NULL, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_constants_get("C", NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_constants_unit(NULL) != NULL ||
        lmmc_lsr_constants_entry(lmmc_lsr_constants_count(),
                                 &constant_name,
                                 &constant_value,
                                 &constant_unit) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_constants_entry(0, NULL, &constant_value, &constant_unit) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_constants_entry(0, &constant_name, NULL, &constant_unit) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_constants_entry(0, &constant_name, &constant_value, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.constants invalid arguments not rejected\n");
        return 1;
    }

    for (size_t i = 0; i < lmmc_lsr_constants_count(); ++i) {
        const char* name = lmmc_lsr_constants_name(i);
        const char* unit = name ? lmmc_lsr_constants_unit(name) : NULL;
        if (!name || !unit ||
            lmmc_lsr_constants_entry(i,
                                     &constant_name,
                                     &constant_value,
                                     &constant_unit) != LMMC_STATUS_OK ||
            strcmp(constant_name, name) != 0 ||
            strcmp(constant_unit, unit) != 0 ||
            lmmc_lsr_constants_get(name, &out) != LMMC_STATUS_OK ||
            !close_real(constant_value, out) ||
            lmmc_lsr_units_convert(1, unit, unit, &out2) != LMMC_STATUS_OK ||
            !close_real(out2, 1)) {
            fprintf(stderr, "std.constants unit grammar mismatch\n");
            return 1;
        }
    }

    for (size_t i = 0; i < sizeof(required_constants) /
                                sizeof(required_constants[0]); ++i) {
        const char* unit =
            lmmc_lsr_constants_unit(required_constants[i].name);
        if (lmmc_lsr_constants_get(required_constants[i].name, &out) !=
                LMMC_STATUS_OK ||
            !close_real(out, required_constants[i].value) ||
            !unit || strcmp(unit, required_constants[i].unit) != 0) {
            fprintf(stderr, "std.constants required value mismatch: %s\n",
                    required_constants[i].name);
            return 1;
        }
    }

    if (lmmc_lsr_math_I(&w) != LMMC_STATUS_OK ||
        !close_real(w.real, z.real) || !close_real(w.imag, z.imag)) {
        fprintf(stderr, "std.math.I alias mismatch\n");
        return 1;
    }

    if (lmmc_lsr_num_equal(1.5, 1.5, &num_equal) != LMMC_STATUS_OK ||
        !num_equal ||
        lmmc_lsr_num_hash(1.5, &num_hash) != LMMC_STATUS_OK ||
        lmmc_lsr_num_hash(1.5, &num_hash_same) != LMMC_STATUS_OK ||
        num_hash != num_hash_same ||
        lmmc_lsr_num_equal(1.5, 2.5, &num_equal) != LMMC_STATUS_OK ||
        num_equal ||
        lmmc_lsr_num_hash(2.5, &num_hash_other) != LMMC_STATUS_OK ||
        num_hash == num_hash_other) {
        fprintf(stderr, "std.math num key equality/hash mismatch\n");
        return 1;
    }
    if (lmmc_lsr_num_equal(0.0, -0.0, &num_equal) != LMMC_STATUS_OK ||
        !num_equal ||
        lmmc_lsr_num_hash(0.0, &num_hash) != LMMC_STATUS_OK ||
        lmmc_lsr_num_hash(-0.0, &num_hash_same) != LMMC_STATUS_OK ||
        num_hash != num_hash_same) {
        fprintf(stderr, "std.math num zero hash normalization mismatch\n");
        return 1;
    }
    if (lmmc_lsr_num_equal(1.0, 1.0, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_num_hash(1.0, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_num_equal(NAN, 1.0, &num_equal) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_num_equal(1.0, INFINITY, &num_equal) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_num_hash(NAN, &num_hash) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_num_hash(INFINITY, &num_hash) !=
            LMMC_STATUS_NUMERICAL_FAILURE) {
        fprintf(stderr, "std.math num key invalid input not rejected\n");
        return 1;
    }

    if (lmmc_lsr_num_set_make(set_a_values, 5, &set_a) != LMMC_STATUS_OK ||
        lmmc_lsr_num_set_make(set_b_values, 3, &set_b) != LMMC_STATUS_OK ||
        lmmc_lsr_num_set_make(NULL, 0, &set_empty) != LMMC_STATUS_OK) {
        fprintf(stderr, "set<num> construction failed\n");
        return 1;
    }
    if (set_a.size != 3 || !num_set_has(&set_a, 1) ||
        !num_set_has(&set_a, 2) || !num_set_has(&set_a, 0) ||
        num_set_has(&set_a, 3) ||
        lmmc_lsr_num_set_contains(&set_a, -0.0, &set_contains) !=
            LMMC_STATUS_OK ||
        !set_contains ||
        set_empty.size != 0 ||
        lmmc_lsr_num_set_subset(&set_empty, &set_a, &set_subset) !=
            LMMC_STATUS_OK ||
        !set_subset) {
        fprintf(stderr, "set<num> membership or empty subset mismatch\n");
        return 1;
    }
    if (lmmc_lsr_num_set_union(&set_a, &set_b, &set_union) !=
            LMMC_STATUS_OK ||
        set_union.size != 4 || !num_set_has(&set_union, 1) ||
        !num_set_has(&set_union, 2) || !num_set_has(&set_union, 3) ||
        !num_set_has(&set_union, 0)) {
        fprintf(stderr, "set<num> union mismatch\n");
        return 1;
    }
    if (lmmc_lsr_num_set_intersection(&set_a,
                                      &set_b,
                                      &set_intersection) != LMMC_STATUS_OK ||
        set_intersection.size != 2 || !num_set_has(&set_intersection, 2) ||
        !num_set_has(&set_intersection, 0)) {
        fprintf(stderr, "set<num> intersection mismatch\n");
        return 1;
    }
    if (lmmc_lsr_num_set_difference(&set_a, &set_b, &set_difference) !=
            LMMC_STATUS_OK ||
        set_difference.size != 1 || !num_set_has(&set_difference, 1)) {
        fprintf(stderr, "set<num> difference mismatch\n");
        return 1;
    }
    if (lmmc_lsr_num_set_symmetric_difference(&set_a, &set_b, &set_xor) !=
            LMMC_STATUS_OK ||
        set_xor.size != 2 || !num_set_has(&set_xor, 1) ||
        !num_set_has(&set_xor, 3)) {
        fprintf(stderr, "set<num> symmetric difference mismatch\n");
        return 1;
    }
    if (lmmc_lsr_num_set_subset(&set_intersection, &set_union, &set_subset) !=
            LMMC_STATUS_OK ||
        !set_subset ||
        lmmc_lsr_num_set_subset(&set_union, &set_intersection, &set_subset) !=
            LMMC_STATUS_OK ||
        set_subset) {
        fprintf(stderr, "set<num> subset mismatch\n");
        return 1;
    }
    if (lmmc_lsr_num_set_make(NULL, 1, &set_xor) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_num_set_make(set_bad_values, 2, &set_xor) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_num_set_contains(NULL, 1, &set_contains) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_num_set_contains(&set_a, NAN, &set_contains) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_num_set_contains(&set_a, 1, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_num_set_subset(NULL, &set_b, &set_subset) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_num_set_subset(&set_a, &set_b, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_num_set_union(NULL, &set_b, &set_xor) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_num_set_union(&set_a, &set_b, &set_a) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "set<num> invalid input not rejected\n");
        return 1;
    }
    lmmc_lsr_num_set_destroy(&set_a);
    lmmc_lsr_num_set_destroy(&set_b);
    lmmc_lsr_num_set_destroy(&set_empty);
    lmmc_lsr_num_set_destroy(&set_union);
    lmmc_lsr_num_set_destroy(&set_intersection);
    lmmc_lsr_num_set_destroy(&set_difference);
    lmmc_lsr_num_set_destroy(&set_xor);
    lmmc_lsr_num_set_destroy(NULL);

    if (lmmc_lsr_bool_equal(1, 1, &bool_equal) != LMMC_STATUS_OK ||
        !bool_equal ||
        lmmc_lsr_bool_hash(1, &bool_hash) != LMMC_STATUS_OK ||
        lmmc_lsr_bool_hash(1, &bool_hash_same) != LMMC_STATUS_OK ||
        bool_hash != bool_hash_same ||
        lmmc_lsr_bool_equal(1, 0, &bool_equal) != LMMC_STATUS_OK ||
        bool_equal ||
        lmmc_lsr_bool_hash(0, &bool_hash_other) != LMMC_STATUS_OK ||
        bool_hash == bool_hash_other) {
        fprintf(stderr, "std.math bool key equality/hash mismatch\n");
        return 1;
    }
    if (lmmc_lsr_bool_equal(2, 1, &bool_equal) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_bool_equal(1, 1, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_bool_hash(2, &bool_hash) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_bool_hash(1, NULL) != LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.math bool key invalid input not rejected\n");
        return 1;
    }
    if (lmmc_lsr_bool_set_make(bool_set_a_values, 3, &bool_set_a) !=
            LMMC_STATUS_OK ||
        lmmc_lsr_bool_set_make(bool_set_b_values, 1, &bool_set_b) !=
            LMMC_STATUS_OK ||
        lmmc_lsr_bool_set_make(NULL, 0, &bool_set_empty) !=
            LMMC_STATUS_OK) {
        fprintf(stderr, "set<bool> construction failed\n");
        return 1;
    }
    if (bool_set_a.size != 2 || !bool_set_has(&bool_set_a, 1) ||
        !bool_set_has(&bool_set_a, 0) ||
        bool_set_empty.size != 0 ||
        lmmc_lsr_bool_set_subset(&bool_set_empty,
                                 &bool_set_a,
                                 &set_subset) != LMMC_STATUS_OK ||
        !set_subset) {
        fprintf(stderr, "set<bool> membership or empty subset mismatch\n");
        return 1;
    }
    if (lmmc_lsr_bool_set_union(&bool_set_a, &bool_set_b, &bool_set_union) !=
            LMMC_STATUS_OK ||
        bool_set_union.size != 2 || !bool_set_has(&bool_set_union, 1) ||
        !bool_set_has(&bool_set_union, 0)) {
        fprintf(stderr, "set<bool> union mismatch\n");
        return 1;
    }
    if (lmmc_lsr_bool_set_intersection(&bool_set_a,
                                       &bool_set_b,
                                       &bool_set_intersection) !=
            LMMC_STATUS_OK ||
        bool_set_intersection.size != 1 ||
        !bool_set_has(&bool_set_intersection, 0)) {
        fprintf(stderr, "set<bool> intersection mismatch\n");
        return 1;
    }
    if (lmmc_lsr_bool_set_difference(&bool_set_a,
                                     &bool_set_b,
                                     &bool_set_difference) !=
            LMMC_STATUS_OK ||
        bool_set_difference.size != 1 ||
        !bool_set_has(&bool_set_difference, 1)) {
        fprintf(stderr, "set<bool> difference mismatch\n");
        return 1;
    }
    if (lmmc_lsr_bool_set_symmetric_difference(&bool_set_a,
                                              &bool_set_b,
                                              &bool_set_xor) !=
            LMMC_STATUS_OK ||
        bool_set_xor.size != 1 || !bool_set_has(&bool_set_xor, 1)) {
        fprintf(stderr, "set<bool> symmetric difference mismatch\n");
        return 1;
    }
    if (lmmc_lsr_bool_set_subset(&bool_set_intersection,
                                 &bool_set_union,
                                 &set_subset) != LMMC_STATUS_OK ||
        !set_subset ||
        lmmc_lsr_bool_set_subset(&bool_set_union,
                                 &bool_set_intersection,
                                 &set_subset) != LMMC_STATUS_OK ||
        set_subset) {
        fprintf(stderr, "set<bool> subset mismatch\n");
        return 1;
    }
    if (lmmc_lsr_bool_set_make(NULL, 1, &bool_set_xor) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_bool_set_make(bool_set_bad_values, 2, &bool_set_xor) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_bool_set_contains(NULL, 1, &set_contains) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_bool_set_contains(&bool_set_a, 2, &set_contains) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_bool_set_contains(&bool_set_a, 1, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_bool_set_subset(NULL, &bool_set_b, &set_subset) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_bool_set_subset(&bool_set_a, &bool_set_b, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_bool_set_union(NULL, &bool_set_b, &bool_set_xor) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_bool_set_union(&bool_set_a,
                                &bool_set_b,
                                &bool_set_a) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "set<bool> invalid input not rejected\n");
        return 1;
    }
    lmmc_lsr_bool_set_destroy(&bool_set_a);
    lmmc_lsr_bool_set_destroy(&bool_set_b);
    lmmc_lsr_bool_set_destroy(&bool_set_empty);
    lmmc_lsr_bool_set_destroy(&bool_set_union);
    lmmc_lsr_bool_set_destroy(&bool_set_intersection);
    lmmc_lsr_bool_set_destroy(&bool_set_difference);
    lmmc_lsr_bool_set_destroy(&bool_set_xor);
    lmmc_lsr_bool_set_destroy(NULL);

    if (lmmc_lsr_text_equal("alpha", "alpha", &text_equal) !=
            LMMC_STATUS_OK ||
        !text_equal ||
        lmmc_lsr_text_hash("alpha", &text_hash) != LMMC_STATUS_OK ||
        lmmc_lsr_text_hash("alpha", &text_hash_same) != LMMC_STATUS_OK ||
        text_hash != text_hash_same ||
        lmmc_lsr_text_equal("alpha", "beta", &text_equal) !=
            LMMC_STATUS_OK ||
        text_equal ||
        lmmc_lsr_text_hash("beta", &text_hash_other) != LMMC_STATUS_OK ||
        text_hash == text_hash_other) {
        fprintf(stderr, "std.math text key equality/hash mismatch\n");
        return 1;
    }
    if (lmmc_lsr_text_equal(NULL, "alpha", &text_equal) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_text_equal("alpha", NULL, &text_equal) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_text_equal("alpha", "alpha", NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_text_hash(NULL, &text_hash) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_text_hash("alpha", NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.math text key invalid input not rejected\n");
        return 1;
    }

    if (lmmc_lsr_text_set_make(text_set_a_values, 3, &text_set_a) !=
            LMMC_STATUS_OK ||
        lmmc_lsr_text_set_make(text_set_b_values, 2, &text_set_b) !=
            LMMC_STATUS_OK ||
        lmmc_lsr_text_set_make(NULL, 0, &text_set_empty) !=
            LMMC_STATUS_OK) {
        fprintf(stderr, "set<text> construction failed\n");
        return 1;
    }
    if (text_set_a.size != 2 || !text_set_has(&text_set_a, "alpha") ||
        !text_set_has(&text_set_a, "beta") ||
        text_set_has(&text_set_a, "gamma") ||
        text_set_empty.size != 0 ||
        lmmc_lsr_text_set_subset(&text_set_empty,
                                 &text_set_a,
                                 &set_subset) != LMMC_STATUS_OK ||
        !set_subset) {
        fprintf(stderr, "set<text> membership mismatch\n");
        return 1;
    }
    if (lmmc_lsr_text_set_union(&text_set_a,
                                &text_set_b,
                                &text_set_union) != LMMC_STATUS_OK ||
        text_set_union.size != 3 ||
        !text_set_has(&text_set_union, "alpha") ||
        !text_set_has(&text_set_union, "beta") ||
        !text_set_has(&text_set_union, "gamma")) {
        fprintf(stderr, "set<text> union mismatch\n");
        return 1;
    }
    if (lmmc_lsr_text_set_intersection(&text_set_a,
                                       &text_set_b,
                                       &text_set_intersection) !=
            LMMC_STATUS_OK ||
        text_set_intersection.size != 1 ||
        !text_set_has(&text_set_intersection, "beta")) {
        fprintf(stderr, "set<text> intersection mismatch\n");
        return 1;
    }
    if (lmmc_lsr_text_set_difference(&text_set_a,
                                     &text_set_b,
                                     &text_set_difference) !=
            LMMC_STATUS_OK ||
        text_set_difference.size != 1 ||
        !text_set_has(&text_set_difference, "alpha")) {
        fprintf(stderr, "set<text> difference mismatch\n");
        return 1;
    }
    if (lmmc_lsr_text_set_symmetric_difference(&text_set_a,
                                              &text_set_b,
                                              &text_set_xor) !=
            LMMC_STATUS_OK ||
        text_set_xor.size != 2 || !text_set_has(&text_set_xor, "alpha") ||
        !text_set_has(&text_set_xor, "gamma")) {
        fprintf(stderr, "set<text> symmetric difference mismatch\n");
        return 1;
    }
    if (lmmc_lsr_text_set_subset(&text_set_intersection,
                                 &text_set_union,
                                 &set_subset) != LMMC_STATUS_OK ||
        !set_subset ||
        lmmc_lsr_text_set_subset(&text_set_union,
                                 &text_set_intersection,
                                 &set_subset) != LMMC_STATUS_OK ||
        set_subset) {
        fprintf(stderr, "set<text> subset mismatch\n");
        return 1;
    }
    if (lmmc_lsr_text_set_make(NULL, 1, &text_set_xor) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_text_set_make(text_set_bad_values, 2, &text_set_xor) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_text_set_contains(NULL, "alpha", &set_contains) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_text_set_contains(&text_set_a, NULL, &set_contains) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_text_set_contains(&text_set_a, "alpha", NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_text_set_subset(NULL, &text_set_b, &set_subset) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_text_set_subset(&text_set_a, &text_set_b, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_text_set_union(NULL, &text_set_b, &text_set_xor) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_text_set_union(&text_set_a,
                                &text_set_b,
                                &text_set_a) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "set<text> invalid input not rejected\n");
        return 1;
    }
    lmmc_lsr_text_set_destroy(&text_set_a);
    lmmc_lsr_text_set_destroy(&text_set_b);
    lmmc_lsr_text_set_destroy(&text_set_empty);
    lmmc_lsr_text_set_destroy(&text_set_union);
    lmmc_lsr_text_set_destroy(&text_set_intersection);
    lmmc_lsr_text_set_destroy(&text_set_difference);
    lmmc_lsr_text_set_destroy(&text_set_xor);
    lmmc_lsr_text_set_destroy(NULL);

    if (lmmc_lsr_math_complex(3, 4, &z) != LMMC_STATUS_OK) {
        fprintf(stderr, "std.math.complex failed\n");
        return 1;
    }

    if (lmmc_lsr_math_real(&z, &out) != LMMC_STATUS_OK ||
        !close_real(out, 3)) {
        fprintf(stderr, "std.math.real mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_imag(&z, &out) != LMMC_STATUS_OK ||
        !close_real(out, 4)) {
        fprintf(stderr, "std.math.imag mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_conj(&z, &w) != LMMC_STATUS_OK ||
        !close_real(w.real, 3) || !close_real(w.imag, -4)) {
        fprintf(stderr, "std.math.conj mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_complex_abs(&z, &out) != LMMC_STATUS_OK ||
        !close_real(out, 5)) {
        fprintf(stderr, "std.math.abs complex mismatch\n");
        return 1;
    }
    if (lmmc_lsr_math_complex(3, 4, &w) != LMMC_STATUS_OK ||
        lmmc_lsr_math_complex_equal(&z, &w, &complex_equal) !=
            LMMC_STATUS_OK ||
        !complex_equal ||
        lmmc_lsr_math_complex_hash(&z, &complex_hash) != LMMC_STATUS_OK ||
        lmmc_lsr_math_complex_hash(&w, &complex_hash_same) != LMMC_STATUS_OK ||
        complex_hash != complex_hash_same ||
        lmmc_lsr_math_complex(3, 5, &w) != LMMC_STATUS_OK ||
        lmmc_lsr_math_complex_equal(&z, &w, &complex_equal) !=
            LMMC_STATUS_OK ||
        complex_equal ||
        lmmc_lsr_math_complex_hash(&w, &complex_hash_other) !=
            LMMC_STATUS_OK ||
        complex_hash == complex_hash_other) {
        fprintf(stderr, "std.math complex key equality/hash mismatch\n");
        return 1;
    }
    z.real = 0.0;
    z.imag = -0.0;
    w.real = -0.0;
    w.imag = 0.0;
    if (lmmc_lsr_math_complex_equal(&z, &w, &complex_equal) !=
            LMMC_STATUS_OK ||
        !complex_equal ||
        lmmc_lsr_math_complex_hash(&z, &complex_hash) != LMMC_STATUS_OK ||
        lmmc_lsr_math_complex_hash(&w, &complex_hash_same) != LMMC_STATUS_OK ||
        complex_hash != complex_hash_same) {
        fprintf(stderr, "std.math complex zero hash normalization mismatch\n");
        return 1;
    }
    if (lmmc_lsr_math_complex(3, 4, &z) != LMMC_STATUS_OK) {
        fprintf(stderr, "std.math.complex restore failed\n");
        return 1;
    }
    if (lmmc_lsr_math_complex(3, 4, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_real(NULL, &out) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_real(&z, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_imag(NULL, &out) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_imag(&z, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_conj(NULL, &w) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_conj(&z, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_complex_abs(NULL, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_complex_abs(&z, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_complex_equal(NULL, &z, &complex_equal) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_complex_equal(&z, NULL, &complex_equal) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_complex_equal(&z, &z, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_complex_hash(NULL, &complex_hash) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_complex_hash(&z, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.math complex invalid arguments not rejected\n");
        return 1;
    }
    z.real = NAN;
    z.imag = 1;
    if (lmmc_lsr_math_complex(NAN, 1, &w) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_real(&z, &out) != LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_imag(&z, &out) != LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_conj(&z, &w) != LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_complex_abs(&z, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_complex_equal(&z, &z, &complex_equal) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_complex_hash(&z, &complex_hash) !=
            LMMC_STATUS_NUMERICAL_FAILURE) {
        fprintf(stderr, "std.math complex non-finite input not rejected\n");
        return 1;
    }
    if (lmmc_lsr_math_complex(3, 4, &z) != LMMC_STATUS_OK) {
        fprintf(stderr, "std.math.complex restore failed\n");
        return 1;
    }

    if (lmmc_lsr_complex_set_make(complex_set_a_values,
                                  4,
                                  &complex_set_a) != LMMC_STATUS_OK ||
        lmmc_lsr_complex_set_make(complex_set_b_values,
                                  3,
                                  &complex_set_b) != LMMC_STATUS_OK ||
        lmmc_lsr_complex_set_make(NULL, 0, &complex_set_empty) !=
            LMMC_STATUS_OK) {
        fprintf(stderr, "set<complex> construction failed\n");
        return 1;
    }
    if (complex_set_a.size != 3 ||
        !complex_set_has(&complex_set_a, 1, 2) ||
        !complex_set_has(&complex_set_a, 0, 0) ||
        !complex_set_has(&complex_set_a, 3, 4) ||
        complex_set_has(&complex_set_a, 5, 6) ||
        complex_set_empty.size != 0 ||
        lmmc_lsr_complex_set_subset(&complex_set_empty,
                                    &complex_set_a,
                                    &set_subset) != LMMC_STATUS_OK ||
        !set_subset) {
        fprintf(stderr, "set<complex> membership or empty subset mismatch\n");
        return 1;
    }
    if (lmmc_lsr_complex_set_union(&complex_set_a,
                                   &complex_set_b,
                                   &complex_set_union) != LMMC_STATUS_OK ||
        complex_set_union.size != 4 ||
        !complex_set_has(&complex_set_union, 1, 2) ||
        !complex_set_has(&complex_set_union, 3, 4) ||
        !complex_set_has(&complex_set_union, 5, 6) ||
        !complex_set_has(&complex_set_union, 0, 0)) {
        fprintf(stderr, "set<complex> union mismatch\n");
        return 1;
    }
    if (lmmc_lsr_complex_set_intersection(&complex_set_a,
                                          &complex_set_b,
                                          &complex_set_intersection) !=
            LMMC_STATUS_OK ||
        complex_set_intersection.size != 2 ||
        !complex_set_has(&complex_set_intersection, 3, 4) ||
        !complex_set_has(&complex_set_intersection, 0, 0)) {
        fprintf(stderr, "set<complex> intersection mismatch\n");
        return 1;
    }
    if (lmmc_lsr_complex_set_difference(&complex_set_a,
                                        &complex_set_b,
                                        &complex_set_difference) !=
            LMMC_STATUS_OK ||
        complex_set_difference.size != 1 ||
        !complex_set_has(&complex_set_difference, 1, 2)) {
        fprintf(stderr, "set<complex> difference mismatch\n");
        return 1;
    }
    if (lmmc_lsr_complex_set_symmetric_difference(&complex_set_a,
                                                 &complex_set_b,
                                                 &complex_set_xor) !=
            LMMC_STATUS_OK ||
        complex_set_xor.size != 2 ||
        !complex_set_has(&complex_set_xor, 1, 2) ||
        !complex_set_has(&complex_set_xor, 5, 6)) {
        fprintf(stderr, "set<complex> symmetric difference mismatch\n");
        return 1;
    }
    if (lmmc_lsr_complex_set_subset(&complex_set_intersection,
                                    &complex_set_union,
                                    &set_subset) != LMMC_STATUS_OK ||
        !set_subset ||
        lmmc_lsr_complex_set_subset(&complex_set_union,
                                    &complex_set_intersection,
                                    &set_subset) != LMMC_STATUS_OK ||
        set_subset) {
        fprintf(stderr, "set<complex> subset mismatch\n");
        return 1;
    }
    if (lmmc_lsr_complex_set_make(NULL, 1, &complex_set_xor) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_complex_set_make(complex_set_bad_values,
                                  2,
                                  &complex_set_xor) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_complex_set_contains(NULL, &z, &set_contains) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_complex_set_contains(&complex_set_a, NULL, &set_contains) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_complex_set_contains(&complex_set_a, &z, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_complex_set_subset(NULL, &complex_set_b, &set_subset) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_complex_set_subset(&complex_set_a, &complex_set_b, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_complex_set_union(NULL, &complex_set_b, &complex_set_xor) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_complex_set_union(&complex_set_a,
                                   &complex_set_b,
                                   &complex_set_a) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "set<complex> invalid input not rejected\n");
        return 1;
    }
    lmmc_lsr_complex_set_destroy(&complex_set_a);
    lmmc_lsr_complex_set_destroy(&complex_set_b);
    lmmc_lsr_complex_set_destroy(&complex_set_empty);
    lmmc_lsr_complex_set_destroy(&complex_set_union);
    lmmc_lsr_complex_set_destroy(&complex_set_intersection);
    lmmc_lsr_complex_set_destroy(&complex_set_difference);
    lmmc_lsr_complex_set_destroy(&complex_set_xor);
    lmmc_lsr_complex_set_destroy(NULL);

    if (lmmc_lsr_math_sqrt(-1, &out) != LMMC_STATUS_OUT_OF_RANGE) {
        fprintf(stderr, "std.math.sqrt domain error not reported\n");
        return 1;
    }
    if (lmmc_lsr_math_log(-1, &out) != LMMC_STATUS_OUT_OF_RANGE ||
        lmmc_lsr_math_log10(0, &out) != LMMC_STATUS_OUT_OF_RANGE ||
        lmmc_lsr_math_log_base(8, 1, &out) != LMMC_STATUS_OUT_OF_RANGE ||
        lmmc_lsr_math_asin(2, &out) != LMMC_STATUS_OUT_OF_RANGE ||
        lmmc_lsr_math_acos(-2, &out) != LMMC_STATUS_OUT_OF_RANGE ||
        lmmc_lsr_math_pow(0, -1, &out) != LMMC_STATUS_OUT_OF_RANGE ||
        lmmc_lsr_math_pow(-1, 0.5, &out) != LMMC_STATUS_OUT_OF_RANGE) {
        fprintf(stderr, "std.math domain errors not reported\n");
        return 1;
    }
    if (lmmc_lsr_error_name(LMMC_STATUS_OUT_OF_RANGE) == NULL ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_OK), "Ok") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_INVALID_ARGUMENT),
               "InvalidArgument") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_OUT_OF_RANGE),
               "DomainError") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_DIMENSION_MISMATCH),
               "DimensionMismatch") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_ALLOCATION_FAILED),
               "ResourceLimit") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_SINGULAR_MATRIX),
               "SingularMatrix") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_NOT_IMPLEMENTED),
               "UnsupportedExpression") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_NOT_POSITIVE_DEFINITE),
               "DomainError") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_CONVERGENCE_FAILED),
               "NumericFailure") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_INDEX_OUT_OF_BOUNDS),
               "InvalidArgument") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_WARNING_MAX_DEPTH),
               "ResourceLimit") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_EMPTY_INPUT),
               "EmptyInput") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_UNIT_STRIP_TYPE_MISMATCH),
               "UnitStripTypeMismatch") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_UNIT_STRIP_OVERFLOW),
               "UnitStripOverflow") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_UNIT_STRIP_INVALID),
               "UnitStripInvalid") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_UNIT_STRIP_LEGACY_SYNTAX),
               "UnitStripLegacySyntax") != 0 ||
        strcmp(lmmc_lsr_error_name(LMMC_STATUS_NUMERICAL_FAILURE),
               "NumericFailure") != 0 ||
        strcmp(lmmc_lsr_error_name((lmmc_status_t)9999),
               "InternalInvariant") != 0) {
        fprintf(stderr, "LSR diagnostic status mapping mismatch\n");
        return 1;
    }

    if (lmmc_lsr_stats_mean(values, 0, &out) != LMMC_STATUS_EMPTY_INPUT ||
        strcmp(lmmc_status_string(LMMC_STATUS_EMPTY_INPUT), "empty input") != 0) {
        fprintf(stderr, "std.stats empty input diagnostic mismatch\n");
        return 1;
    }
    if (lmmc_lsr_stats_median(values, 0, &out) != LMMC_STATUS_EMPTY_INPUT ||
        lmmc_lsr_stats_var(values, 0, &out) != LMMC_STATUS_EMPTY_INPUT ||
        lmmc_lsr_stats_std(values, 0, &out) != LMMC_STATUS_EMPTY_INPUT ||
        lmmc_lsr_stats_quantile(values, 0, 0.5, &out) !=
            LMMC_STATUS_EMPTY_INPUT ||
        lmmc_lsr_stats_quantile(values, 4, -0.1, &out) !=
            LMMC_STATUS_OUT_OF_RANGE ||
        lmmc_lsr_stats_quantile(values, 4, 1.1, &out) !=
            LMMC_STATUS_OUT_OF_RANGE ||
        lmmc_lsr_stats_cov(values, scaled_values, 0, &out) !=
            LMMC_STATUS_EMPTY_INPUT ||
        lmmc_lsr_stats_corr(values, scaled_values, 0, &out) !=
            LMMC_STATUS_EMPTY_INPUT) {
        fprintf(stderr, "std.stats empty input coverage mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_log(exp(1.0), &out) != LMMC_STATUS_OK ||
        !close_real(out, 1)) {
        fprintf(stderr, "std.math.log natural mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_log_base(8, 2, &out) != LMMC_STATUS_OK ||
        !close_real(out, 3)) {
        fprintf(stderr, "std.math.log_base mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_log10(1000, &out) != LMMC_STATUS_OK ||
        !close_real(out, 3)) {
        fprintf(stderr, "std.math.log10 mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_pow(2, 3, &out) != LMMC_STATUS_OK ||
        !close_real(out, 8)) {
        fprintf(stderr, "std.math.pow mismatch\n");
        return 1;
    }

    if (lmmc_lsr_math_exp(1000, &out) != LMMC_STATUS_NUMERICAL_FAILURE) {
        fprintf(stderr, "std.math.exp overflow did not report numeric failure\n");
        return 1;
    }
    if (lmmc_lsr_math_sin(0, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_cos(0, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_tan(0, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_pow(2, 3, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_asin(0, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_acos(0, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_atan(0, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_sqrt(4, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_exp(1, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_ln(1, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_log(1, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_log_base(8, 2, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_log10(10, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_abs(-1, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_floor(1, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_ceil(1, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_round(1, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_clamp(1, 0, 2, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_math_clamp(2, 3, 1, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.math scalar invalid arguments not rejected\n");
        return 1;
    }

    if (lmmc_lsr_math_clamp(5, 1, 3, &out) != LMMC_STATUS_OK ||
        !close_real(out, 3)) {
        fprintf(stderr, "std.math.clamp mismatch\n");
        return 1;
    }
    if (lmmc_lsr_math_asin(NAN, &out) != LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_acos(NAN, &out) != LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_atan(INFINITY, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_sin(INFINITY, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_cos(INFINITY, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_tan(INFINITY, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_pow(INFINITY, 0, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_sqrt(NAN, &out) != LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_exp(INFINITY, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_abs(INFINITY, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_ln(INFINITY, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_log10(INFINITY, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_log_base(INFINITY, 2, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_log_base(8, NAN, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_floor(INFINITY, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_ceil(INFINITY, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_round(INFINITY, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_clamp(NAN, 1, 3, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_clamp(2, NAN, 3, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_math_clamp(2, 1, INFINITY, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE) {
        fprintf(stderr, "std.math non-finite scalar input not rejected\n");
        return 1;
    }

    int dimensionless = 0;
    if (lmmc_lsr_units_convert(36, "km/h", "m/s", &out) != LMMC_STATUS_OK ||
        !close_real(out, 10) ||
        lmmc_lsr_units_convert(10, "km", "m", &out) != LMMC_STATUS_OK ||
        !close_real(out, 10000) ||
        lmmc_lsr_units_convert(10, "km", "km", &out) != LMMC_STATUS_OK ||
        !close_real(out, 10) ||
        lmmc_lsr_units_convert(2, "kg", "g", &out) != LMMC_STATUS_OK ||
        !close_real(out, 2000) ||
        lmmc_lsr_units_convert(3600, "s", "h", &out) != LMMC_STATUS_OK ||
        !close_real(out, 1) ||
        lmmc_lsr_units_convert(1, "N", "kg*m*s^-2", &out) !=
            LMMC_STATUS_OK ||
        !close_real(out, 1) ||
        lmmc_lsr_units_convert(1, "J", "N*m", &out) !=
            LMMC_STATUS_OK ||
        !close_real(out, 1) ||
        lmmc_lsr_units_convert_from_si(10000, "km", &out) !=
            LMMC_STATUS_OK ||
        !close_real(out, 10) ||
        lmmc_lsr_units_convert_num(10000, "km", &out) !=
            LMMC_STATUS_OK ||
        !close_real(out, 10) ||
        lmmc_lsr_units_convert_from_si(10, "km", &out) !=
            LMMC_STATUS_OK ||
        !close_real(out, 0.01) ||
        lmmc_lsr_units_is_dimensionless("1", &dimensionless) !=
            LMMC_STATUS_OK ||
        !dimensionless ||
        lmmc_lsr_units_is_dimensionless_num(12.5, &dimensionless) !=
            LMMC_STATUS_OK ||
        !dimensionless ||
        lmmc_lsr_units_convert(10, "m", "s", &out) !=
            LMMC_STATUS_DIMENSION_MISMATCH ||
        lmmc_lsr_units_strip(12.5, &out) != LMMC_STATUS_OK ||
        !close_real(out, 12.5) ||
        lmmc_lsr_units_strip_num(10, "km", &out) != LMMC_STATUS_OK ||
        !close_real(out, 10000) ||
        lmmc_lsr_units_strip_num(10, "score", &out) != LMMC_STATUS_OK ||
        !close_real(out, 10) ||
        lmmc_lsr_units_strip_scalar(10, &out) != LMMC_STATUS_OK ||
        !close_real(out, 10) ||
        lmmc_lsr_units_convert(10, "score", "score", &out) !=
            LMMC_STATUS_OK ||
        !close_real(out, 10) ||
        lmmc_lsr_units_convert(10, "score", "token", &out) !=
            LMMC_STATUS_DIMENSION_MISMATCH ||
        lmmc_lsr_units_is_dimensionless("score", &dimensionless) !=
            LMMC_STATUS_OK ||
        dimensionless ||
        lmmc_lsr_units_is_dimensionless("score/score", &dimensionless) !=
            LMMC_STATUS_OK ||
        !dimensionless ||
        lmmc_lsr_units_is_dimensionless("m/m", &dimensionless) !=
            LMMC_STATUS_OK ||
        !dimensionless ||
        lmmc_lsr_units_is_dimensionless("m*s^-1", &dimensionless) !=
            LMMC_STATUS_OK ||
        dimensionless ||
        lmmc_lsr_units_convert(1, "m^32", "m^32", &out) !=
            LMMC_STATUS_OK ||
        lmmc_lsr_units_convert(1, "m^32*m", "m", &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_convert(1, "m^33", "m^33", &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_convert(1, "m^999999999999", "m", &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_convert(1, "m/", "m", &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_convert(1, "m*", "m", &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_convert(1, "unknown", "m", &out) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.units adapter mismatch\n");
        return 1;
    }
    if (lmmc_lsr_units_convert(INFINITY, "m", "m", &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_units_convert_from_si(INFINITY, "m", &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_units_convert_num(INFINITY, "m", &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_units_strip(INFINITY, &out) !=
            LMMC_STATUS_UNIT_STRIP_OVERFLOW ||
        lmmc_lsr_units_strip_num(INFINITY, "m", &out) !=
            LMMC_STATUS_UNIT_STRIP_OVERFLOW ||
        lmmc_lsr_units_strip_scalar(INFINITY, &out) !=
            LMMC_STATUS_UNIT_STRIP_OVERFLOW ||
        lmmc_lsr_units_is_dimensionless_num(INFINITY, &dimensionless) !=
            LMMC_STATUS_NUMERICAL_FAILURE) {
        fprintf(stderr, "std.units non-finite input not rejected\n");
        return 1;
    }
    if (lmmc_lsr_units_convert(1, NULL, "m", &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_convert(1, "m", NULL, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_convert(1, "m", "m", NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_convert_from_si(1, NULL, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_convert_from_si(1, "unknown", &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_convert_from_si(1, "m", NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_convert_num(1, NULL, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_convert_num(1, "unknown", &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_convert_num(1, "m", NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_strip(1, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_strip_num(1, NULL, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_strip_num(1, "m", NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_strip_num(1, "unknown", &out) !=
            LMMC_STATUS_UNIT_STRIP_INVALID ||
        lmmc_lsr_units_strip_num(1, "num<m>", &out) !=
            LMMC_STATUS_UNIT_STRIP_LEGACY_SYNTAX ||
        lmmc_lsr_units_strip_scalar(1, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_is_dimensionless_num(1, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_is_dimensionless(NULL, &dimensionless) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_is_dimensionless("m", NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_units_is_dimensionless("unknown", &dimensionless) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.units invalid arguments not rejected\n");
        return 1;
    }

    if (lmmc_lsr_stats_mean(values, 4, &out) != LMMC_STATUS_OK ||
        !close_real(out, 2.5)) {
        fprintf(stderr, "std.stats.mean mismatch\n");
        return 1;
    }

    if (lmmc_lsr_stats_median(values, 4, &out) != LMMC_STATUS_OK ||
        !close_real(out, 2.5)) {
        fprintf(stderr, "std.stats.median mismatch\n");
        return 1;
    }

    if (lmmc_lsr_stats_var(values, 4, &out) != LMMC_STATUS_OK ||
        !close_real(out, 5.0 / 3.0)) {
        fprintf(stderr, "std.stats.var mismatch\n");
        return 1;
    }

    if (lmmc_lsr_stats_std(values, 4, &out) != LMMC_STATUS_OK ||
        !close_real(out, sqrt(5.0 / 3.0))) {
        fprintf(stderr, "std.stats.std mismatch\n");
        return 1;
    }

    if (lmmc_lsr_stats_quantile(values, 4, 0.5, &out) != LMMC_STATUS_OK ||
        !close_real(out, 2.5)) {
        fprintf(stderr, "std.stats.quantile mismatch\n");
        return 1;
    }

    if (lmmc_lsr_stats_cov(values, scaled_values, 4, &out) !=
            LMMC_STATUS_OK ||
        !close_real(out, 10.0 / 3.0)) {
        fprintf(stderr, "std.stats.cov mismatch\n");
        return 1;
    }

    if (lmmc_lsr_stats_corr(values, scaled_values, 4, &out) !=
            LMMC_STATUS_OK ||
        !close_real(out, 1.0)) {
        fprintf(stderr, "std.stats.corr mismatch\n");
        return 1;
    }
    if (lmmc_lsr_stats_mean(values, 0, &out) != LMMC_STATUS_EMPTY_INPUT ||
        lmmc_lsr_stats_median(values, 0, &out) != LMMC_STATUS_EMPTY_INPUT ||
        lmmc_lsr_stats_var(values, 0, &out) != LMMC_STATUS_EMPTY_INPUT ||
        lmmc_lsr_stats_std(values, 0, &out) != LMMC_STATUS_EMPTY_INPUT ||
        lmmc_lsr_stats_quantile(values, 0, 0.5, &out) !=
            LMMC_STATUS_EMPTY_INPUT ||
        lmmc_lsr_stats_cov(values, scaled_values, 0, &out) !=
            LMMC_STATUS_EMPTY_INPUT ||
        lmmc_lsr_stats_corr(values, scaled_values, 0, &out) !=
            LMMC_STATUS_EMPTY_INPUT) {
        fprintf(stderr, "std.stats empty input not rejected\n");
        return 1;
    }
    if (lmmc_lsr_stats_mean(NULL, 4, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_mean(values, 4, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_median(NULL, 4, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_median(values, 4, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_var(NULL, 4, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_var(values, 4, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_std(NULL, 4, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_std(values, 4, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_quantile(NULL, 4, 0.5, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_quantile(values, 4, 0.5, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_cov(NULL, scaled_values, 4, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_cov(values, NULL, 4, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_cov(values, scaled_values, 4, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_corr(NULL, scaled_values, 4, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_corr(values, NULL, 4, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_corr(values, scaled_values, 4, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.stats invalid arguments not rejected\n");
        return 1;
    }
    if (lmmc_lsr_stats_mean(nonfinite_values, 3, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_stats_median(nonfinite_values, 3, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_stats_var(nonfinite_values, 3, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_stats_std(nonfinite_values, 3, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_stats_quantile(values, 4, NAN, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_stats_cov(values, nonfinite_values, 3, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_stats_corr(values, nonfinite_values, 3, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE) {
        fprintf(stderr, "std.stats non-finite input not rejected\n");
        return 1;
    }
    if (lmmc_lsr_stats_corr(constant_values, scaled_values, 4, &out) !=
        LMMC_STATUS_NUMERICAL_FAILURE) {
        fprintf(stderr, "std.stats.corr zero variance was not diagnosed\n");
        return 1;
    }
    if (lmmc_lsr_stats_normal_pdf(0, 0, 1, &out) != LMMC_STATUS_OK ||
        !close_real(out, 0.3989422804014327) ||
        lmmc_lsr_stats_normal_cdf(0, 0, 1, &out) != LMMC_STATUS_OK ||
        !close_real(out, 0.5) ||
        lmmc_lsr_stats_normal_quantile(0.5, 0, 1, &out) !=
            LMMC_STATUS_OK ||
        !close_real(out, 0.0)) {
        fprintf(stderr, "std.stats normal distribution mismatch\n");
        return 1;
    }
    if (lmmc_lsr_stats_t_pdf(0, 1, &out) != LMMC_STATUS_OK ||
        !close_real(out, 1.0 / LMMC_PI) ||
        lmmc_lsr_stats_t_cdf(0, 1, &out) != LMMC_STATUS_OK ||
        !close_real(out, 0.5) ||
        lmmc_lsr_stats_t_quantile(0.5, 1, &out) != LMMC_STATUS_OK ||
        !close_real(out, 0.0)) {
        fprintf(stderr, "std.stats t distribution mismatch\n");
        return 1;
    }
    if (lmmc_lsr_stats_chi2_pdf(2, 2, &out) != LMMC_STATUS_OK ||
        !close_real(out, 0.5 * exp(-1.0)) ||
        lmmc_lsr_stats_chi2_cdf(0, 2, &out) != LMMC_STATUS_OK ||
        !close_real(out, 0.0) ||
        lmmc_lsr_stats_chi2_quantile(1.0 - exp(-1.0), 2, &out) !=
            LMMC_STATUS_OK ||
        fabs((double)(out - 2.0)) > 1e-8) {
        fprintf(stderr, "std.stats chi2 distribution mismatch\n");
        return 1;
    }
    if (lmmc_lsr_stats_f_pdf(1, 2, 2, &out) != LMMC_STATUS_OK ||
        !close_real(out, 0.25) ||
        lmmc_lsr_stats_f_cdf(1, 2, 2, &out) != LMMC_STATUS_OK ||
        !close_real(out, 0.5) ||
        lmmc_lsr_stats_f_quantile(0.5, 2, 2, &out) != LMMC_STATUS_OK ||
        fabs((double)(out - 1.0)) > 1e-8) {
        fprintf(stderr, "std.stats f distribution mismatch\n");
        return 1;
    }
    if (lmmc_lsr_stats_gamma_pdf(2, 1, 2, &out) != LMMC_STATUS_OK ||
        !close_real(out, 0.5 * exp(-1.0)) ||
        lmmc_lsr_stats_gamma_cdf(0, 1, 2, &out) != LMMC_STATUS_OK ||
        !close_real(out, 0.0) ||
        lmmc_lsr_stats_gamma_quantile(1.0 - exp(-1.0), 1, 2, &out) !=
            LMMC_STATUS_OK ||
        fabs((double)(out - 2.0)) > 1e-8) {
        fprintf(stderr, "std.stats gamma distribution mismatch\n");
        return 1;
    }
    if (lmmc_lsr_stats_beta_pdf(0.5, 1, 1, &out) != LMMC_STATUS_OK ||
        !close_real(out, 1.0) ||
        lmmc_lsr_stats_beta_cdf(0.5, 1, 1, &out) != LMMC_STATUS_OK ||
        !close_real(out, 0.5) ||
        lmmc_lsr_stats_beta_quantile(0.5, 1, 1, &out) != LMMC_STATUS_OK ||
        fabs((double)(out - 0.5)) > 1e-8) {
        fprintf(stderr, "std.stats beta distribution mismatch\n");
        return 1;
    }
    if (lmmc_lsr_stats_binomial_pmf(2, 4, 0.5, &out) != LMMC_STATUS_OK ||
        !close_real(out, 0.375) ||
        lmmc_lsr_stats_binomial_cdf(4, 4, 0.5, &out) != LMMC_STATUS_OK ||
        !close_real(out, 1.0) ||
        lmmc_lsr_stats_poisson_pmf(2, 2, &out) != LMMC_STATUS_OK ||
        !close_real(out, 2.0 * exp(-2.0)) ||
        lmmc_lsr_stats_poisson_cdf(0, 1, &out) != LMMC_STATUS_OK ||
        !close_real(out, exp(-1.0))) {
        fprintf(stderr, "std.stats discrete distribution mismatch\n");
        return 1;
    }
    if (lmmc_lsr_stats_normal_pdf(0, 0, 0, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_normal_quantile(0, 0, 1, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_t_pdf(0, 0, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_chi2_pdf(0, 1, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_stats_f_pdf(0, 1, 2, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_stats_gamma_pdf(0, 0.5, 1, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_stats_beta_pdf(0, 0.5, 1, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_stats_binomial_pmf(0, 1, NAN, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_stats_poisson_pmf(0, NAN, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_stats_poisson_pmf(0, 1, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.stats distribution failures not diagnosed\n");
        return 1;
    }
    if (lmmc_lsr_stats_var(single_value, 1, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_std(single_value, 1, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_cov(single_value, single_scaled_value, 1, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_stats_corr(single_value, single_scaled_value, 1, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.stats sample-size boundary was not diagnosed\n");
        return 1;
    }

    if (lmmc_rng_create(&rng) != LMMC_STATUS_OK) {
        fprintf(stderr, "rng create failed\n");
        return 1;
    }

    if (lmmc_lsr_random_seed(rng, 42) != LMMC_STATUS_OK ||
        lmmc_lsr_random_rand(rng, &out) != LMMC_STATUS_OK ||
        !isfinite((double)out) ||
        out < (lmmc_real_t)0 || out >= (lmmc_real_t)1 ||
        lmmc_lsr_random_seed(rng, 42) != LMMC_STATUS_OK ||
        lmmc_lsr_random_rand(rng, &out2) != LMMC_STATUS_OK ||
        !isfinite((double)out2) ||
        out2 < (lmmc_real_t)0 || out2 >= (lmmc_real_t)1 ||
        !close_real(out, out2)) {
        fprintf(stderr, "std.random fixed seed is not reproducible\n");
        lmmc_rng_destroy(rng);
        return 1;
    }

    if (lmmc_lsr_random_seed(NULL, 42) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_random_rand(NULL, &out) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_random_rand(rng, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_random_randint(NULL, 1, 3, &randint_out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_random_randint(rng, 1, 3, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_random_randint(rng, 3, 1, &randint_out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_random_normal(NULL, 0, 1, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_random_normal(rng, 0, 1, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_random_choice(NULL, values, 4, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_random_choice(rng, NULL, 4, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_random_choice(rng, values, 4, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.random invalid arguments not rejected\n");
        lmmc_rng_destroy(rng);
        return 1;
    }

    if (lmmc_lsr_random_randint(rng, 1, 3, &randint_out) != LMMC_STATUS_OK ||
        randint_out < 1 || randint_out > 3) {
        fprintf(stderr, "std.random.randint range mismatch\n");
        lmmc_rng_destroy(rng);
        return 1;
    }
    if (lmmc_lsr_random_seed(rng, 2718) != LMMC_STATUS_OK ||
        lmmc_lsr_random_randint(rng, 1, 3, &randint_out) != LMMC_STATUS_OK ||
        lmmc_lsr_random_seed(rng, 2718) != LMMC_STATUS_OK ||
        lmmc_lsr_random_randint(rng, 1, 3, &randint_out2) != LMMC_STATUS_OK ||
        randint_out != randint_out2) {
        fprintf(stderr, "std.random.randint fixed seed is not reproducible\n");
        lmmc_rng_destroy(rng);
        return 1;
    }

    if (lmmc_lsr_random_seed(rng, 314) != LMMC_STATUS_OK ||
        lmmc_lsr_random_normal(rng, 10, 2, &out) != LMMC_STATUS_OK ||
        !isfinite((double)out) ||
        lmmc_lsr_random_seed(rng, 314) != LMMC_STATUS_OK ||
        lmmc_lsr_random_normal(rng, 10, 2, &out2) != LMMC_STATUS_OK ||
        !close_real(out, out2) ||
        lmmc_lsr_random_normal(rng, 0, 0, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_random_normal(rng, 0, -1, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_random_normal(rng, INFINITY, 1, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_random_normal(rng, 0, NAN, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE) {
        fprintf(stderr, "std.random.normal mismatch\n");
        lmmc_rng_destroy(rng);
        return 1;
    }

    if (lmmc_lsr_random_choice(rng, values, 4, &out) != LMMC_STATUS_OK ||
        out < 1 || out > 4) {
        fprintf(stderr, "std.random.choice mismatch\n");
        lmmc_rng_destroy(rng);
        return 1;
    }
    if (lmmc_lsr_random_seed(rng, 1618) != LMMC_STATUS_OK ||
        lmmc_lsr_random_choice(rng, values, 4, &out) != LMMC_STATUS_OK ||
        lmmc_lsr_random_seed(rng, 1618) != LMMC_STATUS_OK ||
        lmmc_lsr_random_choice(rng, values, 4, &out2) != LMMC_STATUS_OK ||
        !close_real(out, out2)) {
        fprintf(stderr, "std.random.choice fixed seed is not reproducible\n");
        lmmc_rng_destroy(rng);
        return 1;
    }
    if (lmmc_lsr_random_choice(rng, nonfinite_values, 3, &out) !=
        LMMC_STATUS_NUMERICAL_FAILURE) {
        fprintf(stderr, "std.random.choice non-finite input not rejected\n");
        lmmc_rng_destroy(rng);
        return 1;
    }
    if (lmmc_lsr_random_choice(rng, values, 0, &out) !=
        LMMC_STATUS_EMPTY_INPUT) {
        fprintf(stderr, "std.random.choice empty input diagnostic mismatch\n");
        lmmc_rng_destroy(rng);
        return 1;
    }

    lmmc_rng_destroy(rng);

    if (lmmc_lsr_random_default_rand(NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_random_default_randint(4, 1, &randint_out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_random_default_normal(0, 0, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_random_default_choice(values, 0, &out) !=
            LMMC_STATUS_EMPTY_INPUT ||
        lmmc_lsr_random_default_choice(nonfinite_values, 3, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE) {
        fprintf(stderr, "std.random default diagnostics mismatch\n");
        lmmc_lsr_random_default_deinit();
        return 1;
    }
    if (lmmc_lsr_random_default_seed(2718) != LMMC_STATUS_OK ||
        lmmc_lsr_random_default_rand(&out) != LMMC_STATUS_OK ||
        lmmc_lsr_random_default_seed(2718) != LMMC_STATUS_OK ||
        lmmc_lsr_random_default_rand(&out2) != LMMC_STATUS_OK ||
        !close_real(out, out2)) {
        fprintf(stderr, "std.random default rand fixed seed is not reproducible\n");
        lmmc_lsr_random_default_deinit();
        return 1;
    }
    if (lmmc_lsr_random_default_seed(31415) != LMMC_STATUS_OK ||
        lmmc_lsr_random_default_randint(2, 5, &randint_out) !=
            LMMC_STATUS_OK ||
        randint_out < 2 || randint_out > 5 ||
        lmmc_lsr_random_default_normal(0, 1, &out) != LMMC_STATUS_OK ||
        !isfinite((double)out) ||
        lmmc_lsr_random_default_choice(values, 4, &out) !=
            LMMC_STATUS_OK ||
        (out != 1 && out != 2 && out != 3 && out != 4)) {
        fprintf(stderr, "std.random default adapter mismatch\n");
        lmmc_lsr_random_default_deinit();
        return 1;
    }
    lmmc_lsr_random_default_deinit();
    lmmc_lsr_random_default_deinit();

    if (lmmc_lsr_linalg_dot(&vec_a, &vec_b, &out) != LMMC_STATUS_OK ||
        !close_real(out, 32)) {
        fprintf(stderr, "std.linalg.dot mismatch\n");
        return 1;
    }
    if (lmmc_lsr_linalg_norm(&vec_a, &out) != LMMC_STATUS_OK ||
        !close_real(out, sqrt(14.0))) {
        fprintf(stderr, "std.linalg.norm mismatch\n");
        return 1;
    }
    if (lmmc_lsr_linalg_cross(&vec_a, &vec_b, &cross_result) !=
            LMMC_STATUS_OK ||
        cross_result.size != 3 ||
        !close_real(cross_result.data[0], -3) ||
        !close_real(cross_result.data[1], 6) ||
        !close_real(cross_result.data[2], -3)) {
        fprintf(stderr, "std.linalg.cross mismatch\n");
        lmmc_vec_destroy(&cross_result);
        return 1;
    }
    lmmc_vec_destroy(&cross_result);

    if (lmmc_lsr_linalg_vec_add(&vec_a, &vec_b, &cross_result) !=
            LMMC_STATUS_OK ||
        cross_result.size != 3 ||
        !close_real(cross_result.data[0], 5) ||
        !close_real(cross_result.data[1], 7) ||
        !close_real(cross_result.data[2], 9)) {
        fprintf(stderr, "std.linalg vector add mismatch\n");
        lmmc_vec_destroy(&cross_result);
        return 1;
    }
    lmmc_vec_destroy(&cross_result);

    if (lmmc_lsr_linalg_vec_scale(&vec_a, 2, &cross_result) !=
            LMMC_STATUS_OK ||
        cross_result.size != 3 ||
        !close_real(cross_result.data[0], 2) ||
        !close_real(cross_result.data[1], 4) ||
        !close_real(cross_result.data[2], 6)) {
        fprintf(stderr, "std.linalg vector scale mismatch\n");
        lmmc_vec_destroy(&cross_result);
        return 1;
    }
    lmmc_vec_destroy(&cross_result);

    if (lmmc_lsr_linalg_vec_add_scalar(&vec_a, 10, &cross_result) !=
            LMMC_STATUS_OK ||
        cross_result.size != 3 ||
        !close_real(cross_result.data[0], 11) ||
        !close_real(cross_result.data[1], 12) ||
        !close_real(cross_result.data[2], 13)) {
        fprintf(stderr, "std.linalg vector scalar add mismatch\n");
        lmmc_vec_destroy(&cross_result);
        return 1;
    }
    lmmc_vec_destroy(&cross_result);

    if (lmmc_lsr_linalg_vec_sub(&vec_b, &vec_a, &cross_result) !=
            LMMC_STATUS_OK ||
        cross_result.size != 3 ||
        !close_real(cross_result.data[0], 3) ||
        !close_real(cross_result.data[1], 3) ||
        !close_real(cross_result.data[2], 3)) {
        fprintf(stderr, "std.linalg vector sub mismatch\n");
        lmmc_vec_destroy(&cross_result);
        return 1;
    }
    lmmc_vec_destroy(&cross_result);

    if (lmmc_lsr_linalg_vec_sub_scalar(&vec_a, 1, &cross_result) !=
            LMMC_STATUS_OK ||
        cross_result.size != 3 ||
        !close_real(cross_result.data[0], 0) ||
        !close_real(cross_result.data[1], 1) ||
        !close_real(cross_result.data[2], 2)) {
        fprintf(stderr, "std.linalg vector scalar sub mismatch\n");
        lmmc_vec_destroy(&cross_result);
        return 1;
    }
    lmmc_vec_destroy(&cross_result);

    if (lmmc_lsr_linalg_scalar_sub_vec(10, &vec_a, &cross_result) !=
            LMMC_STATUS_OK ||
        cross_result.size != 3 ||
        !close_real(cross_result.data[0], 9) ||
        !close_real(cross_result.data[1], 8) ||
        !close_real(cross_result.data[2], 7)) {
        fprintf(stderr, "std.linalg scalar vector sub mismatch\n");
        lmmc_vec_destroy(&cross_result);
        return 1;
    }
    lmmc_vec_destroy(&cross_result);

    if (lmmc_lsr_linalg_vec_mul(&vec_a, &vec_b, &cross_result) !=
            LMMC_STATUS_OK ||
        cross_result.size != 3 ||
        !close_real(cross_result.data[0], 4) ||
        !close_real(cross_result.data[1], 10) ||
        !close_real(cross_result.data[2], 18)) {
        fprintf(stderr, "std.linalg vector elementwise multiply mismatch\n");
        lmmc_vec_destroy(&cross_result);
        return 1;
    }
    lmmc_vec_destroy(&cross_result);

    if (lmmc_lsr_linalg_vec_mul_scalar(&vec_a, 2, &cross_result) !=
            LMMC_STATUS_OK ||
        cross_result.size != 3 ||
        !close_real(cross_result.data[0], 2) ||
        !close_real(cross_result.data[1], 4) ||
        !close_real(cross_result.data[2], 6)) {
        fprintf(stderr, "std.linalg vector scalar multiply mismatch\n");
        lmmc_vec_destroy(&cross_result);
        return 1;
    }
    lmmc_vec_destroy(&cross_result);

    if (lmmc_lsr_linalg_vec_div(&vec_b, &vec_a, &cross_result) !=
            LMMC_STATUS_OK ||
        cross_result.size != 3 ||
        !close_real(cross_result.data[0], 4) ||
        !close_real(cross_result.data[1], 2.5) ||
        !close_real(cross_result.data[2], 2)) {
        fprintf(stderr, "std.linalg vector elementwise divide mismatch\n");
        lmmc_vec_destroy(&cross_result);
        return 1;
    }
    lmmc_vec_destroy(&cross_result);

    if (lmmc_lsr_linalg_vec_div_scalar(&vec_b, 2, &cross_result) !=
            LMMC_STATUS_OK ||
        cross_result.size != 3 ||
        !close_real(cross_result.data[0], 2) ||
        !close_real(cross_result.data[1], 2.5) ||
        !close_real(cross_result.data[2], 3)) {
        fprintf(stderr, "std.linalg vector scalar divide mismatch\n");
        lmmc_vec_destroy(&cross_result);
        return 1;
    }
    lmmc_vec_destroy(&cross_result);

    if (lmmc_lsr_linalg_scalar_div_vec(12, &vec_a, &cross_result) !=
            LMMC_STATUS_OK ||
        cross_result.size != 3 ||
        !close_real(cross_result.data[0], 12) ||
        !close_real(cross_result.data[1], 6) ||
        !close_real(cross_result.data[2], 4)) {
        fprintf(stderr, "std.linalg scalar vector divide mismatch\n");
        lmmc_vec_destroy(&cross_result);
        return 1;
    }
    lmmc_vec_destroy(&cross_result);

    if (lmmc_lsr_linalg_vec_pow(&vec_pow, &vec_pow_exp, &cross_result) !=
            LMMC_STATUS_OK ||
        cross_result.size != 3 ||
        !close_real(cross_result.data[0], 8) ||
        !close_real(cross_result.data[1], 9) ||
        !close_real(cross_result.data[2], 2)) {
        fprintf(stderr, "std.linalg vector elementwise power mismatch\n");
        lmmc_vec_destroy(&cross_result);
        return 1;
    }
    lmmc_vec_destroy(&cross_result);

    if (lmmc_lsr_linalg_vec_pow_scalar(&vec_pow, 2, &cross_result) !=
            LMMC_STATUS_OK ||
        cross_result.size != 3 ||
        !close_real(cross_result.data[0], 4) ||
        !close_real(cross_result.data[1], 9) ||
        !close_real(cross_result.data[2], 16)) {
        fprintf(stderr, "std.linalg vector scalar power mismatch\n");
        lmmc_vec_destroy(&cross_result);
        return 1;
    }
    lmmc_vec_destroy(&cross_result);

    if (lmmc_lsr_linalg_vec_pow(&vec_negative,
                                &vec_negative_integer_exp,
                                &cross_result) != LMMC_STATUS_OK ||
        cross_result.size != 3 ||
        !close_real(cross_result.data[0], -1) ||
        !close_real(cross_result.data[1], 4) ||
        !close_real(cross_result.data[2], 3)) {
        fprintf(stderr, "std.linalg vector integer power mismatch\n");
        lmmc_vec_destroy(&cross_result);
        return 1;
    }
    lmmc_vec_destroy(&cross_result);

    if (lmmc_lsr_linalg_vec_compare_scalar(&vec_a,
                                           LMMC_LSR_COMPARE_GT,
                                           1,
                                           &bool_vec_result) !=
            LMMC_STATUS_OK ||
        bool_vec_result.size != 3 ||
        bool_vec_result.data[0] != 0 ||
        bool_vec_result.data[1] != 1 ||
        bool_vec_result.data[2] != 1) {
        fprintf(stderr, "std.linalg vector scalar comparison mismatch\n");
        lmmc_lsr_bool_vec_destroy(&bool_vec_result);
        return 1;
    }
    lmmc_lsr_bool_vec_destroy(&bool_vec_result);

    if (lmmc_lsr_linalg_vec_compare(&vec_a,
                                    &vec_b,
                                    LMMC_LSR_COMPARE_LT,
                                    &bool_vec_result) != LMMC_STATUS_OK ||
        bool_vec_result.size != 3 ||
        bool_vec_result.data[0] != 1 ||
        bool_vec_result.data[1] != 1 ||
        bool_vec_result.data[2] != 1) {
        fprintf(stderr, "std.linalg vector comparison mismatch\n");
        lmmc_lsr_bool_vec_destroy(&bool_vec_result);
        return 1;
    }
    lmmc_lsr_bool_vec_destroy(&bool_vec_result);
    lmmc_lsr_bool_vec_destroy(NULL);

    if (lmmc_lsr_linalg_dot(NULL, &vec_b, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_dot(&vec_a, NULL, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_dot(&vec_a, &vec_b, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_cross(NULL, &vec_b, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_cross(&vec_a, NULL, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_cross(&vec_a, &vec_b, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_add(NULL, &vec_b, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_add(&vec_a, NULL, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_add(&vec_a, &vec_b, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_add_scalar(NULL, 10, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_add_scalar(&vec_a, 10, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_sub(NULL, &vec_b, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_sub(&vec_a, NULL, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_sub(&vec_a, &vec_b, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_sub_scalar(NULL, 1, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_sub_scalar(&vec_a, 1, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_scalar_sub_vec(1, NULL, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_scalar_sub_vec(1, &vec_a, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_mul(NULL, &vec_b, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_mul(&vec_a, NULL, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_mul(&vec_a, &vec_b, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_mul_scalar(NULL, 2, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_mul_scalar(&vec_a, 2, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_div(NULL, &vec_b, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_div(&vec_a, NULL, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_div(&vec_a, &vec_b, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_div_scalar(NULL, 2, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_div_scalar(&vec_a, 2, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_scalar_div_vec(2, NULL, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_scalar_div_vec(2, &vec_a, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_pow(NULL, &vec_b, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_pow(&vec_a, NULL, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_pow(&vec_a, &vec_b, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_pow_scalar(NULL, 2, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_pow_scalar(&vec_a, 2, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_compare(NULL,
                                    &vec_b,
                                    LMMC_LSR_COMPARE_EQ,
                                    &bool_vec_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_compare(&vec_a,
                                    NULL,
                                    LMMC_LSR_COMPARE_EQ,
                                    &bool_vec_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_compare(&vec_a,
                                    &vec_b,
                                    LMMC_LSR_COMPARE_EQ,
                                    NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_compare_scalar(NULL,
                                           LMMC_LSR_COMPARE_GT,
                                           1,
                                           &bool_vec_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_compare_scalar(&vec_a,
                                           LMMC_LSR_COMPARE_GT,
                                           1,
                                           NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_compare_scalar(&vec_a,
                                           (lmmc_lsr_compare_op_t)99,
                                           1,
                                           &bool_vec_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_scale(NULL, 2, &cross_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_vec_scale(&vec_a, 2, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_norm(NULL, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_norm(&vec_a, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.linalg vector invalid arguments not rejected\n");
        return 1;
    }
    if (lmmc_lsr_linalg_dot(&vec_a, &vec_short, &out) !=
            LMMC_STATUS_DIMENSION_MISMATCH ||
        lmmc_lsr_linalg_cross(&vec_a, &vec_short, &cross_result) !=
            LMMC_STATUS_DIMENSION_MISMATCH ||
        lmmc_lsr_linalg_vec_add(&vec_a, &vec_short, &cross_result) !=
            LMMC_STATUS_DIMENSION_MISMATCH ||
        lmmc_lsr_linalg_vec_sub(&vec_a, &vec_short, &cross_result) !=
            LMMC_STATUS_DIMENSION_MISMATCH ||
        lmmc_lsr_linalg_vec_mul(&vec_a, &vec_short, &cross_result) !=
            LMMC_STATUS_DIMENSION_MISMATCH ||
        lmmc_lsr_linalg_vec_div(&vec_a, &vec_short, &cross_result) !=
            LMMC_STATUS_DIMENSION_MISMATCH ||
        lmmc_lsr_linalg_vec_pow(&vec_a, &vec_short, &cross_result) !=
            LMMC_STATUS_DIMENSION_MISMATCH ||
        lmmc_lsr_linalg_vec_compare(&vec_a,
                                    &vec_short,
                                    LMMC_LSR_COMPARE_EQ,
                                    &bool_vec_result) !=
            LMMC_STATUS_DIMENSION_MISMATCH) {
        fprintf(stderr, "std.linalg vector dimension mismatch diagnostic mismatch\n");
        return 1;
    }
    if (lmmc_lsr_linalg_dot(&vec_nonfinite, &vec_b, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_cross(&vec_nonfinite, &vec_b, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_add(&vec_nonfinite, &vec_b, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_add_scalar(&vec_a, NAN, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_add_scalar(&vec_nonfinite, 10, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_sub(&vec_nonfinite, &vec_b, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_sub_scalar(&vec_a, NAN, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_sub_scalar(&vec_nonfinite, 1, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_scalar_sub_vec(NAN, &vec_a, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_scalar_sub_vec(1, &vec_nonfinite, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_mul(&vec_nonfinite, &vec_b, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_mul_scalar(&vec_a, NAN, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_mul_scalar(&vec_nonfinite, 2, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_div(&vec_nonfinite, &vec_b, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_div(&vec_b, &vec_zero, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_div_scalar(&vec_a, NAN, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_div_scalar(&vec_a, 0, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_div_scalar(&vec_nonfinite, 2, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_scalar_div_vec(NAN, &vec_a, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_scalar_div_vec(2, &vec_nonfinite, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_scalar_div_vec(2, &vec_zero, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_pow(&vec_nonfinite, &vec_b, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_pow(&vec_negative,
                                &vec_negative_exp,
                                &cross_result) != LMMC_STATUS_OUT_OF_RANGE ||
        lmmc_lsr_linalg_vec_pow(&vec_zero,
                                &vec_zero_negative_exp,
                                &cross_result) != LMMC_STATUS_OUT_OF_RANGE ||
        lmmc_lsr_linalg_vec_pow_scalar(&vec_a, NAN, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_pow_scalar(&vec_negative,
                                       0.5,
                                       &cross_result) !=
            LMMC_STATUS_OUT_OF_RANGE ||
        lmmc_lsr_linalg_vec_compare(&vec_nonfinite,
                                    &vec_b,
                                    LMMC_LSR_COMPARE_EQ,
                                    &bool_vec_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_compare_scalar(&vec_a,
                                           LMMC_LSR_COMPARE_GT,
                                           NAN,
                                           &bool_vec_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_scale(&vec_a, NAN, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_vec_scale(&vec_nonfinite, 2, &cross_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_norm(&vec_nonfinite, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE) {
        fprintf(stderr, "std.linalg vector non-finite input not rejected\n");
        return 1;
    }

    if (lmmc_mat_create(2, 2, &mat) != LMMC_STATUS_OK ||
        lmmc_mat_create(2, 2, &singular) != LMMC_STATUS_OK ||
        lmmc_mat_create(2, 2, &rhs) != LMMC_STATUS_OK ||
        lmmc_mat_create(1, 1, &mismatched_rhs) != LMMC_STATUS_OK ||
        lmmc_mat_create(3, 2, &rectangular) != LMMC_STATUS_OK) {
        fprintf(stderr, "matrix allocation failed\n");
        return 1;
    }
    set_mat_values(&mat, matrix_values);
    set_mat_values(&singular, singular_values);
    set_mat_values(&rhs, rhs_values);
    mismatched_rhs.data[0] = 1;
    set_mat_values(&rectangular, rectangular_values);
    invalid_stride.rows = 2;
    invalid_stride.cols = 2;
    invalid_stride.stride = 1;
    invalid_stride.data = matrix_values;
    invalid_stride.owns_data = 0;

    if (lmmc_lsr_linalg_shape(&mat, &rows, &cols) != LMMC_STATUS_OK ||
        rows != 2 || cols != 2 ||
        lmmc_lsr_linalg_shape_vec(&rectangular, &shape_result) !=
            LMMC_STATUS_OK ||
        shape_result.size != 2 ||
        !close_real(shape_result.data[0], 3) ||
        !close_real(shape_result.data[1], 2)) {
        fprintf(stderr, "std.linalg.shape mismatch\n");
        return 1;
    }
    lmmc_vec_destroy(&shape_result);
    if (lmmc_lsr_linalg_shape(NULL, &rows, &cols) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_shape_vec(NULL, &shape_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_shape_vec(&mat, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_matmul(NULL, &rhs, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_matmul(&mat, NULL, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_matmul(&mat, &rhs, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_add(NULL, &rhs, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_add(&mat, NULL, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_add(&mat, &rhs, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_add_scalar(NULL, 10, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_add_scalar(&mat, 10, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_sub(NULL, &rhs, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_sub(&mat, NULL, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_sub(&mat, &rhs, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_sub_scalar(NULL, 1, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_sub_scalar(&mat, 1, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_scalar_sub_mat(1, NULL, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_scalar_sub_mat(1, &mat, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_mul_elem(NULL, &rhs, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_mul_elem(&mat, NULL, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_mul_elem(&mat, &rhs, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_mul_scalar(NULL, 2, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_mul_scalar(&mat, 2, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_div(NULL, &rhs, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_div(&mat, NULL, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_div(&mat, &rhs, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_div_scalar(NULL, 2, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_div_scalar(&mat, 2, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_scalar_div_mat(2, NULL, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_scalar_div_mat(2, &mat, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_pow_elem(NULL, &rhs, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_pow_elem(&mat, NULL, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_pow_elem(&mat, &rhs, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_pow_scalar(NULL, 2, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_pow_scalar(&mat, 2, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_compare(NULL,
                                    &rhs,
                                    LMMC_LSR_COMPARE_EQ,
                                    &bool_mat_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_compare(&mat,
                                    NULL,
                                    LMMC_LSR_COMPARE_EQ,
                                    &bool_mat_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_compare(&mat,
                                    &rhs,
                                    LMMC_LSR_COMPARE_EQ,
                                    NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_compare_scalar(NULL,
                                           LMMC_LSR_COMPARE_GE,
                                           1,
                                           &bool_mat_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_compare_scalar(&mat,
                                           LMMC_LSR_COMPARE_GE,
                                           1,
                                           NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_compare_scalar(&mat,
                                           (lmmc_lsr_compare_op_t)99,
                                           1,
                                           &bool_mat_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_scale(NULL, 2, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_scale(&mat, 2, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_matvec(NULL, &vec_short, &matvec_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_matvec(&mat, NULL, &matvec_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_matvec(&mat, &vec_short, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_transpose(NULL, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_adjoint(NULL, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_det(NULL, &out) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_trace(NULL, &out) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_rank(NULL, &rank) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_inv(NULL, &result) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_solve_left(NULL, &rhs, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_solve_left(&mat, NULL, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_solve_right(NULL, &mat, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_solve_right(&rhs, NULL, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_eig(NULL, &eig_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_svd(NULL, &svd_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_eig_table(NULL, &eig_table) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_svd_table(NULL, &svd_table) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_shape(&mat, NULL, &cols) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_shape(&mat, &rows, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_transpose(&mat, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_adjoint(&mat, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_det(&mat, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_trace(&mat, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_rank(&mat, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_inv(&mat, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_solve_left(&mat, &rhs, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_solve_right(&rhs, &mat, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_eig(&mat, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_svd(&mat, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_eig_table(&mat, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_svd_table(&mat, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.linalg invalid output arguments not rejected\n");
        return 1;
    }
    if (lmmc_lsr_linalg_eye(0, &result) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_eye(2, NULL) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_diag(NULL, &result) != LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_diag(&vec_a, NULL) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.linalg constructor invalid arguments not rejected\n");
        return 1;
    }
    if (lmmc_lsr_linalg_shape(&invalid_stride, &rows, &cols) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_shape_vec(&invalid_stride, &shape_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_matmul(&invalid_stride, &rhs, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_matmul(&mat, &invalid_stride, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_add(&invalid_stride, &rhs, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_add(&mat, &invalid_stride, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_add_scalar(&invalid_stride, 10, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_sub(&invalid_stride, &rhs, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_sub(&mat, &invalid_stride, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_sub_scalar(&invalid_stride, 1, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_mul_elem(&invalid_stride, &rhs, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_mul_elem(&mat, &invalid_stride, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_div(&invalid_stride, &rhs, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_div(&mat, &invalid_stride, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_div_scalar(&invalid_stride, 2, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_pow_elem(&invalid_stride, &rhs, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_pow_elem(&mat, &invalid_stride, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_pow_scalar(&invalid_stride, 2, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_compare(&invalid_stride,
                                    &rhs,
                                    LMMC_LSR_COMPARE_EQ,
                                    &bool_mat_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_compare(&mat,
                                    &invalid_stride,
                                    LMMC_LSR_COMPARE_EQ,
                                    &bool_mat_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_compare_scalar(&invalid_stride,
                                           LMMC_LSR_COMPARE_GT,
                                           1,
                                           &bool_mat_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_mat_scale(&invalid_stride, 2, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_matvec(&invalid_stride, &vec_short, &matvec_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_transpose(&invalid_stride, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_adjoint(&invalid_stride, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_det(&invalid_stride, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_trace(&invalid_stride, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_rank(&invalid_stride, &rank) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_inv(&invalid_stride, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_solve_left(&invalid_stride, &rhs, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_solve_left(&mat, &invalid_stride, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_solve_right(&invalid_stride, &mat, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_solve_right(&rhs, &invalid_stride, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_eig(&invalid_stride, &eig_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_svd(&invalid_stride, &svd_result) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_eig_table(&invalid_stride, &eig_table) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_svd_table(&invalid_stride, &svd_table) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.linalg invalid matrix descriptor not rejected\n");
        return 1;
    }
    if (lmmc_lsr_linalg_diag(&vec_nonfinite, &result) !=
        LMMC_STATUS_NUMERICAL_FAILURE) {
        fprintf(stderr, "std.linalg.diag non-finite input not rejected\n");
        return 1;
    }

    if (lmmc_lsr_linalg_matvec(&mat, &vec_short, &matvec_result) !=
            LMMC_STATUS_OK ||
        matvec_result.size != 2 ||
        !close_real(matvec_result.data[0], 5) ||
        !close_real(matvec_result.data[1], 11)) {
        fprintf(stderr, "std.linalg.matmul matrix-vector mismatch\n");
        lmmc_vec_destroy(&matvec_result);
        return 1;
    }
    lmmc_vec_destroy(&matvec_result);

    if (lmmc_lsr_linalg_matmul(&mat, &rhs, &result) != LMMC_STATUS_OK ||
        result.rows != 2 || result.cols != 2 ||
        !mat_close_at(&result, 0, 0, 19) ||
        !mat_close_at(&result, 0, 1, 22) ||
        !mat_close_at(&result, 1, 0, 43) ||
        !mat_close_at(&result, 1, 1, 50)) {
        fprintf(stderr, "std.linalg.matmul matrix-matrix mismatch\n");
        lmmc_mat_destroy(&result);
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_mat_add(&mat, &rhs, &result) != LMMC_STATUS_OK ||
        result.rows != 2 || result.cols != 2 ||
        !mat_close_at(&result, 0, 0, 6) ||
        !mat_close_at(&result, 0, 1, 8) ||
        !mat_close_at(&result, 1, 0, 10) ||
        !mat_close_at(&result, 1, 1, 12)) {
        fprintf(stderr, "std.linalg matrix add mismatch\n");
        lmmc_mat_destroy(&result);
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_mat_scale(&mat, 2, &result) != LMMC_STATUS_OK ||
        result.rows != 2 || result.cols != 2 ||
        !mat_close_at(&result, 0, 0, 2) ||
        !mat_close_at(&result, 0, 1, 4) ||
        !mat_close_at(&result, 1, 0, 6) ||
        !mat_close_at(&result, 1, 1, 8)) {
        fprintf(stderr, "std.linalg matrix scale mismatch\n");
        lmmc_mat_destroy(&result);
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_mat_add_scalar(&mat, 10, &result) != LMMC_STATUS_OK ||
        result.rows != 2 || result.cols != 2 ||
        !mat_close_at(&result, 0, 0, 11) ||
        !mat_close_at(&result, 0, 1, 12) ||
        !mat_close_at(&result, 1, 0, 13) ||
        !mat_close_at(&result, 1, 1, 14)) {
        fprintf(stderr, "std.linalg matrix scalar add mismatch\n");
        lmmc_mat_destroy(&result);
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_mat_sub(&rhs, &mat, &result) != LMMC_STATUS_OK ||
        result.rows != 2 || result.cols != 2 ||
        !mat_close_at(&result, 0, 0, 4) ||
        !mat_close_at(&result, 0, 1, 4) ||
        !mat_close_at(&result, 1, 0, 4) ||
        !mat_close_at(&result, 1, 1, 4)) {
        fprintf(stderr, "std.linalg matrix sub mismatch\n");
        lmmc_mat_destroy(&result);
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_mat_sub_scalar(&mat, 1, &result) != LMMC_STATUS_OK ||
        result.rows != 2 || result.cols != 2 ||
        !mat_close_at(&result, 0, 0, 0) ||
        !mat_close_at(&result, 0, 1, 1) ||
        !mat_close_at(&result, 1, 0, 2) ||
        !mat_close_at(&result, 1, 1, 3)) {
        fprintf(stderr, "std.linalg matrix scalar sub mismatch\n");
        lmmc_mat_destroy(&result);
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_scalar_sub_mat(10, &mat, &result) != LMMC_STATUS_OK ||
        result.rows != 2 || result.cols != 2 ||
        !mat_close_at(&result, 0, 0, 9) ||
        !mat_close_at(&result, 0, 1, 8) ||
        !mat_close_at(&result, 1, 0, 7) ||
        !mat_close_at(&result, 1, 1, 6)) {
        fprintf(stderr, "std.linalg scalar matrix sub mismatch\n");
        lmmc_mat_destroy(&result);
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_mat_mul_elem(&mat, &rhs, &result) !=
            LMMC_STATUS_OK ||
        result.rows != 2 || result.cols != 2 ||
        !mat_close_at(&result, 0, 0, 5) ||
        !mat_close_at(&result, 0, 1, 12) ||
        !mat_close_at(&result, 1, 0, 21) ||
        !mat_close_at(&result, 1, 1, 32)) {
        fprintf(stderr, "std.linalg matrix elementwise multiply mismatch\n");
        lmmc_mat_destroy(&result);
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_mat_mul_scalar(&mat, 2, &result) != LMMC_STATUS_OK ||
        result.rows != 2 || result.cols != 2 ||
        !mat_close_at(&result, 0, 0, 2) ||
        !mat_close_at(&result, 0, 1, 4) ||
        !mat_close_at(&result, 1, 0, 6) ||
        !mat_close_at(&result, 1, 1, 8)) {
        fprintf(stderr, "std.linalg matrix scalar multiply mismatch\n");
        lmmc_mat_destroy(&result);
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_mat_div(&rhs, &mat, &result) != LMMC_STATUS_OK ||
        result.rows != 2 || result.cols != 2 ||
        !mat_close_at(&result, 0, 0, 5) ||
        !mat_close_at(&result, 0, 1, 3) ||
        !mat_close_at(&result, 1, 0, 7.0 / 3.0) ||
        !mat_close_at(&result, 1, 1, 2)) {
        fprintf(stderr, "std.linalg matrix elementwise divide mismatch\n");
        lmmc_mat_destroy(&result);
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_mat_div_scalar(&rhs, 2, &result) != LMMC_STATUS_OK ||
        result.rows != 2 || result.cols != 2 ||
        !mat_close_at(&result, 0, 0, 2.5) ||
        !mat_close_at(&result, 0, 1, 3) ||
        !mat_close_at(&result, 1, 0, 3.5) ||
        !mat_close_at(&result, 1, 1, 4)) {
        fprintf(stderr, "std.linalg matrix scalar divide mismatch\n");
        lmmc_mat_destroy(&result);
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_scalar_div_mat(12, &mat, &result) !=
            LMMC_STATUS_OK ||
        result.rows != 2 || result.cols != 2 ||
        !mat_close_at(&result, 0, 0, 12) ||
        !mat_close_at(&result, 0, 1, 6) ||
        !mat_close_at(&result, 1, 0, 4) ||
        !mat_close_at(&result, 1, 1, 3)) {
        fprintf(stderr, "std.linalg scalar matrix divide mismatch\n");
        lmmc_mat_destroy(&result);
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_mat_pow_elem(&mat, &rhs, &result) !=
            LMMC_STATUS_OK ||
        result.rows != 2 || result.cols != 2 ||
        !mat_close_at(&result, 0, 0, 1) ||
        !mat_close_at(&result, 0, 1, 64) ||
        !mat_close_at(&result, 1, 0, 2187) ||
        !mat_close_at(&result, 1, 1, 65536)) {
        fprintf(stderr, "std.linalg matrix elementwise power mismatch\n");
        lmmc_mat_destroy(&result);
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_mat_pow_scalar(&mat, 2, &result) !=
            LMMC_STATUS_OK ||
        result.rows != 2 || result.cols != 2 ||
        !mat_close_at(&result, 0, 0, 1) ||
        !mat_close_at(&result, 0, 1, 4) ||
        !mat_close_at(&result, 1, 0, 9) ||
        !mat_close_at(&result, 1, 1, 16)) {
        fprintf(stderr, "std.linalg matrix scalar power mismatch\n");
        lmmc_mat_destroy(&result);
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_mat_compare_scalar(&mat,
                                           LMMC_LSR_COMPARE_GE,
                                           3,
                                           &bool_mat_result) !=
            LMMC_STATUS_OK ||
        bool_mat_result.rows != 2 || bool_mat_result.cols != 2 ||
        bool_mat_result.data[0] != 0 ||
        bool_mat_result.data[1] != 0 ||
        bool_mat_result.data[bool_mat_result.stride] != 1 ||
        bool_mat_result.data[bool_mat_result.stride + 1] != 1) {
        fprintf(stderr, "std.linalg matrix scalar comparison mismatch\n");
        lmmc_lsr_bool_mat_destroy(&bool_mat_result);
        return 1;
    }
    lmmc_lsr_bool_mat_destroy(&bool_mat_result);

    if (lmmc_lsr_linalg_mat_compare(&mat,
                                    &rhs,
                                    LMMC_LSR_COMPARE_LE,
                                    &bool_mat_result) != LMMC_STATUS_OK ||
        bool_mat_result.rows != 2 || bool_mat_result.cols != 2 ||
        bool_mat_result.data[0] != 1 ||
        bool_mat_result.data[1] != 1 ||
        bool_mat_result.data[bool_mat_result.stride] != 1 ||
        bool_mat_result.data[bool_mat_result.stride + 1] != 1) {
        fprintf(stderr, "std.linalg matrix comparison mismatch\n");
        lmmc_lsr_bool_mat_destroy(&bool_mat_result);
        return 1;
    }
    lmmc_lsr_bool_mat_destroy(&bool_mat_result);
    lmmc_lsr_bool_mat_destroy(NULL);

    mat.data[0] = -1;
    if (lmmc_lsr_linalg_mat_pow_scalar(&mat, 0.5, &result) !=
        LMMC_STATUS_OUT_OF_RANGE) {
        fprintf(stderr, "std.linalg matrix fractional power domain mismatch\n");
        return 1;
    }
    mat.data[0] = 0;
    if (lmmc_lsr_linalg_mat_pow_scalar(&mat, -1, &result) !=
        LMMC_STATUS_OUT_OF_RANGE) {
        fprintf(stderr, "std.linalg matrix zero negative power mismatch\n");
        return 1;
    }
    mat.data[0] = 1;

    rhs.data[0] = 0;
    if (lmmc_lsr_linalg_mat_div(&mat, &rhs, &result) !=
        LMMC_STATUS_NUMERICAL_FAILURE) {
        fprintf(stderr, "std.linalg matrix divide by zero not rejected\n");
        return 1;
    }
    if (lmmc_lsr_linalg_scalar_div_mat(12, &rhs, &result) !=
        LMMC_STATUS_NUMERICAL_FAILURE) {
        fprintf(stderr, "std.linalg scalar matrix divide by zero not rejected\n");
        return 1;
    }
    rhs.data[0] = 5;

    if (lmmc_lsr_linalg_transpose(&mat, &result) != LMMC_STATUS_OK ||
        result.rows != 2 || result.cols != 2 ||
        !mat_close_at(&result, 0, 1, 3) ||
        !mat_close_at(&result, 1, 0, 2)) {
        fprintf(stderr, "std.linalg.transpose mismatch\n");
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_adjoint(&mat, &result) != LMMC_STATUS_OK ||
        !mat_close_at(&result, 0, 1, 3) ||
        !mat_close_at(&result, 1, 0, 2)) {
        fprintf(stderr, "std.linalg.adjoint real mismatch\n");
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_eye(3, &result) != LMMC_STATUS_OK ||
        result.rows != 3 || result.cols != 3 ||
        !mat_close_at(&result, 0, 0, 1) ||
        !mat_close_at(&result, 1, 1, 1) ||
        !mat_close_at(&result, 2, 2, 1) ||
        !mat_close_at(&result, 0, 1, 0) ||
        !mat_close_at(&result, 1, 2, 0)) {
        fprintf(stderr, "std.linalg.eye mismatch\n");
        lmmc_mat_destroy(&result);
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_diag(&vec_a, &result) != LMMC_STATUS_OK ||
        result.rows != 3 || result.cols != 3 ||
        !mat_close_at(&result, 0, 0, 1) ||
        !mat_close_at(&result, 1, 1, 2) ||
        !mat_close_at(&result, 2, 2, 3) ||
        !mat_close_at(&result, 0, 2, 0) ||
        !mat_close_at(&result, 2, 0, 0)) {
        fprintf(stderr, "std.linalg.diag mismatch\n");
        lmmc_mat_destroy(&result);
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_det(&mat, &out) != LMMC_STATUS_OK ||
        !close_real(out, -2)) {
        fprintf(stderr, "std.linalg.det mismatch\n");
        return 1;
    }

    if (lmmc_lsr_linalg_trace(&mat, &out) != LMMC_STATUS_OK ||
        !close_real(out, 5)) {
        fprintf(stderr, "std.linalg.trace mismatch\n");
        return 1;
    }

    if (lmmc_lsr_linalg_inv(&mat, &result) != LMMC_STATUS_OK ||
        !mat_close_at(&result, 0, 0, -2) ||
        !mat_close_at(&result, 0, 1, 1) ||
        !mat_close_at(&result, 1, 0, 1.5) ||
        !mat_close_at(&result, 1, 1, -0.5)) {
        fprintf(stderr, "std.linalg.inv mismatch\n");
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_rank(&rectangular, &rank) != LMMC_STATUS_OK ||
        rank != 1) {
        fprintf(stderr, "std.linalg.rank mismatch\n");
        return 1;
    }

    if (lmmc_lsr_linalg_solve_left(&mat, &rhs, &result) != LMMC_STATUS_OK ||
        !mat_close_at(&result, 0, 0, -3) ||
        !mat_close_at(&result, 0, 1, -4) ||
        !mat_close_at(&result, 1, 0, 4) ||
        !mat_close_at(&result, 1, 1, 5)) {
        fprintf(stderr, "std.linalg.solve_left mismatch\n");
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_solve_right(&rhs, &mat, &result) != LMMC_STATUS_OK ||
        !mat_close_at(&result, 0, 0, -1) ||
        !mat_close_at(&result, 0, 1, 2) ||
        !mat_close_at(&result, 1, 0, -2) ||
        !mat_close_at(&result, 1, 1, 3)) {
        fprintf(stderr, "std.linalg.solve_right mismatch\n");
        return 1;
    }
    lmmc_mat_destroy(&result);

    if (lmmc_lsr_linalg_det(&rectangular, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_trace(&rectangular, &out) !=
            LMMC_STATUS_INVALID_ARGUMENT ||
        lmmc_lsr_linalg_inv(&rectangular, &result) !=
            LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.linalg rectangular matrix diagnostic mismatch\n");
        return 1;
    }

    if (lmmc_lsr_linalg_solve_left(&mat, &mismatched_rhs, &result) !=
            LMMC_STATUS_DIMENSION_MISMATCH ||
        lmmc_lsr_linalg_solve_right(&mismatched_rhs, &mat, &result) !=
            LMMC_STATUS_DIMENSION_MISMATCH ||
        lmmc_lsr_linalg_mat_add(&mat, &rectangular, &result) !=
            LMMC_STATUS_DIMENSION_MISMATCH ||
        lmmc_lsr_linalg_mat_sub(&mat, &rectangular, &result) !=
            LMMC_STATUS_DIMENSION_MISMATCH ||
        lmmc_lsr_linalg_mat_mul_elem(&mat, &rectangular, &result) !=
            LMMC_STATUS_DIMENSION_MISMATCH ||
        lmmc_lsr_linalg_mat_div(&mat, &rectangular, &result) !=
            LMMC_STATUS_DIMENSION_MISMATCH ||
        lmmc_lsr_linalg_mat_pow_elem(&mat, &rectangular, &result) !=
            LMMC_STATUS_DIMENSION_MISMATCH ||
        lmmc_lsr_linalg_mat_compare(&mat,
                                    &rectangular,
                                    LMMC_LSR_COMPARE_EQ,
                                    &bool_mat_result) !=
            LMMC_STATUS_DIMENSION_MISMATCH ||
        lmmc_lsr_linalg_matmul(&mat, &rectangular, &result) !=
            LMMC_STATUS_DIMENSION_MISMATCH ||
        lmmc_lsr_linalg_matvec(&mat, &vec_a, &matvec_result) !=
            LMMC_STATUS_DIMENSION_MISMATCH) {
        fprintf(stderr, "std.linalg dimension mismatch diagnostic mismatch\n");
        return 1;
    }

    if (lmmc_lsr_linalg_inv(&singular, &result) !=
            LMMC_STATUS_SINGULAR_MATRIX ||
        lmmc_lsr_linalg_solve_left(&singular, &rhs, &result) !=
            LMMC_STATUS_SINGULAR_MATRIX ||
        lmmc_lsr_linalg_solve_right(&rhs, &singular, &result) !=
            LMMC_STATUS_SINGULAR_MATRIX) {
        fprintf(stderr, "std.linalg singular matrix diagnostic mismatch\n");
        return 1;
    }

    if (lmmc_lsr_linalg_eig(&mat, &eig_result) != LMMC_STATUS_OK ||
        eig_result.real_parts.size != 2 ||
        eig_result.imag_parts.size != 2 ||
        !close_real(eig_result.imag_parts.data[0], 0) ||
        !close_real(eig_result.imag_parts.data[1], 0)) {
        fprintf(stderr, "std.linalg.eig mismatch\n");
        return 1;
    }
    lmmc_eigen_gen_full_result_destroy(&eig_result);

    if (lmmc_lsr_linalg_eig_table(&mat, &eig_table) != LMMC_STATUS_OK) {
        fprintf(stderr, "std.linalg.eig table failed\n");
        return 1;
    }
    lmmc_lsr_eig_table_destroy(NULL);
    named = lmmc_lsr_eig_table_get(&eig_table, "values_real");
    if (!named || named->rows != 2 || named->cols != 1 ||
        !isfinite((double)named->data[0]) ||
        !isfinite((double)named->data[named->stride]) ||
        lmmc_lsr_eig_table_count(&eig_table) != 4 ||
        strcmp(lmmc_lsr_eig_table_key(&eig_table, 0), "values_real") != 0 ||
        strcmp(lmmc_lsr_eig_table_key(&eig_table, 3), "vectors_imag") != 0 ||
        lmmc_lsr_eig_table_count(NULL) != 0 ||
        lmmc_lsr_eig_table_key(&eig_table, 4) != NULL ||
        lmmc_lsr_eig_table_key(NULL, 0) != NULL ||
        lmmc_lsr_eig_table_get(&eig_table, "missing") != NULL ||
        lmmc_lsr_eig_table_get(&eig_table, NULL) != NULL ||
        lmmc_lsr_eig_table_get(NULL, "values_real") != NULL) {
        fprintf(stderr, "std.linalg.eig table mapping mismatch\n");
        return 1;
    }
    named = lmmc_lsr_eig_table_get(&eig_table, "vectors_real");
    if (!named || named->rows != 2 || named->cols != 2) {
        fprintf(stderr, "std.linalg.eig table vectors mismatch\n");
        return 1;
    }
    lmmc_lsr_eig_table_destroy(&eig_table);

    if (lmmc_lsr_linalg_svd(&mat, &svd_result) != LMMC_STATUS_OK ||
        svd_result.sigma.size != 2 ||
        svd_result.U.rows != 2 ||
        svd_result.Vt.cols != 2 ||
        svd_result.sigma.data[0] < svd_result.sigma.data[1]) {
        fprintf(stderr, "std.linalg.svd mismatch\n");
        return 1;
    }
    lmmc_svd_result_destroy(&svd_result);

    if (lmmc_lsr_linalg_svd_table(&mat, &svd_table) != LMMC_STATUS_OK) {
        fprintf(stderr, "std.linalg.svd table failed\n");
        return 1;
    }
    lmmc_lsr_svd_table_destroy(NULL);
    named = lmmc_lsr_svd_table_get(&svd_table, "S");
    if (!named || named->rows != 2 || named->cols != 2 ||
        named->data[0] < named->data[named->stride + 1] ||
        lmmc_lsr_svd_table_count(&svd_table) != 3 ||
        strcmp(lmmc_lsr_svd_table_key(&svd_table, 0), "U") != 0 ||
        strcmp(lmmc_lsr_svd_table_key(&svd_table, 2), "Vt") != 0 ||
        lmmc_lsr_svd_table_count(NULL) != 0 ||
        lmmc_lsr_svd_table_key(&svd_table, 3) != NULL ||
        lmmc_lsr_svd_table_key(NULL, 0) != NULL ||
        lmmc_lsr_svd_table_get(&svd_table, "sigma") != NULL ||
        lmmc_lsr_svd_table_get(&svd_table, NULL) != NULL ||
        lmmc_lsr_svd_table_get(NULL, "S") != NULL) {
        fprintf(stderr, "std.linalg.svd table mapping mismatch\n");
        return 1;
    }
    named = lmmc_lsr_svd_table_get(&svd_table, "U");
    if (!named || named->rows != 2 || named->cols != 2) {
        fprintf(stderr, "std.linalg.svd table U mismatch\n");
        return 1;
    }
    lmmc_lsr_svd_table_destroy(&svd_table);

    if (lmmc_lsr_linalg_eig(&rectangular, &eig_result) !=
        LMMC_STATUS_INVALID_ARGUMENT) {
        fprintf(stderr, "std.linalg.eig rectangular error mismatch\n");
        return 1;
    }

    mat.data[0] = NAN;
    eig_table.values_real.rows = 123;
    svd_table.U.rows = 123;
    if (lmmc_lsr_linalg_transpose(&mat, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_matmul(&mat, &rhs, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_add(&mat, &rhs, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_add_scalar(&mat, 10, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_add_scalar(&rhs, NAN, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_sub(&mat, &rhs, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_sub_scalar(&mat, 1, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_sub_scalar(&rhs, NAN, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_scalar_sub_mat(1, &mat, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_scalar_sub_mat(NAN, &rhs, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_mul_elem(&mat, &rhs, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_mul_scalar(&mat, 2, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_mul_scalar(&rhs, NAN, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_div(&mat, &rhs, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_div_scalar(&mat, 2, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_div_scalar(&rhs, NAN, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_scalar_div_mat(2, &mat, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_scalar_div_mat(NAN, &rhs, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_pow_elem(&mat, &rhs, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_pow_scalar(&mat, 2, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_pow_scalar(&rhs, NAN, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_compare(&mat,
                                    &rhs,
                                    LMMC_LSR_COMPARE_EQ,
                                    &bool_mat_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_compare_scalar(&mat,
                                           LMMC_LSR_COMPARE_GT,
                                           1,
                                           &bool_mat_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_compare_scalar(&rhs,
                                           LMMC_LSR_COMPARE_GT,
                                           NAN,
                                           &bool_mat_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_scale(&mat, 2, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_mat_scale(&rhs, NAN, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_matvec(&mat, &vec_short, &matvec_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_adjoint(&mat, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_det(&mat, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_trace(&mat, &out) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_rank(&mat, &rank) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_inv(&mat, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_solve_left(&mat, &rhs, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_solve_right(&rhs, &mat, &result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_eig(&mat, &eig_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_svd(&mat, &svd_result) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_eig_table(&mat, &eig_table) !=
            LMMC_STATUS_NUMERICAL_FAILURE ||
        lmmc_lsr_linalg_svd_table(&mat, &svd_table) !=
            LMMC_STATUS_NUMERICAL_FAILURE) {
        fprintf(stderr, "std.linalg non-finite matrix input not rejected\n");
        return 1;
    }
    if (eig_table.values_real.rows != 0 || svd_table.U.rows != 0) {
        fprintf(stderr, "std.linalg table failure outputs not cleared\n");
        return 1;
    }

    lmmc_mat_destroy(&rectangular);
    lmmc_mat_destroy(&mismatched_rhs);
    lmmc_mat_destroy(&rhs);
    lmmc_mat_destroy(&singular);
    lmmc_mat_destroy(&mat);

    return 0;
}
