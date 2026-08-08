/**
 * @file lsr_stdlib.h
 * @brief LSR-facing numeric standard library adapters.
 *
 * These functions provide a stable C ABI shape for Lamina std.math bindings.
 * They wrap existing LMMC numeric and complex primitives without adding
 * symbolic semantics; Expr handling belongs to LMCAS.
 */
#ifndef LMMC_LSR_STDLIB_H
#define LMMC_LSR_STDLIB_H

#include "lmmc/complex.h"
#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/random.h"
#include "lmmc/status.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Map an LMMC status code to a stable LSR diagnostic name. */
const char* lmmc_lsr_error_name(lmmc_status_t status);

/** @brief LSR table<text, matrix> view for std.linalg.eig. */
typedef struct {
    lmmc_mat_t values_real;
    lmmc_mat_t values_imag;
    lmmc_mat_t vectors_real;
    lmmc_mat_t vectors_imag;
} lmmc_lsr_eig_table_t;

/** @brief LSR table<text, matrix> view for std.linalg.svd. */
typedef struct {
    lmmc_mat_t U;
    lmmc_mat_t S;
    lmmc_mat_t Vt;
} lmmc_lsr_svd_table_t;

typedef enum {
    LMMC_LSR_COMPARE_EQ = 0,
    LMMC_LSR_COMPARE_NE = 1,
    LMMC_LSR_COMPARE_LT = 2,
    LMMC_LSR_COMPARE_LE = 3,
    LMMC_LSR_COMPARE_GT = 4,
    LMMC_LSR_COMPARE_GE = 5
} lmmc_lsr_compare_op_t;

typedef struct {
    size_t size;
    uint8_t* data;
    int owns_data;
} lmmc_lsr_bool_vec_t;

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    uint8_t* data;
    int owns_data;
} lmmc_lsr_bool_mat_t;

typedef struct {
    size_t size;
    lmmc_real_t* data;
    int owns_data;
} lmmc_lsr_num_set_t;

typedef struct {
    size_t size;
    lmmc_complex_t* data;
    int owns_data;
} lmmc_lsr_complex_set_t;

typedef struct {
    size_t size;
    uint8_t* data;
    int owns_data;
} lmmc_lsr_bool_set_t;

typedef struct {
    size_t size;
    char** data;
    int owns_data;
} lmmc_lsr_text_set_t;

/** @brief Return the LSR std.math constant pi. */
lmmc_status_t lmmc_lsr_math_pi(lmmc_real_t* out);
/** @brief Return the LSR std.math constant e. */
lmmc_status_t lmmc_lsr_math_e(lmmc_real_t* out);
/** @brief Return the LSR std.math constant phi. */
lmmc_status_t lmmc_lsr_math_phi(lmmc_real_t* out);

/** @brief Return the number of LSR std.constants entries. */
size_t lmmc_lsr_constants_count(void);
/** @brief Return the name of an LSR std.constants entry by index, or NULL. */
const char* lmmc_lsr_constants_name(size_t index);
/** @brief Return an LSR std.constants numeric value by name. */
lmmc_status_t lmmc_lsr_constants_get(const char* name, lmmc_real_t* out);
/** @brief Return an LSR std.constants unit string by name, or NULL. */
const char* lmmc_lsr_constants_unit(const char* name);
/** @brief Return a complete LSR std.constants entry by index. */
lmmc_status_t lmmc_lsr_constants_entry(size_t index,
                                       const char** out_name,
                                       lmmc_real_t* out_value,
                                       const char** out_unit);

/** @brief Return std.math.i, the imaginary unit. */
lmmc_status_t lmmc_lsr_math_i(lmmc_complex_t* out);
/** @brief Return std.math.I, an alias of std.math.i. */
lmmc_status_t lmmc_lsr_math_I(lmmc_complex_t* out);

/** @brief Compare finite numeric values for LSR structural key equality. */
lmmc_status_t lmmc_lsr_num_equal(lmmc_real_t lhs, lmmc_real_t rhs, int* out);
/** @brief Hash a finite numeric value for LSR table keys. */
lmmc_status_t lmmc_lsr_num_hash(lmmc_real_t value, uint64_t* out);
/** @brief Construct a finite numeric set with duplicate values removed. */
lmmc_status_t lmmc_lsr_num_set_make(const lmmc_real_t* values,
                                    size_t count,
                                    lmmc_lsr_num_set_t* out);
void lmmc_lsr_num_set_destroy(lmmc_lsr_num_set_t* set);
lmmc_status_t lmmc_lsr_num_set_contains(const lmmc_lsr_num_set_t* set,
                                        lmmc_real_t value,
                                        int* out);
lmmc_status_t lmmc_lsr_num_set_subset(const lmmc_lsr_num_set_t* lhs,
                                      const lmmc_lsr_num_set_t* rhs,
                                      int* out);
lmmc_status_t lmmc_lsr_num_set_union(const lmmc_lsr_num_set_t* lhs,
                                     const lmmc_lsr_num_set_t* rhs,
                                     lmmc_lsr_num_set_t* out);
lmmc_status_t lmmc_lsr_num_set_intersection(const lmmc_lsr_num_set_t* lhs,
                                            const lmmc_lsr_num_set_t* rhs,
                                            lmmc_lsr_num_set_t* out);
lmmc_status_t lmmc_lsr_num_set_difference(const lmmc_lsr_num_set_t* lhs,
                                          const lmmc_lsr_num_set_t* rhs,
                                          lmmc_lsr_num_set_t* out);
lmmc_status_t lmmc_lsr_num_set_symmetric_difference(
    const lmmc_lsr_num_set_t* lhs,
    const lmmc_lsr_num_set_t* rhs,
    lmmc_lsr_num_set_t* out);
/** @brief Compare LSR bool values for table key equality. */
lmmc_status_t lmmc_lsr_bool_equal(int lhs, int rhs, int* out);
/** @brief Hash an LSR bool value for table keys. */
lmmc_status_t lmmc_lsr_bool_hash(int value, uint64_t* out);
/** @brief Construct a bool set with duplicate values removed. */
lmmc_status_t lmmc_lsr_bool_set_make(const int* values,
                                     size_t count,
                                     lmmc_lsr_bool_set_t* out);
void lmmc_lsr_bool_set_destroy(lmmc_lsr_bool_set_t* set);
lmmc_status_t lmmc_lsr_bool_set_contains(const lmmc_lsr_bool_set_t* set,
                                         int value,
                                         int* out);
lmmc_status_t lmmc_lsr_bool_set_subset(const lmmc_lsr_bool_set_t* lhs,
                                       const lmmc_lsr_bool_set_t* rhs,
                                       int* out);
lmmc_status_t lmmc_lsr_bool_set_union(const lmmc_lsr_bool_set_t* lhs,
                                      const lmmc_lsr_bool_set_t* rhs,
                                      lmmc_lsr_bool_set_t* out);
lmmc_status_t lmmc_lsr_bool_set_intersection(const lmmc_lsr_bool_set_t* lhs,
                                             const lmmc_lsr_bool_set_t* rhs,
                                             lmmc_lsr_bool_set_t* out);
lmmc_status_t lmmc_lsr_bool_set_difference(const lmmc_lsr_bool_set_t* lhs,
                                           const lmmc_lsr_bool_set_t* rhs,
                                           lmmc_lsr_bool_set_t* out);
lmmc_status_t lmmc_lsr_bool_set_symmetric_difference(
    const lmmc_lsr_bool_set_t* lhs,
    const lmmc_lsr_bool_set_t* rhs,
    lmmc_lsr_bool_set_t* out);
/** @brief Compare UTF-8 text values for LSR table key equality. */
lmmc_status_t lmmc_lsr_text_equal(const char* lhs, const char* rhs, int* out);
/** @brief Hash a UTF-8 text value for LSR table keys. */
lmmc_status_t lmmc_lsr_text_hash(const char* value, uint64_t* out);
/** @brief Construct a text set with duplicate values removed. */
lmmc_status_t lmmc_lsr_text_set_make(const char* const* values,
                                     size_t count,
                                     lmmc_lsr_text_set_t* out);
void lmmc_lsr_text_set_destroy(lmmc_lsr_text_set_t* set);
lmmc_status_t lmmc_lsr_text_set_contains(const lmmc_lsr_text_set_t* set,
                                         const char* value,
                                         int* out);
lmmc_status_t lmmc_lsr_text_set_subset(const lmmc_lsr_text_set_t* lhs,
                                       const lmmc_lsr_text_set_t* rhs,
                                       int* out);
lmmc_status_t lmmc_lsr_text_set_union(const lmmc_lsr_text_set_t* lhs,
                                      const lmmc_lsr_text_set_t* rhs,
                                      lmmc_lsr_text_set_t* out);
lmmc_status_t lmmc_lsr_text_set_intersection(const lmmc_lsr_text_set_t* lhs,
                                             const lmmc_lsr_text_set_t* rhs,
                                             lmmc_lsr_text_set_t* out);
lmmc_status_t lmmc_lsr_text_set_difference(const lmmc_lsr_text_set_t* lhs,
                                           const lmmc_lsr_text_set_t* rhs,
                                           lmmc_lsr_text_set_t* out);
lmmc_status_t lmmc_lsr_text_set_symmetric_difference(
    const lmmc_lsr_text_set_t* lhs,
    const lmmc_lsr_text_set_t* rhs,
    lmmc_lsr_text_set_t* out);

/** @brief Construct an LSR complex value from real and imaginary parts. */
lmmc_status_t lmmc_lsr_math_complex(lmmc_real_t real,
                                    lmmc_real_t imag,
                                    lmmc_complex_t* out);
/** @brief Extract the real part of an LSR complex value. */
lmmc_status_t lmmc_lsr_math_real(const lmmc_complex_t* z, lmmc_real_t* out);
/** @brief Extract the imaginary part of an LSR complex value. */
lmmc_status_t lmmc_lsr_math_imag(const lmmc_complex_t* z, lmmc_real_t* out);
/** @brief Compute the complex conjugate for std.math.conj. */
lmmc_status_t lmmc_lsr_math_conj(const lmmc_complex_t* z, lmmc_complex_t* out);
/** @brief Compute the complex absolute value for std.math.abs. */
lmmc_status_t lmmc_lsr_math_complex_abs(const lmmc_complex_t* z,
                                        lmmc_real_t* out);
/** @brief Compare complex values for LSR structural key equality. */
lmmc_status_t lmmc_lsr_math_complex_equal(const lmmc_complex_t* lhs,
                                          const lmmc_complex_t* rhs,
                                          int* out);
/** @brief Hash a finite complex value for LSR table keys. */
lmmc_status_t lmmc_lsr_math_complex_hash(const lmmc_complex_t* z,
                                         uint64_t* out);
/** @brief Construct a finite complex set with duplicate values removed. */
lmmc_status_t lmmc_lsr_complex_set_make(const lmmc_complex_t* values,
                                        size_t count,
                                        lmmc_lsr_complex_set_t* out);
void lmmc_lsr_complex_set_destroy(lmmc_lsr_complex_set_t* set);
lmmc_status_t lmmc_lsr_complex_set_contains(
    const lmmc_lsr_complex_set_t* set,
    const lmmc_complex_t* value,
    int* out);
lmmc_status_t lmmc_lsr_complex_set_subset(
    const lmmc_lsr_complex_set_t* lhs,
    const lmmc_lsr_complex_set_t* rhs,
    int* out);
lmmc_status_t lmmc_lsr_complex_set_union(
    const lmmc_lsr_complex_set_t* lhs,
    const lmmc_lsr_complex_set_t* rhs,
    lmmc_lsr_complex_set_t* out);
lmmc_status_t lmmc_lsr_complex_set_intersection(
    const lmmc_lsr_complex_set_t* lhs,
    const lmmc_lsr_complex_set_t* rhs,
    lmmc_lsr_complex_set_t* out);
lmmc_status_t lmmc_lsr_complex_set_difference(
    const lmmc_lsr_complex_set_t* lhs,
    const lmmc_lsr_complex_set_t* rhs,
    lmmc_lsr_complex_set_t* out);
lmmc_status_t lmmc_lsr_complex_set_symmetric_difference(
    const lmmc_lsr_complex_set_t* lhs,
    const lmmc_lsr_complex_set_t* rhs,
    lmmc_lsr_complex_set_t* out);

lmmc_status_t lmmc_lsr_math_sin(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_math_cos(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_math_tan(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_math_pow(lmmc_real_t x, lmmc_real_t y,
                                lmmc_real_t* out);
lmmc_status_t lmmc_lsr_math_asin(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_math_acos(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_math_atan(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_math_sqrt(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_math_exp(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_math_ln(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_math_log(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_math_log_base(lmmc_real_t x, lmmc_real_t base,
                                     lmmc_real_t* out);
lmmc_status_t lmmc_lsr_math_log10(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_math_abs(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_math_floor(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_math_ceil(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_math_round(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_math_clamp(lmmc_real_t x, lmmc_real_t lo,
                                  lmmc_real_t hi, lmmc_real_t* out);

lmmc_status_t lmmc_lsr_units_convert(lmmc_real_t x,
                                      const char* from_unit,
                                      const char* to_unit,
                                      lmmc_real_t* out);
lmmc_status_t lmmc_lsr_units_convert_from_si(lmmc_real_t x,
                                             const char* to_unit,
                                             lmmc_real_t* out);
lmmc_status_t lmmc_lsr_units_convert_num(lmmc_real_t x,
                                         const char* to_unit,
                                         lmmc_real_t* out);
lmmc_status_t lmmc_lsr_units_strip(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_units_strip_num(lmmc_real_t x,
                                       const char* unit,
                                       lmmc_real_t* out);
lmmc_status_t lmmc_lsr_units_strip_scalar(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_units_is_dimensionless_num(lmmc_real_t x, int* out);
lmmc_status_t lmmc_lsr_units_is_dimensionless(const char* unit, int* out);

lmmc_status_t lmmc_lsr_stats_mean(const lmmc_real_t* values, size_t count,
                                  lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_median(const lmmc_real_t* values, size_t count,
                                    lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_var(const lmmc_real_t* values, size_t count,
                                 lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_std(const lmmc_real_t* values, size_t count,
                                 lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_quantile(const lmmc_real_t* values, size_t count,
                                      lmmc_real_t q, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_cov(const lmmc_real_t* x,
                                 const lmmc_real_t* y,
                                 size_t count,
                                 lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_corr(const lmmc_real_t* x,
                                  const lmmc_real_t* y,
                                  size_t count,
                                  lmmc_real_t* out);

lmmc_status_t lmmc_lsr_random_seed(lmmc_rng_t* rng, uint64_t seed);
lmmc_status_t lmmc_lsr_random_rand(lmmc_rng_t* rng, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_random_randint(lmmc_rng_t* rng, int64_t lo,
                                      int64_t hi, int64_t* out);
lmmc_status_t lmmc_lsr_random_normal(lmmc_rng_t* rng, lmmc_real_t mean,
                                     lmmc_real_t stddev, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_random_choice(lmmc_rng_t* rng,
                                     const lmmc_real_t* values,
                                     size_t count,
                                     lmmc_real_t* out);
void lmmc_lsr_random_default_deinit(void);
lmmc_status_t lmmc_lsr_random_default_seed(uint64_t seed);
lmmc_status_t lmmc_lsr_random_default_rand(lmmc_real_t* out);
lmmc_status_t lmmc_lsr_random_default_randint(int64_t lo,
                                              int64_t hi,
                                              int64_t* out);
lmmc_status_t lmmc_lsr_random_default_normal(lmmc_real_t mean,
                                             lmmc_real_t stddev,
                                             lmmc_real_t* out);
lmmc_status_t lmmc_lsr_random_default_choice(const lmmc_real_t* values,
                                             size_t count,
                                             lmmc_real_t* out);

lmmc_status_t lmmc_lsr_linalg_shape(const lmmc_mat_t* a,
                                    size_t* out_rows,
                                    size_t* out_cols);
lmmc_status_t lmmc_lsr_linalg_shape_vec(const lmmc_mat_t* a,
                                        lmmc_vec_t* out);
lmmc_status_t lmmc_lsr_linalg_eye(size_t n, lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_diag(const lmmc_vec_t* diagonal,
                                   lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_dot(const lmmc_vec_t* a,
                                  const lmmc_vec_t* b,
                                  lmmc_real_t* out);
lmmc_status_t lmmc_lsr_linalg_cross(const lmmc_vec_t* a,
                                    const lmmc_vec_t* b,
                                    lmmc_vec_t* out);
lmmc_status_t lmmc_lsr_linalg_norm(const lmmc_vec_t* x,
                                   lmmc_real_t* out);
lmmc_status_t lmmc_lsr_linalg_vec_add(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out);
lmmc_status_t lmmc_lsr_linalg_vec_add_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out);
lmmc_status_t lmmc_lsr_linalg_vec_sub(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out);
lmmc_status_t lmmc_lsr_linalg_vec_sub_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out);
lmmc_status_t lmmc_lsr_linalg_scalar_sub_vec(lmmc_real_t scalar,
                                             const lmmc_vec_t* x,
                                             lmmc_vec_t* out);
lmmc_status_t lmmc_lsr_linalg_vec_mul(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out);
lmmc_status_t lmmc_lsr_linalg_vec_mul_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out);
lmmc_status_t lmmc_lsr_linalg_vec_div(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out);
lmmc_status_t lmmc_lsr_linalg_vec_div_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out);
lmmc_status_t lmmc_lsr_linalg_scalar_div_vec(lmmc_real_t scalar,
                                             const lmmc_vec_t* x,
                                             lmmc_vec_t* out);
lmmc_status_t lmmc_lsr_linalg_vec_pow(const lmmc_vec_t* base,
                                      const lmmc_vec_t* exponent,
                                      lmmc_vec_t* out);
lmmc_status_t lmmc_lsr_linalg_vec_pow_scalar(const lmmc_vec_t* base,
                                             lmmc_real_t exponent,
                                             lmmc_vec_t* out);
lmmc_status_t lmmc_lsr_linalg_vec_scale(const lmmc_vec_t* x,
                                        lmmc_real_t alpha,
                                        lmmc_vec_t* out);
lmmc_status_t lmmc_lsr_linalg_mat_add(const lmmc_mat_t* a,
                                      const lmmc_mat_t* b,
                                      lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_mat_add_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_mat_sub(const lmmc_mat_t* a,
                                      const lmmc_mat_t* b,
                                      lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_mat_sub_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_scalar_sub_mat(lmmc_real_t scalar,
                                             const lmmc_mat_t* a,
                                             lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_mat_mul_elem(const lmmc_mat_t* a,
                                           const lmmc_mat_t* b,
                                           lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_mat_mul_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_mat_div(const lmmc_mat_t* a,
                                      const lmmc_mat_t* b,
                                      lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_mat_div_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_scalar_div_mat(lmmc_real_t scalar,
                                             const lmmc_mat_t* a,
                                             lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_mat_pow_elem(const lmmc_mat_t* base,
                                           const lmmc_mat_t* exponent,
                                           lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_mat_pow_scalar(const lmmc_mat_t* base,
                                             lmmc_real_t exponent,
                                             lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_mat_scale(const lmmc_mat_t* a,
                                        lmmc_real_t alpha,
                                        lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_vec_compare(const lmmc_vec_t* a,
                                          const lmmc_vec_t* b,
                                          lmmc_lsr_compare_op_t op,
                                          lmmc_lsr_bool_vec_t* out);
lmmc_status_t lmmc_lsr_linalg_vec_compare_scalar(const lmmc_vec_t* a,
                                                 lmmc_lsr_compare_op_t op,
                                                 lmmc_real_t scalar,
                                                 lmmc_lsr_bool_vec_t* out);
lmmc_status_t lmmc_lsr_linalg_mat_compare(const lmmc_mat_t* a,
                                          const lmmc_mat_t* b,
                                          lmmc_lsr_compare_op_t op,
                                          lmmc_lsr_bool_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_mat_compare_scalar(const lmmc_mat_t* a,
                                                 lmmc_lsr_compare_op_t op,
                                                 lmmc_real_t scalar,
                                                 lmmc_lsr_bool_mat_t* out);
void lmmc_lsr_bool_vec_destroy(lmmc_lsr_bool_vec_t* vec);
void lmmc_lsr_bool_mat_destroy(lmmc_lsr_bool_mat_t* mat);
lmmc_status_t lmmc_lsr_linalg_matmul(const lmmc_mat_t* a,
                                     const lmmc_mat_t* b,
                                     lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_matvec(const lmmc_mat_t* a,
                                     const lmmc_vec_t* x,
                                     lmmc_vec_t* out);
lmmc_status_t lmmc_lsr_linalg_transpose(const lmmc_mat_t* a,
                                        lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_adjoint(const lmmc_mat_t* a,
                                      lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_det(const lmmc_mat_t* a,
                                  lmmc_real_t* out);
lmmc_status_t lmmc_lsr_linalg_inv(const lmmc_mat_t* a,
                                  lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_rank(const lmmc_mat_t* a,
                                   size_t* out_rank);
lmmc_status_t lmmc_lsr_linalg_trace(const lmmc_mat_t* a,
                                    lmmc_real_t* out);
lmmc_status_t lmmc_lsr_linalg_solve_left(const lmmc_mat_t* a,
                                         const lmmc_mat_t* b,
                                         lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_solve_right(const lmmc_mat_t* b,
                                          const lmmc_mat_t* a,
                                          lmmc_mat_t* out);
lmmc_status_t lmmc_lsr_linalg_eig(const lmmc_mat_t* a,
                                  lmmc_eigen_gen_full_result_t* out);
lmmc_status_t lmmc_lsr_linalg_svd(const lmmc_mat_t* a,
                                  lmmc_svd_result_t* out);

lmmc_status_t lmmc_lsr_linalg_eig_table(const lmmc_mat_t* a,
                                        lmmc_lsr_eig_table_t* out);
size_t lmmc_lsr_eig_table_count(const lmmc_lsr_eig_table_t* table);
const char* lmmc_lsr_eig_table_key(const lmmc_lsr_eig_table_t* table,
                                   size_t index);
const lmmc_mat_t* lmmc_lsr_eig_table_get(const lmmc_lsr_eig_table_t* table,
                                         const char* key);
void lmmc_lsr_eig_table_destroy(lmmc_lsr_eig_table_t* table);

lmmc_status_t lmmc_lsr_linalg_svd_table(const lmmc_mat_t* a,
                                        lmmc_lsr_svd_table_t* out);
size_t lmmc_lsr_svd_table_count(const lmmc_lsr_svd_table_t* table);
const char* lmmc_lsr_svd_table_key(const lmmc_lsr_svd_table_t* table,
                                   size_t index);
const lmmc_mat_t* lmmc_lsr_svd_table_get(const lmmc_lsr_svd_table_t* table,
                                         const char* key);
void lmmc_lsr_svd_table_destroy(lmmc_lsr_svd_table_t* table);

#ifdef __cplusplus
}
#endif

#endif
