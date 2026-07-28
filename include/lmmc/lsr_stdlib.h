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

/** @brief Return std.math.i, the imaginary unit. */
lmmc_status_t lmmc_lsr_math_i(lmmc_complex_t* out);
/** @brief Return std.math.I, an alias of std.math.i. */
lmmc_status_t lmmc_lsr_math_I(lmmc_complex_t* out);

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
lmmc_status_t lmmc_lsr_math_log(lmmc_real_t x, lmmc_real_t base,
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
lmmc_status_t lmmc_lsr_units_strip(lmmc_real_t x, lmmc_real_t* out);
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

lmmc_status_t lmmc_lsr_linalg_shape(const lmmc_mat_t* a,
                                    size_t* out_rows,
                                    size_t* out_cols);
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
const lmmc_mat_t* lmmc_lsr_eig_table_get(const lmmc_lsr_eig_table_t* table,
                                         const char* key);
void lmmc_lsr_eig_table_destroy(lmmc_lsr_eig_table_t* table);

lmmc_status_t lmmc_lsr_linalg_svd_table(const lmmc_mat_t* a,
                                        lmmc_lsr_svd_table_t* out);
const lmmc_mat_t* lmmc_lsr_svd_table_get(const lmmc_lsr_svd_table_t* table,
                                         const char* key);
void lmmc_lsr_svd_table_destroy(lmmc_lsr_svd_table_t* table);

#ifdef __cplusplus
}
#endif

#endif
