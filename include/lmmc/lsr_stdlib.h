/**
 * @file lsr_stdlib.h
 * @brief 面向 LSR 的数值标准库适配器.
 *
 * 这些函数为 Lamina std.math 绑定提供稳定的 C ABI.
 * 数值与复数原语由 LMMC 提供,Expr 处理与符号语义由 LMCAS 提供.
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

/** @brief 将 LMMC 状态码映射为稳定的 LSR 诊断名称. */
const char* lmmc_lsr_error_name(lmmc_status_t status);

/** @brief std.linalg.eig 使用的 LSR table<text, matrix> 视图. */
typedef struct {
    lmmc_mat_t values_real;
    lmmc_mat_t values_imag;
    lmmc_mat_t vectors_real;
    lmmc_mat_t vectors_imag;
} lmmc_lsr_eig_table_t;

/** @brief std.linalg.svd 使用的 LSR table<text, matrix> 视图. */
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

/** @brief 返回 LSR std.math 常量 pi. */
lmmc_status_t lmmc_lsr_math_pi(lmmc_real_t* out);
/** @brief 返回 LSR std.math 常量 e. */
lmmc_status_t lmmc_lsr_math_e(lmmc_real_t* out);
/** @brief 返回 LSR std.math 常量 phi. */
lmmc_status_t lmmc_lsr_math_phi(lmmc_real_t* out);

/** @brief 返回 LSR std.constants 条目数. */
size_t lmmc_lsr_constants_count(void);
/** @brief 按索引返回 LSR std.constants 条目名;越界索引映射为 NULL. */
const char* lmmc_lsr_constants_name(size_t index);
/** @brief 按名称返回 LSR std.constants 数值. */
lmmc_status_t lmmc_lsr_constants_get(const char* name, lmmc_real_t* out);
/** @brief 按名称返回 LSR std.constants 单位字符串;无单位条目映射为 NULL. */
const char* lmmc_lsr_constants_unit(const char* name);
/** @brief 按索引返回完整的 LSR std.constants 条目. */
lmmc_status_t lmmc_lsr_constants_entry(size_t index,
                                       const char** out_name,
                                       lmmc_real_t* out_value,
                                       const char** out_unit);

/** @brief 返回虚数单位 std.math.I. */
lmmc_status_t lmmc_lsr_math_I(lmmc_complex_t* out);

/** @brief 按 LSR 结构键语义比较有限数值. */
lmmc_status_t lmmc_lsr_num_equal(lmmc_real_t lhs, lmmc_real_t rhs, int* out);
/** @brief 为 LSR 表键计算有限数值哈希. */
lmmc_status_t lmmc_lsr_num_hash(lmmc_real_t value, uint64_t* out);
/** @brief 构造有限数值集合,并合并重复值. */
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
/** @brief 按 LSR 表键语义比较 bool 值. */
lmmc_status_t lmmc_lsr_bool_equal(int lhs, int rhs, int* out);
/** @brief 为 LSR 表键计算 bool 哈希. */
lmmc_status_t lmmc_lsr_bool_hash(int value, uint64_t* out);
/** @brief 构造 bool 集合,并合并重复值. */
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
/** @brief 按 LSR 表键语义比较 UTF-8 文本值. */
lmmc_status_t lmmc_lsr_text_equal(const char* lhs, const char* rhs, int* out);
/** @brief 为 LSR 表键计算 UTF-8 文本哈希. */
lmmc_status_t lmmc_lsr_text_hash(const char* value, uint64_t* out);
/** @brief 构造文本集合,并合并重复值. */
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

/** @brief 由实部与虚部构造 LSR 复数值. */
lmmc_status_t lmmc_lsr_math_complex(lmmc_real_t real,
                                    lmmc_real_t imag,
                                    lmmc_complex_t* out);
/** @brief 提取 LSR 复数值的实部. */
lmmc_status_t lmmc_lsr_math_real(const lmmc_complex_t* z, lmmc_real_t* out);
/** @brief 提取 LSR 复数值的虚部. */
lmmc_status_t lmmc_lsr_math_imag(const lmmc_complex_t* z, lmmc_real_t* out);
/** @brief 为 std.math.conj 计算复共轭. */
lmmc_status_t lmmc_lsr_math_conj(const lmmc_complex_t* z, lmmc_complex_t* out);
/** @brief 为 std.math.abs 计算复数模. */
lmmc_status_t lmmc_lsr_math_complex_abs(const lmmc_complex_t* z,
                                        lmmc_real_t* out);
/** @brief 按 LSR 结构键语义比较复数值. */
lmmc_status_t lmmc_lsr_math_complex_equal(const lmmc_complex_t* lhs,
                                          const lmmc_complex_t* rhs,
                                          int* out);
/** @brief 为 LSR 表键计算有限复数哈希. */
lmmc_status_t lmmc_lsr_math_complex_hash(const lmmc_complex_t* z,
                                         uint64_t* out);
/** @brief 构造有限复数集合,并合并重复值. */
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
lmmc_status_t lmmc_lsr_stats_normal_pdf(lmmc_real_t x, lmmc_real_t mean,
                                        lmmc_real_t stddev,
                                        lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_normal_cdf(lmmc_real_t x, lmmc_real_t mean,
                                        lmmc_real_t stddev,
                                        lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_normal_quantile(lmmc_real_t p,
                                             lmmc_real_t mean,
                                             lmmc_real_t stddev,
                                             lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_t_pdf(lmmc_real_t x, lmmc_real_t df,
                                   lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_t_cdf(lmmc_real_t x, lmmc_real_t df,
                                   lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_t_quantile(lmmc_real_t p, lmmc_real_t df,
                                        lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_chi2_pdf(lmmc_real_t x, lmmc_real_t df,
                                      lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_chi2_cdf(lmmc_real_t x, lmmc_real_t df,
                                      lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_chi2_quantile(lmmc_real_t p, lmmc_real_t df,
                                           lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_f_pdf(lmmc_real_t x, lmmc_real_t df1,
                                   lmmc_real_t df2, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_f_cdf(lmmc_real_t x, lmmc_real_t df1,
                                   lmmc_real_t df2, lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_f_quantile(lmmc_real_t p, lmmc_real_t df1,
                                        lmmc_real_t df2,
                                        lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_gamma_pdf(lmmc_real_t x, lmmc_real_t shape,
                                       lmmc_real_t scale,
                                       lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_gamma_cdf(lmmc_real_t x, lmmc_real_t shape,
                                       lmmc_real_t scale,
                                       lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_gamma_quantile(lmmc_real_t p,
                                            lmmc_real_t shape,
                                            lmmc_real_t scale,
                                            lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_beta_pdf(lmmc_real_t x, lmmc_real_t alpha,
                                      lmmc_real_t beta,
                                      lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_beta_cdf(lmmc_real_t x, lmmc_real_t alpha,
                                      lmmc_real_t beta,
                                      lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_beta_quantile(lmmc_real_t p,
                                           lmmc_real_t alpha,
                                           lmmc_real_t beta,
                                           lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_binomial_pmf(size_t k, size_t n,
                                          lmmc_real_t p,
                                          lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_binomial_cdf(size_t k, size_t n,
                                          lmmc_real_t p,
                                          lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_poisson_pmf(size_t k, lmmc_real_t lambda,
                                         lmmc_real_t* out);
lmmc_status_t lmmc_lsr_stats_poisson_cdf(size_t k, lmmc_real_t lambda,
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
/**
 * @brief Default RNG operations use one allocation-free independent stream
 * per thread. Deinitialization resets only the calling thread's stream.
 */
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
lmmc_status_t lmmc_lsr_linalg_mat_pow_int(const lmmc_mat_t* base,
                                          int64_t exponent,
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
