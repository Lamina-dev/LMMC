/** @file stdlib_sets.h */
#ifndef LMMC_STDLIB_SETS_H
#define LMMC_STDLIB_SETS_H

#include "lmmc/stdlib_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief 按 LSR 规范的结构键语义比较有限数值。 */
lmmc_status_t lmmc_std_num_equal(lmmc_real_t lhs, lmmc_real_t rhs, int* out);
/** @brief 为 LSR 规范的表键计算有限数值哈希。 */
lmmc_status_t lmmc_std_num_hash(lmmc_real_t value, uint64_t* out);
/** @brief 构造去重的有限数值集合。 */
lmmc_status_t lmmc_std_num_set_make(const lmmc_real_t* values,
                                    size_t count,
                                    lmmc_std_num_set_t* out);
void lmmc_std_num_set_destroy(lmmc_std_num_set_t* set);
lmmc_status_t lmmc_std_num_set_contains(const lmmc_std_num_set_t* set,
                                        lmmc_real_t value,
                                        int* out);
lmmc_status_t lmmc_std_num_set_subset(const lmmc_std_num_set_t* lhs,
                                      const lmmc_std_num_set_t* rhs,
                                      int* out);
lmmc_status_t lmmc_std_num_set_union(const lmmc_std_num_set_t* lhs,
                                     const lmmc_std_num_set_t* rhs,
                                     lmmc_std_num_set_t* out);
lmmc_status_t lmmc_std_num_set_intersection(const lmmc_std_num_set_t* lhs,
                                            const lmmc_std_num_set_t* rhs,
                                            lmmc_std_num_set_t* out);
lmmc_status_t lmmc_std_num_set_difference(const lmmc_std_num_set_t* lhs,
                                          const lmmc_std_num_set_t* rhs,
                                          lmmc_std_num_set_t* out);
lmmc_status_t lmmc_std_num_set_symmetric_difference(
    const lmmc_std_num_set_t* lhs,
    const lmmc_std_num_set_t* rhs,
    lmmc_std_num_set_t* out);
/** @brief 按 LSR 规范的表键语义比较 bool 值。 */
lmmc_status_t lmmc_std_bool_equal(int lhs, int rhs, int* out);
/** @brief 为 LSR 规范的表键计算 bool 哈希。 */
lmmc_status_t lmmc_std_bool_hash(int value, uint64_t* out);
/** @brief 构造去重的 bool 集合。 */
lmmc_status_t lmmc_std_bool_set_make(const int* values,
                                     size_t count,
                                     lmmc_std_bool_set_t* out);
void lmmc_std_bool_set_destroy(lmmc_std_bool_set_t* set);
lmmc_status_t lmmc_std_bool_set_contains(const lmmc_std_bool_set_t* set,
                                         int value,
                                         int* out);
lmmc_status_t lmmc_std_bool_set_subset(const lmmc_std_bool_set_t* lhs,
                                       const lmmc_std_bool_set_t* rhs,
                                       int* out);
lmmc_status_t lmmc_std_bool_set_union(const lmmc_std_bool_set_t* lhs,
                                      const lmmc_std_bool_set_t* rhs,
                                      lmmc_std_bool_set_t* out);
lmmc_status_t lmmc_std_bool_set_intersection(const lmmc_std_bool_set_t* lhs,
                                             const lmmc_std_bool_set_t* rhs,
                                             lmmc_std_bool_set_t* out);
lmmc_status_t lmmc_std_bool_set_difference(const lmmc_std_bool_set_t* lhs,
                                           const lmmc_std_bool_set_t* rhs,
                                           lmmc_std_bool_set_t* out);
lmmc_status_t lmmc_std_bool_set_symmetric_difference(
    const lmmc_std_bool_set_t* lhs,
    const lmmc_std_bool_set_t* rhs,
    lmmc_std_bool_set_t* out);
/** @brief 按 LSR 规范的表键语义比较 UTF-8 文本值。 */
lmmc_status_t lmmc_std_text_equal(const char* lhs, const char* rhs, int* out);
/** @brief 为 LSR 规范的表键计算 UTF-8 文本哈希。 */
lmmc_status_t lmmc_std_text_hash(const char* value, uint64_t* out);
/** @brief 构造去重的文本集合。 */
lmmc_status_t lmmc_std_text_set_make(const char* const* values,
                                     size_t count,
                                     lmmc_std_text_set_t* out);
void lmmc_std_text_set_destroy(lmmc_std_text_set_t* set);
lmmc_status_t lmmc_std_text_set_contains(const lmmc_std_text_set_t* set,
                                         const char* value,
                                         int* out);
lmmc_status_t lmmc_std_text_set_subset(const lmmc_std_text_set_t* lhs,
                                       const lmmc_std_text_set_t* rhs,
                                       int* out);
lmmc_status_t lmmc_std_text_set_union(const lmmc_std_text_set_t* lhs,
                                      const lmmc_std_text_set_t* rhs,
                                      lmmc_std_text_set_t* out);
lmmc_status_t lmmc_std_text_set_intersection(const lmmc_std_text_set_t* lhs,
                                             const lmmc_std_text_set_t* rhs,
                                             lmmc_std_text_set_t* out);
lmmc_status_t lmmc_std_text_set_difference(const lmmc_std_text_set_t* lhs,
                                           const lmmc_std_text_set_t* rhs,
                                           lmmc_std_text_set_t* out);
lmmc_status_t lmmc_std_text_set_symmetric_difference(
    const lmmc_std_text_set_t* lhs,
    const lmmc_std_text_set_t* rhs,
    lmmc_std_text_set_t* out);

/** @brief 按 LSR 规范的结构键语义比较复数值。 */
lmmc_status_t lmmc_std_math_complex_equal(const lmmc_complex_t* lhs,
                                          const lmmc_complex_t* rhs,
                                          int* out);
/** @brief 为 LSR 规范的表键计算有限复数哈希。 */
lmmc_status_t lmmc_std_math_complex_hash(const lmmc_complex_t* z,
                                         uint64_t* out);
/** @brief 构造去重的有限复数集合。 */
lmmc_status_t lmmc_std_complex_set_make(const lmmc_complex_t* values,
                                        size_t count,
                                        lmmc_std_complex_set_t* out);
void lmmc_std_complex_set_destroy(lmmc_std_complex_set_t* set);
lmmc_status_t lmmc_std_complex_set_contains(
    const lmmc_std_complex_set_t* set,
    const lmmc_complex_t* value,
    int* out);
lmmc_status_t lmmc_std_complex_set_subset(
    const lmmc_std_complex_set_t* lhs,
    const lmmc_std_complex_set_t* rhs,
    int* out);
lmmc_status_t lmmc_std_complex_set_union(
    const lmmc_std_complex_set_t* lhs,
    const lmmc_std_complex_set_t* rhs,
    lmmc_std_complex_set_t* out);
lmmc_status_t lmmc_std_complex_set_intersection(
    const lmmc_std_complex_set_t* lhs,
    const lmmc_std_complex_set_t* rhs,
    lmmc_std_complex_set_t* out);
lmmc_status_t lmmc_std_complex_set_difference(
    const lmmc_std_complex_set_t* lhs,
    const lmmc_std_complex_set_t* rhs,
    lmmc_std_complex_set_t* out);
lmmc_status_t lmmc_std_complex_set_symmetric_difference(
    const lmmc_std_complex_set_t* lhs,
    const lmmc_std_complex_set_t* rhs,
    lmmc_std_complex_set_t* out);

#ifdef __cplusplus
}
#endif

#endif
