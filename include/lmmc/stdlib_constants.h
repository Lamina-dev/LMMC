/** @file stdlib_constants.h */
#ifndef LMMC_STDLIB_CONSTANTS_H
#define LMMC_STDLIB_CONSTANTS_H

#include "lmmc/complex.h"
#include "lmmc/status.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief 返回 LSR 规范的 std.math 常量 pi。 */
lmmc_status_t lmmc_std_math_pi(lmmc_real_t* out);
/** @brief 返回 LSR 规范的 std.math 常量 e。 */
lmmc_status_t lmmc_std_math_e(lmmc_real_t* out);
/** @brief 返回 LSR 规范的 std.math 常量 phi。 */
lmmc_status_t lmmc_std_math_phi(lmmc_real_t* out);

/** @brief 返回 LSR 规范的 std.constants 条目数。 */
size_t lmmc_std_constants_count(void);
/** @brief 按索引返回 LSR 规范的 std.constants 条目名；索引越界时返回 NULL。 */
const char* lmmc_std_constants_name(size_t index);
/** @brief 按名称返回 LSR 规范的 std.constants 数值。 */
lmmc_status_t lmmc_std_constants_get(const char* name, lmmc_real_t* out);
/** @brief 按名称返回 LSR 规范的 std.constants 单位字符串；条目无单位时返回 NULL。 */
const char* lmmc_std_constants_unit(const char* name);
/** @brief 按索引返回完整的 LSR 规范 std.constants 条目。 */
lmmc_status_t lmmc_std_constants_entry(size_t index,
                                       const char** out_name,
                                       lmmc_real_t* out_value,
                                       const char** out_unit);

/** @brief 返回虚数单位 std.math.I。 */
lmmc_status_t lmmc_std_math_i(lmmc_complex_t* out);

lmmc_status_t lmmc_std_units_convert(lmmc_real_t x,
                                      const char* from_unit,
                                      const char* to_unit,
                                      lmmc_real_t* out);
lmmc_status_t lmmc_std_units_convert_from_si(lmmc_real_t x,
                                             const char* to_unit,
                                             lmmc_real_t* out);
lmmc_status_t lmmc_std_units_convert_num(lmmc_real_t x,
                                         const char* to_unit,
                                         lmmc_real_t* out);
lmmc_status_t lmmc_std_units_strip(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_std_units_strip_num(lmmc_real_t x,
                                       const char* unit,
                                       lmmc_real_t* out);
lmmc_status_t lmmc_std_units_strip_scalar(lmmc_real_t x, lmmc_real_t* out);
lmmc_status_t lmmc_std_units_is_dimensionless_num(lmmc_real_t x, int* out);
lmmc_status_t lmmc_std_units_is_dimensionless(const char* unit, int* out);

#ifdef __cplusplus
}
#endif

#endif
