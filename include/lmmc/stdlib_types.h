/** @file stdlib_types.h */
#ifndef LMMC_STDLIB_TYPES_H
#define LMMC_STDLIB_TYPES_H

#include "lmmc/complex.h"
#include "lmmc/dense_types.h"
#include "lmmc/status.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief 将 LMMC 状态码映射为 LSR 规范定义的稳定诊断名称。 */
const char* lmmc_std_error_name(lmmc_status_t status);

/** @brief std.linalg.eig 使用的 LSR 标准 table<text, matrix> 视图。 */
typedef struct {
    lmmc_mat_t values_real;
    lmmc_mat_t values_imag;
    lmmc_mat_t vectors_real;
    lmmc_mat_t vectors_imag;
} lmmc_std_eig_table_t;

/** @brief std.linalg.svd 使用的 LSR 标准 table<text, matrix> 视图。 */
typedef struct {
    lmmc_mat_t U;
    lmmc_mat_t S;
    lmmc_mat_t Vt;
} lmmc_std_svd_table_t;

typedef enum {
    LMMC_STD_COMPARE_EQ = 0,
    LMMC_STD_COMPARE_NE = 1,
    LMMC_STD_COMPARE_LT = 2,
    LMMC_STD_COMPARE_LE = 3,
    LMMC_STD_COMPARE_GT = 4,
    LMMC_STD_COMPARE_GE = 5
} lmmc_std_compare_op_t;

typedef struct {
    size_t size;
    uint8_t* data;
    int owns_data;
} lmmc_std_bool_vec_t;

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    uint8_t* data;
    int owns_data;
} lmmc_std_bool_mat_t;

typedef struct {
    size_t size;
    lmmc_real_t* data;
    int owns_data;
} lmmc_std_num_set_t;

typedef struct {
    size_t size;
    lmmc_complex_t* data;
    int owns_data;
} lmmc_std_complex_set_t;

typedef struct {
    size_t size;
    uint8_t* data;
    int owns_data;
} lmmc_std_bool_set_t;

typedef struct {
    size_t size;
    char** data;
    int owns_data;
} lmmc_std_text_set_t;

#ifdef __cplusplus
}
#endif

#endif
