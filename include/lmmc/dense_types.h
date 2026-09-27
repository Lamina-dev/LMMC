/** @file dense_types.h */
#ifndef LMMC_DENSE_TYPES_H
#define LMMC_DENSE_TYPES_H

#include <stddef.h>
#include "lmmc/config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 行优先存储的稠密矩阵。
 *
 * 元素 @c (i,j) 位于 @c data[i * stride + j] ,其中 @c stride>=cols 。
 * 当 ::lmmc_mat_t::owns_data 非零时, ::lmmc_mat_destroy 会释放 @c data 。
 */
typedef struct {
    size_t rows;          /**< 行数。 */
    size_t cols;          /**< 列数。 */
    size_t stride;        /**< 行步距(每行实际占用的元素数,>= @c cols )。 */
    lmmc_real_t* data;    /**< 数据缓冲区起始地址。 */
    int owns_data;        /**< 是否拥有缓冲区所有权(非 0 表示销毁时释放)。 */
} lmmc_mat_t;

/**
 * @brief 一维稠密向量。
 *
 * 当 ::lmmc_vec_t::owns_data 非零时, ::lmmc_vec_destroy 会释放 @c data 。
 */
typedef struct {
    size_t size;          /**< 元素个数。 */
    lmmc_real_t* data;    /**< 数据缓冲区起始地址。 */
    int owns_data;        /**< 是否拥有缓冲区所有权。 */
} lmmc_vec_t;

#ifdef __cplusplus
}
#endif

#endif
