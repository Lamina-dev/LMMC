/**
 * @file status.h
 * @brief LMMC 通用状态码定义。
 *
 * 库内几乎所有可能失败的接口都返回 ::lmmc_status_t ，调用方
 * 应根据返回值判断是否成功，并通过 ::lmmc_status_string 获取
 * 可读描述。
 */
#ifndef LMMC_STATUS_H
#define LMMC_STATUS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief LMMC 库统一返回的状态码。
 *
 * 任何返回该枚举的函数若返回非 ::LMMC_STATUS_OK ，
 * 输出参数的内容均不应被使用。
 */
typedef enum {
    LMMC_STATUS_OK = 0,                    /**< 操作成功。 */
    LMMC_STATUS_INVALID_ARGUMENT = 1,      /**< 输入参数非法（空指针、负值等）。 */
    LMMC_STATUS_DIMENSION_MISMATCH = 2,    /**< 矩阵 / 向量维度不匹配。 */
    LMMC_STATUS_ALLOCATION_FAILED = 3,     /**< 内存分配失败。 */
    LMMC_STATUS_SINGULAR_MATRIX = 4,       /**< 矩阵奇异，无法求解。 */
    LMMC_STATUS_NOT_IMPLEMENTED = 5,       /**< 接口暂未实现。 */
    LMMC_STATUS_NUMERICAL_FAILURE = 6,     /**< 数值计算失败（如溢出、除零）。 */
    LMMC_STATUS_NOT_POSITIVE_DEFINITE = 7, /**< 矩阵非正定。 */
    LMMC_STATUS_CONVERGENCE_FAILED = 8,    /**< 迭代算法未在限定步数内收敛。 */
    LMMC_STATUS_OUT_OF_RANGE = 9,          /**< 数值超出允许范围。 */
    LMMC_STATUS_INDEX_OUT_OF_BOUNDS = 10,  /**< 数组 / 矩阵下标越界。 */
    LMMC_STATUS_WARNING_MAX_DEPTH = 11,    /**< 自适应算法达到最大递归深度（结果可能仍可用）。 */
    LMMC_STATUS_EMPTY_INPUT = 12,          /**< 输入集合为空。 */
    LMMC_STATUS_UNIT_STRIP_TYPE_MISMATCH = 13, /**< 量纲剥离作用于非数值类型。 */
    LMMC_STATUS_UNIT_STRIP_OVERFLOW = 14,  /**< 量纲剥离发生表示范围溢出。 */
    LMMC_STATUS_UNIT_STRIP_INVALID = 15,   /**< 量纲剥离的单位转换阶段失败。 */
    LMMC_STATUS_UNIT_STRIP_LEGACY_SYNTAX = 16 /**< 使用了废弃的量纲剥离语法。 */
} lmmc_status_t;

/**
 * @brief 获取状态码对应的可读字符串。
 *
 * @param status 任意 ::lmmc_status_t 取值。
 * @return 指向静态字符串的指针，调用方不得释放，亦不会失效。
 *         未知状态码返回 @c "unknown" 。
 */
const char* lmmc_status_string(lmmc_status_t status);

#ifdef __cplusplus
}
#endif

#endif
