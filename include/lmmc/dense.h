/**
 * @file dense.h
 * @brief 稠密矩阵 / 向量数据结构与基本运算（含 BLAS Level 1 / 2 接口）。
 *
 * 本模块提供按行优先存储的二维稠密矩阵 ::lmmc_mat_t 与一维稠密向量
 * ::lmmc_vec_t ，并涵盖创建、包装、销毁、复制、转置、矩阵乘法、
 * 范数、AXPY 等常用线性代数操作。
 */
#ifndef LMMC_DENSE_H
#define LMMC_DENSE_H

#include <stddef.h>
#include "lmmc/status.h"
#include "lmmc/config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 行优先存储的稠密矩阵。
 *
 * 元素 @c (i,j) 位于 @c data[i * stride + j] ，其中 @c stride>=cols 。
 * 当 ::lmmc_mat_t::owns_data 非零时， ::lmmc_mat_destroy 会释放 @c data 。
 */
typedef struct {
    size_t rows;          /**< 行数。 */
    size_t cols;          /**< 列数。 */
    size_t stride;        /**< 行步距（每行实际占用的元素数，>= @c cols ）。 */
    lmmc_real_t* data;    /**< 数据缓冲区起始地址。 */
    int owns_data;        /**< 是否拥有缓冲区所有权（非 0 表示销毁时释放）。 */
} lmmc_mat_t;

/**
 * @brief 一维稠密向量。
 *
 * 当 ::lmmc_vec_t::owns_data 非零时， ::lmmc_vec_destroy 会释放 @c data 。
 */
typedef struct {
    size_t size;          /**< 元素个数。 */
    lmmc_real_t* data;    /**< 数据缓冲区起始地址。 */
    int owns_data;        /**< 是否拥有缓冲区所有权。 */
} lmmc_vec_t;

/**
 * @brief 分配并初始化一个 @p rows × @p cols 的稠密矩阵，所有元素置零。
 *
 * 内部通过 lmmc_alloc 分配 rows*cols 个 lmmc_real_t 的连续缓冲区，
 * stride 设为 cols（紧凑存储），owns_data 置 1。
 *
 * @param[in]  rows    行数，必须 > 0。
 * @param[in]  cols    列数，必须 > 0。
 * @param[out] out_mat 输出矩阵结构体（调用方提供栈/堆上的结构体地址）。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — rows 或 cols 为 0，或 out_mat 为 NULL，或 rows*cols 溢出。
 * - ::LMMC_STATUS_ALLOCATION_FAILED — 内存分配失败。
 *
 * @par 副作用
 * - 分配堆内存（通过 lmmc_alloc），调用方必须配对调用 ::lmmc_mat_destroy 释放。
 * - out_mat 的所有字段被覆写。
 *
 * @par 线程安全
 * 本函数本身无共享状态，多线程可并发调用（前提是 lmmc_alloc 线程安全）。
 */
lmmc_status_t lmmc_mat_create(size_t rows, size_t cols, lmmc_mat_t* out_mat);

/**
 * @brief 用外部缓冲区构造矩阵视图（不拥有内存，不分配）。
 *
 * 将已有的连续内存区域包装为 lmmc_mat_t，owns_data 置 0，
 * 因此 ::lmmc_mat_destroy 不会释放 data 指针。
 * 调用方需保证 data 的生命周期覆盖该视图的使用期。
 *
 * @param[in]  rows    行数，必须 > 0。
 * @param[in]  cols    列数，必须 > 0。
 * @param[in]  stride  行步距（相邻行首元素间距），要求 >= @p cols 。
 * @param[in]  data    外部缓冲区指针，至少容纳 rows*stride 个 lmmc_real_t。不可为 NULL。
 * @param[out] out_mat 输出矩阵视图。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 参数非法。
 *
 * @par 副作用
 * - 无内存分配。out_mat 的所有字段被覆写。
 * - 对视图的写操作会直接修改 data 指向的外部缓冲区。
 */
lmmc_status_t lmmc_mat_wrap(size_t rows, size_t cols, size_t stride, lmmc_real_t* data, lmmc_mat_t* out_mat);

/**
 * @brief 销毁矩阵，释放其拥有的底层缓冲区。
 *
 * 若 mat->owns_data 非零，调用 lmmc_free 释放 mat->data；
 * 否则仅将结构体字段清零。对 NULL 指针或已销毁（data==NULL）的矩阵安全。
 *
 * @param[in,out] mat 待销毁的矩阵，可为 NULL。
 *
 * @par 副作用
 * - 若 owns_data：释放堆内存，mat->data 置 NULL，rows/cols/stride 清零。
 * - 若 !owns_data：仅清零结构体字段，不释放外部缓冲区。
 * - 销毁后该矩阵不可再用于任何运算（除非重新 create/wrap）。
 */
void lmmc_mat_destroy(lmmc_mat_t* mat);

/**
 * @brief 将矩阵所有元素设为同一值。
 *
 * 遍历 mat 的每个 (i,j) 位置，将 data[i*stride+j] 设为 value。
 *
 * @param[in,out] mat   待填充矩阵，必须已创建且 data 非 NULL。
 * @param[in]     value 填充值。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — mat 或 mat->data 为 NULL。
 *
 * @par 副作用
 * - 就地修改 mat->data 中的所有元素。
 * - 无内存分配/释放。
 */
lmmc_status_t lmmc_mat_fill(lmmc_mat_t* mat, lmmc_real_t value);

/**
 * @brief 逐元素复制矩阵内容：dst = src。
 *
 * src 与 dst 必须维度相同（rows 和 cols 均相等），但 stride 可以不同。
 * 不分配内存，dst 必须已通过 create 或 wrap 初始化。
 *
 * @param[in]  src 源矩阵。
 * @param[out] dst 目标矩阵，维度须与 src 一致。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 任一指针或 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — src 与 dst 维度不同。
 *
 * @par 副作用
 * - 覆写 dst->data 中的全部元素。src 不被修改。
 */
lmmc_status_t lmmc_mat_copy(const lmmc_mat_t* src, lmmc_mat_t* dst);

/**
 * @brief 计算矩阵转置：dst = src^T。
 *
 * dst 的维度必须为 (src->cols × src->rows)。src 与 dst 不可为同一矩阵
 * （不支持就地转置）。
 *
 * @param[in]  src 源矩阵 (m×n)。
 * @param[out] dst 目标矩阵，必须已创建为 (n×m)。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — dst 维度不是 (src->cols × src->rows)。
 *
 * @par 副作用
 * - 覆写 dst->data。src 不被修改。无内存分配。
 */
lmmc_status_t lmmc_mat_transpose_to(const lmmc_mat_t* src, lmmc_mat_t* dst);

/**
 * @brief 通用矩阵-矩阵乘法（GEMM）：@f$C \leftarrow \alpha \cdot \mathrm{op}(A) \cdot \mathrm{op}(B) + \beta \cdot C@f$。
 *
 * op(X) = X（transX==0）或 X^T（transX!=0）。
 * 设 op(A) 为 M×K，op(B) 为 K×N，则 C 必须为 M×N。
 * 当 LMMC_USE_BLAS 编译时，路由到外部 BLAS dgemm；否则使用内置分块三重循环（块大小 64）。
 *
 * @param[in]     alpha  标量乘子 α。
 * @param[in]     A      输入矩阵 A（不被修改）。
 * @param[in]     transA 非零表示对 A 取转置。
 * @param[in]     B      输入矩阵 B（不被修改）。
 * @param[in]     transB 非零表示对 B 取转置。
 * @param[in]     beta   标量乘子 β。beta==0 时 C 的旧值被忽略（可含 NaN）。
 * @param[in,out] C      输出矩阵，维度须为 M×N。C 不可与 A 或 B 别名。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — op(A) 的列数 ≠ op(B) 的行数，或 C 维度不匹配。
 *
 * @par 副作用
 * - 就地修改 C->data。A、B 不被修改。无内存分配。
 */
lmmc_status_t lmmc_mat_gemm(lmmc_real_t alpha, const lmmc_mat_t* A, int transA,
    const lmmc_mat_t* B, int transB, lmmc_real_t beta, lmmc_mat_t* C);

/**
 * @brief 通用矩阵-向量乘法（GEMV）：@f$y \leftarrow \alpha \cdot \mathrm{op}(A) \cdot x + \beta \cdot y@f$。
 *
 * op(A) 为 M×N 时，要求 x->size==N，y->size==M。
 * 当 LMMC_USE_BLAS 编译时路由到外部 BLAS dgemv。
 *
 * @param[in]     alpha  标量乘子 α。
 * @param[in]     A      输入矩阵 A（不被修改）。
 * @param[in]     transA 非零表示对 A 取转置。
 * @param[in]     x      输入向量 x（不被修改）。
 * @param[in]     beta   标量乘子 β。beta==0 时 y 的旧值被忽略。
 * @param[in,out] y      输出向量，长度须为 op(A) 的行数。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 向量长度与矩阵维度不匹配。
 *
 * @par 副作用
 * - 就地修改 y->data。A、x 不被修改。无内存分配。
 */
lmmc_status_t lmmc_mat_gemv(lmmc_real_t alpha, const lmmc_mat_t* A, int transA,
    const lmmc_vec_t* x, lmmc_real_t beta, lmmc_vec_t* y);

/**
 * @brief 简化矩阵乘法：c = a * b。
 *
 * 等价于 lmmc_mat_gemm(1.0, a, 0, b, 0, 0.0, c)。
 * a 为 m×k，b 为 k×n，c 必须已创建为 m×n。c 不可与 a 或 b 别名。
 *
 * @param[in]  a 左矩阵（不被修改）。
 * @param[in]  b 右矩阵（不被修改）。
 * @param[out] c 结果矩阵，旧内容被完全覆写。
 *
 * @return 同 ::lmmc_mat_gemm。
 *
 * @par 副作用
 * - 覆写 c->data。无内存分配。
 */
lmmc_status_t lmmc_mat_mul(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c);

/**
 * @brief 计算矩阵的 Frobenius 范数：@f$\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2}@f$。
 *
 * @param[in]  a        输入矩阵（不被修改）。
 * @param[out] out_norm 输出范数值。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 *
 * @par 副作用
 * - 无。纯计算，不修改任何输入。
 */
lmmc_status_t lmmc_mat_norm_fro(const lmmc_mat_t* a, lmmc_real_t* out_norm);

/**
 * @brief 分配并初始化一个长度为 @p size 的稠密向量，所有元素置零。
 *
 * @param[in]  size    元素个数，必须 > 0。
 * @param[out] out_vec 输出向量结构体。调用方需配对调用 ::lmmc_vec_destroy 释放。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — size==0 或 out_vec==NULL，或 size*sizeof 溢出。
 * - ::LMMC_STATUS_ALLOCATION_FAILED — 内存分配失败。
 *
 * @par 副作用
 * - 分配堆内存（owns_data=1）。out_vec 的所有字段被覆写。
 */
lmmc_status_t lmmc_vec_create(size_t size, lmmc_vec_t* out_vec);

/**
 * @brief 用外部缓冲区构造向量视图（不拥有内存，不分配）。
 *
 * owns_data 置 0，::lmmc_vec_destroy 不会释放 data。
 * 调用方需保证 data 的生命周期覆盖该视图的使用期。
 *
 * @param[in]  size    元素个数，必须 > 0。
 * @param[in]  data    外部缓冲区，至少 size 个 lmmc_real_t。不可为 NULL。
 * @param[out] out_vec 输出向量视图。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 参数非法。
 *
 * @par 副作用
 * - 无内存分配。对视图的写操作直接修改外部 data。
 */
lmmc_status_t lmmc_vec_wrap(size_t size, lmmc_real_t* data, lmmc_vec_t* out_vec);

/**
 * @brief 销毁向量，释放其拥有的底层缓冲区。
 *
 * 若 vec->owns_data 非零则释放 vec->data；否则仅清零字段。
 * 对 NULL 指针安全。销毁后 vec 不可再用于运算。
 *
 * @param[in,out] vec 待销毁的向量，可为 NULL。
 *
 * @par 副作用
 * - 若 owns_data：释放堆内存。
 * - vec->data 置 NULL，size 清零。
 */
void lmmc_vec_destroy(lmmc_vec_t* vec);

/**
 * @brief 将向量所有元素设为同一值。
 *
 * @param[in,out] vec   待填充向量，必须已创建。
 * @param[in]     value 填充值。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — vec 或 vec->data 为 NULL。
 *
 * @par 副作用
 * - 就地修改 vec->data 中的所有元素。无内存分配。
 */
lmmc_status_t lmmc_vec_fill(lmmc_vec_t* vec, lmmc_real_t value);

/**
 * @brief 计算两个等长向量的内积：@f$\text{out\_dot} = \sum_i a_i \cdot b_i@f$。
 *
 * @param[in]  a       第一个向量（不被修改）。
 * @param[in]  b       第二个向量（不被修改），长度须与 a 相同。
 * @param[out] out_dot 输出内积值。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — a->size != b->size。
 *
 * @par 副作用
 * - 无。纯计算，不修改任何输入。
 */
lmmc_status_t lmmc_vec_dot(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_real_t* out_dot);

/**
 * @brief 简化矩阵-向量乘法：y = A * x。
 *
 * 等价于 lmmc_mat_gemv(1.0, a, 0, x, 0.0, y)。
 * a 为 m×n，x->size==n，y->size==m。y 的旧内容被完全覆写。
 *
 * @param[in]  a 输入矩阵（不被修改）。
 * @param[in]  x 输入向量（不被修改）。
 * @param[out] y 输出向量，旧内容被覆写。
 *
 * @return 同 ::lmmc_mat_gemv。
 *
 * @par 副作用
 * - 覆写 y->data。无内存分配。
 */
lmmc_status_t lmmc_mat_vec_mul(const lmmc_mat_t* a, const lmmc_vec_t* x, lmmc_vec_t* y);

/**
 * @brief 计算向量的欧几里得（L2）范数：@f$\|x\|_2 = \sqrt{\sum_i x_i^2}@f$。
 *
 * 当 LMMC_USE_BLAS 编译时路由到 dnrm2。
 *
 * @param[in]  x        输入向量（不被修改），size 必须 > 0。
 * @param[out] out_norm 输出范数值（非负）。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL 或 size==0。
 *
 * @par 副作用
 * - 无。纯计算。
 */
lmmc_status_t lmmc_vec_norm2(const lmmc_vec_t* x, lmmc_real_t* out_norm);

/**
 * @brief 计算向量的无穷范数：@f$\|x\|_\infty = \max_i |x_i|@f$。
 *
 * @param[in]  x        输入向量（不被修改），size 必须 > 0。
 * @param[out] out_norm 输出范数值（非负）。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL 或 size==0。
 *
 * @par 副作用
 * - 无。纯计算。
 */
lmmc_status_t lmmc_vec_norm_inf(const lmmc_vec_t* x, lmmc_real_t* out_norm);

/**
 * @brief 就地向量缩放：@f$x \leftarrow \alpha \cdot x@f$。
 *
 * @param[in,out] x     待缩放向量，size 必须 > 0。
 * @param[in]     alpha 缩放因子。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — x 或 x->data 为 NULL 或 size==0。
 *
 * @par 副作用
 * - 就地修改 x->data 中的所有元素。无内存分配。
 */
lmmc_status_t lmmc_vec_scale(lmmc_vec_t* x, lmmc_real_t alpha);

/**
 * @brief 向量 AXPY 运算：@f$y \leftarrow \alpha \cdot x + y@f$。
 *
 * 当 LMMC_USE_BLAS 编译时路由到 daxpy。x 与 y 长度须相同。
 *
 * @param[in]     alpha 缩放因子。
 * @param[in]     x     输入向量（不被修改）。
 * @param[in,out] y     累加目标向量，就地修改。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL 或 size==0。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — x->size != y->size。
 *
 * @par 副作用
 * - 就地修改 y->data。x 不被修改。无内存分配。
 */
lmmc_status_t lmmc_vec_axpy(lmmc_real_t alpha, const lmmc_vec_t* x, lmmc_vec_t* y);

/**
 * @brief 逐元素复制向量内容：dst = src。
 *
 * src 与 dst 长度须相同。不分配内存。
 *
 * @param[in]  src 源向量（不被修改）。
 * @param[out] dst 目标向量，内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL 或 size==0。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — src->size != dst->size。
 *
 * @par 副作用
 * - 覆写 dst->data。src 不被修改。
 */
lmmc_status_t lmmc_vec_copy(const lmmc_vec_t* src, lmmc_vec_t* dst);

/**
 * @brief 交换两个等长向量的全部元素内容。
 *
 * @param[in,out] x 第一个向量。
 * @param[in,out] y 第二个向量，长度须与 x 相同。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL 或 size==0。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — x->size != y->size。
 *
 * @par 副作用
 * - 就地修改 x->data 和 y->data。无内存分配。
 */
lmmc_status_t lmmc_vec_swap(lmmc_vec_t* x, lmmc_vec_t* y);

/**
 * @brief 计算向量元素绝对值之和（L1 范数）：@f$\sum_i |x_i|@f$。
 *
 * @param[in]  x        输入向量（不被修改），size 必须 > 0。
 * @param[out] out_asum 输出绝对值和（非负）。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL 或 size==0。
 *
 * @par 副作用
 * - 无。纯计算。
 */
lmmc_status_t lmmc_vec_asum(const lmmc_vec_t* x, lmmc_real_t* out_asum);

/**
 * @brief 返回绝对值最大元素的下标（BLAS idamax 语义）。
 *
 * 若有多个相同最大值，返回最小下标。
 *
 * @param[in]  x       输入向量（不被修改），size 必须 > 0。
 * @param[out] out_idx 输出下标（0-based）。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL 或 size==0。
 *
 * @par 副作用
 * - 无。纯计算。
 */
lmmc_status_t lmmc_vec_iamax(const lmmc_vec_t* x, size_t* out_idx);

/**
 * @brief 同维矩阵逐元素加法：c = a + b。
 *
 * a、b、c 维度须完全相同。c 可与 a 或 b 别名（就地加法）。
 *
 * @param[in]  a 输入矩阵（不被修改）。
 * @param[in]  b 输入矩阵（不被修改）。
 * @param[out] c 输出矩阵，内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 维度不一致。
 *
 * @par 副作用
 * - 覆写 c->data。无内存分配。
 */
lmmc_status_t lmmc_mat_add(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c);

/**
 * @brief 同维矩阵逐元素减法：c = a - b。
 *
 * a、b、c 维度须完全相同。c 可与 a 或 b 别名。
 *
 * @param[in]  a 输入矩阵（不被修改）。
 * @param[in]  b 输入矩阵（不被修改）。
 * @param[out] c 输出矩阵，内容被覆写。
 *
 * @return 同 ::lmmc_mat_add。
 *
 * @par 副作用
 * - 覆写 c->data。无内存分配。
 */
lmmc_status_t lmmc_mat_sub(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c);

/**
 * @brief 就地矩阵标量乘：@f$A \leftarrow \alpha \cdot A@f$。
 *
 * @param[in,out] a     待缩放矩阵。
 * @param[in]     alpha 缩放因子。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — a 或 a->data 为 NULL。
 *
 * @par 副作用
 * - 就地修改 a->data 中的所有元素。无内存分配。
 */
lmmc_status_t lmmc_mat_scale(lmmc_mat_t* a, lmmc_real_t alpha);

/**
 * @brief 创建 n×n 单位矩阵（对角线为 1，其余为 0）。
 *
 * 内部调用 ::lmmc_mat_create 分配内存。
 *
 * @param[in]  n       矩阵阶数，必须 > 0。
 * @param[out] out_mat 输出矩阵，调用方需配对调用 ::lmmc_mat_destroy 释放。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — n==0 或 out_mat==NULL。
 * - ::LMMC_STATUS_ALLOCATION_FAILED — 内存分配失败。
 *
 * @par 副作用
 * - 分配堆内存（owns_data=1）。
 */
lmmc_status_t lmmc_mat_identity(size_t n, lmmc_mat_t* out_mat);

/**
 * @brief 计算方阵的迹：@f$\mathrm{tr}(A) = \sum_i A_{ii}@f$。
 *
 * 要求 a->rows == a->cols。
 *
 * @param[in]  a         输入方阵（不被修改）。
 * @param[out] out_trace 输出迹值。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL 或矩阵非方阵。
 *
 * @par 副作用
 * - 无。纯计算。
 */
lmmc_status_t lmmc_mat_trace(const lmmc_mat_t* a, lmmc_real_t* out_trace);

/**
 * @brief 计算方阵行列式（基于内部 LU 分解）。
 *
 * 内部分配临时工作空间进行 LU 分解，不修改输入矩阵。
 * 对 1×1 和 2×2 矩阵有特化快速路径。
 *
 * @param[in]  a       输入方阵（不被修改）。
 * @param[out] out_det 输出行列式值。奇异矩阵时 *out_det = 0。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功（包括奇异情况，此时 *out_det==0）。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL 或矩阵非方阵。
 * - ::LMMC_STATUS_ALLOCATION_FAILED — 临时内存分配失败。
 *
 * @par 副作用
 * - 分配并释放临时工作内存。输入矩阵不被修改。
 */
lmmc_status_t lmmc_mat_det(const lmmc_mat_t* a, lmmc_real_t* out_det);

/**
 * @brief 计算方阵的逆矩阵。
 *
 * 内部通过 LU 分解 + 逐列求解 @f$A \cdot \text{col}_i = e_i@f$ 实现。
 * 若 @p A 奇异，返回 ::LMMC_STATUS_SINGULAR_MATRIX 且 @p A_inv 不被修改。
 *
 * @param[in]  A     输入方阵。
 * @param[out] A_inv 输出逆矩阵，须已创建且与 @p A 同维。
 * @return ::LMMC_STATUS_OK 成功；奇异返回 ::LMMC_STATUS_SINGULAR_MATRIX 。
 */
lmmc_status_t lmmc_mat_inv(const lmmc_mat_t* A, lmmc_mat_t* A_inv);

/**
 * @brief 三角矩阵求解：@f$T x = b@f$ 。
 *
 * @param[in]  T         三角矩阵（上三角或下三角）。
 * @param[in]  upper     非零表示 @p T 为上三角；零表示下三角。
 * @param[in]  diag_unit 非零表示对角线视为 1（单位三角）。
 * @param[in]  b         右端向量。
 * @param[out] x         解向量，须已创建且与 @p b 同长。
 * @return ::LMMC_STATUS_OK 成功；对角线为零返回 ::LMMC_STATUS_SINGULAR_MATRIX 。
 */
lmmc_status_t lmmc_solve_triangular(const lmmc_mat_t* T, int upper, int diag_unit,
    const lmmc_vec_t* b, lmmc_vec_t* x);

/**
 * @brief 逐元素向量 Hadamard 乘积：@f$c_i = a_i \cdot b_i@f$。
 *
 * a、b、c 长度须相同。c 可与 a 或 b 别名（就地运算）。
 *
 * @param[in]  a 输入向量（不被修改）。
 * @param[in]  b 输入向量（不被修改）。
 * @param[out] c 输出向量，内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 维度不一致。
 *
 * @par 副作用
 * - 覆写 c->data。无内存分配。
 */
lmmc_status_t lmmc_vec_hadamard(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_vec_t* c);

/**
 * @brief 逐元素向量除法：@f$c_i = a_i / b_i@f$。
 *
 * 在写入任何输出前，预扫描 b 的所有元素检测零值。
 * a、b、c 长度须相同。c 可与 a 或 b 别名。
 *
 * @param[in]  a 被除向量（不被修改）。
 * @param[in]  b 除数向量（不被修改）。
 * @param[out] c 输出向量，内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 维度不一致。
 * - ::LMMC_STATUS_NUMERICAL_FAILURE — b 中存在零元素。
 *
 * @par 副作用
 * - 覆写 c->data。无内存分配。
 */
lmmc_status_t lmmc_vec_elementwise_div(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_vec_t* c);

/**
 * @brief 逐元素向量幂运算：@f$c_i = a_i^{b_i}@f$。
 *
 * 使用标准 pow() 函数。a、b、c 长度须相同。c 可与 a 或 b 别名。
 *
 * @param[in]  a 底数向量（不被修改）。
 * @param[in]  b 指数向量（不被修改）。
 * @param[out] c 输出向量，内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 维度不一致。
 *
 * @par 副作用
 * - 覆写 c->data。无内存分配。
 */
lmmc_status_t lmmc_vec_elementwise_pow(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_vec_t* c);

/**
 * @brief 逐元素矩阵 Hadamard 乘积：@f$C_{ij} = A_{ij} \cdot B_{ij}@f$。
 *
 * a、b、c 维度须完全相同。c 可与 a 或 b 别名（就地运算）。
 *
 * @param[in]  a 输入矩阵（不被修改）。
 * @param[in]  b 输入矩阵（不被修改）。
 * @param[out] c 输出矩阵，内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 维度不一致。
 *
 * @par 副作用
 * - 覆写 c->data。无内存分配。
 */
lmmc_status_t lmmc_mat_hadamard(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c);

/**
 * @brief 逐元素矩阵除法：@f$C_{ij} = A_{ij} / B_{ij}@f$。
 *
 * 在写入任何输出前，预扫描 b 的所有元素检测零值。
 * a、b、c 维度须完全相同。c 可与 a 或 b 别名。
 *
 * @param[in]  a 被除矩阵（不被修改）。
 * @param[in]  b 除数矩阵（不被修改）。
 * @param[out] c 输出矩阵，内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 维度不一致。
 * - ::LMMC_STATUS_NUMERICAL_FAILURE — b 中存在零元素。
 *
 * @par 副作用
 * - 覆写 c->data。无内存分配。
 */
lmmc_status_t lmmc_mat_elementwise_div(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c);

/**
 * @brief 逐元素矩阵幂运算：@f$C_{ij} = A_{ij}^{B_{ij}}@f$。
 *
 * 使用标准 pow() 函数。a、b、c 维度须完全相同。c 可与 a 或 b 别名。
 *
 * @param[in]  a 底数矩阵（不被修改）。
 * @param[in]  b 指数矩阵（不被修改）。
 * @param[out] c 输出矩阵，内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 维度不一致。
 *
 * @par 副作用
 * - 覆写 c->data。无内存分配。
 */
lmmc_status_t lmmc_mat_elementwise_pow(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c);

/**
 * @brief 逐元素向量大于比较：@f$\text{out}[i] = (a_i > b_i) \;?\; 1 : 0@f$。
 *
 * a、b 长度须相同。out 数组由调用方预分配，长度至少为 a->size。
 *
 * @param[in]  a   输入向量（不被修改）。
 * @param[in]  b   输入向量（不被修改）。
 * @param[out] out 输出整数数组，由调用方预分配。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 维度不一致。
 *
 * @par 副作用
 * - 覆写 out 数组。无内存分配。
 */
lmmc_status_t lmmc_vec_cmp_gt(const lmmc_vec_t* a, const lmmc_vec_t* b, int* out);

/**
 * @brief 逐元素向量小于比较：@f$\text{out}[i] = (a_i < b_i) \;?\; 1 : 0@f$。
 *
 * a、b 长度须相同。out 数组由调用方预分配。
 *
 * @param[in]  a   输入向量（不被修改）。
 * @param[in]  b   输入向量（不被修改）。
 * @param[out] out 输出整数数组，由调用方预分配。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 维度不一致。
 *
 * @par 副作用
 * - 覆写 out 数组。无内存分配。
 */
lmmc_status_t lmmc_vec_cmp_lt(const lmmc_vec_t* a, const lmmc_vec_t* b, int* out);

/**
 * @brief 逐元素向量大于等于比较：@f$\text{out}[i] = (a_i \geq b_i) \;?\; 1 : 0@f$。
 *
 * a、b 长度须相同。out 数组由调用方预分配。
 *
 * @param[in]  a   输入向量（不被修改）。
 * @param[in]  b   输入向量（不被修改）。
 * @param[out] out 输出整数数组，由调用方预分配。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 维度不一致。
 *
 * @par 副作用
 * - 覆写 out 数组。无内存分配。
 */
lmmc_status_t lmmc_vec_cmp_ge(const lmmc_vec_t* a, const lmmc_vec_t* b, int* out);

/**
 * @brief 逐元素向量小于等于比较：@f$\text{out}[i] = (a_i \leq b_i) \;?\; 1 : 0@f$。
 *
 * a、b 长度须相同。out 数组由调用方预分配。
 *
 * @param[in]  a   输入向量（不被修改）。
 * @param[in]  b   输入向量（不被修改）。
 * @param[out] out 输出整数数组，由调用方预分配。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 维度不一致。
 *
 * @par 副作用
 * - 覆写 out 数组。无内存分配。
 */
lmmc_status_t lmmc_vec_cmp_le(const lmmc_vec_t* a, const lmmc_vec_t* b, int* out);

/**
 * @brief 逐元素向量近似相等比较：@f$\text{out}[i] = (|a_i - b_i| \leq \text{tol}) \;?\; 1 : 0@f$。
 *
 * a、b 长度须相同。out 数组由调用方预分配。
 *
 * @param[in]  a   输入向量（不被修改）。
 * @param[in]  b   输入向量（不被修改）。
 * @param[in]  tol 容差阈值（非负）。
 * @param[out] out 输出整数数组，由调用方预分配。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 维度不一致。
 *
 * @par 副作用
 * - 覆写 out 数组。无内存分配。
 */
lmmc_status_t lmmc_vec_cmp_eq(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_real_t tol, int* out);

/**
 * @brief 计算矩阵的 1-范数（最大绝对列和）：@f$\|A\|_1 = \max_j \sum_i |A_{ij}|@f$。
 *
 * 无临时内存分配。
 *
 * @param[in]  a        输入矩阵（不被修改）。
 * @param[out] out_norm 输出范数值。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 *
 * @par 副作用
 * - 无。纯计算，不修改任何输入。
 */
lmmc_status_t lmmc_mat_norm1(const lmmc_mat_t* a, lmmc_real_t* out_norm);

/**
 * @brief 计算矩阵的无穷范数（最大绝对行和）：@f$\|A\|_\infty = \max_i \sum_j |A_{ij}|@f$。
 *
 * 无临时内存分配。
 *
 * @param[in]  a        输入矩阵（不被修改）。
 * @param[out] out_norm 输出范数值。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 *
 * @par 副作用
 * - 无。纯计算，不修改任何输入。
 */
lmmc_status_t lmmc_mat_norm_inf(const lmmc_mat_t* a, lmmc_real_t* out_norm);

/**
 * @brief 计算三维向量叉积：@f$c = a \times b@f$。
 *
 * 公式：
 * - @f$c_0 = a_1 b_2 - a_2 b_1@f$
 * - @f$c_1 = a_2 b_0 - a_0 b_2@f$
 * - @f$c_2 = a_0 b_1 - a_1 b_0@f$
 *
 * 支持别名（c 可与 a 或 b 为同一向量），内部使用临时变量计算。
 *
 * @param[in]  a 输入向量（不被修改），长度必须为 3。
 * @param[in]  b 输入向量（不被修改），长度必须为 3。
 * @param[out] c 输出向量，长度必须为 3，内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — 任一向量长度不为 3。
 *
 * @par 副作用
 * - 覆写 c->data。无内存分配。
 */
lmmc_status_t lmmc_vec_cross(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_vec_t* c);

/**
 * @brief 计算矩阵的数值秩（基于 SVD）。
 *
 * 内部通过 ::lmmc_svd 计算奇异值，统计大于容差 @p tol 的奇异值个数。
 * 当 @p tol <= 0 时，使用默认容差：max(rows, cols) * LMMC_REAL_EPSILON * σ₀，
 * 其中 σ₀ 为最大奇异值。
 *
 * @param[in]  a        输入矩阵（不被修改）。
 * @param[in]  tol      奇异值截断容差；<= 0 时使用默认值。
 * @param[out] out_rank 输出秩值。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL。
 * - ::LMMC_STATUS_ALLOCATION_FAILED — 内存分配失败。
 * - ::LMMC_STATUS_CONVERGENCE_FAILED — SVD 迭代未收敛。
 *
 * @par 副作用
 * - 内部分配并释放 SVD 工作空间。a 不被修改。
 */
lmmc_status_t lmmc_mat_rank(const lmmc_mat_t* a, lmmc_real_t tol, size_t* out_rank);

/**
 * @brief 计算方阵的整数幂：@f$\text{out} = A^n@f$。
 *
 * 使用二进制快速幂算法（重复平方法），乘法次数为 O(log|n|)。
 * - n=0 → 单位矩阵。
 * - n=1 → 直接复制。
 * - n<0 → 先计算 A 的逆矩阵，再对逆矩阵求 |n| 次幂。
 *
 * 内部分配两个 n×n 临时矩阵用于迭代，返回前释放。
 *
 * @param[in]  a   输入方阵（不被修改）。
 * @param[in]  n   整数指数，可为负、零或正。
 * @param[out] out 输出矩阵，须已创建且与 @p a 同维（n×n）。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL，或矩阵非方阵，或 out 维度不匹配。
 * - ::LMMC_STATUS_ALLOCATION_FAILED — 临时内存分配失败。
 * - ::LMMC_STATUS_SINGULAR_MATRIX — n<0 且 A 奇异（不可逆）。
 *
 * @par 副作用
 * - 覆写 out->data。内部分配并释放临时工作矩阵。a 不被修改。
 */
lmmc_status_t lmmc_mat_pow(const lmmc_mat_t* a, int n, lmmc_mat_t* out);

/**
 * @brief 计算矩阵右除：@f$X = B \cdot A^{-1}@f$。
 *
 * 通过转置系统求解：将问题转化为 @f$A^T \cdot X^T = B^T@f$，
 * 对 A^T 进行 LU 分解后逐列求解，最终得到 X。
 *
 * 要求 A 为方阵（n×n），B 的列数等于 A 的行数（即 B 为 m×n），
 * X 的维度为 m×n。
 *
 * @param[in]  B 被除矩阵（m×n，不被修改）。
 * @param[in]  A 除数方阵（n×n，不被修改）。
 * @param[out] X 输出矩阵（m×n），须已创建且维度为 (B->rows × A->cols)。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — 指针为 NULL，或 A 非方阵，或维度不匹配。
 * - ::LMMC_STATUS_ALLOCATION_FAILED — 临时内存分配失败。
 * - ::LMMC_STATUS_SINGULAR_MATRIX — A 奇异（不可逆）。
 *
 * @par 副作用
 * - 覆写 X->data。内部分配并释放临时工作矩阵。A、B 不被修改。
 */
lmmc_status_t lmmc_mat_rdiv(const lmmc_mat_t* B, const lmmc_mat_t* A, lmmc_mat_t* X);

/**
 * @brief 对向量逐元素应用标量函数：@f$\text{out}[i] = \text{func}(\text{in}[i])@f$。
 *
 * 将用户提供的标量函数 @p func 逐元素应用于输入向量 @p in ，结果写入 @p out 。
 * 支持就地操作（ @p in 与 @p out 可指向同一向量）。
 *
 * @param[in]  in   输入向量（不被修改，除非 in == out）。
 * @param[in]  func 标量函数指针，接受一个 lmmc_real_t 参数并返回 lmmc_real_t。不可为 NULL。
 * @param[out] out  输出向量，长度须与 @p in 相同。内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — in、out、in->data、out->data 或 func 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — in->size != out->size。
 *
 * @par 副作用
 * - 覆写 out->data。无内存分配。
 */
lmmc_status_t lmmc_vec_apply(const lmmc_vec_t* in, lmmc_real_t (*func)(lmmc_real_t), lmmc_vec_t* out);

/**
 * @brief 对矩阵逐元素应用标量函数：@f$\text{out}[i][j] = \text{func}(\text{in}[i][j])@f$。
 *
 * 将用户提供的标量函数 @p func 逐元素应用于输入矩阵 @p in ，结果写入 @p out 。
 * 支持就地操作（ @p in 与 @p out 可指向同一矩阵）。遍历时尊重各自的 stride。
 *
 * @param[in]  in   输入矩阵（不被修改，除非 in == out）。
 * @param[in]  func 标量函数指针，接受一个 lmmc_real_t 参数并返回 lmmc_real_t。不可为 NULL。
 * @param[out] out  输出矩阵，维度须与 @p in 相同（rows 和 cols 均相等）。内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — in、out、in->data、out->data 或 func 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — in 与 out 的 rows 或 cols 不同。
 *
 * @par 副作用
 * - 覆写 out->data。无内存分配。
 */
lmmc_status_t lmmc_mat_apply(const lmmc_mat_t* in, lmmc_real_t (*func)(lmmc_real_t), lmmc_mat_t* out);

/**
 * @brief 对向量逐元素求正弦：@f$\text{out}[i] = \sin(\text{in}[i])@f$。
 *
 * 等价于 lmmc_vec_apply(in, sin, out)。
 *
 * @param[in]  in  输入向量（不被修改，除非 in == out）。
 * @param[out] out 输出向量，长度须与 @p in 相同。内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — in、out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — in->size != out->size。
 */
lmmc_status_t lmmc_vec_apply_sin(const lmmc_vec_t* in, lmmc_vec_t* out);

/**
 * @brief 对向量逐元素求余弦：@f$\text{out}[i] = \cos(\text{in}[i])@f$。
 *
 * 等价于 lmmc_vec_apply(in, cos, out)。
 *
 * @param[in]  in  输入向量（不被修改，除非 in == out）。
 * @param[out] out 输出向量，长度须与 @p in 相同。内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — in、out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — in->size != out->size。
 */
lmmc_status_t lmmc_vec_apply_cos(const lmmc_vec_t* in, lmmc_vec_t* out);

/**
 * @brief 对向量逐元素求自然指数：@f$\text{out}[i] = \exp(\text{in}[i])@f$。
 *
 * 等价于 lmmc_vec_apply(in, exp, out)。
 *
 * @param[in]  in  输入向量（不被修改，除非 in == out）。
 * @param[out] out 输出向量，长度须与 @p in 相同。内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — in、out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — in->size != out->size。
 */
lmmc_status_t lmmc_vec_apply_exp(const lmmc_vec_t* in, lmmc_vec_t* out);

/**
 * @brief 对向量逐元素求自然对数：@f$\text{out}[i] = \ln(\text{in}[i])@f$。
 *
 * 等价于 lmmc_vec_apply(in, log, out)。
 *
 * @param[in]  in  输入向量（不被修改，除非 in == out）。
 * @param[out] out 输出向量，长度须与 @p in 相同。内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — in、out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — in->size != out->size。
 */
lmmc_status_t lmmc_vec_apply_log(const lmmc_vec_t* in, lmmc_vec_t* out);

/**
 * @brief 对向量逐元素求平方根：@f$\text{out}[i] = \sqrt{\text{in}[i]}@f$。
 *
 * 等价于 lmmc_vec_apply(in, sqrt, out)。
 *
 * @param[in]  in  输入向量（不被修改，除非 in == out）。
 * @param[out] out 输出向量，长度须与 @p in 相同。内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — in、out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — in->size != out->size。
 */
lmmc_status_t lmmc_vec_apply_sqrt(const lmmc_vec_t* in, lmmc_vec_t* out);

/**
 * @brief 对向量逐元素求绝对值：@f$\text{out}[i] = |\text{in}[i]|@f$。
 *
 * 等价于 lmmc_vec_apply(in, fabs, out)。
 *
 * @param[in]  in  输入向量（不被修改，除非 in == out）。
 * @param[out] out 输出向量，长度须与 @p in 相同。内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — in、out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — in->size != out->size。
 */
lmmc_status_t lmmc_vec_apply_abs(const lmmc_vec_t* in, lmmc_vec_t* out);

/**
 * @brief 对矩阵逐元素求正弦：@f$\text{out}[i][j] = \sin(\text{in}[i][j])@f$。
 *
 * 等价于 lmmc_mat_apply(in, sin, out)。
 *
 * @param[in]  in  输入矩阵（不被修改，除非 in == out）。
 * @param[out] out 输出矩阵，维度须与 @p in 相同。内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — in、out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — in 与 out 的 rows 或 cols 不同。
 */
lmmc_status_t lmmc_mat_apply_sin(const lmmc_mat_t* in, lmmc_mat_t* out);

/**
 * @brief 对矩阵逐元素求余弦：@f$\text{out}[i][j] = \cos(\text{in}[i][j])@f$。
 *
 * 等价于 lmmc_mat_apply(in, cos, out)。
 *
 * @param[in]  in  输入矩阵（不被修改，除非 in == out）。
 * @param[out] out 输出矩阵，维度须与 @p in 相同。内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — in、out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — in 与 out 的 rows 或 cols 不同。
 */
lmmc_status_t lmmc_mat_apply_cos(const lmmc_mat_t* in, lmmc_mat_t* out);

/**
 * @brief 对矩阵逐元素求自然指数：@f$\text{out}[i][j] = \exp(\text{in}[i][j])@f$。
 *
 * 等价于 lmmc_mat_apply(in, exp, out)。
 *
 * @param[in]  in  输入矩阵（不被修改，除非 in == out）。
 * @param[out] out 输出矩阵，维度须与 @p in 相同。内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — in、out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — in 与 out 的 rows 或 cols 不同。
 */
lmmc_status_t lmmc_mat_apply_exp(const lmmc_mat_t* in, lmmc_mat_t* out);

/**
 * @brief 对矩阵逐元素求自然对数：@f$\text{out}[i][j] = \ln(\text{in}[i][j])@f$。
 *
 * 等价于 lmmc_mat_apply(in, log, out)。
 *
 * @param[in]  in  输入矩阵（不被修改，除非 in == out）。
 * @param[out] out 输出矩阵，维度须与 @p in 相同。内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — in、out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — in 与 out 的 rows 或 cols 不同。
 */
lmmc_status_t lmmc_mat_apply_log(const lmmc_mat_t* in, lmmc_mat_t* out);

/**
 * @brief 对矩阵逐元素求平方根：@f$\text{out}[i][j] = \sqrt{\text{in}[i][j]}@f$。
 *
 * 等价于 lmmc_mat_apply(in, sqrt, out)。
 *
 * @param[in]  in  输入矩阵（不被修改，除非 in == out）。
 * @param[out] out 输出矩阵，维度须与 @p in 相同。内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — in、out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — in 与 out 的 rows 或 cols 不同。
 */
lmmc_status_t lmmc_mat_apply_sqrt(const lmmc_mat_t* in, lmmc_mat_t* out);

/**
 * @brief 对矩阵逐元素求绝对值：@f$\text{out}[i][j] = |\text{in}[i][j]|@f$。
 *
 * 等价于 lmmc_mat_apply(in, fabs, out)。
 *
 * @param[in]  in  输入矩阵（不被修改，除非 in == out）。
 * @param[out] out 输出矩阵，维度须与 @p in 相同。内容被覆写。
 *
 * @return
 * - ::LMMC_STATUS_OK — 成功。
 * - ::LMMC_STATUS_INVALID_ARGUMENT — in、out 或其 data 为 NULL。
 * - ::LMMC_STATUS_DIMENSION_MISMATCH — in 与 out 的 rows 或 cols 不同。
 */
lmmc_status_t lmmc_mat_apply_abs(const lmmc_mat_t* in, lmmc_mat_t* out);

#ifdef __cplusplus
}
#endif

#endif
