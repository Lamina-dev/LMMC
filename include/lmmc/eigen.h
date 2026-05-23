#ifndef LMMC_EIGEN_H
#define LMMC_EIGEN_H

#include "lmmc/dense.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 对称特征值分解结果 */
typedef struct {
    lmmc_vec_t eigenvalues;   /* n 个特征值，升序排列 */
    lmmc_mat_t eigenvectors;  /* n×n 正交矩阵，列为特征向量 */
} lmmc_eigen_sym_result_t;

/* 一般特征值分解结果 */
typedef struct {
    lmmc_vec_t real_parts;    /* n 个特征值实部 */
    lmmc_vec_t imag_parts;    /* n 个特征值虚部 */
} lmmc_eigen_gen_result_t;

/* SVD 分解结果 */
typedef struct {
    lmmc_mat_t U;             /* m×m 左奇异向量矩阵 */
    lmmc_vec_t sigma;         /* min(m,n) 个奇异值，降序排列 */
    lmmc_mat_t Vt;            /* n×n 右奇异向量矩阵的转置 */
} lmmc_svd_result_t;

/* 对称特征值分解 */
lmmc_status_t lmmc_eigen_symmetric(
    const lmmc_mat_t* a,
    lmmc_eigen_sym_result_t* out_result
);

/* 一般特征值分解 */
lmmc_status_t lmmc_eigen_general(
    const lmmc_mat_t* a,
    lmmc_eigen_gen_result_t* out_result
);

/* 奇异值分解 */
lmmc_status_t lmmc_svd(
    const lmmc_mat_t* a,
    lmmc_svd_result_t* out_result
);

/* Moore-Penrose 伪逆 */
lmmc_status_t lmmc_pinv(
    const lmmc_mat_t* a,
    lmmc_real_t tol,
    lmmc_mat_t* out_pinv
);

/* 条件数 */
lmmc_status_t lmmc_cond(
    const lmmc_mat_t* a,
    lmmc_real_t* out_cond
);

/* 资源释放 */
void lmmc_eigen_sym_result_destroy(lmmc_eigen_sym_result_t* result);
void lmmc_eigen_gen_result_destroy(lmmc_eigen_gen_result_t* result);
void lmmc_svd_result_destroy(lmmc_svd_result_t* result);

#ifdef __cplusplus
}
#endif

#endif
