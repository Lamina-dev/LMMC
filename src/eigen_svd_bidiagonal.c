/** @file eigen_svd_bidiagonal.c
 * @brief Householder 双对角约化及左右正交因子累积。
 */
#include "internal/eigen_svd_internal.h"
#include "internal.h"
#include "eigen_internal.h"
#include "memory_bridge.h"


static void accumulate_reflector(lmmc_mat_t *factor, size_t start,
                                 size_t length, const lmmc_real_t *vector,
                                 lmmc_real_t tau)
{
    if (tau == 0.0) return;
    for (size_t row = 0; row < factor->rows; ++row) {
        lmmc_real_t sum = MAT_ELEM(factor, row, start);
        for (size_t i = 1; i < length; ++i) {
            sum += vector[i] * MAT_ELEM(factor, row, start + i);
        }
        sum *= tau;
        MAT_ELEM(factor, row, start) -= sum;
        for (size_t i = 1; i < length; ++i) {
            MAT_ELEM(factor, row, start + i) -= sum * vector[i];
        }
    }
}

static lmmc_real_t reduce_column(lmmc_mat_t *work, lmmc_mat_t *u,
                                 size_t k, lmmc_real_t *vector)
{
    const size_t length = work->rows - k;
    lmmc_real_t tau, beta;
    for (size_t i = 0; i < length; ++i) {
        vector[i] = MAT_ELEM(work, k + i, k);
    }
    householder_make(vector, length, &tau, &beta);
    if (k + 1 < work->cols) {
        householder_apply_left(work, k, length, k + 1, work->cols, vector, tau);
    }
    accumulate_reflector(u, k, length, vector, tau);
    return beta;
}

static lmmc_real_t reduce_row(lmmc_mat_t *work, lmmc_mat_t *v,
                              size_t k, lmmc_real_t *vector)
{
    const size_t length = work->cols - k - 1;
    lmmc_real_t tau, beta;
    for (size_t j = 0; j < length; ++j) {
        vector[j] = MAT_ELEM(work, k, k + 1 + j);
    }
    householder_make(vector, length, &tau, &beta);
    if (k + 1 < work->rows) {
        householder_apply_right(work, k + 1, work->rows, k + 1,
                                length, vector, tau);
    }
    accumulate_reflector(v, k + 1, length, vector, tau);
    return beta;
}

lmmc_status_t lmmc_svd_bidiagonalize(const lmmc_mat_t *a,
                                   lmmc_real_t input_scale,
                                   lmmc_real_t *d, lmmc_real_t *e,
                                   lmmc_mat_t *u, lmmc_mat_t *v)
{
    lmmc_mat_t work;
    lmmc_status_t status = lmmc_mat_create(a->rows, a->cols, &work);
    if (status != LMMC_STATUS_OK) return status;
    for (size_t i = 0; i < a->rows; ++i) {
        for (size_t j = 0; j < a->cols; ++j) {
            MAT_ELEM(&work, i, j) = MAT_ELEM(a, i, j) / input_scale;
        }
    }
    lmmc_fill_identity_unchecked(u);
    lmmc_fill_identity_unchecked(v);
    lmmc_real_t *vector = lmmc_memory_alloc_array(a->rows, sizeof(*vector));
    if (!vector) {
        lmmc_mat_destroy(&work);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    for (size_t k = 0; k < a->cols; ++k) {
        d[k] = reduce_column(&work, u, k, vector);
        e[k] = (k + 1 < a->cols) ? reduce_row(&work, v, k, vector) : 0.0;
    }
    lmmc_memory_free(vector);
    lmmc_mat_destroy(&work);
    return LMMC_STATUS_OK;
}
