/** @file eigen_svd_factorization.c
 * @brief 高矩阵 SVD 工作区、符号归一化与有序因子组装。
 */
#include "internal/eigen_svd_internal.h"
#include "eigen_internal.h"
#include "memory_bridge.h"

typedef struct {
    lmmc_mat_t v;
    lmmc_real_t *d;
    lmmc_real_t *e;
} svd_workspace_t;

static void destroy_workspace(svd_workspace_t *work)
{
    lmmc_mat_destroy(&work->v);
    lmmc_memory_free(work->d);
    lmmc_memory_free(work->e);
}

static lmmc_status_t create_factors(size_t m, size_t n,
                                   lmmc_svd_result_t *result)
{
    lmmc_status_t status = lmmc_mat_create(m, m, &result->U);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_create(n, &result->sigma);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_mat_create(n, n, &result->Vt);
}

static lmmc_status_t create_workspace(size_t n, svd_workspace_t *work)
{
    lmmc_status_t status = lmmc_mat_create(n, n, &work->v);
    if (status != LMMC_STATUS_OK) return status;
    work->d = lmmc_memory_alloc_array(n, sizeof(*work->d));
    work->e = lmmc_memory_alloc_array(n, sizeof(*work->e));
    if (!work->d || !work->e) return LMMC_STATUS_ALLOCATION_FAILED;
    return LMMC_STATUS_OK;
}

static void normalize_signs(lmmc_real_t *d, lmmc_mat_t *v)
{
    for (size_t i = 0; i < v->cols; ++i) {
        if (d[i] < 0.0) {
            d[i] = -d[i];
            for (size_t k = 0; k < v->rows; ++k) {
                MAT_ELEM(v, k, i) = -MAT_ELEM(v, k, i);
            }
        }
    }
}

static void descending_order(const lmmc_real_t *d, size_t n, size_t *order)
{
    for (size_t i = 0; i < n; ++i) order[i] = i;
    for (size_t i = 0; i + 1 < n; ++i) {
        size_t maximum = i;
        for (size_t j = i + 1; j < n; ++j) {
            if (d[order[j]] > d[order[maximum]]) maximum = j;
        }
        if (maximum != i) {
            size_t tmp = order[i];
            order[i] = order[maximum];
            order[maximum] = tmp;
        }
    }
}

static lmmc_status_t reorder_left_vectors(lmmc_mat_t *u, size_t n,
                                         const size_t *order)
{
    const size_t m = u->rows;
    lmmc_real_t *columns = lmmc_memory_alloc_array_2d(m, n, sizeof(*columns));
    if (!columns) return LMMC_STATUS_ALLOCATION_FAILED;
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            columns[j * m + i] = MAT_ELEM(u, i, j);
        }
    }
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            MAT_ELEM(u, i, j) = columns[order[j] * m + i];
        }
    }
    lmmc_memory_free(columns);
    return LMMC_STATUS_OK;
}

static lmmc_status_t assemble_factors(svd_workspace_t *work,
                                     lmmc_svd_result_t *result)
{
    const size_t n = result->sigma.size;
    normalize_signs(work->d, &work->v);
    size_t *order = lmmc_memory_alloc_array(n, sizeof(*order));
    if (!order) return LMMC_STATUS_ALLOCATION_FAILED;
    descending_order(work->d, n, order);
    for (size_t i = 0; i < n; ++i) {
        result->sigma.data[i] = work->d[order[i]];
        for (size_t k = 0; k < n; ++k) {
            MAT_ELEM(&result->Vt, i, k) = MAT_ELEM(&work->v, k, order[i]);
        }
    }
    lmmc_status_t status = reorder_left_vectors(&result->U, n, order);
    lmmc_memory_free(order);
    return status;
}

lmmc_status_t lmmc_svd_tall(const lmmc_mat_t *a, lmmc_real_t input_scale,
                          lmmc_svd_result_t *result)
{
    svd_workspace_t work = {0};
    lmmc_svd_result_t factors = {0};
    lmmc_status_t status = create_factors(a->rows, a->cols, &factors);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = create_workspace(a->cols, &work);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_svd_bidiagonalize(a, input_scale, work.d, work.e,
                                   &factors.U, &work.v);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_svd_diagonalize(work.d, work.e, a->cols, &factors.U, &work.v);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = assemble_factors(&work, &factors);
cleanup:
    destroy_workspace(&work);
    if (status == LMMC_STATUS_OK) {
        *result = factors;
    } else {
        lmmc_svd_result_destroy(&factors);
    }
    return status;
}
