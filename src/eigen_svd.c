/**
 * @file eigen_svd.c
 * @brief 奇异值分解、事务式伪逆与条件数计算。
 */
#include <math.h>
#include <string.h>
#include "internal/eigen_svd_internal.h"
#include "internal.h"
#include "eigen_internal.h"
#include "lmmc/linear_algebra.h"

static lmmc_status_t restore_singular_value_scale(lmmc_vec_t *sigma,
                                                  lmmc_real_t scale)
{
    for (size_t i = 0; i < sigma->size; ++i) {
        sigma->data[i] *= scale;
        if (!isfinite(sigma->data[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t transpose_factors(const lmmc_svd_result_t *source,
                                      lmmc_svd_result_t *result)
{
    lmmc_status_t status = lmmc_mat_create(source->Vt.rows, source->Vt.cols,
                                          &result->U);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    status = lmmc_vec_create(source->sigma.size, &result->sigma);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    status = lmmc_mat_create(source->U.rows, source->U.cols, &result->Vt);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    status = lmmc_mat_transpose_to(&source->Vt, &result->U);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    for (size_t i = 0; i < source->sigma.size; ++i) {
        result->sigma.data[i] = source->sigma.data[i];
    }
    return lmmc_mat_transpose_to(&source->U, &result->Vt);
}

static lmmc_status_t svd_wide(const lmmc_mat_t *a, lmmc_real_t scale,
                             lmmc_svd_result_t *result)
{
    lmmc_mat_t transpose;
    lmmc_svd_result_t temporary = {0};
    lmmc_status_t status = lmmc_mat_create(a->cols, a->rows, &transpose);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    status = lmmc_mat_transpose_to(a, &transpose);
    if (status == LMMC_STATUS_OK) {
        status = lmmc_svd_tall(&transpose, scale, &temporary);
    }
    lmmc_mat_destroy(&transpose);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    status = restore_singular_value_scale(&temporary.sigma, scale);
    if (status == LMMC_STATUS_OK) {
        status = transpose_factors(&temporary, result);
    }
    lmmc_svd_result_destroy(&temporary);
    return status;
}

lmmc_status_t lmmc_svd(const lmmc_mat_t *a, lmmc_svd_result_t *out_result)
{
    lmmc_real_t scale;
    if (!lmmc_mat_descriptor_is_valid(a) || !out_result) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_status_t status = lmmc_eigen_matrix_scale(a, &scale);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    lmmc_svd_result_t result = {0};
    if (a->rows >= a->cols) {
        status = lmmc_svd_tall(a, scale, &result);
        if (status == LMMC_STATUS_OK) {
            status = restore_singular_value_scale(&result.sigma, scale);
        }
    } else {
        status = svd_wide(a, scale, &result);
    }
    if (status == LMMC_STATUS_OK) {
        *out_result = result;
    } else {
        lmmc_svd_result_destroy(&result);
    }
    return status;
}

static int matrix_is_finite(const lmmc_mat_t *matrix)
{
    for (size_t i = 0; i < matrix->rows; ++i) {
        for (size_t j = 0; j < matrix->cols; ++j) {
            if (!isfinite(MAT_ELEM(matrix, i, j))) {
                return 0;
            }
        }
    }
    return 1;
}

/** @brief 复用私有 SVD 结果，将奇异值改写为倒数。 */
static lmmc_status_t prepare_reciprocals(lmmc_svd_result_t *svd,
                                        lmmc_real_t tol)
{
    if (!matrix_is_finite(&svd->U) || !matrix_is_finite(&svd->Vt)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    const size_t m = svd->U.rows;
    const size_t n = svd->Vt.rows;
    if (tol <= 0.0) {
        lmmc_real_t dimension = (lmmc_real_t)((m > n) ? m : n);
        tol = LMMC_REAL_EPSILON * dimension * svd->sigma.data[0];
    }
    if (!isfinite(tol)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    for (size_t k = 0; k < svd->sigma.size; ++k) {
        lmmc_real_t singular = svd->sigma.data[k];
        if (!isfinite(singular) || singular < 0.0) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        lmmc_real_t reciprocal = (singular > tol) ? 1.0 / singular : 0.0;
        if (!isfinite(reciprocal)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        svd->sigma.data[k] = reciprocal;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t reconstruct_entry(const lmmc_svd_result_t *svd,
                                       size_t i, size_t j, lmmc_real_t *value)
{
    lmmc_real_t sum = 0.0;
    for (size_t k = 0; k < svd->sigma.size; ++k) {
        const lmmc_real_t reciprocal = svd->sigma.data[k];
        if (reciprocal == 0.0) {
            continue;
        }
        lmmc_real_t product = MAT_ELEM(&svd->Vt, k, i) * MAT_ELEM(&svd->U, j, k);
        if (!isfinite(product)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        product *= reciprocal;
        if (!isfinite(product)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        sum += product;
        if (!isfinite(sum)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    *value = sum;
    return LMMC_STATUS_OK;
}

static lmmc_status_t reconstruct_pinv(const lmmc_svd_result_t *svd,
                                      lmmc_mat_t *candidate)
{
    for (size_t i = 0; i < candidate->rows; ++i) {
        for (size_t j = 0; j < candidate->cols; ++j) {
            lmmc_status_t status = reconstruct_entry(svd, i, j,
                                                     &MAT_ELEM(candidate, i, j));
            if (status != LMMC_STATUS_OK) {
                return status;
            }
        }
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t validate_pinv(const lmmc_mat_t *a, lmmc_real_t tol,
                                  const lmmc_mat_t *out_pinv)
{
    if (!a || !out_pinv || !a->data || !out_pinv->data) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows == 0 || a->cols == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (out_pinv->rows != a->cols || out_pinv->cols != a->rows) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    if (!lmmc_mat_descriptor_is_valid(a) || !lmmc_mat_descriptor_is_valid(out_pinv)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return isfinite(tol) ? LMMC_STATUS_OK : LMMC_STATUS_NUMERICAL_FAILURE;
}

lmmc_status_t lmmc_pinv(const lmmc_mat_t *a, lmmc_real_t tol, lmmc_mat_t *out_pinv)
{
    lmmc_status_t status = validate_pinv(a, tol, out_pinv);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    lmmc_svd_result_t svd;
    status = lmmc_svd(a, &svd);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    lmmc_mat_t candidate = {0};
    status = prepare_reciprocals(&svd, tol);
    if (status != LMMC_STATUS_OK) {
        goto cleanup;
    }
    status = lmmc_mat_create(a->cols, a->rows, &candidate);
    if (status != LMMC_STATUS_OK) {
        goto cleanup;
    }
    status = reconstruct_pinv(&svd, &candidate);
    if (status == LMMC_STATUS_OK) {
        for (size_t i = 0; i < candidate.rows; ++i) {
            memcpy(out_pinv->data + i * out_pinv->stride,
                   candidate.data + i * candidate.stride,
                   candidate.cols * sizeof(*candidate.data));
        }
    }
cleanup:
    lmmc_mat_destroy(&candidate);
    lmmc_svd_result_destroy(&svd);
    return status;
}

lmmc_status_t lmmc_cond(const lmmc_mat_t *a, lmmc_real_t *out_cond)
{
    if (!a || !out_cond || !a->data) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows == 0 || a->cols == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_svd_result_t svd;
    lmmc_status_t status = lmmc_svd(a, &svd);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    size_t p = svd.sigma.size;
    if (p == 0) {
        lmmc_svd_result_destroy(&svd);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_real_t s_max = svd.sigma.data[0];
    lmmc_real_t s_min = svd.sigma.data[p - 1];
    *out_cond = (s_min <= 0.0) ? INFINITY : s_max / s_min;
    lmmc_svd_result_destroy(&svd);
    return LMMC_STATUS_OK;
}

void lmmc_svd_result_destroy(lmmc_svd_result_t *result)
{
    if (!result) {
        return;
    }
    lmmc_mat_destroy(&result->U);
    lmmc_vec_destroy(&result->sigma);
    lmmc_mat_destroy(&result->Vt);
}
