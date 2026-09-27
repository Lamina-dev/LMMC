#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "internal.h"
#include "eigen_internal.h"
#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/linear_algebra.h"
#include "eigen_general_internal.h"

/** @brief 释放通用特征分解结果持有的向量与矩阵缓冲区。 */

void lmmc_eigen_gen_full_result_destroy(lmmc_eigen_gen_full_result_t *result) {
    if (!result) {
        return;
    }
    lmmc_vec_destroy(&result->real_parts);
    lmmc_vec_destroy(&result->imag_parts);
    lmmc_mat_destroy(&result->vectors_real);
    lmmc_mat_destroy(&result->vectors_imag);
}

static lmmc_status_t create_full_result(size_t n, lmmc_eigen_gen_full_result_t* result)
{
    memset(result, 0, sizeof(*result));
    lmmc_status_t status = lmmc_vec_create(n, &result->real_parts);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    status = lmmc_vec_create(n, &result->imag_parts);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    status = lmmc_mat_create(n, n, &result->vectors_real);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    return lmmc_mat_create(n, n, &result->vectors_imag);
}

static lmmc_status_t eigenvectors_large(const lmmc_mat_t* a, lmmc_real_t matrix_scale,
                                        lmmc_eigen_gen_full_result_t* out_result)
{
    size_t n = a->rows;
    lmmc_mat_t scaled_a = {0};
    lmmc_eigen_gen_result_t eigenvalues = {0};
    lmmc_status_t status = lmmc_mat_create(n, n, &scaled_a);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    lmmc_eigen_copy_scaled(a, matrix_scale, &scaled_a);
    status = lmmc_eigen_general(&scaled_a, &eigenvalues);
    if (status != LMMC_STATUS_OK) {
        goto cleanup;
    }
    for (size_t i = 0; i < n; ++i) {
        out_result->real_parts.data[i] = eigenvalues.real_parts.data[i];
        out_result->imag_parts.data[i] = eigenvalues.imag_parts.data[i];
    }
    lmmc_eigen_gen_result_destroy(&eigenvalues);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            MAT_ELEM(&out_result->vectors_imag, i, j) = 0.0;
        }
    }
    status = lmmc_eigen_inverse_vectors(&scaled_a, out_result);
    if (status == LMMC_STATUS_OK) {
        status = lmmc_eigen_restore_scale(&out_result->real_parts,
                                          &out_result->imag_parts, matrix_scale);
    }
cleanup:
    lmmc_mat_destroy(&scaled_a);
    return status;
}

lmmc_status_t lmmc_eigen_general_full(const lmmc_mat_t* a,
                                     lmmc_eigen_gen_full_result_t* out_result)
{
    lmmc_real_t matrix_scale;
    if (!lmmc_mat_descriptor_is_valid(a) || !out_result) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_status_t status = lmmc_eigen_matrix_scale(a, &matrix_scale);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    size_t n = a->rows;
    status = create_full_result(n, out_result);
    if (status != LMMC_STATUS_OK) {
        goto fail;
    }
    if (n == 1) {
        out_result->real_parts.data[0] = MAT_ELEM(a, 0, 0);
        out_result->imag_parts.data[0] = 0.0;
        MAT_ELEM(&out_result->vectors_real, 0, 0) = 1.0;
        MAT_ELEM(&out_result->vectors_imag, 0, 0) = 0.0;
        return LMMC_STATUS_OK;
    }
if (n == 2) {
        lmmc_eigen_vectors_2x2(a, matrix_scale, out_result);
        return LMMC_STATUS_OK;
    }
    status = eigenvectors_large(a, matrix_scale, out_result);
    if (status == LMMC_STATUS_OK) {
        return status;
    }
fail:
    lmmc_eigen_gen_full_result_destroy(out_result);
    memset(out_result, 0, sizeof(*out_result));
    return status;
}
