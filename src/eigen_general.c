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

void lmmc_eigen_copy_scaled(const lmmc_mat_t* a, lmmc_real_t scale, lmmc_mat_t* result)
{
    for (size_t i = 0; i < a->rows; ++i) {
        for (size_t j = 0; j < a->cols; ++j) {
            MAT_ELEM(result, i, j) = MAT_ELEM(a, i, j) / scale;
        }
    }
}

lmmc_status_t lmmc_eigen_restore_scale(lmmc_vec_t* real, lmmc_vec_t* imag,
                                       lmmc_real_t scale)
{
    for (size_t i = 0; i < real->size; ++i) {
        real->data[i] *= scale;
        imag->data[i] *= scale;
        if (!isfinite(real->data[i]) || !isfinite(imag->data[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    return LMMC_STATUS_OK;
}
void lmmc_eigen_values_2x2(
    lmmc_real_t a00, lmmc_real_t a01,
    lmmc_real_t a10, lmmc_real_t a11,
    lmmc_real_t *real0, lmmc_real_t *imag0,
    lmmc_real_t *real1, lmmc_real_t *imag1)
{
    const lmmc_real_t trace = a00 + a11;
    const lmmc_real_t determinant = a00 * a11 - a01 * a10;
    const lmmc_real_t discriminant =
        fma(trace, trace, -4.0 * determinant);
    if (discriminant >= 0.0) {
        const lmmc_real_t root = sqrt(discriminant);
        const lmmc_real_t q =
            0.5 * (trace + copysign(root, trace));
        *real0 = q;
        *real1 = q == 0.0 ? 0.0 : determinant / q;
        *imag0 = 0.0;
        *imag1 = 0.0;
    } else {
        const lmmc_real_t imaginary = sqrt(-discriminant) * 0.5;
        *real0 = trace * 0.5;
        *real1 = trace * 0.5;
        *imag0 = imaginary;
        *imag1 = -imaginary;
    }
}

/**
 * @brief 从实 Schur 形提取特征值.
 *
 * 1x1 对角块产生实特征值,2x2 对角块产生共轭复特征值对.
 */
static void extract_eigenvalues_from_schur(const lmmc_mat_t *H, size_t n,
                                           lmmc_real_t *re, lmmc_real_t *im) {
    size_t i = 0;
    while (i < n) {
        if (i + 1 == n || MAT_ELEM(H, i + 1, i) == 0.0) {
            /** 1x1 块产生实特征值. */
            re[i] = MAT_ELEM(H, i, i);
            im[i] = 0.0;
            i++;
        } else {
            /** 2x2 块产生共轭复特征值对. */
            lmmc_real_t a11 = MAT_ELEM(H, i, i);
            lmmc_real_t a12 = MAT_ELEM(H, i, i + 1);
            lmmc_real_t a21 = MAT_ELEM(H, i + 1, i);
            lmmc_real_t a22 = MAT_ELEM(H, i + 1, i + 1);
            lmmc_eigen_values_2x2(
                a11, a12, a21, a22,
                &re[i], &im[i], &re[i + 1], &im[i + 1]);
            i += 2;
        }
    }
}
static void eigenvalues_small(const lmmc_mat_t* a, lmmc_real_t matrix_scale,
                               lmmc_eigen_gen_result_t* out_result)
{
    if (a->rows == 1) {
        out_result->real_parts.data[0] = MAT_ELEM(a, 0, 0);
        out_result->imag_parts.data[0] = 0.0;
        return;
    }
        lmmc_real_t a00 = MAT_ELEM(a, 0, 0) / matrix_scale;
        lmmc_real_t a01 = MAT_ELEM(a, 0, 1) / matrix_scale;
        lmmc_real_t a10 = MAT_ELEM(a, 1, 0) / matrix_scale;
        lmmc_real_t a11 = MAT_ELEM(a, 1, 1) / matrix_scale;
        lmmc_eigen_values_2x2(
            a00, a01, a10, a11,
            &out_result->real_parts.data[0],
            &out_result->imag_parts.data[0],
            &out_result->real_parts.data[1],
            &out_result->imag_parts.data[1]);
        out_result->real_parts.data[0] *= matrix_scale;
        out_result->imag_parts.data[0] *= matrix_scale;
        out_result->real_parts.data[1] *= matrix_scale;
        out_result->imag_parts.data[1] *= matrix_scale;
}

static lmmc_status_t eigenvalues_large(const lmmc_mat_t* a, lmmc_real_t matrix_scale,
                                       lmmc_eigen_gen_result_t* out_result)
{
    size_t n = a->rows;
    lmmc_mat_t H, Q_mat;
    lmmc_status_t status = lmmc_mat_create(n, n, &H);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    status = lmmc_mat_create(n, n, &Q_mat);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&H);
        return status;
    }
    lmmc_eigen_copy_scaled(a, matrix_scale, &H);
    status = lmmc_eigen_schur_reduce(&H, &Q_mat);
    if (status == LMMC_STATUS_OK) {
        extract_eigenvalues_from_schur(&H, n,
            out_result->real_parts.data, out_result->imag_parts.data);
        status = lmmc_eigen_restore_scale(&out_result->real_parts,
                                          &out_result->imag_parts, matrix_scale);
    }
    lmmc_mat_destroy(&H);
    lmmc_mat_destroy(&Q_mat);
    return status;
}

lmmc_status_t lmmc_eigen_general(const lmmc_mat_t* a,
                                lmmc_eigen_gen_result_t* out_result)
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
    status = lmmc_vec_create(n, &out_result->real_parts);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    status = lmmc_vec_create(n, &out_result->imag_parts);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&out_result->real_parts);
        return status;
    }
if (n <= 2) {
        eigenvalues_small(a, matrix_scale, out_result);
        return LMMC_STATUS_OK;
    }
    status = eigenvalues_large(a, matrix_scale, out_result);
    if (status != LMMC_STATUS_OK) {
        lmmc_eigen_gen_result_destroy(out_result);
    }
    return status;
}
void lmmc_eigen_gen_result_destroy(lmmc_eigen_gen_result_t *result)
{
    if (!result) {
        return;
    }
    lmmc_vec_destroy(&result->real_parts);
    lmmc_vec_destroy(&result->imag_parts);
}
