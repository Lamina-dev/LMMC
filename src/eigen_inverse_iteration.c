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

/** @brief 特征向量计算阶段独占的工作区，各移位复用同一缓冲区。 */
typedef struct {
    lmmc_mat_t shifted;
    size_t* pivots;
    lmmc_vec_t rhs;
    lmmc_vec_t solution;
} lmmc_eigen_inverse_workspace_t;

static void inverse_workspace_destroy(lmmc_eigen_inverse_workspace_t* work)
{
    lmmc_vec_destroy(&work->solution);
    lmmc_vec_destroy(&work->rhs);
    if (work->pivots) {
        lmmc_memory_free(work->pivots);
    }
    lmmc_mat_destroy(&work->shifted);
    memset(work, 0, sizeof(*work));
}

static lmmc_status_t inverse_workspace_resize(lmmc_eigen_inverse_workspace_t* work, size_t n)
{
    if (work->shifted.rows == n) {
        return LMMC_STATUS_OK;
    }
    inverse_workspace_destroy(work);
    lmmc_status_t status = lmmc_mat_create(n, n, &work->shifted);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    work->pivots = (size_t*)lmmc_memory_alloc_array(n, sizeof(size_t));
    if (!work->pivots) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    status = lmmc_vec_create(n, &work->rhs);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    return lmmc_vec_create(n, &work->solution);
}

static lmmc_real_t inverse_matrix_norm(const lmmc_mat_t* A)
{
    lmmc_real_t norm_A = 0.0;
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->rows; j++) {
            norm_A += MAT_ELEM(A, i, j) * MAT_ELEM(A, i, j);
        }
    }
    norm_A = sqrt(norm_A);
    if (norm_A < 1.0) {
        norm_A = 1.0;
    }
    return norm_A;
}

static void fill_real_shift(const lmmc_mat_t* A, lmmc_real_t shift, lmmc_mat_t* shifted)
{
    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < A->rows; j++) {
            MAT_ELEM(shifted, i, j) = MAT_ELEM(A, i, j) - ((i == j) ? shift : 0.0);
        }
    }
}

static lmmc_status_t factor_real_shift(const lmmc_mat_t* A, lmmc_real_t mu,
                                       lmmc_eigen_inverse_workspace_t* work)
{
    const lmmc_real_t eps = 2.2204460492503131e-16;
    lmmc_real_t norm_A = inverse_matrix_norm(A);
    lmmc_real_t perturb = eps * norm_A * (lmmc_real_t)A->rows;
    fill_real_shift(A, mu + perturb, &work->shifted);
    lmmc_status_t status = lmmc_lu_decompose_inplace(&work->shifted, work->pivots, NULL);
    if (status == LMMC_STATUS_SINGULAR_MATRIX) {
        perturb = sqrt(eps) * norm_A;
        fill_real_shift(A, mu + perturb, &work->shifted);
        status = lmmc_lu_decompose_inplace(&work->shifted, work->pivots, NULL);
    }
    return status;
}

static void fill_complex_shift(const lmmc_mat_t* A, lmmc_real_t alpha_p,
                                lmmc_real_t beta, lmmc_mat_t* big)
{
    size_t n = A->rows;
    size_t n2 = 2 * n;
    size_t i, j;
    for (i = 0; i < n2; i++) {
        for (j = 0; j < n2; j++) {
            MAT_ELEM(big, i, j) = 0.0;
        }
    }

    /** @brief 左上块为 A - alpha_p*I。 */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            MAT_ELEM(big, i, j) = MAT_ELEM(A, i, j) - ((i == j) ? alpha_p : 0.0);
        }
    }

    /** @brief 右上块为 beta*I。 */
    for (i = 0; i < n; i++) {
        MAT_ELEM(big, i, n + i) = beta;
    }

    /** @brief 左下块为 -beta*I。 */
    for (i = 0; i < n; i++) {
        MAT_ELEM(big, n + i, i) = -beta;
    }

    /** @brief 右下块为 A - alpha_p*I。 */
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            MAT_ELEM(big, n + i, n + j) = MAT_ELEM(A, i, j) - ((i == j) ? alpha_p : 0.0);
        }
    }
}

static void retry_complex_shift(const lmmc_mat_t* A, lmmc_real_t alpha_p,
                                 lmmc_real_t beta, lmmc_mat_t* big)
{
    size_t n = A->rows;
    size_t i, j;
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                lmmc_real_t aij = MAT_ELEM(A, i, j) - ((i == j) ? alpha_p : 0.0);
                MAT_ELEM(big, i, j) = aij;
                MAT_ELEM(big, n + i, n + j) = aij;
            }
        }
        for (i = 0; i < n; i++) {
            MAT_ELEM(big, i, n + i) = beta;
            MAT_ELEM(big, n + i, i) = -beta;
        }
}

static lmmc_status_t factor_complex_shift(const lmmc_mat_t* A, lmmc_real_t alpha,
                                          lmmc_real_t beta, lmmc_eigen_inverse_workspace_t* work)
{
    const lmmc_real_t eps = 2.2204460492503131e-16;
    lmmc_real_t norm_A = inverse_matrix_norm(A);
    lmmc_real_t perturb = eps * norm_A * (lmmc_real_t)A->rows;
    fill_complex_shift(A, alpha + perturb, beta, &work->shifted);
    lmmc_status_t status = lmmc_lu_decompose_inplace(&work->shifted, work->pivots, NULL);
    if (status == LMMC_STATUS_SINGULAR_MATRIX) {
        retry_complex_shift(A, alpha + sqrt(eps) * norm_A, beta, &work->shifted);
        status = lmmc_lu_decompose_inplace(&work->shifted, work->pivots, NULL);
    }
    return status;
}

static lmmc_real_t inverse_vector_norm(const lmmc_vec_t* vector)
{
    lmmc_real_t nrm = 0.0;
    for (size_t i = 0; i < vector->size; i++) nrm += vector->data[i] * vector->data[i];
    return sqrt(nrm);
}

static void inverse_start_vector(size_t n, lmmc_eigen_inverse_workspace_t* work)
{
    for (size_t i = 0; i < n; i++) {
        work->rhs.data[i] = ((i % 2 == 0) ? 1.0 : -1.0) / (lmmc_real_t)(i + 1);
    }
    if (work->rhs.size != n) {
        for (size_t i = 0; i < n; i++) {
            work->rhs.data[n + i] = 1.0 / (lmmc_real_t)(n + i + 1);
        }
    }
    lmmc_real_t nrm = inverse_vector_norm(&work->rhs);
    if (nrm > 0.0) {
        for (size_t i = 0; i < work->rhs.size; i++) work->rhs.data[i] /= nrm;
    }
}

static lmmc_status_t inverse_iterate(lmmc_eigen_inverse_workspace_t* work)
{
    const size_t max_iter = 20;
    for (size_t iter = 0; iter < max_iter; iter++) {
        lmmc_status_t status = lmmc_lu_solve(&work->shifted, work->pivots,
                                           &work->rhs, &work->solution);
        if (status != LMMC_STATUS_OK) {
            return status;
        }
        lmmc_real_t nrm = inverse_vector_norm(&work->solution);
        if (nrm == 0.0) {
            nrm = 1.0;
        }
        for (size_t i = 0; i < work->solution.size; i++) work->solution.data[i] /= nrm;
        for (size_t i = 0; i < work->solution.size; i++) {
            work->rhs.data[i] = work->solution.data[i];
        }
    }
    return LMMC_STATUS_OK;
}

static void store_real_vector(lmmc_eigen_gen_full_result_t* result, size_t column,
                               const lmmc_eigen_inverse_workspace_t* work)
{
    for (size_t j = 0; j < result->real_parts.size; j++) {
        MAT_ELEM(&result->vectors_real, j, column) = work->solution.data[j];
        MAT_ELEM(&result->vectors_imag, j, column) = 0.0;
    }
}

static void store_complex_vectors(lmmc_eigen_gen_full_result_t* result, size_t column,
                                   const lmmc_eigen_inverse_workspace_t* work)
{
    size_t n = result->real_parts.size;
    for (size_t j = 0; j < n; j++) {
        MAT_ELEM(&result->vectors_real, j, column) = work->solution.data[j];
        MAT_ELEM(&result->vectors_imag, j, column) = work->solution.data[n + j];
        MAT_ELEM(&result->vectors_real, j, column + 1) = work->solution.data[j];
        MAT_ELEM(&result->vectors_imag, j, column + 1) = -work->solution.data[n + j];
    }
}

lmmc_status_t lmmc_eigen_inverse_vectors(const lmmc_mat_t* a,
                                         lmmc_eigen_gen_full_result_t* result)
{
    size_t n = a->rows;
    lmmc_eigen_inverse_workspace_t work = {0};
    lmmc_status_t status = LMMC_STATUS_OK;
    for (size_t i = 0; i < n;) {
        int is_real = result->imag_parts.data[i] == 0.0;
        status = inverse_workspace_resize(&work, is_real ? n : 2 * n);
        if (status != LMMC_STATUS_OK) {
            break;
        }
        if (is_real) {
            status = factor_real_shift(a, result->real_parts.data[i], &work);
        } else {
            status = factor_complex_shift(a, result->real_parts.data[i],
                                           result->imag_parts.data[i], &work);
        }
        if (status != LMMC_STATUS_OK) {
            break;
        }
        inverse_start_vector(n, &work);
        status = inverse_iterate(&work);
        if (status != LMMC_STATUS_OK) {
            break;
        }
        if (is_real) {
            store_real_vector(result, i, &work);
            ++i;
        } else {
            store_complex_vectors(result, i, &work);
            i += 2;
        }
    }
    inverse_workspace_destroy(&work);
    return status;
}
