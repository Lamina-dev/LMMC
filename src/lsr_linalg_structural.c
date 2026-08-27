#include "lmmc/lsr_stdlib.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/linear_algebra.h"
#include "memory_bridge.h"

#include "lsr_stdlib_internal.h"

lmmc_status_t lmmc_lsr_linalg_shape(const lmmc_mat_t* a,
                                    size_t* out_rows,
                                    size_t* out_cols)
{
    if (!lmmc_lsr_mat_valid(a) || !out_rows || !out_cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *out_rows = a->rows;
    *out_cols = a->cols;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_shape_vec(const lmmc_mat_t* a,
                                        lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!lmmc_lsr_mat_valid(a) || !out) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    status = lmmc_vec_create(2, out);
    if (status != LMMC_STATUS_OK) return status;
    out->data[0] = (lmmc_real_t)a->rows;
    out->data[1] = (lmmc_real_t)a->cols;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_eye(size_t n, lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (n == 0 || !out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_mat_identity(n, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_diag(const lmmc_vec_t* diagonal,
                                   lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(diagonal);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_vec_to_diag(diagonal, diagonal->size, diagonal->size, out);
}

lmmc_status_t lmmc_lsr_linalg_dot(const lmmc_vec_t* a,
                                  const lmmc_vec_t* b,
                                  lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->size != b->size) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_vec_dot(a, b, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_linalg_cross(const lmmc_vec_t* a,
                                    const lmmc_vec_t* b,
                                    lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->size != 3 || b->size != 3) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_vec_create(3, out);
    if (status != LMMC_STATUS_OK) return status;
    out->data[0] = a->data[1] * b->data[2] - a->data[2] * b->data[1];
    out->data[1] = a->data[2] * b->data[0] - a->data[0] * b->data[2];
    out->data[2] = a->data[0] * b->data[1] - a->data[1] * b->data[0];
    status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_norm(const lmmc_vec_t* x,
                                   lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_vec(x);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_norm2(x, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

void lmmc_lsr_bool_vec_destroy(lmmc_lsr_bool_vec_t* vec)
{
    if (!vec) return;
    if (vec->owns_data && vec->data) lmmc_free(vec->data);
    vec->size = 0;
    vec->data = NULL;
    vec->owns_data = 0;
}

void lmmc_lsr_bool_mat_destroy(lmmc_lsr_bool_mat_t* mat)
{
    if (!mat) return;
    if (mat->owns_data && mat->data) lmmc_free(mat->data);
    mat->rows = 0;
    mat->cols = 0;
    mat->stride = 0;
    mat->data = NULL;
    mat->owns_data = 0;
}

lmmc_status_t lmmc_lsr_linalg_matmul(const lmmc_mat_t* a,
                                     const lmmc_mat_t* b,
                                     lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->cols != b->rows) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_mat_create(a->rows, b->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_mul(a, b, out);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_matvec(const lmmc_mat_t* a,
                                     const lmmc_vec_t* x,
                                     lmmc_vec_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_vec(x);
    if (status != LMMC_STATUS_OK) return status;
    if (a->cols != x->size) return LMMC_STATUS_DIMENSION_MISMATCH;
    status = lmmc_vec_create(a->rows, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_vec_mul(a, x, out);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_vec(out);
    if (status != LMMC_STATUS_OK) lmmc_vec_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_transpose(const lmmc_mat_t* a,
                                        lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_create(a->cols, a->rows, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_transpose_to(a, out);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_adjoint(const lmmc_mat_t* a,
                                      lmmc_mat_t* out)
{
    return lmmc_lsr_linalg_transpose(a, out);
}

lmmc_status_t lmmc_lsr_linalg_det(const lmmc_mat_t* a,
                                  lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_det(a, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_linalg_inv(const lmmc_mat_t* a,
                                  lmmc_mat_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    if (a->rows != a->cols) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_mat_create(a->rows, a->cols, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_inv(a, out);
    if (status == LMMC_STATUS_OK) status = lmmc_lsr_require_finite_mat(out);
    if (status != LMMC_STATUS_OK) lmmc_mat_destroy(out);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_rank(const lmmc_mat_t* a,
                                   size_t* out_rank)
{
    size_t rows, cols, count, rank, col;
    lmmc_real_t* work;
    lmmc_real_t scale = (lmmc_real_t)0;
    lmmc_real_t tol;

    if (!out_rank) return LMMC_STATUS_INVALID_ARGUMENT;
    lmmc_status_t input_status = lmmc_lsr_require_finite_mat(a);
    if (input_status != LMMC_STATUS_OK) return input_status;
    if (a->rows != 0 && a->cols > ((size_t)-1) / a->rows) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    rows = a->rows;
    cols = a->cols;
    count = rows * cols;
    work = (lmmc_real_t*)malloc(count * sizeof(lmmc_real_t));
    if (!work) return LMMC_STATUS_ALLOCATION_FAILED;

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            lmmc_real_t value = a->data[i * a->stride + j];
            work[i * cols + j] = value;
            if (fabs((double)value) > fabs((double)scale)) scale = value;
        }
    }

    if (scale == (lmmc_real_t)0) {
        free(work);
        *out_rank = 0;
        return LMMC_STATUS_OK;
    }
    tol = (lmmc_real_t)(1e-12 * fabs((double)scale) *
                        (double)(rows > cols ? rows : cols));

    rank = 0;
    for (col = 0; col < cols && rank < rows; ++col) {
        size_t pivot = rank;
        lmmc_real_t pivot_abs =
            (lmmc_real_t)fabs((double)work[pivot * cols + col]);
        for (size_t row = rank + 1; row < rows; ++row) {
            lmmc_real_t candidate =
                (lmmc_real_t)fabs((double)work[row * cols + col]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if (pivot_abs <= tol) continue;

        if (pivot != rank) {
            for (size_t j = col; j < cols; ++j) {
                lmmc_real_t tmp = work[rank * cols + j];
                work[rank * cols + j] = work[pivot * cols + j];
                work[pivot * cols + j] = tmp;
            }
        }

        for (size_t row = rank + 1; row < rows; ++row) {
            lmmc_real_t factor = work[row * cols + col] /
                                 work[rank * cols + col];
            work[row * cols + col] = (lmmc_real_t)0;
            for (size_t j = col + 1; j < cols; ++j) {
                work[row * cols + j] -= factor * work[rank * cols + j];
            }
        }
        ++rank;
    }

    free(work);
    *out_rank = rank;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_trace(const lmmc_mat_t* a,
                                    lmmc_real_t* out)
{
    lmmc_real_t value;
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_trace(a, &value);
    if (status != LMMC_STATUS_OK) return status;
    return lmmc_lsr_store_finite_real(value, out);
}

lmmc_status_t lmmc_lsr_linalg_solve_left(const lmmc_mat_t* a,
                                         const lmmc_mat_t* b,
                                         lmmc_mat_t* out)
{
    lmmc_mat_t lu = {0};
    lmmc_vec_t rhs = {0};
    lmmc_vec_t sol = {0};
    size_t* pivots = NULL;
    lmmc_status_t status;

    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->rows != a->cols || b->rows != a->rows) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    status = lmmc_mat_create(a->rows, a->cols, &lu);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_mat_copy(a, &lu);
    if (status != LMMC_STATUS_OK) goto cleanup;

    pivots = (size_t*)malloc(a->rows * sizeof(size_t));
    if (!pivots) {
        status = LMMC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }
    status = lmmc_lu_decompose_inplace(&lu, pivots, NULL);
    if (status != LMMC_STATUS_OK) goto cleanup;

    status = lmmc_mat_create(a->cols, b->cols, out);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_vec_create(b->rows, &rhs);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(out);
        goto cleanup;
    }
    status = lmmc_vec_create(a->cols, &sol);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(out);
        goto cleanup;
    }

    for (size_t col = 0; col < b->cols; ++col) {
        for (size_t row = 0; row < b->rows; ++row) {
            rhs.data[row] = b->data[row * b->stride + col];
        }
        status = lmmc_lu_solve(&lu, pivots, &rhs, &sol);
        if (status != LMMC_STATUS_OK) {
            lmmc_mat_destroy(out);
            goto cleanup;
        }
        for (size_t row = 0; row < a->cols; ++row) {
            out->data[row * out->stride + col] = sol.data[row];
        }
    }

cleanup:
    lmmc_vec_destroy(&sol);
    lmmc_vec_destroy(&rhs);
    if (pivots) free(pivots);
    lmmc_mat_destroy(&lu);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_solve_right(const lmmc_mat_t* b,
                                          const lmmc_mat_t* a,
                                          lmmc_mat_t* out)
{
    lmmc_mat_t at = {0};
    lmmc_mat_t bt = {0};
    lmmc_mat_t xt = {0};
    lmmc_status_t status;

    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_mat(b);
    if (status != LMMC_STATUS_OK) return status;
    if (a->rows != a->cols || b->cols != a->rows) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    status = lmmc_lsr_linalg_transpose(a, &at);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_lsr_linalg_transpose(b, &bt);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_lsr_linalg_solve_left(&at, &bt, &xt);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_lsr_linalg_transpose(&xt, out);

cleanup:
    lmmc_mat_destroy(&xt);
    lmmc_mat_destroy(&bt);
    lmmc_mat_destroy(&at);
    return status;
}

lmmc_status_t lmmc_lsr_linalg_eig(const lmmc_mat_t* a,
                                  lmmc_eigen_gen_full_result_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    if (a->rows != a->cols) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_eigen_general_full(a, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_eig_result(out);
    if (status != LMMC_STATUS_OK) {
        lmmc_eigen_gen_full_result_destroy(out);
        return status;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_svd(const lmmc_mat_t* a,
                                  lmmc_svd_result_t* out)
{
    lmmc_status_t status;
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_svd(a, out);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_require_finite_svd_result(out);
    if (status != LMMC_STATUS_OK) {
        lmmc_svd_result_destroy(out);
        return status;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lsr_linalg_eig_table(const lmmc_mat_t* a,
                                        lmmc_lsr_eig_table_t* out)
{
    lmmc_status_t status;
    lmmc_eigen_gen_full_result_t raw = {0};
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    memset(out, 0, sizeof(*out));
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_linalg_eig(a, &raw);
    if (status != LMMC_STATUS_OK) return status;

    status = lmmc_lsr_vec_to_column(&raw.real_parts, &out->values_real);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_lsr_vec_to_column(&raw.imag_parts, &out->values_imag);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_lsr_copy_mat(&raw.vectors_real, &out->vectors_real);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_lsr_copy_mat(&raw.vectors_imag, &out->vectors_imag);
    if (status != LMMC_STATUS_OK) goto cleanup;

cleanup:
    lmmc_eigen_gen_full_result_destroy(&raw);
    if (status != LMMC_STATUS_OK) lmmc_lsr_eig_table_destroy(out);
    return status;
}

const lmmc_mat_t* lmmc_lsr_eig_table_get(const lmmc_lsr_eig_table_t* table,
                                         const char* key)
{
    if (!table || !key) return NULL;
    if (strcmp(key, "values_real") == 0) return &table->values_real;
    if (strcmp(key, "values_imag") == 0) return &table->values_imag;
    if (strcmp(key, "vectors_real") == 0) return &table->vectors_real;
    if (strcmp(key, "vectors_imag") == 0) return &table->vectors_imag;
    return NULL;
}

size_t lmmc_lsr_eig_table_count(const lmmc_lsr_eig_table_t* table)
{
    return table ? 4u : 0u;
}

const char* lmmc_lsr_eig_table_key(const lmmc_lsr_eig_table_t* table,
                                   size_t index)
{
    static const char* keys[] = {
        "values_real",
        "values_imag",
        "vectors_real",
        "vectors_imag",
    };
    if (!table || index >= sizeof(keys) / sizeof(keys[0])) return NULL;
    return keys[index];
}

void lmmc_lsr_eig_table_destroy(lmmc_lsr_eig_table_t* table)
{
    if (!table) return;
    lmmc_mat_destroy(&table->values_real);
    lmmc_mat_destroy(&table->values_imag);
    lmmc_mat_destroy(&table->vectors_real);
    lmmc_mat_destroy(&table->vectors_imag);
}

lmmc_status_t lmmc_lsr_linalg_svd_table(const lmmc_mat_t* a,
                                        lmmc_lsr_svd_table_t* out)
{
    lmmc_status_t status;
    lmmc_svd_result_t raw = {0};
    if (!out) return LMMC_STATUS_INVALID_ARGUMENT;
    memset(out, 0, sizeof(*out));
    status = lmmc_lsr_require_finite_mat(a);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_lsr_linalg_svd(a, &raw);
    if (status != LMMC_STATUS_OK) return status;

    status = lmmc_lsr_copy_mat(&raw.U, &out->U);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_lsr_vec_to_diag(&raw.sigma, raw.U.cols, raw.Vt.rows, &out->S);
    if (status != LMMC_STATUS_OK) goto cleanup;
    status = lmmc_lsr_copy_mat(&raw.Vt, &out->Vt);
    if (status != LMMC_STATUS_OK) goto cleanup;

cleanup:
    lmmc_svd_result_destroy(&raw);
    if (status != LMMC_STATUS_OK) lmmc_lsr_svd_table_destroy(out);
    return status;
}

const lmmc_mat_t* lmmc_lsr_svd_table_get(const lmmc_lsr_svd_table_t* table,
                                         const char* key)
{
    if (!table || !key) return NULL;
    if (strcmp(key, "U") == 0) return &table->U;
    if (strcmp(key, "S") == 0) return &table->S;
    if (strcmp(key, "Vt") == 0) return &table->Vt;
    return NULL;
}

size_t lmmc_lsr_svd_table_count(const lmmc_lsr_svd_table_t* table)
{
    return table ? 3u : 0u;
}

const char* lmmc_lsr_svd_table_key(const lmmc_lsr_svd_table_t* table,
                                   size_t index)
{
    static const char* keys[] = {"U", "S", "Vt"};
    if (!table || index >= sizeof(keys) / sizeof(keys[0])) return NULL;
    return keys[index];
}

void lmmc_lsr_svd_table_destroy(lmmc_lsr_svd_table_t* table)
{
    if (!table) return;
    lmmc_mat_destroy(&table->U);
    lmmc_mat_destroy(&table->S);
    lmmc_mat_destroy(&table->Vt);
}
