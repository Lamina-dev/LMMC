/**
 * @file dense.c
 * @brief 稠密矩阵 / 向量运算实现。
 */
#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "internal.h"
#include "blas_backend.h"
#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/linear_algebra.h"

static int lmmc_mul_overflow_size(size_t a, size_t b, size_t* out) {
    if (a == 0 || b == 0) {
        *out = 0;
        return 0;
    }
    if (a > ((size_t)-1) / b) {
        return 1;
    }
    *out = a * b;
    return 0;
}

lmmc_status_t lmmc_mat_create(size_t rows, size_t cols, lmmc_mat_t* out_mat) {
    size_t n_elem = 0;
    size_t n_bytes = 0;
    lmmc_real_t* data = NULL;

    if (out_mat == NULL || rows == 0 || cols == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lmmc_mul_overflow_size(rows, cols, &n_elem) || lmmc_mul_overflow_size(n_elem, sizeof(lmmc_real_t), &n_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    data = (lmmc_real_t*)lmmc_alloc(n_bytes);
    if (data == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(data, 0, n_bytes);

    for (size_t i = 0; i < rows * cols; ++i) {
        LMMC_REAL_INIT(&data[i]);
    }

    out_mat->rows = rows;
    out_mat->cols = cols;
    out_mat->stride = cols;
    out_mat->data = data;
    out_mat->owns_data = 1;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_wrap(size_t rows, size_t cols, size_t stride, lmmc_real_t* data, lmmc_mat_t* out_mat) {
    if (out_mat == NULL || data == NULL || rows == 0 || cols == 0 || stride < cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    out_mat->rows = rows;
    out_mat->cols = cols;
    out_mat->stride = stride;
    out_mat->data = data;
    out_mat->owns_data = 0;
    return LMMC_STATUS_OK;
}

void lmmc_mat_destroy(lmmc_mat_t* mat) {
    if (mat == NULL) {
        return;
    }
    if (mat->owns_data && mat->data != NULL) {
        for (size_t i = 0; i < mat->rows * mat->cols; ++i) {
            LMMC_REAL_CLEAR(&mat->data[i]);
        }
        lmmc_free(mat->data);
    }
    mat->rows = 0;
    mat->cols = 0;
    mat->stride = 0;
    mat->data = NULL;
    mat->owns_data = 0;
}

lmmc_status_t lmmc_mat_fill(lmmc_mat_t* mat, lmmc_real_t value) {
    size_t i = 0;
    size_t j = 0;
    if (mat == NULL || mat->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    for (i = 0; i < mat->rows; ++i) {
        for (j = 0; j < mat->cols; ++j) {
            LMMC_REAL_SET(&mat->data[i * mat->stride + j], &value);
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_copy(const lmmc_mat_t* src, lmmc_mat_t* dst) {
    size_t i = 0;
    size_t j = 0;
    if (src == NULL || dst == NULL || src->data == NULL || dst->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (src->rows != dst->rows || src->cols != dst->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    for (i = 0; i < src->rows; ++i) {
        for (j = 0; j < src->cols; ++j) {
            LMMC_REAL_SET(&dst->data[i * dst->stride + j], &src->data[i * src->stride + j]);
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_transpose_to(const lmmc_mat_t* src, lmmc_mat_t* dst) {
    size_t i = 0;
    size_t j = 0;
    if (src == NULL || dst == NULL || src->data == NULL || dst->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (dst->rows != src->cols || dst->cols != src->rows) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    for (i = 0; i < src->rows; ++i) {
        for (j = 0; j < src->cols; ++j) {
            LMMC_REAL_SET(&dst->data[j * dst->stride + i], &src->data[i * src->stride + j]);
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_gemm(lmmc_real_t alpha, const lmmc_mat_t* A, int transA,
    const lmmc_mat_t* B, int transB, lmmc_real_t beta, lmmc_mat_t* C) {
    if (A == NULL || B == NULL || C == NULL ||
        A->data == NULL || B->data == NULL || C->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Determine effective dimensions of op(A) and op(B) */
    size_t M = transA ? A->cols : A->rows;
    size_t K_A = transA ? A->rows : A->cols;
    size_t K_B = transB ? B->cols : B->rows;
    size_t N = transB ? B->rows : B->cols;

    /* Dimension validation */
    if (K_A != K_B) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    if (C->rows != M || C->cols != N) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    size_t K = K_A;

    const lmmc_real_t* restrict a_data = A->data;
    const lmmc_real_t* restrict b_data = B->data;
    lmmc_real_t* restrict c_data = C->data;

    size_t a_stride = A->stride;
    size_t b_stride = B->stride;
    size_t c_stride = C->stride;

#ifdef LMMC_USE_BLAS
    {
        /* Route through BLAS dgemm with transpose support */
        lmmc_blas_dgemm_ex(transA, transB,
                           M, N, K,
                           alpha,
                           a_data, a_stride,
                           b_data, b_stride,
                           beta,
                           c_data, c_stride);
        return LMMC_STATUS_OK;
    }
#endif

    /* Scale C by beta */
    if (beta == 0.0) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                LMMC_REAL_SET_D(&c_data[i * c_stride + j], 0.0);
            }
        }
    } else if (beta != 1.0) {
        lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul);
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                LMMC_REAL_MUL(&tmp_mul, &beta, &c_data[i * c_stride + j]);
                LMMC_REAL_SET(&c_data[i * c_stride + j], &tmp_mul);
            }
        }
        LMMC_REAL_CLEAR(&tmp_mul);
    }

    /* If alpha is zero, we're done */
    if (alpha == 0.0) {
        return LMMC_STATUS_OK;
    }

    /* Native triple-loop: C += alpha * op(A) * op(B) */
    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul);
    lmmc_real_t tmp_sum; LMMC_REAL_INIT(&tmp_sum);
    lmmc_real_t alpha_a_ik; LMMC_REAL_INIT(&alpha_a_ik);

    size_t bs = 64;
    for (size_t ii = 0; ii < M; ii += bs) {
        size_t i_end = (ii + bs > M) ? M : ii + bs;
        for (size_t kk = 0; kk < K; kk += bs) {
            size_t k_end = (kk + bs > K) ? K : kk + bs;
            for (size_t jj = 0; jj < N; jj += bs) {
                size_t j_end = (jj + bs > N) ? N : jj + bs;
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t k = kk; k < k_end; ++k) {
                        /* Get op(A)[i,k] */
                        lmmc_real_t a_ik;
                        if (transA) {
                            a_ik = a_data[k * a_stride + i];
                        } else {
                            a_ik = a_data[i * a_stride + k];
                        }
                        /* alpha * op(A)[i,k] */
                        LMMC_REAL_MUL(&alpha_a_ik, &alpha, &a_ik);

                        for (size_t j = jj; j < j_end; ++j) {
                            /* Get op(B)[k,j] */
                            lmmc_real_t b_kj;
                            if (transB) {
                                b_kj = b_data[j * b_stride + k];
                            } else {
                                b_kj = b_data[k * b_stride + j];
                            }
                            LMMC_REAL_MUL(&tmp_mul, &alpha_a_ik, &b_kj);
                            LMMC_REAL_ADD(&tmp_sum, &c_data[i * c_stride + j], &tmp_mul);
                            LMMC_REAL_SET(&c_data[i * c_stride + j], &tmp_sum);
                        }
                    }
                }
            }
        }
    }

    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&tmp_sum);
    LMMC_REAL_CLEAR(&alpha_a_ik);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_mul(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c) {
    return lmmc_mat_gemm(1.0, a, 0, b, 0, 0.0, c);
}

lmmc_status_t lmmc_mat_norm_fro(const lmmc_mat_t* a, lmmc_real_t* out_norm) {
    size_t i = 0;
    size_t j = 0;
    lmmc_real_t sum;
    lmmc_real_t tmp_mul;
    lmmc_real_t tmp_sum;

    if (a == NULL || out_norm == NULL || a->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    LMMC_REAL_INIT(&sum);
    LMMC_REAL_INIT(&tmp_mul);
    LMMC_REAL_INIT(&tmp_sum);
    LMMC_REAL_SET_D(&sum, 0.0);

    for (i = 0; i < a->rows; ++i) {
        for (j = 0; j < a->cols; ++j) {
            LMMC_REAL_MUL(&tmp_mul, &a->data[i * a->stride + j], &a->data[i * a->stride + j]);
            LMMC_REAL_ADD(&tmp_sum, &sum, &tmp_mul);
            LMMC_REAL_SET(&sum, &tmp_sum);
        }
    }
    LMMC_REAL_SQRT(out_norm, &sum);

    LMMC_REAL_CLEAR(&sum);
    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&tmp_sum);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_create(size_t size, lmmc_vec_t* out_vec) {
    size_t n_bytes = 0;
    lmmc_real_t* data = NULL;
    if (out_vec == NULL || size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lmmc_mul_overflow_size(size, sizeof(lmmc_real_t), &n_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    data = (lmmc_real_t*)lmmc_alloc(n_bytes);
    if (data == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(data, 0, n_bytes);

    for (size_t i = 0; i < size; ++i) {
        LMMC_REAL_INIT(&data[i]);
    }

    out_vec->size = size;
    out_vec->data = data;
    out_vec->owns_data = 1;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_wrap(size_t size, lmmc_real_t* data, lmmc_vec_t* out_vec) {
    if (out_vec == NULL || data == NULL || size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    out_vec->size = size;
    out_vec->data = data;
    out_vec->owns_data = 0;
    return LMMC_STATUS_OK;
}

void lmmc_vec_destroy(lmmc_vec_t* vec) {
    if (vec == NULL) {
        return;
    }
    if (vec->owns_data && vec->data != NULL) {
        for (size_t i = 0; i < vec->size; ++i) {
            LMMC_REAL_CLEAR(&vec->data[i]);
        }
        lmmc_free(vec->data);
    }
    vec->size = 0;
    vec->data = NULL;
    vec->owns_data = 0;
}

lmmc_status_t lmmc_vec_fill(lmmc_vec_t* vec, lmmc_real_t value) {
    size_t i = 0;
    if (vec == NULL || vec->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    for (i = 0; i < vec->size; ++i) {
        LMMC_REAL_SET(&vec->data[i], &value);
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_dot(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_real_t* out_dot) {
    size_t i = 0;
    if (a == NULL || b == NULL || out_dot == NULL || a->data == NULL || b->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->size != b->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    lmmc_real_t sum; LMMC_REAL_INIT(&sum);
    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul);
    lmmc_real_t tmp_sum; LMMC_REAL_INIT(&tmp_sum);
    LMMC_REAL_SET_D(&sum, 0.0);

    for (i = 0; i < a->size; ++i) {
        LMMC_REAL_MUL(&tmp_mul, &a->data[i], &b->data[i]);
        LMMC_REAL_ADD(&tmp_sum, &sum, &tmp_mul);
        LMMC_REAL_SET(&sum, &tmp_sum);
    }
    LMMC_REAL_SET(out_dot, &sum);

    LMMC_REAL_CLEAR(&sum);
    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&tmp_sum);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_gemv(lmmc_real_t alpha, const lmmc_mat_t* A, int transA,
    const lmmc_vec_t* x, lmmc_real_t beta, lmmc_vec_t* y) {
    if (A == NULL || x == NULL || y == NULL ||
        A->data == NULL || x->data == NULL || y->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Determine effective dimensions of op(A) */
    size_t M = transA ? A->cols : A->rows;
    size_t N = transA ? A->rows : A->cols;

    /* Dimension validation */
    if (N != x->size || M != y->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    const lmmc_real_t* restrict a_data = A->data;
    size_t a_stride = A->stride;

#ifdef LMMC_USE_BLAS
    {
        char trans_char = transA ? 'T' : 'N';
        lmmc_blas_dgemv(trans_char, A->rows, A->cols,
                        alpha,
                        a_data, a_stride,
                        x->data, 1,
                        beta,
                        y->data, 1);
        return LMMC_STATUS_OK;
    }
#endif

    /* Scale y by beta */
    if (beta == 0.0) {
        for (size_t i = 0; i < M; ++i) {
            LMMC_REAL_SET_D(&y->data[i], 0.0);
        }
    } else if (beta != 1.0) {
        lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul);
        for (size_t i = 0; i < M; ++i) {
            LMMC_REAL_MUL(&tmp_mul, &beta, &y->data[i]);
            LMMC_REAL_SET(&y->data[i], &tmp_mul);
        }
        LMMC_REAL_CLEAR(&tmp_mul);
    }

    /* If alpha is zero, we're done */
    if (alpha == 0.0) {
        return LMMC_STATUS_OK;
    }

    /* Native loop: y += alpha * op(A) * x */
    lmmc_real_t sum; LMMC_REAL_INIT(&sum);
    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul);
    lmmc_real_t tmp_sum; LMMC_REAL_INIT(&tmp_sum);
    lmmc_real_t alpha_sum; LMMC_REAL_INIT(&alpha_sum);

    for (size_t i = 0; i < M; ++i) {
        LMMC_REAL_SET_D(&sum, 0.0);
        for (size_t j = 0; j < N; ++j) {
            lmmc_real_t a_ij;
            if (transA) {
                a_ij = a_data[j * a_stride + i];
            } else {
                a_ij = a_data[i * a_stride + j];
            }
            LMMC_REAL_MUL(&tmp_mul, &a_ij, &x->data[j]);
            LMMC_REAL_ADD(&tmp_sum, &sum, &tmp_mul);
            LMMC_REAL_SET(&sum, &tmp_sum);
        }
        /* y[i] += alpha * sum */
        LMMC_REAL_MUL(&alpha_sum, &alpha, &sum);
        LMMC_REAL_ADD(&tmp_sum, &y->data[i], &alpha_sum);
        LMMC_REAL_SET(&y->data[i], &tmp_sum);
    }

    LMMC_REAL_CLEAR(&sum);
    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&tmp_sum);
    LMMC_REAL_CLEAR(&alpha_sum);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_vec_mul(const lmmc_mat_t* a, const lmmc_vec_t* x, lmmc_vec_t* y) {
    return lmmc_mat_gemv(1.0, a, 0, x, 0.0, y);
}


lmmc_status_t lmmc_vec_norm2(const lmmc_vec_t* x, lmmc_real_t* out_norm) {
    if (x == NULL || out_norm == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

#ifdef LMMC_USE_BLAS

    *out_norm = lmmc_blas_dnrm2(x->size, x->data, 1);
    return LMMC_STATUS_OK;
#else
    lmmc_real_t sum; LMMC_REAL_INIT(&sum);
    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul);
    lmmc_real_t tmp_sum; LMMC_REAL_INIT(&tmp_sum);
    LMMC_REAL_SET_D(&sum, 0.0);

    for (size_t i = 0; i < x->size; ++i) {
        LMMC_REAL_MUL(&tmp_mul, &x->data[i], &x->data[i]);
        LMMC_REAL_ADD(&tmp_sum, &sum, &tmp_mul);
        LMMC_REAL_SET(&sum, &tmp_sum);
    }
    LMMC_REAL_SQRT(out_norm, &sum);

    LMMC_REAL_CLEAR(&sum);
    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&tmp_sum);
    return LMMC_STATUS_OK;
#endif
}

lmmc_status_t lmmc_vec_norm_inf(const lmmc_vec_t* x, lmmc_real_t* out_norm) {
    if (x == NULL || out_norm == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_real_t max_val; LMMC_REAL_INIT(&max_val);
    lmmc_real_t abs_val; LMMC_REAL_INIT(&abs_val);
    LMMC_REAL_SET_D(&max_val, 0.0);

    for (size_t i = 0; i < x->size; ++i) {
        LMMC_REAL_ABS(&abs_val, &x->data[i]);
        max_val = lmmc_max(max_val, abs_val);
    }
    LMMC_REAL_SET(out_norm, &max_val);

    LMMC_REAL_CLEAR(&max_val);
    LMMC_REAL_CLEAR(&abs_val);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_scale(lmmc_vec_t* x, lmmc_real_t alpha) {
    if (x == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul);

    for (size_t i = 0; i < x->size; ++i) {
        LMMC_REAL_MUL(&tmp_mul, &x->data[i], &alpha);
        LMMC_REAL_SET(&x->data[i], &tmp_mul);
    }

    LMMC_REAL_CLEAR(&tmp_mul);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_axpy(lmmc_real_t alpha, const lmmc_vec_t* x, lmmc_vec_t* y) {
    if (x == NULL || y == NULL || x->data == NULL || y->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size == 0 || y->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size != y->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

#ifdef LMMC_USE_BLAS

    lmmc_blas_daxpy(x->size, alpha, x->data, 1, y->data, 1);
    return LMMC_STATUS_OK;
#else
    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul);
    lmmc_real_t tmp_sum; LMMC_REAL_INIT(&tmp_sum);

    for (size_t i = 0; i < x->size; ++i) {
        LMMC_REAL_MUL(&tmp_mul, &alpha, &x->data[i]);
        LMMC_REAL_ADD(&tmp_sum, &y->data[i], &tmp_mul);
        LMMC_REAL_SET(&y->data[i], &tmp_sum);
    }

    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&tmp_sum);
    return LMMC_STATUS_OK;
#endif
}

lmmc_status_t lmmc_vec_copy(const lmmc_vec_t* src, lmmc_vec_t* dst) {
    if (src == NULL || dst == NULL || src->data == NULL || dst->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (src->size == 0 || dst->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (src->size != dst->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    for (size_t i = 0; i < src->size; ++i) {
        LMMC_REAL_SET(&dst->data[i], &src->data[i]);
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_swap(lmmc_vec_t* x, lmmc_vec_t* y) {
    if (x == NULL || y == NULL || x->data == NULL || y->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size == 0 || y->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size != y->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    for (size_t i = 0; i < x->size; ++i) {
        lmmc_swap(&x->data[i], &y->data[i]);
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_asum(const lmmc_vec_t* x, lmmc_real_t* out_asum) {
    if (x == NULL || out_asum == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_real_t sum; LMMC_REAL_INIT(&sum);
    lmmc_real_t abs_val; LMMC_REAL_INIT(&abs_val);
    lmmc_real_t tmp_sum; LMMC_REAL_INIT(&tmp_sum);
    LMMC_REAL_SET_D(&sum, 0.0);

    for (size_t i = 0; i < x->size; ++i) {
        LMMC_REAL_ABS(&abs_val, &x->data[i]);
        LMMC_REAL_ADD(&tmp_sum, &sum, &abs_val);
        LMMC_REAL_SET(&sum, &tmp_sum);
    }
    LMMC_REAL_SET(out_asum, &sum);

    LMMC_REAL_CLEAR(&sum);
    LMMC_REAL_CLEAR(&abs_val);
    LMMC_REAL_CLEAR(&tmp_sum);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_iamax(const lmmc_vec_t* x, size_t* out_idx) {
    if (x == NULL || out_idx == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_real_t max_val; LMMC_REAL_INIT(&max_val);
    lmmc_real_t abs_val; LMMC_REAL_INIT(&abs_val);
    size_t max_idx = 0;

    LMMC_REAL_ABS(&max_val, &x->data[0]);

    for (size_t i = 1; i < x->size; ++i) {
        LMMC_REAL_ABS(&abs_val, &x->data[i]);
        if (LMMC_REAL_CMP(&abs_val, &max_val) > 0) {
            LMMC_REAL_SET(&max_val, &abs_val);
            max_idx = i;
        }
    }
    *out_idx = max_idx;

    LMMC_REAL_CLEAR(&max_val);
    LMMC_REAL_CLEAR(&abs_val);
    return LMMC_STATUS_OK;
}


lmmc_status_t lmmc_mat_add(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c) {
    if (a == NULL || b == NULL || c == NULL ||
        a->data == NULL || b->data == NULL || c->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != b->rows || a->cols != b->cols ||
        a->rows != c->rows || a->cols != c->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    lmmc_real_t tmp_sum; LMMC_REAL_INIT(&tmp_sum);

    for (size_t i = 0; i < a->rows; ++i) {
        for (size_t j = 0; j < a->cols; ++j) {
            LMMC_REAL_ADD(&tmp_sum,
                          &a->data[i * a->stride + j],
                          &b->data[i * b->stride + j]);
            LMMC_REAL_SET(&c->data[i * c->stride + j], &tmp_sum);
        }
    }

    LMMC_REAL_CLEAR(&tmp_sum);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_sub(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c) {
    if (a == NULL || b == NULL || c == NULL ||
        a->data == NULL || b->data == NULL || c->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != b->rows || a->cols != b->cols ||
        a->rows != c->rows || a->cols != c->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    lmmc_real_t tmp_sub; LMMC_REAL_INIT(&tmp_sub);

    for (size_t i = 0; i < a->rows; ++i) {
        for (size_t j = 0; j < a->cols; ++j) {
            LMMC_REAL_SUB(&tmp_sub,
                          &a->data[i * a->stride + j],
                          &b->data[i * b->stride + j]);
            LMMC_REAL_SET(&c->data[i * c->stride + j], &tmp_sub);
        }
    }

    LMMC_REAL_CLEAR(&tmp_sub);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_scale(lmmc_mat_t* a, lmmc_real_t alpha) {
    if (a == NULL || a->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul);

    for (size_t i = 0; i < a->rows; ++i) {
        for (size_t j = 0; j < a->cols; ++j) {
            LMMC_REAL_MUL(&tmp_mul, &a->data[i * a->stride + j], &alpha);
            LMMC_REAL_SET(&a->data[i * a->stride + j], &tmp_mul);
        }
    }

    LMMC_REAL_CLEAR(&tmp_mul);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_identity(size_t n, lmmc_mat_t* out_mat) {
    if (out_mat == NULL || n == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_status_t status = lmmc_mat_create(n, n, out_mat);
    if (status != LMMC_STATUS_OK) {
        return status;
    }

    lmmc_real_t one; LMMC_REAL_INIT(&one);
    LMMC_REAL_SET_D(&one, 1.0);

    for (size_t i = 0; i < n; ++i) {
        LMMC_REAL_SET(&out_mat->data[i * out_mat->stride + i], &one);
    }

    LMMC_REAL_CLEAR(&one);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_trace(const lmmc_mat_t* a, lmmc_real_t* out_trace) {
    if (a == NULL || out_trace == NULL || a->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_real_t sum; LMMC_REAL_INIT(&sum);
    lmmc_real_t tmp_sum; LMMC_REAL_INIT(&tmp_sum);
    LMMC_REAL_SET_D(&sum, 0.0);

    for (size_t i = 0; i < a->rows; ++i) {
        LMMC_REAL_ADD(&tmp_sum, &sum, &a->data[i * a->stride + i]);
        LMMC_REAL_SET(&sum, &tmp_sum);
    }
    LMMC_REAL_SET(out_trace, &sum);

    LMMC_REAL_CLEAR(&sum);
    LMMC_REAL_CLEAR(&tmp_sum);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_det(const lmmc_mat_t* a, lmmc_real_t* out_det) {
    if (a == NULL || out_det == NULL || a->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    size_t n = a->rows;


    if (n == 1) {
        LMMC_REAL_SET(out_det, &a->data[0]);
        return LMMC_STATUS_OK;
    }


    if (n == 2) {
        lmmc_real_t ad; LMMC_REAL_INIT(&ad);
        lmmc_real_t bc; LMMC_REAL_INIT(&bc);
        lmmc_real_t result; LMMC_REAL_INIT(&result);

        LMMC_REAL_MUL(&ad, &a->data[0 * a->stride + 0], &a->data[1 * a->stride + 1]);
        LMMC_REAL_MUL(&bc, &a->data[0 * a->stride + 1], &a->data[1 * a->stride + 0]);
        LMMC_REAL_SUB(&result, &ad, &bc);
        LMMC_REAL_SET(out_det, &result);

        LMMC_REAL_CLEAR(&ad);
        LMMC_REAL_CLEAR(&bc);
        LMMC_REAL_CLEAR(&result);
        return LMMC_STATUS_OK;
    }


    size_t n_elem = n * n;
    size_t n_bytes = n_elem * sizeof(lmmc_real_t);
    lmmc_real_t* lu = (lmmc_real_t*)lmmc_alloc(n_bytes);
    if (lu == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }


    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            lu[i * n + j] = a->data[i * a->stride + j];
        }
    }

    int sign = 1;

    lmmc_real_t abs_val; LMMC_REAL_INIT(&abs_val);
    lmmc_real_t max_val; LMMC_REAL_INIT(&max_val);
    lmmc_real_t factor; LMMC_REAL_INIT(&factor);
    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul);
    lmmc_real_t tmp_sub; LMMC_REAL_INIT(&tmp_sub);

    for (size_t k = 0; k < n; ++k) {

        size_t pivot_row = k;
        LMMC_REAL_ABS(&max_val, &lu[k * n + k]);

        for (size_t i = k + 1; i < n; ++i) {
            LMMC_REAL_ABS(&abs_val, &lu[i * n + k]);
            if (LMMC_REAL_CMP(&abs_val, &max_val) > 0) {
                LMMC_REAL_SET(&max_val, &abs_val);
                pivot_row = i;
            }
        }


        lmmc_real_t zero; LMMC_REAL_INIT(&zero);
        LMMC_REAL_SET_D(&zero, 0.0);
        if (LMMC_REAL_CMP(&max_val, &zero) == 0) {
            LMMC_REAL_SET_D(out_det, 0.0);
            LMMC_REAL_CLEAR(&zero);
            LMMC_REAL_CLEAR(&abs_val);
            LMMC_REAL_CLEAR(&max_val);
            LMMC_REAL_CLEAR(&factor);
            LMMC_REAL_CLEAR(&tmp_mul);
            LMMC_REAL_CLEAR(&tmp_sub);
            lmmc_free(lu);
            return LMMC_STATUS_OK;
        }
        LMMC_REAL_CLEAR(&zero);


        if (pivot_row != k) {
            for (size_t j = 0; j < n; ++j) {
                lmmc_real_t tmp = lu[k * n + j];
                lu[k * n + j] = lu[pivot_row * n + j];
                lu[pivot_row * n + j] = tmp;
            }
            sign = -sign;
        }


        for (size_t i = k + 1; i < n; ++i) {
            LMMC_REAL_DIV(&factor, &lu[i * n + k], &lu[k * n + k]);
            LMMC_REAL_SET(&lu[i * n + k], &factor);

            for (size_t j = k + 1; j < n; ++j) {
                LMMC_REAL_MUL(&tmp_mul, &factor, &lu[k * n + j]);
                LMMC_REAL_SUB(&tmp_sub, &lu[i * n + j], &tmp_mul);
                LMMC_REAL_SET(&lu[i * n + j], &tmp_sub);
            }
        }
    }


    lmmc_real_t det; LMMC_REAL_INIT(&det);
    LMMC_REAL_SET_D(&det, (double)sign);

    lmmc_real_t tmp_prod; LMMC_REAL_INIT(&tmp_prod);
    for (size_t i = 0; i < n; ++i) {
        LMMC_REAL_MUL(&tmp_prod, &det, &lu[i * n + i]);
        LMMC_REAL_SET(&det, &tmp_prod);
    }
    LMMC_REAL_SET(out_det, &det);

    LMMC_REAL_CLEAR(&det);
    LMMC_REAL_CLEAR(&tmp_prod);
    LMMC_REAL_CLEAR(&abs_val);
    LMMC_REAL_CLEAR(&max_val);
    LMMC_REAL_CLEAR(&factor);
    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&tmp_sub);
    lmmc_free(lu);

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_inv(const lmmc_mat_t* A, lmmc_mat_t* A_inv) {
    lmmc_mat_t lu_mat;
    size_t* pivots = NULL;
    lmmc_vec_t ei;
    lmmc_vec_t col;
    size_t n, i, j;
    lmmc_status_t status;

    if (A == NULL || A_inv == NULL || A->data == NULL || A_inv->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (A->rows != A->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (A_inv->rows != A->rows || A_inv->cols != A->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    n = A->rows;

    /* Copy A into workspace for LU decomposition */
    status = lmmc_mat_create(n, n, &lu_mat);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    status = lmmc_mat_copy(A, &lu_mat);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&lu_mat);
        return status;
    }

    /* Allocate pivot array */
    pivots = (size_t*)lmmc_alloc(n * sizeof(size_t));
    if (pivots == NULL) {
        lmmc_mat_destroy(&lu_mat);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /* LU decompose with partial pivoting */
    status = lmmc_lu_decompose_inplace(&lu_mat, pivots, NULL);
    if (status != LMMC_STATUS_OK) {
        /* Singular or other failure — leave A_inv unmodified */
        lmmc_free(pivots);
        lmmc_mat_destroy(&lu_mat);
        return status;
    }

    /* Create temporary vectors for solving */
    status = lmmc_vec_create(n, &ei);
    if (status != LMMC_STATUS_OK) {
        lmmc_free(pivots);
        lmmc_mat_destroy(&lu_mat);
        return status;
    }
    status = lmmc_vec_create(n, &col);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&ei);
        lmmc_free(pivots);
        lmmc_mat_destroy(&lu_mat);
        return status;
    }

    /* Solve A * col_i = e_i for each column of the identity */
    for (i = 0; i < n; ++i) {
        /* Set up e_i (standard basis vector) */
        for (j = 0; j < n; ++j) {
            LMMC_REAL_SET_D(&ei.data[j], 0.0);
        }
        LMMC_REAL_SET_D(&ei.data[i], 1.0);

        /* Solve using LU factors */
        status = lmmc_lu_solve(&lu_mat, pivots, &ei, &col);
        if (status != LMMC_STATUS_OK) {
            lmmc_vec_destroy(&col);
            lmmc_vec_destroy(&ei);
            lmmc_free(pivots);
            lmmc_mat_destroy(&lu_mat);
            return status;
        }

        /* Copy solution into column i of A_inv */
        for (j = 0; j < n; ++j) {
            LMMC_REAL_SET(&A_inv->data[j * A_inv->stride + i], &col.data[j]);
        }
    }

    lmmc_vec_destroy(&col);
    lmmc_vec_destroy(&ei);
    lmmc_free(pivots);
    lmmc_mat_destroy(&lu_mat);

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_solve_triangular(const lmmc_mat_t* T, int upper, int diag_unit,
    const lmmc_vec_t* b, lmmc_vec_t* x) {
    size_t n, i, j;
    lmmc_real_t sum, tmp, diag_val, abs_diag, eps;

    if (T == NULL || b == NULL || x == NULL || T->data == NULL || b->data == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (T->rows != T->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (b->size != T->rows || x->size != T->rows) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    n = T->rows;
    LMMC_REAL_INIT(&sum);
    LMMC_REAL_INIT(&tmp);
    LMMC_REAL_INIT(&diag_val);
    LMMC_REAL_INIT(&abs_diag);
    LMMC_REAL_INIT(&eps);
    LMMC_REAL_SET_D(&eps, 1e-15);

    if (upper) {
        /* Back substitution for upper triangular */
        for (i = n; i-- > 0;) {
            LMMC_REAL_SET(&sum, &b->data[i]);
            for (j = i + 1; j < n; ++j) {
                LMMC_REAL_MUL(&tmp, &T->data[i * T->stride + j], &x->data[j]);
                LMMC_REAL_SUB(&sum, &sum, &tmp);
            }
            if (diag_unit) {
                LMMC_REAL_SET(&x->data[i], &sum);
            } else {
                LMMC_REAL_SET(&diag_val, &T->data[i * T->stride + i]);
                LMMC_REAL_ABS(&abs_diag, &diag_val);
                if (LMMC_REAL_CMP(&abs_diag, &eps) <= 0) {
                    LMMC_REAL_CLEAR(&sum);
                    LMMC_REAL_CLEAR(&tmp);
                    LMMC_REAL_CLEAR(&diag_val);
                    LMMC_REAL_CLEAR(&abs_diag);
                    LMMC_REAL_CLEAR(&eps);
                    return LMMC_STATUS_SINGULAR_MATRIX;
                }
                LMMC_REAL_DIV(&x->data[i], &sum, &diag_val);
            }
        }
    } else {
        /* Forward substitution for lower triangular */
        for (i = 0; i < n; ++i) {
            LMMC_REAL_SET(&sum, &b->data[i]);
            for (j = 0; j < i; ++j) {
                LMMC_REAL_MUL(&tmp, &T->data[i * T->stride + j], &x->data[j]);
                LMMC_REAL_SUB(&sum, &sum, &tmp);
            }
            if (diag_unit) {
                LMMC_REAL_SET(&x->data[i], &sum);
            } else {
                LMMC_REAL_SET(&diag_val, &T->data[i * T->stride + i]);
                LMMC_REAL_ABS(&abs_diag, &diag_val);
                if (LMMC_REAL_CMP(&abs_diag, &eps) <= 0) {
                    LMMC_REAL_CLEAR(&sum);
                    LMMC_REAL_CLEAR(&tmp);
                    LMMC_REAL_CLEAR(&diag_val);
                    LMMC_REAL_CLEAR(&abs_diag);
                    LMMC_REAL_CLEAR(&eps);
                    return LMMC_STATUS_SINGULAR_MATRIX;
                }
                LMMC_REAL_DIV(&x->data[i], &sum, &diag_val);
            }
        }
    }

    LMMC_REAL_CLEAR(&sum);
    LMMC_REAL_CLEAR(&tmp);
    LMMC_REAL_CLEAR(&diag_val);
    LMMC_REAL_CLEAR(&abs_diag);
    LMMC_REAL_CLEAR(&eps);

    return LMMC_STATUS_OK;
}
