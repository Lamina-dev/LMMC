/**
 * @file dense_products.c
 * @brief 稠密矩阵乘法与向量内积实现（GEMM / GEMV / dot / axpy）。
 */

#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/linear_algebra.h"

lmmc_status_t lmmc_mat_gemm(lmmc_real_t alpha, const lmmc_mat_t* A, int transA,
    const lmmc_mat_t* B, int transB, lmmc_real_t beta, lmmc_mat_t* C) {
    if (!lmmc_mat_descriptor_is_valid(A) ||
        !lmmc_mat_descriptor_is_valid(B) ||
        !lmmc_mat_descriptor_is_valid(C)) {
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
    {
        lmmc_storage_envelope_t a_envelope;
        lmmc_storage_envelope_t b_envelope;
        lmmc_storage_envelope_t c_envelope;
        if (!lmmc_storage_envelope_checked(
                A->data, A->rows, A->cols, A->stride,
                sizeof(lmmc_real_t), &a_envelope) ||
            !lmmc_storage_envelope_checked(
                B->data, B->rows, B->cols, B->stride,
                sizeof(lmmc_real_t), &b_envelope) ||
            !lmmc_storage_envelope_checked(
                C->data, C->rows, C->cols, C->stride,
                sizeof(lmmc_real_t), &c_envelope)) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        if (lmmc_storage_envelopes_overlap(&c_envelope, &a_envelope) ||
            lmmc_storage_envelopes_overlap(&c_envelope, &b_envelope)) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
    }


    size_t K = K_A;

    const lmmc_real_t* restrict a_data = A->data;
    const lmmc_real_t* restrict b_data = B->data;
    lmmc_real_t* restrict c_data = C->data;

    size_t a_stride = A->stride;
    size_t b_stride = B->stride;
    size_t c_stride = C->stride;


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

lmmc_status_t lmmc_vec_dot(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_real_t* out_dot) {
    size_t i = 0;
    if (!lmmc_vec_descriptor_is_valid(a) ||
        !lmmc_vec_descriptor_is_valid(b) || out_dot == NULL) {
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
    if (!lmmc_mat_descriptor_is_valid(A) ||
        !lmmc_vec_descriptor_is_valid(x) ||
        !lmmc_vec_descriptor_is_valid(y)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Determine effective dimensions of op(A) */
    size_t M = transA ? A->cols : A->rows;
    size_t N = transA ? A->rows : A->cols;

    /* Dimension validation */
    if (N != x->size || M != y->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    {
        lmmc_storage_envelope_t a_envelope;
        lmmc_storage_envelope_t x_envelope;
        lmmc_storage_envelope_t y_envelope;
        if (!lmmc_storage_envelope_checked(A->data, A->rows, A->cols,
                                           A->stride, sizeof(lmmc_real_t),
                                           &a_envelope) ||
            !lmmc_storage_envelope_checked(x->data, 1, x->size, x->size,
                                           sizeof(lmmc_real_t), &x_envelope) ||
            !lmmc_storage_envelope_checked(y->data, 1, y->size, y->size,
                                           sizeof(lmmc_real_t), &y_envelope)) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
        if (lmmc_storage_envelopes_overlap(&y_envelope, &a_envelope) ||
            lmmc_storage_envelopes_overlap(&y_envelope, &x_envelope)) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
    }

    const lmmc_real_t* restrict a_data = A->data;
    size_t a_stride = A->stride;


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

lmmc_status_t lmmc_vec_axpy(lmmc_real_t alpha, const lmmc_vec_t* x, lmmc_vec_t* y) {
    if (!lmmc_vec_descriptor_is_valid(x) ||
        !lmmc_vec_descriptor_is_valid(y)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (x->size != y->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

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
}

