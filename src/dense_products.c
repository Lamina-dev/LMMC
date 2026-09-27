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

static lmmc_status_t lmmc_gemm_storage_check(
    const lmmc_mat_t* A, const lmmc_mat_t* B, const lmmc_mat_t* C) {
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
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_gemv_storage_check(
    const lmmc_mat_t* A, const lmmc_vec_t* x, const lmmc_vec_t* y) {
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
    return LMMC_STATUS_OK;
}

static void lmmc_gemm_scale(lmmc_mat_t* C, lmmc_real_t beta) {
    if (beta == 0.0) {
        for (size_t i = 0; i < C->rows; ++i) {
            for (size_t j = 0; j < C->cols; ++j) {
                LMMC_REAL_SET_D(&C->data[i * C->stride + j], 0.0);
            }
        }
    } else if (beta != 1.0) {
        for (size_t i = 0; i < C->rows; ++i) {
            for (size_t j = 0; j < C->cols; ++j) {
                lmmc_real_t scaled;
                LMMC_REAL_MUL(&scaled, &beta, &C->data[i * C->stride + j]);
                LMMC_REAL_SET(&C->data[i * C->stride + j], &scaled);
            }
        }
    }
}

static void lmmc_gemv_scale(lmmc_vec_t* y, lmmc_real_t beta) {
    if (beta == 0.0) {
        for (size_t i = 0; i < y->size; ++i) {
            LMMC_REAL_SET_D(&y->data[i], 0.0);
        }
    } else if (beta != 1.0) {
        for (size_t i = 0; i < y->size; ++i) {
            lmmc_real_t scaled;
            LMMC_REAL_MUL(&scaled, &beta, &y->data[i]);
            LMMC_REAL_SET(&y->data[i], &scaled);
        }
    }
}

static void lmmc_gemm_block(lmmc_real_t alpha, const lmmc_mat_t* A,
    int transA, const lmmc_mat_t* B, int transB, lmmc_mat_t* C,
    size_t ii, size_t kk, size_t jj) {
    size_t K = A->cols;
    if (transA) {
        K = A->rows;
    }
    size_t i_end = ii + 64;
    if (i_end > C->rows) {
        i_end = C->rows;
    }
    size_t k_end = kk + 64;
    if (k_end > K) {
        k_end = K;
    }
    size_t j_end = jj + 64;
    if (j_end > C->cols) {
        j_end = C->cols;
    }
    for (size_t i = ii; i < i_end; ++i) {
        for (size_t k = kk; k < k_end; ++k) {
            lmmc_real_t a_ik;
            if (transA) {
                a_ik = A->data[k * A->stride + i];
            } else {
                a_ik = A->data[i * A->stride + k];
            }
            lmmc_real_t alpha_a_ik;
            LMMC_REAL_MUL(&alpha_a_ik, &alpha, &a_ik);
            for (size_t j = jj; j < j_end; ++j) {
                lmmc_real_t b_kj;
                if (transB) {
                    b_kj = B->data[j * B->stride + k];
                } else {
                    b_kj = B->data[k * B->stride + j];
                }
                lmmc_real_t tmp_mul;
                lmmc_real_t tmp_sum;
                LMMC_REAL_MUL(&tmp_mul, &alpha_a_ik, &b_kj);
                LMMC_REAL_ADD(&tmp_sum, &C->data[i * C->stride + j], &tmp_mul);
                LMMC_REAL_SET(&C->data[i * C->stride + j], &tmp_sum);
            }
        }
    }
}

static void lmmc_gemv_accumulate(lmmc_real_t alpha, const lmmc_mat_t* A,
    int transA, const lmmc_vec_t* x, lmmc_vec_t* y) {
    const lmmc_real_t* restrict a_data = A->data;
    size_t a_stride = A->stride;
    size_t M = y->size;
    size_t N = x->size;
    /** @brief 累加矩阵向量乘积：y += alpha * op(A) * x。 */
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
        /** @brief 累加当前行：y[i] += alpha * sum。 */
        LMMC_REAL_MUL(&alpha_sum, &alpha, &sum);
        LMMC_REAL_ADD(&tmp_sum, &y->data[i], &alpha_sum);
        LMMC_REAL_SET(&y->data[i], &tmp_sum);
    }

    LMMC_REAL_CLEAR(&sum);
    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&tmp_sum);
    LMMC_REAL_CLEAR(&alpha_sum);
}

static void lmmc_gemm_accumulate(lmmc_real_t alpha, const lmmc_mat_t* A,
    int transA, const lmmc_mat_t* B, int transB, lmmc_mat_t* C, size_t K) {
    /** @brief 按 ii、kk、jj、i、k、j 的顺序累加。 */
    for (size_t ii = 0; ii < C->rows; ii += 64) {
        for (size_t kk = 0; kk < K; kk += 64) {
            for (size_t jj = 0; jj < C->cols; jj += 64) {
                lmmc_gemm_block(alpha, A, transA, B, transB, C, ii, kk, jj);
            }
        }
    }
}

lmmc_status_t lmmc_mat_gemm(lmmc_real_t alpha, const lmmc_mat_t* A, int transA,
    const lmmc_mat_t* B, int transB, lmmc_real_t beta, lmmc_mat_t* C) {
    if (!lmmc_mat_descriptor_is_valid(A) ||
        !lmmc_mat_descriptor_is_valid(B) ||
        !lmmc_mat_descriptor_is_valid(C)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    size_t M = A->rows;
    size_t K_A = A->cols;
    size_t K_B = B->rows;
    size_t N = B->cols;
    if (transA) {
        M = A->cols;
        K_A = A->rows;
    }
    if (transB) {
        K_B = B->cols;
        N = B->rows;
    }

    if (K_A != K_B) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    if (C->rows != M || C->cols != N) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    lmmc_status_t status = lmmc_gemm_storage_check(A, B, C);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    lmmc_gemm_scale(C, beta);
    if (alpha == 0.0) {
        return LMMC_STATUS_OK;
    }
    lmmc_gemm_accumulate(alpha, A, transA, B, transB, C, K_A);
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
    lmmc_status_t status = lmmc_gemv_storage_check(A, x, y);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    lmmc_gemv_scale(y, beta);
    if (alpha != 0.0) {
        lmmc_gemv_accumulate(alpha, A, transA, x, y);
    }
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
