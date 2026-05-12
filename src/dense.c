#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "lmmc/config.h"
#include "lmmc/dense.h"

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

lmmc_status_t lmmc_mat_mul(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c) {
    if (a == NULL || b == NULL || c == NULL || a->data == NULL || b->data == NULL || c->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->cols != b->rows || c->rows != a->rows || c->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    const lmmc_real_t* restrict a_data = a->data;
    const lmmc_real_t* restrict b_data = b->data;
    lmmc_real_t* restrict c_data = c->data;
    
    size_t a_stride = a->stride;
    size_t b_stride = b->stride;
    size_t c_stride = c->stride;
    
    size_t M = a->rows;
    size_t K_dim = a->cols;
    size_t N = b->cols;

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            LMMC_REAL_SET_D(&c_data[i * c_stride + j], 0.0);
        }
    }

    size_t bs = 64;
    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul);
    lmmc_real_t tmp_sum; LMMC_REAL_INIT(&tmp_sum);

    for (size_t ii = 0; ii < M; ii += bs) {
        size_t i_end = (ii + bs > M) ? M : ii + bs;
        for (size_t kk = 0; kk < K_dim; kk += bs) {
            size_t k_end = (kk + bs > K_dim) ? K_dim : kk + bs;
            for (size_t jj = 0; jj < N; jj += bs) {
                size_t j_end = (jj + bs > N) ? N : jj + bs;
                for (size_t i = ii; i < i_end; ++i) {
                    for (size_t k = kk; k < k_end; ++k) {
                        lmmc_real_t a_ik; LMMC_REAL_INIT(&a_ik);
                        LMMC_REAL_SET(&a_ik, &a_data[i * a_stride + k]);
                        
                        size_t j = jj;
                        for (; j + 3 < j_end; j += 4) {
                            LMMC_REAL_MUL(&tmp_mul, &a_ik, &b_data[k * b_stride + j]);
                            LMMC_REAL_ADD(&tmp_sum, &c_data[i * c_stride + j], &tmp_mul);
                            LMMC_REAL_SET(&c_data[i * c_stride + j], &tmp_sum);

                            LMMC_REAL_MUL(&tmp_mul, &a_ik, &b_data[k * b_stride + j + 1]);
                            LMMC_REAL_ADD(&tmp_sum, &c_data[i * c_stride + j + 1], &tmp_mul);
                            LMMC_REAL_SET(&c_data[i * c_stride + j + 1], &tmp_sum);

                            LMMC_REAL_MUL(&tmp_mul, &a_ik, &b_data[k * b_stride + j + 2]);
                            LMMC_REAL_ADD(&tmp_sum, &c_data[i * c_stride + j + 2], &tmp_mul);
                            LMMC_REAL_SET(&c_data[i * c_stride + j + 2], &tmp_sum);

                            LMMC_REAL_MUL(&tmp_mul, &a_ik, &b_data[k * b_stride + j + 3]);
                            LMMC_REAL_ADD(&tmp_sum, &c_data[i * c_stride + j + 3], &tmp_mul);
                            LMMC_REAL_SET(&c_data[i * c_stride + j + 3], &tmp_sum);
                        }
                        for (; j < j_end; ++j) {
                            LMMC_REAL_MUL(&tmp_mul, &a_ik, &b_data[k * b_stride + j]);
                            LMMC_REAL_ADD(&tmp_sum, &c_data[i * c_stride + j], &tmp_mul);
                            LMMC_REAL_SET(&c_data[i * c_stride + j], &tmp_sum);
                        }
                        LMMC_REAL_CLEAR(&a_ik);
                    }
                }
            }
        }
    }

    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&tmp_sum);
    return LMMC_STATUS_OK;
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

lmmc_status_t lmmc_mat_vec_mul(const lmmc_mat_t* a, const lmmc_vec_t* x, lmmc_vec_t* y) {
    size_t i = 0;
    size_t j = 0;
    if (a == NULL || x == NULL || y == NULL || a->data == NULL || x->data == NULL || y->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->cols != x->size || a->rows != y->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    lmmc_real_t sum; LMMC_REAL_INIT(&sum);
    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul);
    lmmc_real_t tmp_sum; LMMC_REAL_INIT(&tmp_sum);

    for (i = 0; i < a->rows; ++i) {
        LMMC_REAL_SET_D(&sum, 0.0);
        for (j = 0; j < a->cols; ++j) {
            LMMC_REAL_MUL(&tmp_mul, &a->data[i * a->stride + j], &x->data[j]);
            LMMC_REAL_ADD(&tmp_sum, &sum, &tmp_mul);
            LMMC_REAL_SET(&sum, &tmp_sum);
        }
        LMMC_REAL_SET(&y->data[i], &sum);
    }

    LMMC_REAL_CLEAR(&sum);
    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&tmp_sum);
    return LMMC_STATUS_OK;
}
