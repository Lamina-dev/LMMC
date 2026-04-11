#include <math.h>
#include <string.h>
#include "memory_bridge.h"
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
    double* data = NULL;

    if (out_mat == NULL || rows == 0 || cols == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lmmc_mul_overflow_size(rows, cols, &n_elem) || lmmc_mul_overflow_size(n_elem, sizeof(double), &n_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    data = (double*)lmmc_alloc(n_bytes);
    if (data == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(data, 0, n_bytes);

    out_mat->rows = rows;
    out_mat->cols = cols;
    out_mat->stride = cols;
    out_mat->data = data;
    out_mat->owns_data = 1;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_wrap(size_t rows, size_t cols, size_t stride, double* data, lmmc_mat_t* out_mat) {
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
        lmmc_free(mat->data);
    }
    mat->rows = 0;
    mat->cols = 0;
    mat->stride = 0;
    mat->data = NULL;
    mat->owns_data = 0;
}

lmmc_status_t lmmc_mat_fill(lmmc_mat_t* mat, double value) {
    size_t i = 0;
    size_t j = 0;
    if (mat == NULL || mat->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    for (i = 0; i < mat->rows; ++i) {
        for (j = 0; j < mat->cols; ++j) {
            mat->data[i * mat->stride + j] = value;
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_copy(const lmmc_mat_t* src, lmmc_mat_t* dst) {
    size_t i = 0;
    if (src == NULL || dst == NULL || src->data == NULL || dst->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (src->rows != dst->rows || src->cols != dst->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    for (i = 0; i < src->rows; ++i) {
        memcpy(dst->data + i * dst->stride, src->data + i * src->stride, src->cols * sizeof(double));
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
            dst->data[j * dst->stride + i] = src->data[i * src->stride + j];
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_mul(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c) {
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
    if (a == NULL || b == NULL || c == NULL || a->data == NULL || b->data == NULL || c->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->cols != b->rows || c->rows != a->rows || c->cols != b->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    for (i = 0; i < c->rows; ++i) {
        for (j = 0; j < c->cols; ++j) {
            double sum = 0.0;
            for (k = 0; k < a->cols; ++k) {
                sum += a->data[i * a->stride + k] * b->data[k * b->stride + j];
            }
            c->data[i * c->stride + j] = sum;
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_mat_norm_fro(const lmmc_mat_t* a, double* out_norm) {
    size_t i = 0;
    size_t j = 0;
    double sum = 0.0;
    if (a == NULL || out_norm == NULL || a->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    for (i = 0; i < a->rows; ++i) {
        for (j = 0; j < a->cols; ++j) {
            double v = a->data[i * a->stride + j];
            sum += v * v;
        }
    }
    *out_norm = sqrt(sum);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_create(size_t size, lmmc_vec_t* out_vec) {
    size_t n_bytes = 0;
    double* data = NULL;
    if (out_vec == NULL || size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lmmc_mul_overflow_size(size, sizeof(double), &n_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    data = (double*)lmmc_alloc(n_bytes);
    if (data == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(data, 0, n_bytes);

    out_vec->size = size;
    out_vec->data = data;
    out_vec->owns_data = 1;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_wrap(size_t size, double* data, lmmc_vec_t* out_vec) {
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
        lmmc_free(vec->data);
    }
    vec->size = 0;
    vec->data = NULL;
    vec->owns_data = 0;
}

lmmc_status_t lmmc_vec_fill(lmmc_vec_t* vec, double value) {
    size_t i = 0;
    if (vec == NULL || vec->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    for (i = 0; i < vec->size; ++i) {
        vec->data[i] = value;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_vec_dot(const lmmc_vec_t* a, const lmmc_vec_t* b, double* out_dot) {
    size_t i = 0;
    double sum = 0.0;
    if (a == NULL || b == NULL || out_dot == NULL || a->data == NULL || b->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->size != b->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    for (i = 0; i < a->size; ++i) {
        sum += a->data[i] * b->data[i];
    }
    *out_dot = sum;
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
    for (i = 0; i < a->rows; ++i) {
        double sum = 0.0;
        for (j = 0; j < a->cols; ++j) {
            sum += a->data[i * a->stride + j] * x->data[j];
        }
        y->data[i] = sum;
    }
    return LMMC_STATUS_OK;
}
