#include "precond_internal.h"

lmmc_status_t lmmc_precond_create_none(size_t size, lmmc_precond_t* out_precond) {
    if (out_precond == NULL || size == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    out_precond->type = LMMC_PRECOND_NONE;
    out_precond->size = size;
    out_precond->impl = NULL;
    out_precond->owns_data = 0;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_jacobi_diagonals(const lmmc_sparse_mat_t* a,
    lmmc_real_t* inverse) {
    for (size_t i = 0; i < a->rows; ++i) {
        size_t pos;
        if (!lmmc_find_col_pos(a->col_idx, a->row_ptr[i], a->row_ptr[i + 1], i, &pos)) {
            return LMMC_STATUS_SINGULAR_MATRIX;
        }
        lmmc_real_t diag = a->values[pos];
        if (!isfinite(diag) || fabs(diag) <= 1e-15) {
            return LMMC_STATUS_SINGULAR_MATRIX;
        }
        inverse[i] = 1.0 / diag;
        if (!isfinite(inverse[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_precond_create_jacobi(const lmmc_sparse_mat_t* a,
    lmmc_precond_t* out_precond) {
    lmmc_status_t st = lmmc_sparse_validate(a);
    if (out_precond == NULL || st != LMMC_STATUS_OK || a->format != LMMC_SPARSE_CSR) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    if (a->rows > (size_t)-1 / sizeof(lmmc_real_t)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_real_t* inverse = lmmc_memory_alloc(a->rows * sizeof(lmmc_real_t));
    if (inverse == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    st = lmmc_jacobi_diagonals(a, inverse);
    if (st != LMMC_STATUS_OK) {
        lmmc_memory_free(inverse);
        return st;
    }
    out_precond->type = LMMC_PRECOND_JACOBI;
    out_precond->size = a->rows;
    out_precond->impl = inverse;
    out_precond->owns_data = 1;
    return LMMC_STATUS_OK;
}

void lmmc_precond_destroy(lmmc_precond_t* precond) {
    if (precond == NULL) {
        return;
    }
    if (precond->owns_data && precond->impl != NULL) {
        if (precond->type == LMMC_PRECOND_ILU0 || precond->type == LMMC_PRECOND_ILUT) {
            lmmc_ilu_impl_destroy(precond->impl);
        }
        else {
            lmmc_memory_free(precond->impl);
        }
    }
    precond->type = LMMC_PRECOND_NONE;
    precond->size = 0;
    precond->impl = NULL;
    precond->owns_data = 0;
}

static lmmc_status_t lmmc_jacobi_apply(const lmmc_precond_t* precond,
    const lmmc_vec_t* rhs, lmmc_vec_t* out) {
    if (precond->impl == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    const lmmc_real_t* inverse = precond->impl;
    for (size_t i = 0; i < rhs->size; ++i) {
        lmmc_real_t value = rhs->data[i] * inverse[i];
        if (!isfinite(rhs->data[i]) || !isfinite(value)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        out->data[i] = value;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_precond_apply(const lmmc_precond_t* precond,
    const lmmc_vec_t* rhs, lmmc_vec_t* out) {
    if (precond == NULL || rhs == NULL || out == NULL || rhs->data == NULL || out->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (rhs->size != precond->size || out->size != precond->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    switch (precond->type) {
    case LMMC_PRECOND_NONE:
        return lmmc_vec_copy(rhs, out);
    case LMMC_PRECOND_JACOBI:
        return lmmc_jacobi_apply(precond, rhs, out);
    case LMMC_PRECOND_ILU0:
    case LMMC_PRECOND_ILUT:
        return lmmc_ilu_apply(precond, rhs, out);
    default:
        return LMMC_STATUS_NOT_IMPLEMENTED;
    }
}
