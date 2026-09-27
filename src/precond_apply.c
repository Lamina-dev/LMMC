#include "precond_internal.h"

static lmmc_status_t lmmc_ilu_forward(const lmmc_precond_ilu_impl_t* impl,
    const lmmc_vec_t* rhs) {
    for (size_t i = 0; i < impl->size; ++i) {
        lmmc_real_t sum = rhs->data[i];
        for (size_t p = impl->row_ptr[i]; p < impl->row_ptr[i + 1]; ++p) {
            size_t j = impl->col_idx[p];
            if (j >= i) {
                continue;
            }
            lmmc_real_t product = impl->lu_values[p] * impl->y_arr[j];
            sum = sum - product;
        }
        if (!isfinite(sum)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        impl->y_arr[i] = sum;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_ilu_backward(const lmmc_precond_ilu_impl_t* impl,
    lmmc_vec_t* out) {
    for (size_t i = impl->size; i > 0; --i) {
        size_t row = i - 1;
        lmmc_real_t sum = impl->y_arr[row];
        for (size_t p = impl->row_ptr[row]; p < impl->row_ptr[row + 1]; ++p) {
            size_t j = impl->col_idx[p];
            if (j <= row) {
                continue;
            }
            lmmc_real_t product = impl->lu_values[p] * out->data[j];
            sum = sum - product;
        }
        lmmc_real_t diag = impl->lu_values[impl->diag_pos[row]];
        if (!isfinite(sum) || !isfinite(diag) || fabs(diag) <= 1e-15) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        out->data[row] = sum / diag;
        if (!isfinite(out->data[row])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_ilu_apply(const lmmc_precond_t* precond,
    const lmmc_vec_t* rhs, lmmc_vec_t* out) {
    const lmmc_precond_ilu_impl_t* impl = precond->impl;
    if (impl == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (impl->size != precond->size || impl->diag_pos == NULL ||
        impl->row_ptr == NULL || impl->y_arr == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    memset(impl->y_arr, 0, impl->size * sizeof(lmmc_real_t));
    lmmc_status_t st = lmmc_ilu_forward(impl, rhs);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    return lmmc_ilu_backward(impl, out);
}
