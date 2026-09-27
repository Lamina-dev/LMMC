#include "precond_internal.h"

static lmmc_status_t lmmc_ilut_load_row(const lmmc_sparse_mat_t* a,
    size_t i, lmmc_ilu_row_t* row) {
    row->count = 0;
    for (size_t p = a->row_ptr[i]; p < a->row_ptr[i + 1]; ++p) {
        size_t col = a->col_idx[p];
        lmmc_real_t value = a->values[p];
        if (!isfinite(value)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        if (!row->present[col]) {
            row->present[col] = 1;
            row->processed[col] = 0;
            row->active[row->count++] = col;
            row->values[col] = value;
        } else {
            row->values[col] = row->values[col] + value;
            if (!isfinite(row->values[col])) {
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
        }
    }
    return LMMC_STATUS_OK;
}

static size_t lmmc_ilut_next_lower(const lmmc_ilu_row_t* row, size_t i) {
    size_t k = (size_t)-1;
    for (size_t p = 0; p < row->count; ++p) {
        size_t col = row->active[p];
        if (!row->present[col] || row->processed[col] || col >= i) {
            continue;
        }
        if (k == (size_t)-1 || col < k) {
            k = col;
        }
    }
    return k;
}

static lmmc_status_t lmmc_ilut_subtract_upper(const lmmc_precond_ilu_impl_t* impl,
    lmmc_ilu_row_t* row, size_t k, lmmc_real_t multiplier) {
    for (size_t p = impl->diag_pos[k] + 1; p < impl->row_ptr[k + 1]; ++p) {
        size_t j = impl->col_idx[p];
        lmmc_real_t delta = multiplier * impl->lu_values[p];
        if (j <= k) {
            continue;
        }
        if (!isfinite(delta)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        if (!row->present[j]) {
            row->present[j] = 1;
            row->processed[j] = 0;
            row->active[row->count++] = j;
            row->values[j] = -delta;
        } else {
            row->values[j] = row->values[j] - delta;
        }
        if (!isfinite(row->values[j])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_ilut_eliminate_entry(const lmmc_precond_ilu_impl_t* impl,
    lmmc_ilu_row_t* row, size_t k, lmmc_real_t drop_tol) {
    row->processed[k] = 1;
    lmmc_real_t aik = row->values[k];
    if (!isfinite(aik)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (fabs(aik) <= drop_tol) {
        row->present[k] = 0;
        row->values[k] = 0.0;
        return LMMC_STATUS_OK;
    }
    if (impl->diag_pos[k] == (size_t)-1 || impl->diag_pos[k] >= impl->nnz) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    lmmc_real_t diag = impl->lu_values[impl->diag_pos[k]];
    if (!isfinite(diag) || fabs(diag) <= 1e-15) {
        return LMMC_STATUS_SINGULAR_MATRIX;
    }
    lmmc_real_t multiplier = aik / diag;
    if (!isfinite(multiplier)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    row->values[k] = multiplier;
    return lmmc_ilut_subtract_upper(impl, row, k, multiplier);
}

static lmmc_status_t lmmc_ilut_eliminate_row(const lmmc_precond_ilu_impl_t* impl,
    lmmc_ilu_row_t* row, size_t i, lmmc_real_t drop_tol) {
    for (;;) {
        size_t k = lmmc_ilut_next_lower(row, i);
        if (k == (size_t)-1) {
            return LMMC_STATUS_OK;
        }
        lmmc_status_t st = lmmc_ilut_eliminate_entry(impl, row, k, drop_tol);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
    }
}

static lmmc_status_t lmmc_ilut_append_entries(lmmc_precond_ilu_impl_t* impl,
    const lmmc_ilu_row_t* row, const size_t* cols, size_t count) {
    for (size_t p = 0; p < count; ++p) {
        size_t col = cols[p];
        lmmc_real_t value = row->values[col];
        if (!isfinite(value)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        impl->col_idx[impl->nnz] = col;
        impl->lu_values[impl->nnz] = value;
        impl->nnz += 1;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_ilut_store_row(lmmc_precond_ilu_impl_t* impl,
    lmmc_ilu_row_t* row, size_t i, lmmc_real_t drop_tol, size_t fill) {
    if (!row->present[i]) {
        row->present[i] = 1;
        row->processed[i] = 0;
        row->active[row->count++] = i;
        row->values[i] = 0.0;
    }
    lmmc_real_t diag = row->values[i];
    if (!isfinite(diag) || fabs(diag) <= 1e-15) {
        return LMMC_STATUS_SINGULAR_MATRIX;
    }
    size_t lower_count, upper_count;
    lmmc_ilu_select(row, i, 1, drop_tol, fill, row->lower, &lower_count);
    lmmc_ilu_select(row, i, 0, drop_tol, fill, row->upper, &upper_count);
    size_t needed = lower_count + 1 + upper_count;
    if (impl->nnz > (size_t)-1 - needed) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_status_t st = lmmc_ilu_impl_reserve(impl, impl->nnz + needed);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_ilut_append_entries(impl, row, row->lower, lower_count);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    impl->diag_pos[i] = impl->nnz;
    impl->col_idx[impl->nnz] = i;
    impl->lu_values[impl->nnz] = diag;
    impl->nnz += 1;
    st = lmmc_ilut_append_entries(impl, row, row->upper, upper_count);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    impl->row_ptr[i + 1] = impl->nnz;
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_ilut_factor(const lmmc_sparse_mat_t* a,
    lmmc_precond_ilu_impl_t* impl, lmmc_ilu_row_t* row,
    lmmc_real_t drop_tol, size_t fill) {
    size_t capacity = a->nnz;
    if (capacity < 16) {
        capacity = 16;
    }
    if (capacity < a->rows && a->rows <= (size_t)-1 - capacity) {
        capacity += a->rows;
    }
    lmmc_status_t st = lmmc_ilu_impl_reserve(impl, capacity);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    impl->row_ptr[0] = 0;
    for (size_t i = 0; i < a->rows; ++i) {
        st = lmmc_ilut_load_row(a, i, row);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        st = lmmc_ilut_eliminate_row(impl, row, i, drop_tol);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        st = lmmc_ilut_store_row(impl, row, i, drop_tol, fill);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        lmmc_ilu_row_reset(row);
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_precond_create_ilut(const lmmc_sparse_mat_t* a,
    lmmc_real_t drop_tol, size_t max_fill_per_row, lmmc_precond_t* out_precond) {
    lmmc_status_t st = lmmc_sparse_validate(a);
    if (out_precond == NULL || st != LMMC_STATUS_OK ||
        a->format != LMMC_SPARSE_CSR || !isfinite(drop_tol) ||
        drop_tol < 0.0 || max_fill_per_row == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }
    size_t bytes;
    st = lmmc_ilu_row_bytes(a->rows, max_fill_per_row, &bytes);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    lmmc_precond_ilu_impl_t* impl = NULL;
    lmmc_ilu_row_t row = {0};
    st = lmmc_ilu_impl_create(a->rows, &impl);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_ilu_row_create(a->rows, max_fill_per_row, bytes, &row);
    if (st == LMMC_STATUS_OK) {
        st = lmmc_ilut_factor(a, impl, &row, drop_tol, max_fill_per_row);
    }
    lmmc_memory_free(row.values);
    if (st != LMMC_STATUS_OK) {
        lmmc_ilu_impl_destroy(impl);
        return st;
    }
    out_precond->type = LMMC_PRECOND_ILUT;
    out_precond->size = a->rows;
    out_precond->impl = impl;
    out_precond->owns_data = 1;
    return LMMC_STATUS_OK;
}
