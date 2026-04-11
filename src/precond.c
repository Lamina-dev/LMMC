#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "lmmc/precond.h"

typedef struct {
    size_t size;
    size_t nnz;
    size_t capacity;
    size_t* row_ptr;
    size_t* col_idx;
    double* lu_values;
    size_t* diag_pos;
} lmmc_precond_ilu_impl_t;

static int lmmc_is_finite_number(double v) {
    return isfinite(v) ? 1 : 0;
}

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

static lmmc_status_t lmmc_sparse_validate_csr_basic(const lmmc_sparse_mat_t* a) {
    size_t i = 0;
    if (a == NULL || a->row_ptr == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows == 0 || a->cols == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->nnz > 0 && (a->col_idx == NULL || a->values == NULL)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->row_ptr[0] != 0 || a->row_ptr[a->rows] != a->nnz) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    for (i = 0; i < a->rows; ++i) {
        if (a->row_ptr[i] > a->row_ptr[i + 1]) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
    }
    for (i = 0; i < a->nnz; ++i) {
        if (a->col_idx[i] >= a->cols) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }
    }
    return LMMC_STATUS_OK;
}

static int lmmc_find_col_pos(
    const size_t* col_idx,
    size_t start,
    size_t end,
    size_t target_col,
    size_t* out_pos
) {
    size_t p = 0;

    if (col_idx == NULL || out_pos == NULL || start > end) {
        return 0;
    }

    for (p = start; p < end; ++p) {
        if (col_idx[p] == target_col) {
            *out_pos = p;
            return 1;
        }
    }
    return 0;
}

static void lmmc_ilu_impl_destroy(lmmc_precond_ilu_impl_t* impl) {
    if (impl == NULL) {
        return;
    }

    lmmc_free(impl->diag_pos);
    lmmc_free(impl->lu_values);
    lmmc_free(impl->col_idx);
    lmmc_free(impl->row_ptr);
    lmmc_free(impl);
}

static void lmmc_sort_size_t_asc(size_t* values, size_t count) {
    size_t i = 0;

    if (values == NULL || count < 2) {
        return;
    }

    for (i = 1; i < count; ++i) {
        size_t key = values[i];
        size_t j = i;
        while (j > 0 && values[j - 1] > key) {
            values[j] = values[j - 1];
            --j;
        }
        values[j] = key;
    }
}

static void lmmc_select_top_abs(
    const size_t* active_cols,
    size_t active_count,
    const unsigned char* present,
    const double* workspace,
    size_t row,
    int lower_side,
    double drop_tol,
    size_t max_keep,
    size_t* out_cols,
    size_t* out_count
) {
    size_t i = 0;
    size_t selected = 0;

    if (active_cols == NULL || present == NULL || workspace == NULL || out_cols == NULL || out_count == NULL ||
        max_keep == 0) {
        if (out_count != NULL) {
            *out_count = 0;
        }
        return;
    }

    for (i = 0; i < active_count; ++i) {
        size_t col = active_cols[i];
        double v = 0.0;
        double av = 0.0;

        if (!present[col]) {
            continue;
        }

        if (lower_side) {
            if (col >= row) {
                continue;
            }
        } else {
            if (col <= row) {
                continue;
            }
        }

        v = workspace[col];
        av = fabs(v);
        if (av <= drop_tol) {
            continue;
        }

        if (selected < max_keep) {
            out_cols[selected++] = col;
        } else {
            size_t min_idx = 0;
            double min_abs = fabs(workspace[out_cols[0]]);
            size_t s = 0;
            for (s = 1; s < selected; ++s) {
                double cur_abs = fabs(workspace[out_cols[s]]);
                if (cur_abs < min_abs) {
                    min_abs = cur_abs;
                    min_idx = s;
                }
            }

            if (av > min_abs) {
                out_cols[min_idx] = col;
            }
        }
    }

    lmmc_sort_size_t_asc(out_cols, selected);
    *out_count = selected;
}

static lmmc_status_t lmmc_ilu_impl_reserve(lmmc_precond_ilu_impl_t* impl, size_t required_capacity) {
    size_t new_capacity = 0;
    size_t idx_bytes = 0;
    size_t val_bytes = 0;
    size_t* new_col_idx = NULL;
    double* new_lu_values = NULL;

    if (impl == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    if (required_capacity <= impl->capacity) {
        return LMMC_STATUS_OK;
    }

    new_capacity = (impl->capacity == 0) ? 16 : impl->capacity;
    while (new_capacity < required_capacity) {
        size_t doubled = 0;
        if (lmmc_mul_overflow_size(new_capacity, 2, &doubled)) {
            new_capacity = required_capacity;
            break;
        }
        new_capacity = doubled;
    }

    if (lmmc_mul_overflow_size(new_capacity, sizeof(size_t), &idx_bytes) ||
        lmmc_mul_overflow_size(new_capacity, sizeof(double), &val_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    new_col_idx = (size_t*)lmmc_alloc(idx_bytes);
    new_lu_values = (double*)lmmc_alloc(val_bytes);
    if (new_col_idx == NULL || new_lu_values == NULL) {
        lmmc_free(new_lu_values);
        lmmc_free(new_col_idx);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    if (impl->nnz > 0) {
        memcpy(new_col_idx, impl->col_idx, impl->nnz * sizeof(size_t));
        memcpy(new_lu_values, impl->lu_values, impl->nnz * sizeof(double));
    }

    lmmc_free(impl->lu_values);
    lmmc_free(impl->col_idx);
    impl->col_idx = new_col_idx;
    impl->lu_values = new_lu_values;
    impl->capacity = new_capacity;
    return LMMC_STATUS_OK;
}

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

lmmc_status_t lmmc_precond_create_jacobi(const lmmc_sparse_mat_t* a, lmmc_precond_t* out_precond) {
    size_t i = 0;
    size_t bytes = 0;
    double* diag_inv = NULL;
    lmmc_status_t st = lmmc_sparse_validate_csr_basic(a);

    if (out_precond == NULL || st != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    if (a->rows > ((size_t)-1) / sizeof(double)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    bytes = a->rows * sizeof(double);

    diag_inv = (double*)lmmc_alloc(bytes);
    if (diag_inv == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(diag_inv, 0, bytes);

    for (i = 0; i < a->rows; ++i) {
        size_t p = 0;
        int found = 0;
        double diag = 0.0;

        for (p = a->row_ptr[i]; p < a->row_ptr[i + 1]; ++p) {
            if (a->col_idx[p] == i) {
                diag = a->values[p];
                found = 1;
                break;
            }
        }

        if (!found || !lmmc_is_finite_number(diag) || fabs(diag) <= 1e-15) {
            lmmc_free(diag_inv);
            return LMMC_STATUS_SINGULAR_MATRIX;
        }

        diag_inv[i] = 1.0 / diag;
        if (!lmmc_is_finite_number(diag_inv[i])) {
            lmmc_free(diag_inv);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }

    out_precond->type = LMMC_PRECOND_JACOBI;
    out_precond->size = a->rows;
    out_precond->impl = diag_inv;
    out_precond->owns_data = 1;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_precond_create_ilu0(const lmmc_sparse_mat_t* a, lmmc_precond_t* out_precond) {
    size_t i = 0;
    size_t row_ptr_bytes = 0;
    size_t idx_bytes = 0;
    size_t val_bytes = 0;
    size_t diag_bytes = 0;
    lmmc_precond_ilu_impl_t* impl = NULL;
    lmmc_status_t st = lmmc_sparse_validate_csr_basic(a);

    if (out_precond == NULL || st != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    if (lmmc_mul_overflow_size(a->rows + 1, sizeof(size_t), &row_ptr_bytes) ||
        lmmc_mul_overflow_size(a->nnz, sizeof(size_t), &idx_bytes) ||
        lmmc_mul_overflow_size(a->nnz, sizeof(double), &val_bytes) ||
        lmmc_mul_overflow_size(a->rows, sizeof(size_t), &diag_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    impl = (lmmc_precond_ilu_impl_t*)lmmc_alloc(sizeof(lmmc_precond_ilu_impl_t));
    if (impl == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(impl, 0, sizeof(lmmc_precond_ilu_impl_t));

    impl->size = a->rows;
    impl->nnz = a->nnz;
    impl->capacity = a->nnz;

    impl->row_ptr = (size_t*)lmmc_alloc(row_ptr_bytes);
    impl->diag_pos = (size_t*)lmmc_alloc(diag_bytes);
    if (impl->row_ptr == NULL || impl->diag_pos == NULL) {
        lmmc_ilu_impl_destroy(impl);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    if (a->nnz > 0) {
        impl->col_idx = (size_t*)lmmc_alloc(idx_bytes);
        impl->lu_values = (double*)lmmc_alloc(val_bytes);
        if (impl->col_idx == NULL || impl->lu_values == NULL) {
            lmmc_ilu_impl_destroy(impl);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
        memcpy(impl->col_idx, a->col_idx, idx_bytes);
        memcpy(impl->lu_values, a->values, val_bytes);
    }

    memcpy(impl->row_ptr, a->row_ptr, row_ptr_bytes);

    for (i = 0; i < a->rows; ++i) {
        size_t p = 0;
        int found = 0;
        impl->diag_pos[i] = (size_t)-1;
        for (p = impl->row_ptr[i]; p < impl->row_ptr[i + 1]; ++p) {
            if (impl->col_idx[p] == i) {
                impl->diag_pos[i] = p;
                found = 1;
                break;
            }
        }
        if (!found) {
            lmmc_ilu_impl_destroy(impl);
            return LMMC_STATUS_SINGULAR_MATRIX;
        }
    }

    for (i = 0; i < a->rows; ++i) {
        size_t row_start = impl->row_ptr[i];
        size_t row_end = impl->row_ptr[i + 1];
        size_t p = 0;

        for (p = row_start; p < row_end; ++p) {
            size_t j = impl->col_idx[p];
            double sum = impl->lu_values[p];
            size_t q = 0;

            if (j >= i) {
                continue;
            }

            for (q = row_start; q < row_end; ++q) {
                size_t k = impl->col_idx[q];
                size_t pos_kj = 0;

                if (k >= j) {
                    continue;
                }
                if (lmmc_find_col_pos(impl->col_idx, impl->row_ptr[k], impl->row_ptr[k + 1], j, &pos_kj)) {
                    sum -= impl->lu_values[q] * impl->lu_values[pos_kj];
                }
            }

            {
                double diag_j = impl->lu_values[impl->diag_pos[j]];
                if (!lmmc_is_finite_number(sum) || !lmmc_is_finite_number(diag_j) || fabs(diag_j) <= 1e-15) {
                    lmmc_ilu_impl_destroy(impl);
                    return LMMC_STATUS_SINGULAR_MATRIX;
                }
                impl->lu_values[p] = sum / diag_j;
                if (!lmmc_is_finite_number(impl->lu_values[p])) {
                    lmmc_ilu_impl_destroy(impl);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
            }
        }

        for (p = row_start; p < row_end; ++p) {
            size_t j = impl->col_idx[p];
            double sum = impl->lu_values[p];
            size_t q = 0;

            if (j < i) {
                continue;
            }

            for (q = row_start; q < row_end; ++q) {
                size_t k = impl->col_idx[q];
                size_t pos_kj = 0;

                if (k >= i) {
                    continue;
                }
                if (lmmc_find_col_pos(impl->col_idx, impl->row_ptr[k], impl->row_ptr[k + 1], j, &pos_kj)) {
                    sum -= impl->lu_values[q] * impl->lu_values[pos_kj];
                }
            }

            if (!lmmc_is_finite_number(sum)) {
                lmmc_ilu_impl_destroy(impl);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
            impl->lu_values[p] = sum;
        }

        {
            double diag_i = impl->lu_values[impl->diag_pos[i]];
            if (!lmmc_is_finite_number(diag_i) || fabs(diag_i) <= 1e-15) {
                lmmc_ilu_impl_destroy(impl);
                return LMMC_STATUS_SINGULAR_MATRIX;
            }
        }
    }

    out_precond->type = LMMC_PRECOND_ILU0;
    out_precond->size = a->rows;
    out_precond->impl = impl;
    out_precond->owns_data = 1;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_precond_create_ilut(
    const lmmc_sparse_mat_t* a,
    double drop_tol,
    size_t max_fill_per_row,
    lmmc_precond_t* out_precond
) {
    size_t i = 0;
    size_t n = 0;
    size_t row_ptr_bytes = 0;
    size_t diag_bytes = 0;
    size_t workspace_bytes = 0;
    size_t mark_bytes = 0;
    size_t active_bytes = 0;
    size_t keep_bytes = 0;
    double* workspace = NULL;
    unsigned char* present = NULL;
    unsigned char* processed = NULL;
    size_t* active_cols = NULL;
    size_t* lower_keep_cols = NULL;
    size_t* upper_keep_cols = NULL;
    size_t active_count = 0;
    lmmc_precond_ilu_impl_t* impl = NULL;
    lmmc_status_t st = lmmc_sparse_validate_csr_basic(a);

    if (out_precond == NULL || st != LMMC_STATUS_OK || !lmmc_is_finite_number(drop_tol) ||
        drop_tol < 0.0 || max_fill_per_row == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    n = a->rows;
    if (lmmc_mul_overflow_size(n + 1, sizeof(size_t), &row_ptr_bytes) ||
        lmmc_mul_overflow_size(n, sizeof(size_t), &diag_bytes) ||
        lmmc_mul_overflow_size(n, sizeof(double), &workspace_bytes) ||
        lmmc_mul_overflow_size(n, sizeof(unsigned char), &mark_bytes) ||
        lmmc_mul_overflow_size(n, sizeof(size_t), &active_bytes) ||
        lmmc_mul_overflow_size(max_fill_per_row, sizeof(size_t), &keep_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    impl = (lmmc_precond_ilu_impl_t*)lmmc_alloc(sizeof(lmmc_precond_ilu_impl_t));
    if (impl == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(impl, 0, sizeof(lmmc_precond_ilu_impl_t));

    impl->size = n;
    impl->nnz = 0;
    impl->capacity = 0;

    impl->row_ptr = (size_t*)lmmc_alloc(row_ptr_bytes);
    impl->diag_pos = (size_t*)lmmc_alloc(diag_bytes);
    workspace = (double*)lmmc_alloc(workspace_bytes);
    present = (unsigned char*)lmmc_alloc(mark_bytes);
    processed = (unsigned char*)lmmc_alloc(mark_bytes);
    active_cols = (size_t*)lmmc_alloc(active_bytes);
    lower_keep_cols = (size_t*)lmmc_alloc(keep_bytes);
    upper_keep_cols = (size_t*)lmmc_alloc(keep_bytes);

    if (impl->row_ptr == NULL || impl->diag_pos == NULL || workspace == NULL || present == NULL ||
        processed == NULL || active_cols == NULL || lower_keep_cols == NULL || upper_keep_cols == NULL) {
        st = LMMC_STATUS_ALLOCATION_FAILED;
        goto fail;
    }

    memset(workspace, 0, workspace_bytes);
    memset(present, 0, mark_bytes);
    memset(processed, 0, mark_bytes);

    {
        size_t initial_capacity = a->nnz;
        if (initial_capacity < 16) {
            initial_capacity = 16;
        }
        if (initial_capacity < n && n <= ((size_t)-1) - initial_capacity) {
            initial_capacity += n;
        }

        st = lmmc_ilu_impl_reserve(impl, initial_capacity);
        if (st != LMMC_STATUS_OK) {
            goto fail;
        }
    }

    impl->row_ptr[0] = 0;

    for (i = 0; i < n; ++i) {
        size_t lower_keep_count = 0;
        size_t upper_keep_count = 0;
        double diag_i = 0.0;
        size_t p = 0;
        active_count = 0;

        for (p = a->row_ptr[i]; p < a->row_ptr[i + 1]; ++p) {
            size_t col = a->col_idx[p];
            double v = a->values[p];

            if (!lmmc_is_finite_number(v)) {
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto fail;
            }

            if (!present[col]) {
                present[col] = 1;
                processed[col] = 0;
                active_cols[active_count++] = col;
                workspace[col] = v;
            } else {
                workspace[col] += v;
                if (!lmmc_is_finite_number(workspace[col])) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto fail;
                }
            }
        }

        while (1) {
            size_t k = (size_t)-1;
            size_t idx = 0;

            for (idx = 0; idx < active_count; ++idx) {
                size_t col = active_cols[idx];
                if (!present[col] || processed[col] || col >= i) {
                    continue;
                }

                if (k == (size_t)-1 || col < k) {
                    k = col;
                }
            }

            if (k == (size_t)-1) {
                break;
            }

            processed[k] = 1;

            {
                double aik = workspace[k];
                double diag_k = 0.0;
                double lik = 0.0;
                size_t u_start = 0;
                size_t u_end = 0;

                if (!lmmc_is_finite_number(aik)) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto fail;
                }

                if (fabs(aik) <= drop_tol) {
                    present[k] = 0;
                    workspace[k] = 0.0;
                    continue;
                }

                if (impl->diag_pos[k] == (size_t)-1 || impl->diag_pos[k] >= impl->nnz) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto fail;
                }

                diag_k = impl->lu_values[impl->diag_pos[k]];
                if (!lmmc_is_finite_number(diag_k) || fabs(diag_k) <= 1e-15) {
                    st = LMMC_STATUS_SINGULAR_MATRIX;
                    goto fail;
                }

                lik = aik / diag_k;
                if (!lmmc_is_finite_number(lik)) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto fail;
                }
                workspace[k] = lik;

                u_start = impl->diag_pos[k] + 1;
                u_end = impl->row_ptr[k + 1];

                for (p = u_start; p < u_end; ++p) {
                    size_t j = impl->col_idx[p];
                    double delta = lik * impl->lu_values[p];

                    if (j <= k) {
                        continue;
                    }
                    if (!lmmc_is_finite_number(delta)) {
                        st = LMMC_STATUS_NUMERICAL_FAILURE;
                        goto fail;
                    }

                    if (!present[j]) {
                        present[j] = 1;
                        processed[j] = 0;
                        active_cols[active_count++] = j;
                        workspace[j] = -delta;
                    } else {
                        workspace[j] -= delta;
                    }

                    if (!lmmc_is_finite_number(workspace[j])) {
                        st = LMMC_STATUS_NUMERICAL_FAILURE;
                        goto fail;
                    }
                }
            }
        }

        if (!present[i]) {
            present[i] = 1;
            processed[i] = 0;
            active_cols[active_count++] = i;
            workspace[i] = 0.0;
        }

        diag_i = workspace[i];
        if (!lmmc_is_finite_number(diag_i) || fabs(diag_i) <= 1e-15) {
            st = LMMC_STATUS_SINGULAR_MATRIX;
            goto fail;
        }

        lmmc_select_top_abs(
            active_cols,
            active_count,
            present,
            workspace,
            i,
            1,
            drop_tol,
            max_fill_per_row,
            lower_keep_cols,
            &lower_keep_count
        );

        lmmc_select_top_abs(
            active_cols,
            active_count,
            present,
            workspace,
            i,
            0,
            drop_tol,
            max_fill_per_row,
            upper_keep_cols,
            &upper_keep_count
        );

        {
            size_t row_need = lower_keep_count + 1 + upper_keep_count;
            size_t required = 0;
            size_t idx = 0;

            if (impl->nnz > ((size_t)-1) - row_need) {
                st = LMMC_STATUS_INVALID_ARGUMENT;
                goto fail;
            }
            required = impl->nnz + row_need;

            st = lmmc_ilu_impl_reserve(impl, required);
            if (st != LMMC_STATUS_OK) {
                goto fail;
            }

            for (idx = 0; idx < lower_keep_count; ++idx) {
                size_t col = lower_keep_cols[idx];
                double v = workspace[col];
                if (!lmmc_is_finite_number(v)) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto fail;
                }
                impl->col_idx[impl->nnz] = col;
                impl->lu_values[impl->nnz] = v;
                impl->nnz += 1;
            }

            impl->diag_pos[i] = impl->nnz;
            impl->col_idx[impl->nnz] = i;
            impl->lu_values[impl->nnz] = diag_i;
            impl->nnz += 1;

            for (idx = 0; idx < upper_keep_count; ++idx) {
                size_t col = upper_keep_cols[idx];
                double v = workspace[col];
                if (!lmmc_is_finite_number(v)) {
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto fail;
                }
                impl->col_idx[impl->nnz] = col;
                impl->lu_values[impl->nnz] = v;
                impl->nnz += 1;
            }

            impl->row_ptr[i + 1] = impl->nnz;
        }

        for (p = 0; p < active_count; ++p) {
            size_t col = active_cols[p];
            present[col] = 0;
            processed[col] = 0;
            workspace[col] = 0.0;
        }
    }

    out_precond->type = LMMC_PRECOND_ILUT;
    out_precond->size = n;
    out_precond->impl = impl;
    out_precond->owns_data = 1;

    lmmc_free(upper_keep_cols);
    lmmc_free(lower_keep_cols);
    lmmc_free(active_cols);
    lmmc_free(processed);
    lmmc_free(present);
    lmmc_free(workspace);
    return LMMC_STATUS_OK;

fail:
    lmmc_free(upper_keep_cols);
    lmmc_free(lower_keep_cols);
    lmmc_free(active_cols);
    lmmc_free(processed);
    lmmc_free(present);
    lmmc_free(workspace);
    lmmc_ilu_impl_destroy(impl);
    return st;
}

lmmc_status_t lmmc_precond_apply(const lmmc_precond_t* precond, const lmmc_vec_t* rhs, lmmc_vec_t* out) {
    size_t i = 0;

    if (precond == NULL || rhs == NULL || out == NULL || rhs->data == NULL || out->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (rhs->size != precond->size || out->size != precond->size) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    if (precond->type == LMMC_PRECOND_NONE) {
        memcpy(out->data, rhs->data, rhs->size * sizeof(double));
        return LMMC_STATUS_OK;
    }

    if (precond->type == LMMC_PRECOND_JACOBI) {
        const double* diag_inv = NULL;
        if (precond->impl == NULL) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }

        diag_inv = (const double*)precond->impl;
        for (i = 0; i < rhs->size; ++i) {
            double v = rhs->data[i] * diag_inv[i];
            if (!lmmc_is_finite_number(rhs->data[i]) || !lmmc_is_finite_number(v)) {
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
            out->data[i] = v;
        }
        return LMMC_STATUS_OK;
    }

    if (precond->type == LMMC_PRECOND_ILU0 || precond->type == LMMC_PRECOND_ILUT) {
        const lmmc_precond_ilu_impl_t* impl = NULL;
        size_t bytes = 0;
        double* y = NULL;

        if (precond->impl == NULL) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }

        impl = (const lmmc_precond_ilu_impl_t*)precond->impl;
        if (impl->size != precond->size || impl->diag_pos == NULL || impl->row_ptr == NULL) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }

        if (lmmc_mul_overflow_size(impl->size, sizeof(double), &bytes)) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }

        y = (double*)lmmc_alloc(bytes);
        if (y == NULL) {
            return LMMC_STATUS_ALLOCATION_FAILED;
        }

        for (i = 0; i < impl->size; ++i) {
            size_t p = 0;
            double sum = rhs->data[i];

            for (p = impl->row_ptr[i]; p < impl->row_ptr[i + 1]; ++p) {
                size_t j = impl->col_idx[p];
                if (j >= i) {
                    continue;
                }
                sum -= impl->lu_values[p] * y[j];
            }

            if (!lmmc_is_finite_number(sum)) {
                lmmc_free(y);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
            y[i] = sum;
        }

        for (i = impl->size; i > 0; --i) {
            size_t row = i - 1;
            size_t p = 0;
            double sum = y[row];
            double diag = 0.0;

            for (p = impl->row_ptr[row]; p < impl->row_ptr[row + 1]; ++p) {
                size_t j = impl->col_idx[p];
                if (j <= row) {
                    continue;
                }
                sum -= impl->lu_values[p] * out->data[j];
            }

            diag = impl->lu_values[impl->diag_pos[row]];
            if (!lmmc_is_finite_number(sum) || !lmmc_is_finite_number(diag) || fabs(diag) <= 1e-15) {
                lmmc_free(y);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }

            out->data[row] = sum / diag;
            if (!lmmc_is_finite_number(out->data[row])) {
                lmmc_free(y);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
        }

        lmmc_free(y);
        return LMMC_STATUS_OK;
    }

    return LMMC_STATUS_NOT_IMPLEMENTED;
}

void lmmc_precond_destroy(lmmc_precond_t* precond) {
    if (precond == NULL) {
        return;
    }

    if (precond->owns_data && precond->impl != NULL) {
        if (precond->type == LMMC_PRECOND_JACOBI) {
            lmmc_free(precond->impl);
        } else if (precond->type == LMMC_PRECOND_ILU0 || precond->type == LMMC_PRECOND_ILUT) {
            lmmc_ilu_impl_destroy((lmmc_precond_ilu_impl_t*)precond->impl);
        } else {
            lmmc_free(precond->impl);
        }
    }

    precond->type = LMMC_PRECOND_NONE;
    precond->size = 0;
    precond->impl = NULL;
    precond->owns_data = 0;
}
