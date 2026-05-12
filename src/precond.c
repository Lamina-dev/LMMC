#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "lmmc/config.h"
#include "lmmc/precond.h"

typedef struct {
    size_t size;
    size_t nnz;
    size_t capacity;
    size_t* row_ptr;
    size_t* col_idx;
    lmmc_real_t* lu_values;
    size_t* diag_pos;
    lmmc_real_t* y_arr;
} lmmc_precond_ilu_impl_t;

static int lmmc_is_finite_number(const lmmc_real_t* v) {
    return LMMC_REAL_IS_FINITE(v) ? 1 : 0;
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

    if (impl->lu_values != NULL) {
        size_t k;
        for (k = 0; k < impl->capacity; ++k) {
            LMMC_REAL_CLEAR(&impl->lu_values[k]);
        }
        lmmc_free(impl->lu_values);
    }
    if (impl->y_arr != NULL) {
        size_t k;
        for (k = 0; k < impl->size; ++k) {
            LMMC_REAL_CLEAR(&impl->y_arr[k]);
        }
        lmmc_free(impl->y_arr);
    }
    lmmc_free(impl->diag_pos);
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
    const lmmc_real_t* workspace,
    size_t row,
    int lower_side,
    lmmc_real_t drop_tol,
    size_t max_keep,
    size_t* out_cols,
    size_t* out_count
) {
    size_t i = 0;
    size_t selected = 0;

    lmmc_real_t v; LMMC_REAL_INIT(&v);
    lmmc_real_t av; LMMC_REAL_INIT(&av);
    if (active_cols == NULL || present == NULL || workspace == NULL || out_cols == NULL || out_count == NULL ||
        max_keep == 0) {
        if (out_count != NULL) {
            *out_count = 0;
        }
        LMMC_REAL_CLEAR(&av);
        LMMC_REAL_CLEAR(&v);
        return;
    }
    LMMC_REAL_SET_D(&v, 0.0);
    LMMC_REAL_SET_D(&av, 0.0);

    for (i = 0; i < active_count; ++i) {
        size_t col = active_cols[i];

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

        LMMC_REAL_SET(&v, &workspace[col]);
        LMMC_REAL_ABS(&av, &v);

        if (LMMC_REAL_CMP(&av, &drop_tol) <= 0) {
            continue;
        }

        if (selected < max_keep) {
            out_cols[selected++] = col;
        } else {
            size_t min_idx = 0;
            lmmc_real_t min_abs; LMMC_REAL_INIT(&min_abs);
            LMMC_REAL_ABS(&min_abs, &workspace[out_cols[0]]);

            size_t s = 0;
            for (s = 1; s < selected; ++s) {
                lmmc_real_t cur_abs; LMMC_REAL_INIT(&cur_abs);
                LMMC_REAL_ABS(&cur_abs, &workspace[out_cols[s]]);
                
                if (LMMC_REAL_CMP(&cur_abs, &min_abs) < 0) {
                    LMMC_REAL_SET(&min_abs, &cur_abs);
                    min_idx = s;
                }
                LMMC_REAL_CLEAR(&cur_abs);
            }

            if (LMMC_REAL_CMP(&av, &min_abs) > 0) {
                out_cols[min_idx] = col;
            }
            LMMC_REAL_CLEAR(&min_abs);
        }
    }

    lmmc_sort_size_t_asc(out_cols, selected);
    *out_count = selected;

    LMMC_REAL_CLEAR(&av);
    LMMC_REAL_CLEAR(&v);
}

static lmmc_status_t lmmc_ilu_impl_reserve(lmmc_precond_ilu_impl_t* impl, size_t required_capacity) {
    size_t new_capacity = 0;
    size_t idx_bytes = 0;
    size_t val_bytes = 0;
    size_t* new_col_idx = NULL;
    lmmc_real_t* new_lu_values = NULL;

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
        lmmc_mul_overflow_size(new_capacity, sizeof(lmmc_real_t), &val_bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    new_col_idx = (size_t*)lmmc_alloc(idx_bytes);
    new_lu_values = (lmmc_real_t*)lmmc_alloc(val_bytes);
    if (new_col_idx == NULL || new_lu_values == NULL) {
        lmmc_free(new_lu_values);
        lmmc_free(new_col_idx);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    size_t k;
    for (k = 0; k < new_capacity; ++k) {
        LMMC_REAL_INIT(&new_lu_values[k]);
        LMMC_REAL_SET_D(&new_lu_values[k], 0.0);
    }

    if (impl->nnz > 0) {
        memcpy(new_col_idx, impl->col_idx, impl->nnz * sizeof(size_t));
        for (k = 0; k < impl->nnz; ++k) {
            LMMC_REAL_SET(&new_lu_values[k], &impl->lu_values[k]);
        }
    }

    if (impl->lu_values != NULL) {
        for (k = 0; k < impl->capacity; ++k) {
            LMMC_REAL_CLEAR(&impl->lu_values[k]);
        }
        lmmc_free(impl->lu_values);
    }

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
    lmmc_real_t* diag_inv = NULL;
    lmmc_status_t st = lmmc_sparse_validate_csr_basic(a);

    if (out_precond == NULL || st != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    if (a->rows > ((size_t)-1) / sizeof(lmmc_real_t)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    bytes = a->rows * sizeof(lmmc_real_t);

    diag_inv = (lmmc_real_t*)lmmc_alloc(bytes);
    if (diag_inv == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    for (i = 0; i < a->rows; ++i) {
        LMMC_REAL_INIT(&diag_inv[i]);
        LMMC_REAL_SET_D(&diag_inv[i], 0.0);
    }

    lmmc_real_t diag; LMMC_REAL_INIT(&diag); LMMC_REAL_SET_D(&diag, 0.0);
    lmmc_real_t eps_15; LMMC_REAL_INIT(&eps_15); LMMC_REAL_SET_D(&eps_15, 1e-15);
    lmmc_real_t one; LMMC_REAL_INIT(&one); LMMC_REAL_SET_D(&one, 1.0);
    lmmc_real_t abs_diag; LMMC_REAL_INIT(&abs_diag); LMMC_REAL_SET_D(&abs_diag, 0.0);

    for (i = 0; i < a->rows; ++i) {
        size_t p = 0;
        int found = 0;
        LMMC_REAL_SET_D(&diag, 0.0);

        for (p = a->row_ptr[i]; p < a->row_ptr[i + 1]; ++p) {
            if (a->col_idx[p] == i) {
                LMMC_REAL_SET(&diag, &a->values[p]);
                found = 1;
                break;
            }
        }

        LMMC_REAL_ABS(&abs_diag, &diag);

        if (!found || !lmmc_is_finite_number(&diag) || LMMC_REAL_CMP(&abs_diag, &eps_15) <= 0) {
            LMMC_REAL_CLEAR(&abs_diag);
            LMMC_REAL_CLEAR(&one);
            LMMC_REAL_CLEAR(&eps_15);
            LMMC_REAL_CLEAR(&diag);
            for (size_t k = 0; k < a->rows; ++k) LMMC_REAL_CLEAR(&diag_inv[k]);
            lmmc_free(diag_inv);
            return LMMC_STATUS_SINGULAR_MATRIX;
        }

        LMMC_REAL_DIV(&diag_inv[i], &one, &diag);

        if (!lmmc_is_finite_number(&diag_inv[i])) {
            LMMC_REAL_CLEAR(&abs_diag);
            LMMC_REAL_CLEAR(&one);
            LMMC_REAL_CLEAR(&eps_15);
            LMMC_REAL_CLEAR(&diag);
            for (size_t k = 0; k < a->rows; ++k) LMMC_REAL_CLEAR(&diag_inv[k]);
            lmmc_free(diag_inv);
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }

    LMMC_REAL_CLEAR(&abs_diag);
    LMMC_REAL_CLEAR(&one);
    LMMC_REAL_CLEAR(&eps_15);
    LMMC_REAL_CLEAR(&diag);

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
    size_t y_bytes = 0;
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
        lmmc_mul_overflow_size(a->nnz, sizeof(lmmc_real_t), &val_bytes) ||
        lmmc_mul_overflow_size(a->rows, sizeof(size_t), &diag_bytes) ||
        lmmc_mul_overflow_size(a->rows, sizeof(lmmc_real_t), &y_bytes)) {
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
    impl->y_arr = (lmmc_real_t*)lmmc_alloc(y_bytes);
    if (impl->row_ptr == NULL || impl->diag_pos == NULL || impl->y_arr == NULL) {
        lmmc_ilu_impl_destroy(impl);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    for (i = 0; i < a->rows; ++i) {
        LMMC_REAL_INIT(&impl->y_arr[i]);
        LMMC_REAL_SET_D(&impl->y_arr[i], 0.0);
    }

    if (a->nnz > 0) {
        impl->col_idx = (size_t*)lmmc_alloc(idx_bytes);
        impl->lu_values = (lmmc_real_t*)lmmc_alloc(val_bytes);
        if (impl->col_idx == NULL || impl->lu_values == NULL) {
            lmmc_ilu_impl_destroy(impl);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
        memcpy(impl->col_idx, a->col_idx, idx_bytes);
        for(size_t k=0; k < a->nnz; k++) {
            LMMC_REAL_INIT(&impl->lu_values[k]);
            LMMC_REAL_SET(&impl->lu_values[k], &a->values[k]);
        }
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

    lmmc_real_t sum; LMMC_REAL_INIT(&sum); LMMC_REAL_SET_D(&sum, 0.0);
    lmmc_real_t diag; LMMC_REAL_INIT(&diag); LMMC_REAL_SET_D(&diag, 0.0);
    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul); LMMC_REAL_SET_D(&tmp_mul, 0.0);
    lmmc_real_t tmp_sub; LMMC_REAL_INIT(&tmp_sub); LMMC_REAL_SET_D(&tmp_sub, 0.0);
    lmmc_real_t eps_15; LMMC_REAL_INIT(&eps_15); LMMC_REAL_SET_D(&eps_15, 1e-15);
    lmmc_real_t abs_diag; LMMC_REAL_INIT(&abs_diag); LMMC_REAL_SET_D(&abs_diag, 0.0);

    for (i = 0; i < a->rows; ++i) {
        size_t row_start = impl->row_ptr[i];
        size_t row_end = impl->row_ptr[i + 1];
        size_t p = 0;

        for (p = row_start; p < row_end; ++p) {
            size_t j = impl->col_idx[p];
            LMMC_REAL_SET(&sum, &impl->lu_values[p]);
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
                    LMMC_REAL_MUL(&tmp_mul, &impl->lu_values[q], &impl->lu_values[pos_kj]);
                    LMMC_REAL_SUB(&tmp_sub, &sum, &tmp_mul);
                    LMMC_REAL_SET(&sum, &tmp_sub);
                }
            }

            {
                LMMC_REAL_SET(&diag, &impl->lu_values[impl->diag_pos[j]]);
                LMMC_REAL_ABS(&abs_diag, &diag);
                if (!lmmc_is_finite_number(&sum) || !lmmc_is_finite_number(&diag) || LMMC_REAL_CMP(&abs_diag, &eps_15) <= 0) {
                    LMMC_REAL_CLEAR(&abs_diag); LMMC_REAL_CLEAR(&eps_15);
                    LMMC_REAL_CLEAR(&tmp_sub); LMMC_REAL_CLEAR(&tmp_mul);
                    LMMC_REAL_CLEAR(&diag); LMMC_REAL_CLEAR(&sum);
                    lmmc_ilu_impl_destroy(impl);
                    return LMMC_STATUS_SINGULAR_MATRIX;
                }
                LMMC_REAL_DIV(&impl->lu_values[p], &sum, &diag);
                if (!lmmc_is_finite_number(&impl->lu_values[p])) {
                    LMMC_REAL_CLEAR(&abs_diag); LMMC_REAL_CLEAR(&eps_15);
                    LMMC_REAL_CLEAR(&tmp_sub); LMMC_REAL_CLEAR(&tmp_mul);
                    LMMC_REAL_CLEAR(&diag); LMMC_REAL_CLEAR(&sum);
                    lmmc_ilu_impl_destroy(impl);
                    return LMMC_STATUS_NUMERICAL_FAILURE;
                }
            }
        }

        for (p = row_start; p < row_end; ++p) {
            size_t j = impl->col_idx[p];
            LMMC_REAL_SET(&sum, &impl->lu_values[p]);
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
                    LMMC_REAL_MUL(&tmp_mul, &impl->lu_values[q], &impl->lu_values[pos_kj]);
                    LMMC_REAL_SUB(&tmp_sub, &sum, &tmp_mul);
                    LMMC_REAL_SET(&sum, &tmp_sub);
                }
            }

            if (!lmmc_is_finite_number(&sum)) {
                LMMC_REAL_CLEAR(&abs_diag); LMMC_REAL_CLEAR(&eps_15);
                LMMC_REAL_CLEAR(&tmp_sub); LMMC_REAL_CLEAR(&tmp_mul);
                LMMC_REAL_CLEAR(&diag); LMMC_REAL_CLEAR(&sum);
                lmmc_ilu_impl_destroy(impl);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
            LMMC_REAL_SET(&impl->lu_values[p], &sum);
        }

        {
            LMMC_REAL_SET(&diag, &impl->lu_values[impl->diag_pos[i]]);
            LMMC_REAL_ABS(&abs_diag, &diag);
            if (!lmmc_is_finite_number(&diag) || LMMC_REAL_CMP(&abs_diag, &eps_15) <= 0) {
                LMMC_REAL_CLEAR(&abs_diag); LMMC_REAL_CLEAR(&eps_15);
                LMMC_REAL_CLEAR(&tmp_sub); LMMC_REAL_CLEAR(&tmp_mul);
                LMMC_REAL_CLEAR(&diag); LMMC_REAL_CLEAR(&sum);
                lmmc_ilu_impl_destroy(impl);
                return LMMC_STATUS_SINGULAR_MATRIX;
            }
        }
    }

    LMMC_REAL_CLEAR(&abs_diag);
    LMMC_REAL_CLEAR(&eps_15);
    LMMC_REAL_CLEAR(&tmp_sub);
    LMMC_REAL_CLEAR(&tmp_mul);
    LMMC_REAL_CLEAR(&diag);
    LMMC_REAL_CLEAR(&sum);

    out_precond->type = LMMC_PRECOND_ILU0;
    out_precond->size = a->rows;
    out_precond->impl = impl;
    out_precond->owns_data = 1;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_precond_create_ilut(
    const lmmc_sparse_mat_t* a,
    lmmc_real_t drop_tol,
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
    size_t y_bytes = 0;
    lmmc_real_t* workspace = NULL;
    unsigned char* present = NULL;
    unsigned char* processed = NULL;
    size_t* active_cols = NULL;
    size_t* lower_keep_cols = NULL;
    size_t* upper_keep_cols = NULL;
    size_t active_count = 0;
    lmmc_precond_ilu_impl_t* impl = NULL;
    lmmc_status_t st = lmmc_sparse_validate_csr_basic(a);
    
    lmmc_real_t zero; LMMC_REAL_INIT(&zero); LMMC_REAL_SET_D(&zero, 0.0);
    lmmc_real_t eps_15; LMMC_REAL_INIT(&eps_15); LMMC_REAL_SET_D(&eps_15, 1e-15);
    lmmc_real_t tmp_add; LMMC_REAL_INIT(&tmp_add); LMMC_REAL_SET_D(&tmp_add, 0.0);
    lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul); LMMC_REAL_SET_D(&tmp_mul, 0.0);
    lmmc_real_t tmp_sub; LMMC_REAL_INIT(&tmp_sub); LMMC_REAL_SET_D(&tmp_sub, 0.0);

    if (out_precond == NULL || st != LMMC_STATUS_OK || !lmmc_is_finite_number(&drop_tol) ||
        LMMC_REAL_CMP(&drop_tol, &zero) < 0 || max_fill_per_row == 0) {
        st = LMMC_STATUS_INVALID_ARGUMENT; goto fail_early;
    }
    if (a->rows != a->cols) {
        st = LMMC_STATUS_DIMENSION_MISMATCH; goto fail_early;
    }

    n = a->rows;
    if (lmmc_mul_overflow_size(n + 1, sizeof(size_t), &row_ptr_bytes) ||
        lmmc_mul_overflow_size(n, sizeof(size_t), &diag_bytes) ||
        lmmc_mul_overflow_size(n, sizeof(lmmc_real_t), &workspace_bytes) ||
        lmmc_mul_overflow_size(n, sizeof(unsigned char), &mark_bytes) ||
        lmmc_mul_overflow_size(n, sizeof(size_t), &active_bytes) ||
        lmmc_mul_overflow_size(max_fill_per_row, sizeof(size_t), &keep_bytes) ||
        lmmc_mul_overflow_size(n, sizeof(lmmc_real_t), &y_bytes)) {
        st = LMMC_STATUS_INVALID_ARGUMENT; goto fail_early;
    }

    impl = (lmmc_precond_ilu_impl_t*)lmmc_alloc(sizeof(lmmc_precond_ilu_impl_t));
    if (impl == NULL) {
        st = LMMC_STATUS_ALLOCATION_FAILED; goto fail_early;
    }
    memset(impl, 0, sizeof(lmmc_precond_ilu_impl_t));

    impl->size = n;
    impl->nnz = 0;
    impl->capacity = 0;

    impl->row_ptr = (size_t*)lmmc_alloc(row_ptr_bytes);
    impl->diag_pos = (size_t*)lmmc_alloc(diag_bytes);
    impl->y_arr = (lmmc_real_t*)lmmc_alloc(y_bytes);
    workspace = (lmmc_real_t*)lmmc_alloc(workspace_bytes);
    present = (unsigned char*)lmmc_alloc(mark_bytes);
    processed = (unsigned char*)lmmc_alloc(mark_bytes);
    active_cols = (size_t*)lmmc_alloc(active_bytes);
    lower_keep_cols = (size_t*)lmmc_alloc(keep_bytes);
    upper_keep_cols = (size_t*)lmmc_alloc(keep_bytes);

    if (impl->row_ptr == NULL || impl->diag_pos == NULL || impl->y_arr == NULL || workspace == NULL || present == NULL ||
        processed == NULL || active_cols == NULL || lower_keep_cols == NULL || upper_keep_cols == NULL) {
        st = LMMC_STATUS_ALLOCATION_FAILED;
        goto fail;
    }

    size_t kw;
    for(kw=0; kw<n; ++kw) {
        LMMC_REAL_INIT(&workspace[kw]);
        LMMC_REAL_SET_D(&workspace[kw], 0.0);
        LMMC_REAL_INIT(&impl->y_arr[kw]);
        LMMC_REAL_SET_D(&impl->y_arr[kw], 0.0);
    }
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
        lmmc_real_t diag_i; LMMC_REAL_INIT(&diag_i); LMMC_REAL_SET_D(&diag_i, 0.0);
        size_t p = 0;
        active_count = 0;

        for (p = a->row_ptr[i]; p < a->row_ptr[i + 1]; ++p) {
            size_t col = a->col_idx[p];
            lmmc_real_t v; LMMC_REAL_INIT(&v); LMMC_REAL_SET(&v, &a->values[p]);

            if (!lmmc_is_finite_number(&v)) {
                LMMC_REAL_CLEAR(&v); LMMC_REAL_CLEAR(&diag_i);
                st = LMMC_STATUS_NUMERICAL_FAILURE;
                goto fail;
            }

            if (!present[col]) {
                present[col] = 1;
                processed[col] = 0;
                active_cols[active_count++] = col;
                LMMC_REAL_SET(&workspace[col], &v);
            } else {
                LMMC_REAL_ADD(&tmp_add, &workspace[col], &v);
                LMMC_REAL_SET(&workspace[col], &tmp_add);
                if (!lmmc_is_finite_number(&workspace[col])) {
                    LMMC_REAL_CLEAR(&v); LMMC_REAL_CLEAR(&diag_i);
                    st = LMMC_STATUS_NUMERICAL_FAILURE;
                    goto fail;
                }
            }
            LMMC_REAL_CLEAR(&v);
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
                lmmc_real_t aik; LMMC_REAL_INIT(&aik); LMMC_REAL_SET(&aik, &workspace[k]);
                lmmc_real_t diag_k; LMMC_REAL_INIT(&diag_k); LMMC_REAL_SET_D(&diag_k, 0.0);
                lmmc_real_t lik; LMMC_REAL_INIT(&lik); LMMC_REAL_SET_D(&lik, 0.0);
                lmmc_real_t abs_aik; LMMC_REAL_INIT(&abs_aik);
                lmmc_real_t abs_diag_k; LMMC_REAL_INIT(&abs_diag_k);
                
                size_t u_start = 0;
                size_t u_end = 0;

                if (!lmmc_is_finite_number(&aik)) {
                    LMMC_REAL_CLEAR(&abs_diag_k); LMMC_REAL_CLEAR(&abs_aik);
                    LMMC_REAL_CLEAR(&lik); LMMC_REAL_CLEAR(&diag_k); LMMC_REAL_CLEAR(&aik);
                    LMMC_REAL_CLEAR(&diag_i);
                    st = LMMC_STATUS_NUMERICAL_FAILURE; goto fail;
                }

                LMMC_REAL_ABS(&abs_aik, &aik);
                if (LMMC_REAL_CMP(&abs_aik, &drop_tol) <= 0) {
                    present[k] = 0;
                    LMMC_REAL_SET_D(&workspace[k], 0.0);
                    LMMC_REAL_CLEAR(&abs_diag_k); LMMC_REAL_CLEAR(&abs_aik);
                    LMMC_REAL_CLEAR(&lik); LMMC_REAL_CLEAR(&diag_k); LMMC_REAL_CLEAR(&aik);
                    continue;
                }

                if (impl->diag_pos[k] == (size_t)-1 || impl->diag_pos[k] >= impl->nnz) {
                    LMMC_REAL_CLEAR(&abs_diag_k); LMMC_REAL_CLEAR(&abs_aik);
                    LMMC_REAL_CLEAR(&lik); LMMC_REAL_CLEAR(&diag_k); LMMC_REAL_CLEAR(&aik);
                    LMMC_REAL_CLEAR(&diag_i);
                    st = LMMC_STATUS_NUMERICAL_FAILURE; goto fail;
                }

                LMMC_REAL_SET(&diag_k, &impl->lu_values[impl->diag_pos[k]]);
                LMMC_REAL_ABS(&abs_diag_k, &diag_k);
                if (!lmmc_is_finite_number(&diag_k) || LMMC_REAL_CMP(&abs_diag_k, &eps_15) <= 0) {
                    LMMC_REAL_CLEAR(&abs_diag_k); LMMC_REAL_CLEAR(&abs_aik);
                    LMMC_REAL_CLEAR(&lik); LMMC_REAL_CLEAR(&diag_k); LMMC_REAL_CLEAR(&aik);
                    LMMC_REAL_CLEAR(&diag_i);
                    st = LMMC_STATUS_SINGULAR_MATRIX; goto fail;
                }

                LMMC_REAL_DIV(&lik, &aik, &diag_k);
                if (!lmmc_is_finite_number(&lik)) {
                    LMMC_REAL_CLEAR(&abs_diag_k); LMMC_REAL_CLEAR(&abs_aik);
                    LMMC_REAL_CLEAR(&lik); LMMC_REAL_CLEAR(&diag_k); LMMC_REAL_CLEAR(&aik);
                    LMMC_REAL_CLEAR(&diag_i);
                    st = LMMC_STATUS_NUMERICAL_FAILURE; goto fail;
                }
                LMMC_REAL_SET(&workspace[k], &lik);

                u_start = impl->diag_pos[k] + 1;
                u_end = impl->row_ptr[k + 1];

                for (p = u_start; p < u_end; ++p) {
                    size_t j = impl->col_idx[p];
                    lmmc_real_t delta; LMMC_REAL_INIT(&delta);
                    LMMC_REAL_MUL(&delta, &lik, &impl->lu_values[p]);

                    if (j <= k) { LMMC_REAL_CLEAR(&delta); continue; }
                    if (!lmmc_is_finite_number(&delta)) {
                        LMMC_REAL_CLEAR(&delta);
                        LMMC_REAL_CLEAR(&abs_diag_k); LMMC_REAL_CLEAR(&abs_aik);
                        LMMC_REAL_CLEAR(&lik); LMMC_REAL_CLEAR(&diag_k); LMMC_REAL_CLEAR(&aik);
                        LMMC_REAL_CLEAR(&diag_i);
                        st = LMMC_STATUS_NUMERICAL_FAILURE; goto fail;
                    }

                    if (!present[j]) {
                        present[j] = 1;
                        processed[j] = 0;
                        active_cols[active_count++] = j;
                        LMMC_REAL_NEG(&workspace[j], &delta);
                    } else {
                        LMMC_REAL_SUB(&tmp_sub, &workspace[j], &delta);
                        LMMC_REAL_SET(&workspace[j], &tmp_sub);
                    }

                    if (!lmmc_is_finite_number(&workspace[j])) {
                        LMMC_REAL_CLEAR(&delta);
                        LMMC_REAL_CLEAR(&abs_diag_k); LMMC_REAL_CLEAR(&abs_aik);
                        LMMC_REAL_CLEAR(&lik); LMMC_REAL_CLEAR(&diag_k); LMMC_REAL_CLEAR(&aik);
                        LMMC_REAL_CLEAR(&diag_i);
                        st = LMMC_STATUS_NUMERICAL_FAILURE; goto fail;
                    }
                    LMMC_REAL_CLEAR(&delta);
                }

                LMMC_REAL_CLEAR(&abs_diag_k); LMMC_REAL_CLEAR(&abs_aik);
                LMMC_REAL_CLEAR(&lik); LMMC_REAL_CLEAR(&diag_k); LMMC_REAL_CLEAR(&aik);
            }
        }

        if (!present[i]) {
            present[i] = 1;
            processed[i] = 0;
            active_cols[active_count++] = i;
            LMMC_REAL_SET_D(&workspace[i], 0.0);
        }

        LMMC_REAL_SET(&diag_i, &workspace[i]);
        lmmc_real_t abs_diag_i; LMMC_REAL_INIT(&abs_diag_i);
        LMMC_REAL_ABS(&abs_diag_i, &diag_i);
        
        if (!lmmc_is_finite_number(&diag_i) || LMMC_REAL_CMP(&abs_diag_i, &eps_15) <= 0) {
            LMMC_REAL_CLEAR(&abs_diag_i); LMMC_REAL_CLEAR(&diag_i);
            st = LMMC_STATUS_SINGULAR_MATRIX; goto fail;
        }
        LMMC_REAL_CLEAR(&abs_diag_i);

        lmmc_select_top_abs(
            active_cols, active_count, present, workspace, i, 1,
            drop_tol, max_fill_per_row, lower_keep_cols, &lower_keep_count
        );

        lmmc_select_top_abs(
            active_cols, active_count, present, workspace, i, 0,
            drop_tol, max_fill_per_row, upper_keep_cols, &upper_keep_count
        );

        {
            size_t row_need = lower_keep_count + 1 + upper_keep_count;
            size_t required = 0;
            size_t idx = 0;

            if (impl->nnz > ((size_t)-1) - row_need) {
                LMMC_REAL_CLEAR(&diag_i);
                st = LMMC_STATUS_INVALID_ARGUMENT; goto fail;
            }
            required = impl->nnz + row_need;

            st = lmmc_ilu_impl_reserve(impl, required);
            if (st != LMMC_STATUS_OK) {
                LMMC_REAL_CLEAR(&diag_i);
                goto fail;
            }

            for (idx = 0; idx < lower_keep_count; ++idx) {
                size_t col = lower_keep_cols[idx];
                lmmc_real_t v; LMMC_REAL_INIT(&v); LMMC_REAL_SET(&v, &workspace[col]);
                if (!lmmc_is_finite_number(&v)) {
                    LMMC_REAL_CLEAR(&v); LMMC_REAL_CLEAR(&diag_i);
                    st = LMMC_STATUS_NUMERICAL_FAILURE; goto fail;
                }
                impl->col_idx[impl->nnz] = col;
                LMMC_REAL_SET(&impl->lu_values[impl->nnz], &v);
                impl->nnz += 1;
                LMMC_REAL_CLEAR(&v);
            }

            impl->diag_pos[i] = impl->nnz;
            impl->col_idx[impl->nnz] = i;
            LMMC_REAL_SET(&impl->lu_values[impl->nnz], &diag_i);
            impl->nnz += 1;

            for (idx = 0; idx < upper_keep_count; ++idx) {
                size_t col = upper_keep_cols[idx];
                lmmc_real_t v; LMMC_REAL_INIT(&v); LMMC_REAL_SET(&v, &workspace[col]);
                if (!lmmc_is_finite_number(&v)) {
                    LMMC_REAL_CLEAR(&v); LMMC_REAL_CLEAR(&diag_i);
                    st = LMMC_STATUS_NUMERICAL_FAILURE; goto fail;
                }
                impl->col_idx[impl->nnz] = col;
                LMMC_REAL_SET(&impl->lu_values[impl->nnz], &v);
                impl->nnz += 1;
                LMMC_REAL_CLEAR(&v);
            }

            impl->row_ptr[i + 1] = impl->nnz;
        }

        LMMC_REAL_CLEAR(&diag_i);

        for (p = 0; p < active_count; ++p) {
            size_t col = active_cols[p];
            present[col] = 0;
            processed[col] = 0;
            LMMC_REAL_SET_D(&workspace[col], 0.0);
        }
    }

    out_precond->type = LMMC_PRECOND_ILUT;
    out_precond->size = n;
    out_precond->impl = impl;
    out_precond->owns_data = 1;

    LMMC_REAL_CLEAR(&tmp_sub); LMMC_REAL_CLEAR(&tmp_mul); LMMC_REAL_CLEAR(&tmp_add);
    LMMC_REAL_CLEAR(&eps_15); LMMC_REAL_CLEAR(&zero);

    lmmc_free(upper_keep_cols);
    lmmc_free(lower_keep_cols);
    lmmc_free(active_cols);
    lmmc_free(processed);
    lmmc_free(present);
    if (workspace != NULL) {
        for(size_t kw=0; kw<n; ++kw) {
            LMMC_REAL_CLEAR(&workspace[kw]);
        }
        lmmc_free(workspace);
    }
    return LMMC_STATUS_OK;

fail:
    if (workspace != NULL) {
        for(size_t kw=0; kw<n; ++kw) {
            LMMC_REAL_CLEAR(&workspace[kw]);
        }
        lmmc_free(workspace);
    }
    lmmc_free(upper_keep_cols);
    lmmc_free(lower_keep_cols);
    lmmc_free(active_cols);
    lmmc_free(processed);
    lmmc_free(present);
    lmmc_ilu_impl_destroy(impl);
fail_early:
    LMMC_REAL_CLEAR(&tmp_sub); LMMC_REAL_CLEAR(&tmp_mul); LMMC_REAL_CLEAR(&tmp_add);
    LMMC_REAL_CLEAR(&eps_15); LMMC_REAL_CLEAR(&zero);
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
        for(i = 0; i < rhs->size; ++i) {
            LMMC_REAL_SET(&out->data[i], &rhs->data[i]);
        }
        return LMMC_STATUS_OK;
    }

    if (precond->type == LMMC_PRECOND_JACOBI) {
        const lmmc_real_t* diag_inv = NULL;
        if (precond->impl == NULL) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }

        diag_inv = (const lmmc_real_t*)precond->impl;
        for (i = 0; i < rhs->size; ++i) {
            lmmc_real_t v; LMMC_REAL_INIT(&v);
            LMMC_REAL_MUL(&v, &rhs->data[i], &diag_inv[i]);
            if (!lmmc_is_finite_number(&rhs->data[i]) || !lmmc_is_finite_number(&v)) {
                LMMC_REAL_CLEAR(&v);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
            LMMC_REAL_SET(&out->data[i], &v);
            LMMC_REAL_CLEAR(&v);
        }
        return LMMC_STATUS_OK;
    }

    if (precond->type == LMMC_PRECOND_ILU0 || precond->type == LMMC_PRECOND_ILUT) {
        const lmmc_precond_ilu_impl_t* impl = NULL;
        lmmc_real_t* y_arr = NULL;

        if (precond->impl == NULL) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }

        impl = (const lmmc_precond_ilu_impl_t*)precond->impl;
        if (impl->size != precond->size || impl->diag_pos == NULL || impl->row_ptr == NULL || impl->y_arr == NULL) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }

        y_arr = impl->y_arr;
        for (i = 0; i < impl->size; ++i) {
            LMMC_REAL_SET_D(&y_arr[i], 0.0);
        }

        lmmc_real_t sum; LMMC_REAL_INIT(&sum); LMMC_REAL_SET_D(&sum, 0.0);
        lmmc_real_t tmp_mul; LMMC_REAL_INIT(&tmp_mul); LMMC_REAL_SET_D(&tmp_mul, 0.0);
        lmmc_real_t tmp_sub; LMMC_REAL_INIT(&tmp_sub); LMMC_REAL_SET_D(&tmp_sub, 0.0);
        lmmc_real_t diag; LMMC_REAL_INIT(&diag); LMMC_REAL_SET_D(&diag, 0.0);
        lmmc_real_t abs_diag; LMMC_REAL_INIT(&abs_diag); LMMC_REAL_SET_D(&abs_diag, 0.0);
        lmmc_real_t eps_15; LMMC_REAL_INIT(&eps_15); LMMC_REAL_SET_D(&eps_15, 1e-15);

        for (i = 0; i < impl->size; ++i) {
            size_t p = 0;
            LMMC_REAL_SET(&sum, &rhs->data[i]);

            for (p = impl->row_ptr[i]; p < impl->row_ptr[i + 1]; ++p) {
                size_t j = impl->col_idx[p];
                if (j >= i) {
                    continue;
                }
                LMMC_REAL_MUL(&tmp_mul, &impl->lu_values[p], &y_arr[j]);
                LMMC_REAL_SUB(&tmp_sub, &sum, &tmp_mul);
                LMMC_REAL_SET(&sum, &tmp_sub);
            }

            if (!lmmc_is_finite_number(&sum)) {
                LMMC_REAL_CLEAR(&eps_15); LMMC_REAL_CLEAR(&abs_diag);
                LMMC_REAL_CLEAR(&diag); LMMC_REAL_CLEAR(&tmp_sub);
                LMMC_REAL_CLEAR(&tmp_mul); LMMC_REAL_CLEAR(&sum);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
            LMMC_REAL_SET(&y_arr[i], &sum);
        }

        for (i = impl->size; i > 0; --i) {
            size_t row = i - 1;
            size_t p = 0;
            LMMC_REAL_SET(&sum, &y_arr[row]);
            LMMC_REAL_SET_D(&diag, 0.0);

            for (p = impl->row_ptr[row]; p < impl->row_ptr[row + 1]; ++p) {
                size_t j = impl->col_idx[p];
                if (j <= row) {
                    continue;
                }
                LMMC_REAL_MUL(&tmp_mul, &impl->lu_values[p], &out->data[j]);
                LMMC_REAL_SUB(&tmp_sub, &sum, &tmp_mul);
                LMMC_REAL_SET(&sum, &tmp_sub);
            }

            LMMC_REAL_SET(&diag, &impl->lu_values[impl->diag_pos[row]]);
            LMMC_REAL_ABS(&abs_diag, &diag);
            if (!lmmc_is_finite_number(&sum) || !lmmc_is_finite_number(&diag) || LMMC_REAL_CMP(&abs_diag, &eps_15) <= 0) {
                LMMC_REAL_CLEAR(&eps_15); LMMC_REAL_CLEAR(&abs_diag);
                LMMC_REAL_CLEAR(&diag); LMMC_REAL_CLEAR(&tmp_sub);
                LMMC_REAL_CLEAR(&tmp_mul); LMMC_REAL_CLEAR(&sum);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }

            LMMC_REAL_DIV(&out->data[row], &sum, &diag);
            if (!lmmc_is_finite_number(&out->data[row])) {
                LMMC_REAL_CLEAR(&eps_15); LMMC_REAL_CLEAR(&abs_diag);
                LMMC_REAL_CLEAR(&diag); LMMC_REAL_CLEAR(&tmp_sub);
                LMMC_REAL_CLEAR(&tmp_mul); LMMC_REAL_CLEAR(&sum);
                return LMMC_STATUS_NUMERICAL_FAILURE;
            }
        }

        LMMC_REAL_CLEAR(&eps_15); LMMC_REAL_CLEAR(&abs_diag);
        LMMC_REAL_CLEAR(&diag); LMMC_REAL_CLEAR(&tmp_sub);
        LMMC_REAL_CLEAR(&tmp_mul); LMMC_REAL_CLEAR(&sum);
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
            lmmc_real_t* arr = (lmmc_real_t*)precond->impl;
            size_t k;
            for (k = 0; k < precond->size; ++k) {
                LMMC_REAL_CLEAR(&arr[k]);
            }
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
