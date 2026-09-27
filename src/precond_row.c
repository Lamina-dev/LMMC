#include "precond_internal.h"

lmmc_status_t lmmc_ilu_row_bytes(size_t n, size_t fill, size_t* bytes) {
    size_t values, indices, marks;
    if (!lmmc_safe_mul_size(n, sizeof(lmmc_real_t), &values) ||
        !lmmc_safe_mul_size(fill, 2, &indices)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_safe_add_size(indices, n, &indices) ||
        !lmmc_safe_mul_size(indices, sizeof(size_t), &indices)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_safe_mul_size(n, 2, &marks)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    const size_t alignment = _Alignof(size_t);
    size_t padding = (alignment - values % alignment) % alignment;
    if (!lmmc_safe_add_size(values, padding, &values) ||
        !lmmc_safe_add_size(values, indices, bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_safe_add_size(*bytes, marks, bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_ilu_row_create(size_t n, size_t fill, size_t bytes,
    lmmc_ilu_row_t* row) {
    row->values = lmmc_memory_alloc(bytes);
    if (row->values == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(row->values, 0, bytes);
    size_t values = n * sizeof(lmmc_real_t);
    const size_t alignment = _Alignof(size_t);
    values += (alignment - values % alignment) % alignment;
    row->active = (size_t*)((unsigned char*)row->values + values);
    row->lower = row->active + n;
    row->upper = row->lower + fill;
    row->present = (unsigned char*)(row->upper + fill);
    row->processed = row->present + n;
    row->count = 0;
    return LMMC_STATUS_OK;
}

void lmmc_ilu_row_reset(lmmc_ilu_row_t* row) {
    for (size_t p = 0; p < row->count; ++p) {
        size_t col = row->active[p];
        row->present[col] = 0;
        row->processed[col] = 0;
        row->values[col] = 0.0;
    }
    row->count = 0;
}

static void lmmc_ilu_sort_columns(size_t* values, size_t count) {
    for (size_t i = 1; i < count; ++i) {
        size_t key = values[i];
        size_t j = i;
        while (j > 0 && values[j - 1] > key) {
            values[j] = values[j - 1];
            --j;
        }
        values[j] = key;
    }
}

static size_t lmmc_ilu_smallest_entry(const lmmc_real_t* values,
    const size_t* cols, size_t count) {
    size_t minimum = 0;
    lmmc_real_t minimum_abs = fabs(values[cols[0]]);
    for (size_t i = 1; i < count; ++i) {
        lmmc_real_t magnitude = fabs(values[cols[i]]);
        if (magnitude < minimum_abs) {
            minimum_abs = magnitude;
            minimum = i;
        }
    }
    return minimum;
}

void lmmc_ilu_select(const lmmc_ilu_row_t* row, size_t i, int lower,
    lmmc_real_t drop_tol, size_t max_keep, size_t* cols, size_t* count) {
    size_t selected = 0;
    for (size_t p = 0; p < row->count; ++p) {
        size_t col = row->active[p];
        if (!row->present[col]) {
            continue;
        }
        if (lower ? col >= i : col <= i) {
            continue;
        }
        lmmc_real_t magnitude = fabs(row->values[col]);
        if (magnitude <= drop_tol) {
            continue;
        }
        if (selected < max_keep) {
            cols[selected++] = col;
        }
        else {
            size_t minimum = lmmc_ilu_smallest_entry(row->values, cols, selected);
            if (magnitude > fabs(row->values[cols[minimum]])) {
                cols[minimum] = col;
            }
        }
    }
    lmmc_ilu_sort_columns(cols, selected);
    *count = selected;
}
