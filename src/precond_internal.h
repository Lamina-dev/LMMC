#ifndef LMMC_PRECOND_INTERNAL_H
#define LMMC_PRECOND_INTERNAL_H

#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/precond.h"
#include "sparse_internal.h"

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

/** @brief 行工作区仅分配一次，以下指针均借用该内存。 */
typedef struct {
    lmmc_real_t* values;
    size_t* active;
    size_t* lower;
    size_t* upper;
    unsigned char* present;
    unsigned char* processed;
    size_t count;
} lmmc_ilu_row_t;

int lmmc_find_col_pos(const size_t* cols, size_t start, size_t end,
    size_t col, size_t* pos);
void lmmc_ilu_impl_destroy(lmmc_precond_ilu_impl_t* impl);
lmmc_status_t lmmc_ilu_impl_create(size_t size, lmmc_precond_ilu_impl_t** out);
lmmc_status_t lmmc_ilu_impl_reserve(lmmc_precond_ilu_impl_t* impl, size_t required);
lmmc_status_t lmmc_ilu_row_bytes(size_t n, size_t fill, size_t* bytes);
lmmc_status_t lmmc_ilu_row_create(size_t n, size_t fill, size_t bytes,
    lmmc_ilu_row_t* row);
void lmmc_ilu_row_reset(lmmc_ilu_row_t* row);
void lmmc_ilu_select(const lmmc_ilu_row_t* row, size_t i, int lower,
    lmmc_real_t drop_tol, size_t max_keep, size_t* cols, size_t* count);
lmmc_status_t lmmc_ilu_apply(const lmmc_precond_t* precond,
    const lmmc_vec_t* rhs, lmmc_vec_t* out);

#endif
