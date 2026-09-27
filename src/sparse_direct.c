#include <string.h>
#include <math.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/sparse.h"
#include "sparse_direct_internal.h"
#include "sparse_internal.h"

lmmc_status_t lmmc_sparse_direct_csc(const lmmc_sparse_mat_t* a,
                                lmmc_sparse_mat_t* csc_out,
                                int* needs_free)
{
    if (csc_out == NULL || needs_free == NULL ||
        lmmc_sparse_validate(a) != LMMC_STATUS_OK) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    *needs_free = 0;
    if (a->format == LMMC_SPARSE_CSC) {
        *csc_out = *a;
        return LMMC_STATUS_OK;
    }
    *needs_free = 1;
    return lmmc_sparse_to_csc(a, csc_out);
}

lmmc_status_t lmmc_sparse_factor_capacity(size_t** idx, lmmc_real_t** vals,
                                            size_t* capacity, size_t needed)
{
    size_t new_cap;
    size_t idx_bytes;
    size_t value_bytes;

    if (needed < *capacity) {
        return LMMC_STATUS_OK;
    }
    if (!lmmc_safe_mul_size(*capacity, 2, &new_cap) || new_cap <= needed) {
        if (!lmmc_safe_add_size(needed, 1, &new_cap)) {
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
    }
    if (!lmmc_safe_mul_size(new_cap, sizeof(size_t), &idx_bytes) ||
        !lmmc_safe_mul_size(new_cap, sizeof(lmmc_real_t), &value_bytes)) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    size_t* new_idx = (size_t*)lmmc_memory_realloc(*idx, idx_bytes);
    if (new_idx == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    *idx = new_idx;
    lmmc_real_t* new_vals = (lmmc_real_t*)lmmc_memory_realloc(*vals, value_bytes);
    if (new_vals == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    *vals = new_vals;
    *capacity = new_cap;
    return LMMC_STATUS_OK;
}
