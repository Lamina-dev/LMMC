#ifndef LMMC_TENSOR_H
#define LMMC_TENSOR_H

#include <stddef.h>
#include "lmmc/dense.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t dim0;
    size_t dim1;
    size_t dim2;
    size_t stride0;
    size_t stride1;
    size_t stride2;
    lmmc_real_t* data;
    int owns_data;
} lmmc_tensor_t;

lmmc_status_t lmmc_tensor3_create(size_t dim0, size_t dim1, size_t dim2, lmmc_tensor_t* out_tensor);
lmmc_status_t lmmc_tensor3_wrap(
    size_t dim0,
    size_t dim1,
    size_t dim2,
    size_t stride0,
    size_t stride1,
    size_t stride2,
    lmmc_real_t* data,
    lmmc_tensor_t* out_tensor
);
void lmmc_tensor_destroy(lmmc_tensor_t* tensor);

lmmc_status_t lmmc_tensor_fill(lmmc_tensor_t* tensor, lmmc_real_t value);
lmmc_status_t lmmc_tensor_set(lmmc_tensor_t* tensor, size_t i, size_t j, size_t k, lmmc_real_t value);
lmmc_status_t lmmc_tensor_get(const lmmc_tensor_t* tensor, size_t i, size_t j, size_t k, lmmc_real_t* out_value);
lmmc_status_t lmmc_tensor_norm_fro(const lmmc_tensor_t* tensor, lmmc_real_t* out_norm);

lmmc_status_t lmmc_tensor_add(const lmmc_tensor_t* a, const lmmc_tensor_t* b, lmmc_tensor_t* out_tensor);
lmmc_status_t lmmc_tensor_sub(const lmmc_tensor_t* a, const lmmc_tensor_t* b, lmmc_tensor_t* out_tensor);
lmmc_status_t lmmc_tensor_mul(const lmmc_tensor_t* a, const lmmc_tensor_t* b, lmmc_tensor_t* out_tensor);
lmmc_status_t lmmc_tensor_div(const lmmc_tensor_t* a, const lmmc_tensor_t* b, lmmc_tensor_t* out_tensor);
lmmc_status_t lmmc_tensor_scale(const lmmc_tensor_t* tensor, lmmc_real_t alpha, lmmc_tensor_t* out_tensor);

lmmc_status_t lmmc_tensor_sum(const lmmc_tensor_t* tensor, lmmc_real_t* out_sum);
lmmc_status_t lmmc_tensor_max(const lmmc_tensor_t* tensor, lmmc_real_t* out_max);
lmmc_status_t lmmc_tensor_min(const lmmc_tensor_t* tensor, lmmc_real_t* out_min);
lmmc_status_t lmmc_tensor_sum_axis(const lmmc_tensor_t* tensor, size_t axis, lmmc_mat_t* out_matrix);

lmmc_status_t lmmc_tensor_reshape_view(
    const lmmc_tensor_t* tensor,
    size_t new_dim0,
    size_t new_dim1,
    size_t new_dim2,
    lmmc_tensor_t* out_view
);

lmmc_status_t lmmc_tensor_slice_view(
    const lmmc_tensor_t* tensor,
    size_t begin0,
    size_t end0,
    size_t begin1,
    size_t end1,
    size_t begin2,
    size_t end2,
    lmmc_tensor_t* out_view
);

#ifdef __cplusplus
}
#endif

#endif
