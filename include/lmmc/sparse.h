#ifndef LMMC_SPARSE_H
#define LMMC_SPARSE_H

#include <stddef.h>
#include "lmmc/dense.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t rows;
    size_t cols;
    size_t nnz;
    size_t* row_ptr;
    size_t* col_idx;
    double* values;
    int owns_data;
} lmmc_sparse_mat_t;

lmmc_status_t lmmc_sparse_create_csr(size_t rows, size_t cols, size_t nnz, lmmc_sparse_mat_t* out_sparse);
lmmc_status_t lmmc_sparse_wrap_csr(
    size_t rows,
    size_t cols,
    size_t nnz,
    size_t* row_ptr,
    size_t* col_idx,
    double* values,
    lmmc_sparse_mat_t* out_sparse
);
void lmmc_sparse_destroy(lmmc_sparse_mat_t* sparse);

lmmc_status_t lmmc_sparse_from_dense(const lmmc_mat_t* dense, double eps, lmmc_sparse_mat_t* out_sparse);
lmmc_status_t lmmc_sparse_to_dense(const lmmc_sparse_mat_t* sparse, lmmc_mat_t* out_dense);
lmmc_status_t lmmc_sparse_mat_vec_mul(const lmmc_sparse_mat_t* sparse, const lmmc_vec_t* x, lmmc_vec_t* y);
lmmc_status_t lmmc_sparse_transpose(const lmmc_sparse_mat_t* sparse, lmmc_sparse_mat_t* out_transposed);
lmmc_status_t lmmc_sparse_mat_mat_mul_dense(const lmmc_sparse_mat_t* sparse, const lmmc_mat_t* b, lmmc_mat_t* c);

#ifdef __cplusplus
}
#endif

#endif
