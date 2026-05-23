#ifndef LMMC_SPARSE_H
#define LMMC_SPARSE_H

#include <stddef.h>
#include "lmmc/dense.h"
#include "lmmc/status.h"
#include "lmmc/config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LMMC_SPARSE_CSR = 0,
    LMMC_SPARSE_CSC = 1
} lmmc_sparse_format_t;

typedef struct {
    size_t rows;
    size_t cols;
    size_t nnz;
    size_t* row_ptr;  /* outer_ptr: row_ptr for CSR, col_ptr for CSC */
    size_t* col_idx;  /* inner_idx: col_idx for CSR, row_idx for CSC */
    lmmc_real_t* values;
    lmmc_sparse_format_t format;
    int owns_data;
} lmmc_sparse_mat_t;

/* Sparse Builder for easy matrix construction (COO based) */
typedef struct lmmc_sparse_builder_t lmmc_sparse_builder_t;

lmmc_status_t lmmc_sparse_builder_create(size_t rows, size_t cols, size_t initial_capacity, lmmc_sparse_builder_t** out_builder);
lmmc_status_t lmmc_sparse_builder_add(lmmc_sparse_builder_t* builder, size_t row, size_t col, lmmc_real_t value);
lmmc_status_t lmmc_sparse_builder_build(lmmc_sparse_builder_t* builder, lmmc_sparse_format_t format, lmmc_sparse_mat_t* out_sparse);
void lmmc_sparse_builder_destroy(lmmc_sparse_builder_t* builder);

lmmc_status_t lmmc_sparse_create_csr(size_t rows, size_t cols, size_t nnz, lmmc_sparse_mat_t* out_sparse);
lmmc_status_t lmmc_sparse_create_csc(size_t rows, size_t cols, size_t nnz, lmmc_sparse_mat_t* out_sparse);
lmmc_status_t lmmc_sparse_wrap_csr(
    size_t rows,
    size_t cols,
    size_t nnz,
    size_t* row_ptr,
    size_t* col_idx,
    lmmc_real_t* values,
    lmmc_sparse_mat_t* out_sparse
);
lmmc_status_t lmmc_sparse_wrap_csc(
    size_t rows,
    size_t cols,
    size_t nnz,
    size_t* col_ptr,
    size_t* row_idx,
    lmmc_real_t* values,
    lmmc_sparse_mat_t* out_sparse
);
void lmmc_sparse_destroy(lmmc_sparse_mat_t* sparse);

lmmc_status_t lmmc_sparse_from_dense(const lmmc_mat_t* dense, lmmc_real_t eps, lmmc_sparse_mat_t* out_sparse);
lmmc_status_t lmmc_sparse_to_dense(const lmmc_sparse_mat_t* sparse, lmmc_mat_t* out_dense);
lmmc_status_t lmmc_sparse_mat_vec_mul(const lmmc_sparse_mat_t* sparse, const lmmc_vec_t* x, lmmc_vec_t* y);
lmmc_status_t lmmc_sparse_transpose(const lmmc_sparse_mat_t* sparse, lmmc_sparse_mat_t* out_transposed);
lmmc_status_t lmmc_sparse_mat_mat_mul_dense(const lmmc_sparse_mat_t* sparse, const lmmc_mat_t* b, lmmc_mat_t* c);
lmmc_status_t lmmc_sparse_mat_mat_mul_sparse(const lmmc_sparse_mat_t* a, const lmmc_sparse_mat_t* b, lmmc_sparse_mat_t* c);

lmmc_status_t lmmc_sparse_to_csc(const lmmc_sparse_mat_t* src, lmmc_sparse_mat_t* dst);
lmmc_status_t lmmc_sparse_to_csr(const lmmc_sparse_mat_t* src, lmmc_sparse_mat_t* dst);

/* === 稀疏 LU 分解 === */
typedef struct lmmc_sparse_lu_t lmmc_sparse_lu_t;  /* 不透明类型 */

lmmc_status_t lmmc_sparse_lu_symbolic(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_lu_t** out_lu
);

lmmc_status_t lmmc_sparse_lu_numeric(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_lu_t* lu
);

lmmc_status_t lmmc_sparse_lu_solve(
    const lmmc_sparse_lu_t* lu,
    const lmmc_vec_t* b,
    lmmc_vec_t* x
);

void lmmc_sparse_lu_destroy(lmmc_sparse_lu_t* lu);

/* === 稀疏 Cholesky 分解 === */
typedef struct lmmc_sparse_chol_t lmmc_sparse_chol_t;  /* 不透明类型 */

lmmc_status_t lmmc_sparse_chol_symbolic(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_chol_t** out_chol
);

lmmc_status_t lmmc_sparse_chol_numeric(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_chol_t* chol
);

lmmc_status_t lmmc_sparse_chol_solve(
    const lmmc_sparse_chol_t* chol,
    const lmmc_vec_t* b,
    lmmc_vec_t* x
);

void lmmc_sparse_chol_destroy(lmmc_sparse_chol_t* chol);

/* === COO 格式 === */
typedef struct {
    size_t rows;
    size_t cols;
    size_t nnz;
    size_t capacity;
    size_t* row_idx;
    size_t* col_idx;
    lmmc_real_t* values;
} lmmc_sparse_coo_t;

lmmc_status_t lmmc_sparse_coo_create(
    size_t rows, size_t cols, size_t capacity,
    lmmc_sparse_coo_t* out_coo
);

lmmc_status_t lmmc_sparse_coo_add_entry(
    lmmc_sparse_coo_t* coo,
    size_t row, size_t col, lmmc_real_t value
);

lmmc_status_t lmmc_sparse_coo_to_csr(
    const lmmc_sparse_coo_t* coo,
    lmmc_sparse_mat_t* out_csr
);

lmmc_status_t lmmc_sparse_coo_to_csc(
    const lmmc_sparse_coo_t* coo,
    lmmc_sparse_mat_t* out_csc
);

void lmmc_sparse_coo_destroy(lmmc_sparse_coo_t* coo);

/* === 稀疏矩阵工具运算 === */
lmmc_status_t lmmc_sparse_add(
    lmmc_real_t alpha, const lmmc_sparse_mat_t* a,
    lmmc_real_t beta, const lmmc_sparse_mat_t* b,
    lmmc_sparse_mat_t* out_c
);

lmmc_status_t lmmc_sparse_scale(
    lmmc_sparse_mat_t* a,
    lmmc_real_t alpha
);

lmmc_status_t lmmc_sparse_norm_fro(
    const lmmc_sparse_mat_t* a,
    lmmc_real_t* out_norm
);

lmmc_status_t lmmc_sparse_diag(
    const lmmc_sparse_mat_t* a,
    lmmc_vec_t* out_diag
);

#ifdef __cplusplus
}
#endif

#endif
