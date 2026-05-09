#ifndef LMMC_DENSE_H
#define LMMC_DENSE_H

#include <stddef.h>
#include "lmmc/status.h"
#include "lmmc/config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    lmmc_real_t* data;
    int owns_data;
} lmmc_mat_t;

typedef struct {
    size_t size;
    lmmc_real_t* data;
    int owns_data;
} lmmc_vec_t;

lmmc_status_t lmmc_mat_create(size_t rows, size_t cols, lmmc_mat_t* out_mat);
lmmc_status_t lmmc_mat_wrap(size_t rows, size_t cols, size_t stride, lmmc_real_t* data, lmmc_mat_t* out_mat);
void lmmc_mat_destroy(lmmc_mat_t* mat);
lmmc_status_t lmmc_mat_fill(lmmc_mat_t* mat, lmmc_real_t value);
lmmc_status_t lmmc_mat_copy(const lmmc_mat_t* src, lmmc_mat_t* dst);
lmmc_status_t lmmc_mat_transpose_to(const lmmc_mat_t* src, lmmc_mat_t* dst);
lmmc_status_t lmmc_mat_mul(const lmmc_mat_t* a, const lmmc_mat_t* b, lmmc_mat_t* c);
lmmc_status_t lmmc_mat_norm_fro(const lmmc_mat_t* a, lmmc_real_t* out_norm);

lmmc_status_t lmmc_vec_create(size_t size, lmmc_vec_t* out_vec);
lmmc_status_t lmmc_vec_wrap(size_t size, lmmc_real_t* data, lmmc_vec_t* out_vec);
void lmmc_vec_destroy(lmmc_vec_t* vec);
lmmc_status_t lmmc_vec_fill(lmmc_vec_t* vec, lmmc_real_t value);
lmmc_status_t lmmc_vec_dot(const lmmc_vec_t* a, const lmmc_vec_t* b, lmmc_real_t* out_dot);
lmmc_status_t lmmc_mat_vec_mul(const lmmc_mat_t* a, const lmmc_vec_t* x, lmmc_vec_t* y);

#ifdef __cplusplus
}
#endif

#endif
