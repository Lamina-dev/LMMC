#ifndef LMMC_TEST_STDLIB_COMMON_H
#define LMMC_TEST_STDLIB_COMMON_H

#include "test_common.h"

#include "lmmc/lmmc.h"

#include <math.h>
#include <string.h>

static inline int close_real(lmmc_real_t a, lmmc_real_t b)
{
    return fabs((double)(a - b)) <= 1e-12;
}

static inline void set_mat_values(lmmc_mat_t* mat, const lmmc_real_t* values)
{
    size_t k = 0;
    for (size_t i = 0; i < mat->rows; ++i) {
        for (size_t j = 0; j < mat->cols; ++j) {
            mat->data[i * mat->stride + j] = values[k++];
        }
    }
}

static inline int mat_close_at(const lmmc_mat_t* mat, size_t row, size_t col,
                               lmmc_real_t expected)
{
    return close_real(mat->data[row * mat->stride + col], expected);
}

#endif
