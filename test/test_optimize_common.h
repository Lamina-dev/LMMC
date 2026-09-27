#ifndef LMMC_TEST_OPTIMIZE_COMMON_H
#define LMMC_TEST_OPTIMIZE_COMMON_H

#include "lmmc/dense_types.h"
#include "lmmc/status.h"

static inline lmmc_status_t lmmc_test_identity_system_f(
    const lmmc_vec_t *x, lmmc_vec_t *f, void *user_data) {
    (void)user_data;
    for (size_t i = 0; i < x->size; ++i) {
        f->data[i] = x->data[i];
    }
    return LMMC_STATUS_OK;
}

static inline lmmc_status_t lmmc_test_identity_system_j(
    const lmmc_vec_t *x, lmmc_mat_t *j, void *user_data) {
    (void)x;
    (void)user_data;
    for (size_t i = 0; i < j->rows; ++i) {
        for (size_t column = 0; column < j->cols; ++column) {
            j->data[i * j->stride + column] = (i == column) ? 1.0 : 0.0;
        }
    }
    return LMMC_STATUS_OK;
}

static inline lmmc_status_t lmmc_test_scalar_square_system_f(
    const lmmc_vec_t *x, lmmc_vec_t *f, void *user_data) {
    (void)user_data;
    f->data[0] = x->data[0] * x->data[0] - 2.0;
    return LMMC_STATUS_OK;
}

#endif
