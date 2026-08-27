/**
 * @file eigen_internal.c
 * @brief 特征值 / SVD 模块内部共享的 Householder 与 Givens 辅助函数。
 */

#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "internal.h"
#include "eigen_internal.h"
#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/linear_algebra.h"

void householder_make(lmmc_real_t *x, size_t len, lmmc_real_t *tau_out,
                             lmmc_real_t *beta_out) {
    size_t i;
    if (len == 0) { *tau_out = 0.0; *beta_out = 0.0; return; }
    if (len == 1) { *tau_out = 0.0; *beta_out = x[0]; return; }
    lmmc_real_t sigma = 0.0;
    for (i = 1; i < len; i++) sigma += x[i] * x[i];
    lmmc_real_t alpha = x[0];
    if (sigma == 0.0) { *tau_out = 0.0; *beta_out = alpha; return; }
    lmmc_real_t mu = sqrt(alpha * alpha + sigma);

    lmmc_real_t beta = (alpha >= 0.0) ? -mu : mu;
    lmmc_real_t v0 = alpha - beta;
    *beta_out = beta;

    lmmc_real_t v0_sq = v0 * v0;
    *tau_out = 2.0 * v0_sq / (sigma + v0_sq);
    lmmc_real_t inv = 1.0 / v0;
    for (i = 1; i < len; i++) x[i] *= inv;
    x[0] = 1.0;
}

void householder_apply_left(lmmc_mat_t *M, size_t i0, size_t len,
                                   size_t j0, size_t j1,
                                   const lmmc_real_t *v, lmmc_real_t tau) {
    size_t i, j;
    if (tau == 0.0) return;
    for (j = j0; j < j1; j++) {
        lmmc_real_t s = MAT_ELEM(M, i0, j);
        for (i = 1; i < len; i++) s += v[i] * MAT_ELEM(M, i0 + i, j);
        s *= tau;
        MAT_ELEM(M, i0, j) -= s;
        for (i = 1; i < len; i++) MAT_ELEM(M, i0 + i, j) -= s * v[i];
    }
}

void householder_apply_right(lmmc_mat_t *M, size_t i0, size_t i1,
                                    size_t j0, size_t len,
                                    const lmmc_real_t *v, lmmc_real_t tau) {
    size_t i, j;
    if (tau == 0.0) return;
    for (i = i0; i < i1; i++) {
        lmmc_real_t s = MAT_ELEM(M, i, j0);
        for (j = 1; j < len; j++) s += v[j] * MAT_ELEM(M, i, j0 + j);
        s *= tau;
        MAT_ELEM(M, i, j0) -= s;
        for (j = 1; j < len; j++) MAT_ELEM(M, i, j0 + j) -= s * v[j];
    }
}

void givens_right(lmmc_mat_t *M, size_t rows, size_t p, size_t q,
                         lmmc_real_t c, lmmc_real_t s) {
    size_t i;
    for (i = 0; i < rows; i++) {
        lmmc_real_t mp = MAT_ELEM(M, i, p);
        lmmc_real_t mq = MAT_ELEM(M, i, q);
        MAT_ELEM(M, i, p) =  c * mp + s * mq;
        MAT_ELEM(M, i, q) = -s * mp + c * mq;
    }
}

void givens_compute(lmmc_real_t a, lmmc_real_t b,
                           lmmc_real_t *c, lmmc_real_t *s, lmmc_real_t *r) {
    if (b == 0.0) { *c = (a >= 0.0) ? 1.0 : -1.0; *s = 0.0; *r = lmmc_abs(a); }
    else if (a == 0.0) { *c = 0.0; *s = (b >= 0.0) ? 1.0 : -1.0; *r = lmmc_abs(b); }
    else {
        lmmc_real_t h = sqrt(a * a + b * b);
        *r = h;
        *c = a / h;
        *s = b / h;
    }
}

