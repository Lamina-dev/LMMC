#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "lmmc/linear_algebra.h"

static double lmmc_absd(double x) {
    return x < 0.0 ? -x : x;
}

static void lmmc_swap_rows(lmmc_mat_t* a, size_t r0, size_t r1) {
    size_t j = 0;
    if (r0 == r1) {
        return;
    }
    for (j = 0; j < a->cols; ++j) {
        double tmp = a->data[r0 * a->stride + j];
        a->data[r0 * a->stride + j] = a->data[r1 * a->stride + j];
        a->data[r1 * a->stride + j] = tmp;
    }
}

lmmc_status_t lmmc_lu_decompose_inplace(lmmc_mat_t* a, size_t* pivots, size_t* out_swap_count) {
    size_t n = 0;
    size_t k = 0;
    size_t swap_count = 0;
    const double eps = 1e-15;

    if (a == NULL || pivots == NULL || a->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    n = a->rows;
    for (k = 0; k < n; ++k) {
        size_t i = 0;
        size_t p = k;
        double maxv = lmmc_absd(a->data[k * a->stride + k]);

        for (i = k + 1; i < n; ++i) {
            double v = lmmc_absd(a->data[i * a->stride + k]);
            if (v > maxv) {
                maxv = v;
                p = i;
            }
        }

        if (maxv <= eps) {
            return LMMC_STATUS_SINGULAR_MATRIX;
        }

        pivots[k] = p;
        if (p != k) {
            lmmc_swap_rows(a, p, k);
            ++swap_count;
        }

        {
            double akk = a->data[k * a->stride + k];
            for (i = k + 1; i < n; ++i) {
                size_t j = 0;
                a->data[i * a->stride + k] /= akk;
                for (j = k + 1; j < n; ++j) {
                    a->data[i * a->stride + j] -= a->data[i * a->stride + k] * a->data[k * a->stride + j];
                }
            }
        }
    }

    if (out_swap_count != NULL) {
        *out_swap_count = swap_count;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_lu_solve(const lmmc_mat_t* lu, const size_t* pivots, const lmmc_vec_t* b, lmmc_vec_t* x) {
    size_t n = 0;
    size_t i = 0;

    if (lu == NULL || pivots == NULL || b == NULL || x == NULL || lu->data == NULL || b->data == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (lu->rows != lu->cols || b->size != lu->rows || x->size != lu->rows) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    n = lu->rows;
    for (i = 0; i < n; ++i) {
        x->data[i] = b->data[i];
    }

    for (i = 0; i < n; ++i) {
        size_t p = pivots[i];
        if (p != i) {
            double tmp = x->data[i];
            x->data[i] = x->data[p];
            x->data[p] = tmp;
        }
    }

    for (i = 0; i < n; ++i) {
        size_t j = 0;
        for (j = 0; j < i; ++j) {
            x->data[i] -= lu->data[i * lu->stride + j] * x->data[j];
        }
    }

    for (i = n; i-- > 0;) {
        size_t j = 0;
        for (j = i + 1; j < n; ++j) {
            x->data[i] -= lu->data[i * lu->stride + j] * x->data[j];
        }
        if (lmmc_absd(lu->data[i * lu->stride + i]) <= 1e-15) {
            return LMMC_STATUS_SINGULAR_MATRIX;
        }
        x->data[i] /= lu->data[i * lu->stride + i];
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_qr_decompose_inplace(lmmc_mat_t* a, double* tau, size_t tau_size) {
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;

    if (a == NULL || tau == NULL || a->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    m = a->rows;
    n = a->cols;
    if (m == 0 || n == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (tau_size < (m < n ? m : n)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (k = 0; k < (m < n ? m : n); ++k) {
        size_t i = 0;
        size_t j = 0;
        double sigma = 0.0;
        double alpha = a->data[k * a->stride + k];

        for (i = k + 1; i < m; ++i) {
            double v = a->data[i * a->stride + k];
            sigma += v * v;
        }

        if (sigma == 0.0) {
            tau[k] = 0.0;
            continue;
        }

        {
            double norm = sqrt(alpha * alpha + sigma);
            double beta = (alpha <= 0.0) ? norm : -norm;
            double v0 = alpha - beta;
            a->data[k * a->stride + k] = beta;

            for (i = k + 1; i < m; ++i) {
                a->data[i * a->stride + k] /= v0;
            }

            tau[k] = (beta - alpha) / beta;

            for (j = k + 1; j < n; ++j) {
                double dot = a->data[k * a->stride + j];
                for (i = k + 1; i < m; ++i) {
                    dot += a->data[i * a->stride + k] * a->data[i * a->stride + j];
                }
                dot *= tau[k];

                a->data[k * a->stride + j] -= dot;
                for (i = k + 1; i < m; ++i) {
                    a->data[i * a->stride + j] -= a->data[i * a->stride + k] * dot;
                }
            }
        }
    }

    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_qr_solve(const lmmc_mat_t* qr, const double* tau, const lmmc_vec_t* b, lmmc_vec_t* x) {
    size_t m = 0;
    size_t n = 0;
    size_t k = 0;
    double* y = NULL;

    if (qr == NULL || tau == NULL || b == NULL || x == NULL || qr->data == NULL || b->data == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    m = qr->rows;
    n = qr->cols;
    if (m < n || b->size != m || x->size != n) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    y = (double*)lmmc_alloc(m * sizeof(double));
    if (y == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(y, b->data, m * sizeof(double));

    for (k = 0; k < n; ++k) {
        size_t i = 0;
        double dot = y[k];
        for (i = k + 1; i < m; ++i) {
            dot += qr->data[i * qr->stride + k] * y[i];
        }
        dot *= tau[k];
        y[k] -= dot;
        for (i = k + 1; i < m; ++i) {
            y[i] -= qr->data[i * qr->stride + k] * dot;
        }
    }

    for (k = n; k-- > 0;) {
        size_t j = 0;
        double rhs = y[k];
        for (j = k + 1; j < n; ++j) {
            rhs -= qr->data[k * qr->stride + j] * x->data[j];
        }
        if (lmmc_absd(qr->data[k * qr->stride + k]) <= 1e-15) {
            lmmc_free(y);
            return LMMC_STATUS_SINGULAR_MATRIX;
        }
        x->data[k] = rhs / qr->data[k * qr->stride + k];
    }

    lmmc_free(y);
    return LMMC_STATUS_OK;
}
