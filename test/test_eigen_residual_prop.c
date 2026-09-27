/**
 * @file test_eigen_residual_prop.c
 * 特征对残差精度属性测试。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "lmmc/lmmc.h"
#include "test_common.h"

#define RESIDUAL_TOL 1e-8
#define MAT_ELEM(mat, i, j) ((mat)->data[(i) * (mat)->stride + (j)])

static int eigenpair_matrix_vector_row(const lmmc_mat_t *A, size_t n, size_t i,
                                       const lmmc_real_t *vr, const lmmc_real_t *vi, double *Avr_i, double *Avi_i) {
    for (size_t j = 0; j < n; j++) {
        const double aij = MAT_ELEM(A, i, j);
        if (!isfinite(aij)) {
            return 1;
        }
        *Avr_i += aij * vr[j];
        *Avi_i += aij * vi[j];
        if (!isfinite(*Avr_i) || !isfinite(*Avi_i)) {
            return 1;
        }
    }
    return 0;
}

/**
 * @brief Verify eigenpair residual for a single eigenpair.
 *
 * For complex eigenvalue lambda = re + i*im with eigenvector v = vr + i*vi:
 *   A*(vr + i*vi) should equal (re + i*im)*(vr + i*vi)
 *   Real part residual: A*vr - (re*vr - im*vi)
 *   Imag part residual: A*vi - (re*vi + im*vr)
 *
 * @return 0 if residual is within tolerance, 1 otherwise.
 */
static int eigenpair_residual(const lmmc_mat_t *A, size_t n,
                              lmmc_real_t re, lmmc_real_t im,
                              const lmmc_real_t *vr, const lmmc_real_t *vi,
                              double norm_A, double norm_v, double tol) {
    double residual = 0.0;
    for (size_t i = 0; i < n; i++) {
        double Avr_i = 0.0;
        double Avi_i = 0.0;
        if (eigenpair_matrix_vector_row(A, n, i, vr, vi, &Avr_i, &Avi_i) != 0) {
            return 1;
        }
        const double lv_real = re * vr[i] - im * vi[i];
        const double lv_imag = re * vi[i] + im * vr[i];
        if (!isfinite(lv_real) || !isfinite(lv_imag)) {
            return 1;
        }

        const double dr = Avr_i - lv_real;
        const double di = Avi_i - lv_imag;
        if (!isfinite(dr) || !isfinite(di)) {
            return 1;
        }
        residual = hypot(hypot(residual, dr), di);
    }
    const double bound = tol * (1.0 + norm_A) * norm_v;
    if (!isfinite(bound) || bound < 0.0) {
        return 1;
    }
    return lmmc_test_nearly_equal(residual, 0.0, bound) ? 0 : 1;
}

static int verify_eigenpair(const lmmc_mat_t *A, size_t n,
                            lmmc_real_t re, lmmc_real_t im,
                            const lmmc_real_t *vr, const lmmc_real_t *vi,
                            double norm_A, double tol) {
    if (!isfinite(re) || !isfinite(im)) {
        return 1;
    }
    if (!isfinite(norm_A) || norm_A < 0.0 || !isfinite(tol) || tol < 0.0) {
        return 1;
    }

    double norm_v = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(vr[i]) || !isfinite(vi[i])) {
            return 1;
        }
        norm_v = hypot(hypot(norm_v, vr[i]), vi[i]);
    }
    if (!isfinite(norm_v) || norm_v < 1e-15) {
        return 1;
    }

    return eigenpair_residual(A, n, re, im, vr, vi, norm_A, norm_v, tol);
}

static void report_eigenpair_residual(size_t n, int trial, const lmmc_mat_t *A,
                                      double norm_A, const lmmc_real_t *vr, const lmmc_real_t *vi, size_t k,
                                      lmmc_real_t re, lmmc_real_t im) {
    double norm_v_sq = 0.0;
    for (size_t j = 0; j < n; j++) {
        norm_v_sq += vr[j] * vr[j] + vi[j] * vi[j];
    }
    double norm_v = sqrt(norm_v_sq);

    double residual_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        double Avr_i = 0.0, Avi_i = 0.0;
        for (size_t j = 0; j < n; j++) {
            double aij = MAT_ELEM(A, i, j);
            Avr_i += aij * vr[j];
            Avi_i += aij * vi[j];
        }
        double lv_real = re * vr[i] - im * vi[i];
        double lv_imag = re * vi[i] + im * vr[i];
        double dr = Avr_i - lv_real;
        double di = Avi_i - lv_imag;
        residual_sq += dr * dr + di * di;
    }
    double residual = sqrt(residual_sq);
    double bound = RESIDUAL_TOL * (1.0 + norm_A) * norm_v;
    fail_msg("trial %d (n=%" PRIuMAX "): eigenpair %" PRIuMAX " failed validation "
             "(residual=%.3e, bound=%.3e, lambda=%.6f + %.6fi, ||v||=%.3e)",
             trial, (uintmax_t)(n), (uintmax_t)(k), residual, bound, re, im, norm_v);
}

typedef struct {
    lmmc_rng_t *rng;
    lmmc_mat_t A;
    lmmc_eigen_gen_full_result_t result;
} eigenpair_fixture_t;

static int setup_eigenpair(void **state) {
    eigenpair_fixture_t *f = calloc(1, sizeof(*f));
    assert_non_null(f);
    *state = f;
    return 0;
}

static void release_eigenpair(eigenpair_fixture_t *f) {
    lmmc_eigen_gen_full_result_destroy(&f->result);
    lmmc_mat_destroy(&f->A);
}

static int teardown_eigenpair(void **state) {
    eigenpair_fixture_t *f = *state;
    release_eigenpair(f);
    lmmc_rng_destroy(f->rng);
    free(f);
    return 0;
}

static void test_eigenpair_residual(void **state) {
    eigenpair_fixture_t *f = *state;
    const size_t sizes[] = {3, 4, 5, 6, 7, 8, 10, 12, 15, 18, 20};
    lmmc_real_t vr[20], vi[20];
    assert_int_equal(lmmc_rng_create(&f->rng), LMMC_STATUS_OK);
    assert_int_equal(lmmc_rng_seed(f->rng, UINT64_C(0x45524553)),
                     LMMC_STATUS_OK);

    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++) {
        const size_t n = sizes[si];
        for (int trial = 1; trial <= 8; trial++) {
            assert_int_equal(lmmc_mat_create(n, n, &f->A), LMMC_STATUS_OK);
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    lmmc_real_t val;
                    assert_int_equal(
                        lmmc_rng_uniform(f->rng, -2.0, 2.0, &val),
                        LMMC_STATUS_OK);
                    MAT_ELEM(&f->A, i, j) = val;
                }
            }
            assert_int_equal(lmmc_eigen_general_full(&f->A, &f->result), LMMC_STATUS_OK);
            const double norm_A = lmmc_test_frobenius_norm(&f->A);
            assert_true(isfinite(norm_A) && norm_A >= 0.0);

            for (size_t k = 0; k < n; k++) {
                for (size_t j = 0; j < n; j++) {
                    vr[j] = MAT_ELEM(&f->result.vectors_real, j, k);
                    vi[j] = MAT_ELEM(&f->result.vectors_imag, j, k);
                }
                const lmmc_real_t re = f->result.real_parts.data[k];
                const lmmc_real_t im = f->result.imag_parts.data[k];
                if (verify_eigenpair(&f->A, n, re, im, vr, vi, norm_A, RESIDUAL_TOL) != 0) {
                    report_eigenpair_residual(n, trial, &f->A, norm_A, vr, vi, k, re, im);
                }
            }
            release_eigenpair(f);
        }
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_eigenpair_residual,
                                        setup_eigenpair, teardown_eigenpair),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
