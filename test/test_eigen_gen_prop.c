/**
 * @file test_eigen_gen_prop.c
 * @brief 针对 LMMC 中 eigen gen prop 相关接口的单元测试。
 *
 * @internal
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/status.h"


#define RNG_SEED 0xC0FFEEULL


#define TRACE_TOL 1e-7

#define STRUCT_TOL 1e-9

#define IMAG_ZERO_TOL 1e-9

#define CHECK(cond, msg, ...) do { \
    if (!(cond)) { \
        printf("  FAIL: " msg "\n", ##__VA_ARGS__); \
        return 1; \
    } \
} while (0)


static unsigned long long g_rng_state = RNG_SEED;

static void rng_seed(unsigned long long s) { g_rng_state = (s == 0) ? 1ULL : s; }

static unsigned long long rng_u64(void)
{
    unsigned long long x = g_rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_rng_state = x;
    return x * 2685821657736338717ULL;
}

static double rng_uniform(double lo, double hi)
{

    double u = (double)(rng_u64() >> 11) * (1.0 / 9007199254740992.0);
    return lo + (hi - lo) * u;
}


static lmmc_status_t fill_mat_from_array(lmmc_mat_t *mat, const lmmc_real_t *src, size_t n)
{
    size_t i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            mat->data[i * mat->stride + j] = src[i * n + j];
        }
    }
    return LMMC_STATUS_OK;
}

static double mat_trace(const lmmc_real_t *A, size_t n)
{
    size_t i;
    double t = 0.0;
    for (i = 0; i < n; i++) t += A[i * n + i];
    return t;
}


static int mat_det_via_lu(const lmmc_real_t *A, size_t n, double *out_det)
{
    double *M;
    size_t i, j, k, p;
    double sign;
    double det;

    M = (double *)malloc(n * n * sizeof(double));
    if (!M) return -1;
    for (i = 0; i < n * n; i++) M[i] = (double)A[i];

    sign = 1.0;
    for (k = 0; k < n; k++) {

        p = k;
        for (i = k + 1; i < n; i++) {
            if (fabs(M[i * n + k]) > fabs(M[p * n + k])) p = i;
        }
        if (fabs(M[p * n + k]) < 1e-300) { free(M); *out_det = 0.0; return 0; }
        if (p != k) {
            for (j = 0; j < n; j++) {
                double tmp = M[k * n + j];
                M[k * n + j] = M[p * n + j];
                M[p * n + j] = tmp;
            }
            sign = -sign;
        }
        for (i = k + 1; i < n; i++) {
            double f = M[i * n + k] / M[k * n + k];
            for (j = k + 1; j < n; j++) {
                M[i * n + j] -= f * M[k * n + j];
            }
            M[i * n + k] = 0.0;
        }
    }
    det = sign;
    for (i = 0; i < n; i++) det *= M[i * n + i];
    free(M);
    *out_det = det;
    return 0;
}


static int check_conjugate_pairs(const lmmc_eigen_gen_result_t *result, double tol)
{
    size_t n = result->real_parts.size;
    int *matched = (int *)calloc(n, sizeof(int));
    size_t i, j;
    if (!matched) {
        printf("  FAIL: allocation failure in check_conjugate_pairs\n");
        return 1;
    }
    for (i = 0; i < n; i++) {
        double im;
        double re;
        int found;
        if (matched[i]) continue;
        im = result->imag_parts.data[i];
        if (fabs(im) <= IMAG_ZERO_TOL) { matched[i] = 1; continue; }
        re = result->real_parts.data[i];
        found = 0;
        for (j = 0; j < n; j++) {
            double rj, ij;
            if (j == i || matched[j]) continue;
            rj = result->real_parts.data[j];
            ij = result->imag_parts.data[j];
            if (fabs(re - rj) <= tol && fabs(im + ij) <= tol) {
                matched[i] = 1;
                matched[j] = 1;
                found = 1;
                break;
            }
        }
        if (!found) {
            printf("  FAIL: no conjugate partner for lambda_%zu = %.6f + %.6fi\n",
                   i, re, im);
            free(matched);
            return 1;
        }
    }
    free(matched);
    return 0;
}


static int check_trace_property(const lmmc_real_t *A, size_t n,
                                const lmmc_eigen_gen_result_t *result,
                                double tol, const char *label)
{
    size_t i;
    double sum = 0.0;
    double tr = mat_trace(A, n);
    double scale;
    double diff;
    for (i = 0; i < n; i++) sum += result->real_parts.data[i];
    scale = fabs(tr) + 1.0;
    diff = fabs(sum - tr);
    if (diff > tol * scale) {
        printf("  FAIL: [%s] sum(real_parts)=%.10g but trace(A)=%.10g, diff=%.3e tol=%.3e\n",
               label, sum, tr, diff, tol * scale);
        return 1;
    }
    return 0;
}


static int check_det_property(const lmmc_real_t *A, size_t n,
                              const lmmc_eigen_gen_result_t *result,
                              double tol, const char *label)
{
    size_t i, j;
    int *used = NULL;
    double prod = 1.0;
    double det = 0.0;
    double scale;
    double diff;
    if (mat_det_via_lu(A, n, &det) != 0) {
        printf("  FAIL: [%s] det computation allocation failure\n", label);
        return 1;
    }

    used = (int *)calloc(n, sizeof(int));
    if (!used) {
        printf("  FAIL: [%s] allocation failure in det check\n", label);
        return 1;
    }

    for (i = 0; i < n; i++) {
        double re_i, im_i;
        if (used[i]) continue;
        re_i = result->real_parts.data[i];
        im_i = result->imag_parts.data[i];
        if (fabs(im_i) <= IMAG_ZERO_TOL) {
            prod *= re_i;
            used[i] = 1;
            continue;
        }

        for (j = i + 1; j < n; j++) {
            double re_j, im_j;
            if (used[j]) continue;
            re_j = result->real_parts.data[j];
            im_j = result->imag_parts.data[j];
            if (fabs(re_i - re_j) <= 1e-6 && fabs(im_i + im_j) <= 1e-6) {
                prod *= (re_i * re_i + im_i * im_i);
                used[i] = 1;
                used[j] = 1;
                break;
            }
        }
        if (!used[i]) {
            printf("  FAIL: [%s] unmatched complex eigenvalue at index %zu\n", label, i);
            free(used);
            return 1;
        }
    }
    free(used);

    scale = fabs(det) + 1.0;
    diff = fabs(prod - det);
    if (diff > tol * scale) {
        printf("  FAIL: [%s] product of eigenvalues = %.10g, det(A) = %.10g, "
               "diff=%.3e tol=%.3e\n",
               label, prod, det, diff, tol * scale);
        return 1;
    }
    return 0;
}


static int run_eigen_general(const lmmc_real_t *A, size_t n,
                             lmmc_eigen_gen_result_t *out, const char *label)
{
    lmmc_mat_t mat;
    lmmc_status_t st;
    st = lmmc_mat_create(n, n, &mat);
    if (st != LMMC_STATUS_OK) {
        printf("  FAIL: [%s] mat_create failed (%d)\n", label, (int)st);
        return 1;
    }
    fill_mat_from_array(&mat, A, n);

    memset(out, 0, sizeof(*out));
    st = lmmc_eigen_general(&mat, out);
    lmmc_mat_destroy(&mat);
    if (st != LMMC_STATUS_OK) {
        printf("  FAIL: [%s] lmmc_eigen_general returned %d\n", label, (int)st);
        return 1;
    }
    return 0;
}


static int test_rotation_matrix(void)
{

    const double theta = 0.7;
    const double c = cos(theta);
    const double s = sin(theta);
    lmmc_real_t A[4] = {
        (lmmc_real_t)c, -(lmmc_real_t)s,
        (lmmc_real_t)s,  (lmmc_real_t)c,
    };
    lmmc_eigen_gen_result_t result;
    int rc = 0;
    if (run_eigen_general(A, 2, &result, "rotation_2x2") != 0) return 1;


    {
        size_t found_pos = 0, found_neg = 0;
        size_t i;
        for (i = 0; i < 2; i++) {
            double re = result.real_parts.data[i];
            double im = result.imag_parts.data[i];
            if (fabs(re - c) <= STRUCT_TOL && fabs(im - s) <= STRUCT_TOL) found_pos = 1;
            if (fabs(re - c) <= STRUCT_TOL && fabs(im + s) <= STRUCT_TOL) found_neg = 1;
        }
        if (!found_pos || !found_neg) {
            printf("  FAIL: rotation matrix eigenvalues not cos±i*sin (found_pos=%zu, found_neg=%zu, got [%.6f%+.6fi, %.6f%+.6fi])\n",
                   found_pos, found_neg,
                   result.real_parts.data[0], result.imag_parts.data[0],
                   result.real_parts.data[1], result.imag_parts.data[1]);
            rc = 1;
        }
    }
    if (rc == 0) rc = check_conjugate_pairs(&result, STRUCT_TOL);
    if (rc == 0) rc = check_trace_property(A, 2, &result, STRUCT_TOL, "rotation_2x2");
    if (rc == 0) rc = check_det_property(A, 2, &result, STRUCT_TOL, "rotation_2x2");

    lmmc_eigen_gen_result_destroy(&result);
    return rc;
}


static int test_diagonal_matrix(void)
{
    lmmc_real_t A[9] = {
        2.5, 0.0, 0.0,
        0.0, -1.5, 0.0,
        0.0, 0.0, 4.0,
    };
    lmmc_eigen_gen_result_t result;
    int rc = 0;
    if (run_eigen_general(A, 3, &result, "diagonal_3x3") != 0) return 1;


    {
        size_t i;
        for (i = 0; i < 3; i++) {
            if (fabs(result.imag_parts.data[i]) > STRUCT_TOL) {
                printf("  FAIL: diagonal matrix produced nonzero imag part at i=%zu (%.6f)\n",
                       i, result.imag_parts.data[i]);
                rc = 1;
            }
        }
    }

    if (rc == 0) {
        double expected[3] = { 2.5, -1.5, 4.0 };
        int matched[3] = {0, 0, 0};
        size_t i, j;
        for (i = 0; i < 3; i++) {
            int found = 0;
            for (j = 0; j < 3; j++) {
                if (matched[j]) continue;
                if (fabs(result.real_parts.data[i] - expected[j]) <= STRUCT_TOL) {
                    matched[j] = 1;
                    found = 1;
                    break;
                }
            }
            if (!found) {
                printf("  FAIL: diagonal eigenvalue %.6f not in expected set\n",
                       result.real_parts.data[i]);
                rc = 1;
                break;
            }
        }
    }
    if (rc == 0) rc = check_trace_property(A, 3, &result, STRUCT_TOL, "diagonal_3x3");
    if (rc == 0) rc = check_det_property(A, 3, &result, STRUCT_TOL, "diagonal_3x3");

    lmmc_eigen_gen_result_destroy(&result);
    return rc;
}


static int test_random_matrices(void)
{
    int trials = 32;
    int t;
    rng_seed(RNG_SEED);
    for (t = 0; t < trials; t++) {
        size_t n;
        size_t i;
        lmmc_real_t A[25];
        lmmc_eigen_gen_result_t result;
        char label[32];
        int rc;


        n = (size_t)(2 + (int)(rng_u64() % 4));

        for (i = 0; i < n * n; i++) {
            A[i] = (lmmc_real_t)rng_uniform(-2.0, 2.0);
        }
        snprintf(label, sizeof(label), "random_%dx%d_#%d", (int)n, (int)n, t);

        if (run_eigen_general(A, n, &result, label) != 0) return 1;

        rc = check_conjugate_pairs(&result, TRACE_TOL);
        if (rc == 0) rc = check_trace_property(A, n, &result, TRACE_TOL, label);

        lmmc_eigen_gen_result_destroy(&result);
        if (rc != 0) {
            printf("    (counterexample matrix, n=%d):\n", (int)n);
            for (i = 0; i < n; i++) {
                size_t j;
                printf("      [");
                for (j = 0; j < n; j++) {
                    printf("%s%.6f", j > 0 ? ", " : "", (double)A[i * n + j]);
                }
                printf("]\n");
            }
            return 1;
        }
    }
    return 0;
}


typedef int (*test_func_t)(void);
typedef struct { const char *name; test_func_t func; } test_entry_t;

int main(void)
{
    test_entry_t tests[] = {
        {"rotation_matrix",   test_rotation_matrix},
        {"diagonal_matrix",   test_diagonal_matrix},
        {"random_matrices",   test_random_matrices},
    };
    size_t n_tests = sizeof(tests) / sizeof(tests[0]);
    size_t i;
    size_t n_passed = 0;
    size_t n_failed = 0;

    printf("=== Property 9: 一般特征值方程验证 ===\n");
    printf("Validates: Requirements 6.1, 6.4\n\n");

    for (i = 0; i < n_tests; i++) {
        printf("[%zu/%zu] %s ... ", i + 1, n_tests, tests[i].name);
        fflush(stdout);
        if (tests[i].func() == 0) {
            printf("PASS\n");
            n_passed++;
        } else {
            printf("FAIL\n");
            n_failed++;
        }
    }

    printf("\n=== Results: %zu passed, %zu failed ===\n", n_passed, n_failed);
    return (n_failed > 0) ? 1 : 0;
}
