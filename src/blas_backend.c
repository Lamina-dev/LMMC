/**
 * @file blas_backend.c
 * @brief 外部 BLAS 后端封装实现。
 *
 * 仅当编译时定义 @c LMMC_USE_BLAS 时启用。
 * 当未定义 LMMC_USE_BLAS 时，所有桥接函数为空操作（no-op）。
 */
#ifdef LMMC_USE_BLAS

#include "blas_backend.h"
#include <cblas.h>
#include <stdlib.h>

/* ========================================================================
 * LAPACK Fortran prototypes (column-major, pass-by-pointer convention).
 * Most vendor LAPACK libraries export these symbols with trailing underscore.
 * ======================================================================== */
extern void dgetrf_(const int* m, const int* n, double* A, const int* lda,
                    int* ipiv, int* info);
extern void dgetrs_(const char* trans, const int* n, const int* nrhs,
                    const double* A, const int* lda, const int* ipiv,
                    double* B, const int* ldb, int* info);
extern void dpotrf_(const char* uplo, const int* n, double* A, const int* lda,
                    int* info);
extern void dpotrs_(const char* uplo, const int* n, const int* nrhs,
                    const double* A, const int* lda, double* B, const int* ldb,
                    int* info);
extern void dgeqrf_(const int* m, const int* n, double* A, const int* lda,
                    double* tau, double* work, const int* lwork, int* info);
extern void dgesvd_(const char* jobu, const char* jobvt, const int* m,
                    const int* n, double* A, const int* lda, double* S,
                    double* U, const int* ldu, double* VT, const int* ldvt,
                    double* work, const int* lwork, int* info);


/* ========================================================================
 * BLAS Bridges
 * ======================================================================== */

void lmmc_blas_dgemm(
    size_t m, size_t n, size_t k,
    lmmc_real_t alpha,
    const lmmc_real_t* A, size_t lda,
    const lmmc_real_t* B, size_t ldb,
    lmmc_real_t beta,
    lmmc_real_t* C, size_t ldc)
{
    const int im  = (int)m;
    const int in  = (int)n;
    const int ik  = (int)k;
    const int ilda = (int)lda;
    const int ildb = (int)ldb;
    const int ildc = (int)ldc;

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                im, in, ik,
                alpha,
                A, ilda,
                B, ildb,
                beta,
                C, ildc);
}


void lmmc_blas_dgemm_ex(
    int transA, int transB,
    size_t m, size_t n, size_t k,
    lmmc_real_t alpha,
    const lmmc_real_t* A, size_t lda,
    const lmmc_real_t* B, size_t ldb,
    lmmc_real_t beta,
    lmmc_real_t* C, size_t ldc)
{
    const int im  = (int)m;
    const int in  = (int)n;
    const int ik  = (int)k;
    const int ilda = (int)lda;
    const int ildb = (int)ldb;
    const int ildc = (int)ldc;

    CBLAS_TRANSPOSE cblas_transA = transA ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE cblas_transB = transB ? CblasTrans : CblasNoTrans;

    cblas_dgemm(CblasRowMajor, cblas_transA, cblas_transB,
                im, in, ik,
                alpha,
                A, ilda,
                B, ildb,
                beta,
                C, ildc);
}


void lmmc_blas_dgemv(
    char trans, size_t m, size_t n,
    lmmc_real_t alpha,
    const lmmc_real_t* A, size_t lda,
    const lmmc_real_t* x, size_t incx,
    lmmc_real_t beta,
    lmmc_real_t* y, size_t incy)
{
    const int im  = (int)m;
    const int in  = (int)n;
    const int ilda = (int)lda;
    const int iincx = (int)incx;
    const int iincy = (int)incy;

    CBLAS_TRANSPOSE cblas_trans = (trans == 'T' || trans == 't') ? CblasTrans : CblasNoTrans;

    cblas_dgemv(CblasRowMajor, cblas_trans,
                im, in,
                alpha,
                A, ilda,
                x, iincx,
                beta,
                y, iincy);
}


lmmc_real_t lmmc_blas_ddot(
    size_t n, const lmmc_real_t* x, size_t incx,
    const lmmc_real_t* y, size_t incy)
{
    const int in    = (int)n;
    const int iincx = (int)incx;
    const int iincy = (int)incy;

    return cblas_ddot(in, x, iincx, y, iincy);
}


void lmmc_blas_dscal(
    size_t n, lmmc_real_t alpha,
    lmmc_real_t* x, size_t incx)
{
    const int in    = (int)n;
    const int iincx = (int)incx;

    cblas_dscal(in, alpha, x, iincx);
}


void lmmc_blas_dcopy(
    size_t n, const lmmc_real_t* x, size_t incx,
    lmmc_real_t* y, size_t incy)
{
    const int in    = (int)n;
    const int iincx = (int)incx;
    const int iincy = (int)incy;

    cblas_dcopy(in, x, iincx, y, iincy);
}


void lmmc_blas_dtrsm(
    char side, char uplo, char trans, char diag,
    size_t m, size_t n,
    lmmc_real_t alpha,
    const lmmc_real_t* A, size_t lda,
    lmmc_real_t* B, size_t ldb)
{
    const int im  = (int)m;
    const int in  = (int)n;
    const int ilda = (int)lda;
    const int ildb = (int)ldb;

    CBLAS_SIDE cblas_side = (side == 'L' || side == 'l') ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == 'T' || trans == 't') ? CblasTrans : CblasNoTrans;
    CBLAS_DIAG cblas_diag = (diag == 'U' || diag == 'u') ? CblasUnit : CblasNonUnit;

    cblas_dtrsm(CblasRowMajor, cblas_side, cblas_uplo, cblas_trans, cblas_diag,
                im, in,
                alpha,
                A, ilda,
                B, ildb);
}


void lmmc_blas_dsyrk(
    char uplo, char trans,
    size_t n, size_t k,
    lmmc_real_t alpha,
    const lmmc_real_t* A, size_t lda,
    lmmc_real_t beta,
    lmmc_real_t* C, size_t ldc)
{
    const int in  = (int)n;
    const int ik  = (int)k;
    const int ilda = (int)lda;
    const int ildc = (int)ldc;

    CBLAS_UPLO cblas_uplo = (uplo == 'U' || uplo == 'u') ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = (trans == 'T' || trans == 't') ? CblasTrans : CblasNoTrans;

    cblas_dsyrk(CblasRowMajor, cblas_uplo, cblas_trans,
                in, ik,
                alpha,
                A, ilda,
                beta,
                C, ildc);
}


void lmmc_blas_daxpy(
    size_t n, lmmc_real_t alpha,
    const lmmc_real_t* x, size_t incx,
    lmmc_real_t* y, size_t incy)
{
    const int in    = (int)n;
    const int iincx = (int)incx;
    const int iincy = (int)incy;

    cblas_daxpy(in, alpha, x, iincx, y, iincy);
}


lmmc_real_t lmmc_blas_dnrm2(
    size_t n, const lmmc_real_t* x, size_t incx)
{
    const int in    = (int)n;
    const int iincx = (int)incx;

    return cblas_dnrm2(in, x, iincx);
}


/* ========================================================================
 * LAPACK Bridges
 *
 * Each bridge validates dimensions before calling the vendor.
 * Returns LAPACK info convention:
 *   info = 0  success
 *   info < 0  the |info|-th argument was invalid (dimension check failed)
 *   info > 0  algorithm-specific failure
 * ======================================================================== */

int lmmc_lapack_getrf(int m, int n, double* A, int lda, int* ipiv)
{
    int info = 0;

    /* Validate dimensions */
    if (m < 0) return -1;
    if (n < 0) return -2;
    if (!A && (m > 0 && n > 0)) return -3;
    if (lda < 1 || (m > 0 && lda < m)) return -4;
    if (!ipiv && (m > 0 && n > 0)) return -5;
    if (m == 0 || n == 0) return 0;

    dgetrf_(&m, &n, A, &lda, ipiv, &info);
    return info;
}


int lmmc_lapack_getrs(char trans, int n, int nrhs, const double* A, int lda,
                      const int* ipiv, double* B, int ldb)
{
    int info = 0;

    /* Validate dimensions */
    if (trans != 'N' && trans != 'n' && trans != 'T' && trans != 't' &&
        trans != 'C' && trans != 'c') return -1;
    if (n < 0) return -2;
    if (nrhs < 0) return -3;
    if (!A && n > 0) return -4;
    if (lda < 1 || (n > 0 && lda < n)) return -5;
    if (!ipiv && n > 0) return -6;
    if (!B && (n > 0 && nrhs > 0)) return -7;
    if (ldb < 1 || (n > 0 && ldb < n)) return -8;
    if (n == 0 || nrhs == 0) return 0;

    dgetrs_(&trans, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info);
    return info;
}


int lmmc_lapack_potrf(char uplo, int n, double* A, int lda)
{
    int info = 0;

    /* Validate dimensions */
    if (uplo != 'U' && uplo != 'u' && uplo != 'L' && uplo != 'l') return -1;
    if (n < 0) return -2;
    if (!A && n > 0) return -3;
    if (lda < 1 || (n > 0 && lda < n)) return -4;
    if (n == 0) return 0;

    dpotrf_(&uplo, &n, A, &lda, &info);
    return info;
}


int lmmc_lapack_potrs(char uplo, int n, int nrhs, const double* A, int lda,
                      double* B, int ldb)
{
    int info = 0;

    /* Validate dimensions */
    if (uplo != 'U' && uplo != 'u' && uplo != 'L' && uplo != 'l') return -1;
    if (n < 0) return -2;
    if (nrhs < 0) return -3;
    if (!A && n > 0) return -4;
    if (lda < 1 || (n > 0 && lda < n)) return -5;
    if (!B && (n > 0 && nrhs > 0)) return -6;
    if (ldb < 1 || (n > 0 && ldb < n)) return -7;
    if (n == 0 || nrhs == 0) return 0;

    dpotrs_(&uplo, &n, &nrhs, A, &lda, B, &ldb, &info);
    return info;
}


int lmmc_lapack_geqrf(int m, int n, double* A, int lda, double* tau)
{
    int info = 0;
    double work_query;
    int lwork = -1;
    double* work = NULL;

    /* Validate dimensions */
    if (m < 0) return -1;
    if (n < 0) return -2;
    if (!A && (m > 0 && n > 0)) return -3;
    if (lda < 1 || (m > 0 && lda < m)) return -4;
    if (!tau && (m > 0 && n > 0)) return -5;
    if (m == 0 || n == 0) return 0;

    /* Workspace query */
    dgeqrf_(&m, &n, A, &lda, tau, &work_query, &lwork, &info);
    if (info != 0) return info;

    lwork = (int)work_query;
    work = (double*)malloc((size_t)lwork * sizeof(double));
    if (!work) return -100; /* allocation failure */

    /* Actual computation */
    dgeqrf_(&m, &n, A, &lda, tau, work, &lwork, &info);
    free(work);
    return info;
}


int lmmc_lapack_gesvd(char jobu, char jobvt, int m, int n, double* A, int lda,
                      double* S, double* U, int ldu, double* VT, int ldvt)
{
    int info = 0;
    double work_query;
    int lwork = -1;
    double* work = NULL;
    int minmn = m < n ? m : n;

    /* Validate dimensions */
    if (jobu != 'A' && jobu != 'a' && jobu != 'S' && jobu != 's' &&
        jobu != 'O' && jobu != 'o' && jobu != 'N' && jobu != 'n') return -1;
    if (jobvt != 'A' && jobvt != 'a' && jobvt != 'S' && jobvt != 's' &&
        jobvt != 'O' && jobvt != 'o' && jobvt != 'N' && jobvt != 'n') return -2;
    if (m < 0) return -3;
    if (n < 0) return -4;
    if (!A && (m > 0 && n > 0)) return -5;
    if (lda < 1 || (m > 0 && lda < m)) return -6;
    if (!S && minmn > 0) return -7;
    /* U validation */
    if ((jobu == 'A' || jobu == 'a' || jobu == 'S' || jobu == 's') && !U && m > 0)
        return -8;
    if (ldu < 1) {
        if (jobu == 'A' || jobu == 'a' || jobu == 'S' || jobu == 's')
            return -9;
    } else if ((jobu == 'A' || jobu == 'a') && ldu < m) {
        return -9;
    } else if ((jobu == 'S' || jobu == 's') && ldu < m) {
        return -9;
    }
    /* VT validation */
    if ((jobvt == 'A' || jobvt == 'a' || jobvt == 'S' || jobvt == 's') && !VT && n > 0)
        return -10;
    if (ldvt < 1) {
        if (jobvt == 'A' || jobvt == 'a' || jobvt == 'S' || jobvt == 's')
            return -11;
    } else if ((jobvt == 'A' || jobvt == 'a') && ldvt < n) {
        return -11;
    } else if ((jobvt == 'S' || jobvt == 's') && ldvt < minmn) {
        return -11;
    }
    if (m == 0 || n == 0) return 0;

    /* Workspace query */
    dgesvd_(&jobu, &jobvt, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt,
            &work_query, &lwork, &info);
    if (info != 0) return info;

    lwork = (int)work_query;
    work = (double*)malloc((size_t)lwork * sizeof(double));
    if (!work) return -100; /* allocation failure */

    /* Actual computation */
    dgesvd_(&jobu, &jobvt, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt,
            work, &lwork, &info);
    free(work);
    return info;
}

#endif /* LMMC_USE_BLAS */
