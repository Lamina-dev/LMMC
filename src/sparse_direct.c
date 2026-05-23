/**
 * @file sparse_direct.c
 * @brief Sparse LU factorization with partial pivoting (left-looking algorithm).
 *
 * Implements:
 *   - lmmc_sparse_lu_symbolic: Symbolic analysis (determine non-zero pattern)
 *   - lmmc_sparse_lu_numeric:  Numerical factorization (left-looking with partial pivoting)
 *   - lmmc_sparse_lu_solve:    Forward/back substitution with permutations
 *   - lmmc_sparse_lu_destroy:  Free all allocated memory
 *
 * Requirements: 8.1–8.7
 */

#include <string.h>
#include <math.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/sparse.h"

/* Internal struct definition */
struct lmmc_sparse_lu_t {
    size_t n;
    size_t* col_perm;
    size_t* row_perm;
    size_t* L_col_ptr;
    size_t* U_col_ptr;
    size_t* L_row_idx;
    lmmc_real_t* L_values;
    size_t* U_row_idx;
    lmmc_real_t* U_values;
    size_t L_nnz;
    size_t U_nnz;
};

/* ========================================================================
 * Helper: Convert input matrix to CSC if needed
 * ======================================================================== */
static lmmc_status_t ensure_csc(const lmmc_sparse_mat_t* a,
                                lmmc_sparse_mat_t* csc_out,
                                int* needs_free)
{
    *needs_free = 0;
    if (a->format == LMMC_SPARSE_CSC) {
        /* Already CSC - just copy the struct (shallow) */
        *csc_out = *a;
        return LMMC_STATUS_OK;
    }
    /* Convert CSR to CSC */
    *needs_free = 1;
    return lmmc_sparse_to_csc(a, csc_out);
}

/* ========================================================================
 * lmmc_sparse_lu_symbolic
 * ======================================================================== */
lmmc_status_t lmmc_sparse_lu_symbolic(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_lu_t** out_lu)
{
    lmmc_sparse_lu_t* lu = NULL;
    size_t n;

    /* Parameter validation */
    if (a == NULL || out_lu == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows == 0 || a->cols == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    /* Square matrix check (Requirement 8.6) */
    if (a->rows != a->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = a->rows;

    /* Allocate the LU structure */
    lu = (lmmc_sparse_lu_t*)lmmc_alloc(sizeof(lmmc_sparse_lu_t));
    if (lu == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(lu, 0, sizeof(lmmc_sparse_lu_t));
    lu->n = n;

    /* Allocate permutation arrays */
    lu->col_perm = (size_t*)lmmc_alloc(n * sizeof(size_t));
    lu->row_perm = (size_t*)lmmc_alloc(n * sizeof(size_t));
    if (lu->col_perm == NULL || lu->row_perm == NULL) {
        lmmc_sparse_lu_destroy(lu);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /* Initialize permutations to identity (no reordering in symbolic phase) */
    for (size_t i = 0; i < n; i++) {
        lu->col_perm[i] = i;
        lu->row_perm[i] = i;
    }

    /* Allocate column pointers for L and U.
     * For the simplified approach, we allocate dense-like storage:
     * L is lower triangular (at most n*(n-1)/2 + n entries)
     * U is upper triangular (at most n*(n+1)/2 entries)
     * We allocate maximum possible and trim after numeric phase.
     */
    lu->L_col_ptr = (size_t*)lmmc_alloc((n + 1) * sizeof(size_t));
    lu->U_col_ptr = (size_t*)lmmc_alloc((n + 1) * sizeof(size_t));
    if (lu->L_col_ptr == NULL || lu->U_col_ptr == NULL) {
        lmmc_sparse_lu_destroy(lu);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(lu->L_col_ptr, 0, (n + 1) * sizeof(size_t));
    memset(lu->U_col_ptr, 0, (n + 1) * sizeof(size_t));

    /* For the simplified symbolic analysis, we estimate the maximum non-zero
     * pattern. The actual pattern will be determined during numeric phase.
     * We allocate n*n worst case for small matrices. For larger matrices,
     * we use a heuristic of 10*nnz(A) as upper bound. */
    {
        size_t max_nnz;
        size_t n_sq = n * n;
        size_t heuristic = a->nnz * 10;

        if (n <= 128) {
            /* For small matrices, allocate dense-like storage */
            max_nnz = n_sq;
        } else {
            /* For larger matrices, use heuristic */
            max_nnz = (heuristic < n_sq) ? heuristic : n_sq;
        }

        lu->L_row_idx = (size_t*)lmmc_alloc(max_nnz * sizeof(size_t));
        lu->L_values = (lmmc_real_t*)lmmc_alloc(max_nnz * sizeof(lmmc_real_t));
        lu->U_row_idx = (size_t*)lmmc_alloc(max_nnz * sizeof(size_t));
        lu->U_values = (lmmc_real_t*)lmmc_alloc(max_nnz * sizeof(lmmc_real_t));

        if (lu->L_row_idx == NULL || lu->L_values == NULL ||
            lu->U_row_idx == NULL || lu->U_values == NULL) {
            lmmc_sparse_lu_destroy(lu);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
    }

    lu->L_nnz = 0;
    lu->U_nnz = 0;

    *out_lu = lu;
    return LMMC_STATUS_OK;
}

/* ========================================================================
 * lmmc_sparse_lu_numeric
 *
 * Left-looking LU factorization with partial pivoting.
 * Works column by column: for each column j, solve L * u_j = a_j
 * using previously computed columns of L.
 * ======================================================================== */
lmmc_status_t lmmc_sparse_lu_numeric(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_lu_t* lu)
{
    lmmc_sparse_mat_t csc;
    int csc_needs_free = 0;
    lmmc_status_t status = LMMC_STATUS_OK;
    size_t n;
    lmmc_real_t* col_dense = NULL;  /* Dense workspace for current column */
    int* nonzero_flag = NULL;       /* Flags for non-zero positions */
    size_t* nonzero_list = NULL;    /* List of non-zero row indices */
    size_t* piv_inv = NULL;         /* Inverse of row permutation */

    /* Parameter validation */
    if (a == NULL || lu == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != lu->n) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = lu->n;

    /* Ensure we have CSC format */
    status = ensure_csc(a, &csc, &csc_needs_free);
    if (status != LMMC_STATUS_OK) {
        return status;
    }

    /* Allocate workspace */
    col_dense = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    nonzero_flag = (int*)lmmc_alloc(n * sizeof(int));
    nonzero_list = (size_t*)lmmc_alloc(n * sizeof(size_t));
    piv_inv = (size_t*)lmmc_alloc(n * sizeof(size_t));

    if (col_dense == NULL || nonzero_flag == NULL ||
        nonzero_list == NULL || piv_inv == NULL) {
        status = LMMC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }

    /* Initialize inverse permutation */
    for (size_t i = 0; i < n; i++) {
        piv_inv[i] = i;
    }

    /* Reset LU nnz counters */
    lu->L_nnz = 0;
    lu->U_nnz = 0;

    /* Process each column j */
    for (size_t j = 0; j < n; j++) {
        size_t nz_count = 0;

        /* Clear workspace */
        memset(col_dense, 0, n * sizeof(lmmc_real_t));
        memset(nonzero_flag, 0, n * sizeof(int));

        /* Scatter column j of A (with current row permutation applied) into dense workspace.
         * In CSC format: row_ptr is col_ptr, col_idx is row_idx */
        {
            size_t col_start = csc.row_ptr[j];
            size_t col_end = csc.row_ptr[j + 1];
            for (size_t p = col_start; p < col_end; p++) {
                size_t row = csc.col_idx[p];
                /* Apply inverse row permutation */
                size_t prow = piv_inv[row];
                col_dense[prow] = csc.values[p];
                if (!nonzero_flag[prow]) {
                    nonzero_flag[prow] = 1;
                    nonzero_list[nz_count++] = prow;
                }
            }
        }

        /* Left-looking: for each previously computed column k < j,
         * if col_dense[k] != 0, subtract L(:,k) * col_dense[k] from col_dense */
        for (size_t k = 0; k < j; k++) {
            if (col_dense[k] == 0.0) continue;

            lmmc_real_t ukj = col_dense[k]; /* This is U(k,j) */

            /* Subtract L(:,k) * ukj from col_dense */
            size_t L_start = lu->L_col_ptr[k];
            size_t L_end = lu->L_col_ptr[k + 1];
            for (size_t p = L_start; p < L_end; p++) {
                size_t row = lu->L_row_idx[p];
                lmmc_real_t old_val = col_dense[row];
                col_dense[row] = old_val - lu->L_values[p] * ukj;
                if (!nonzero_flag[row]) {
                    nonzero_flag[row] = 1;
                    nonzero_list[nz_count++] = row;
                }
            }
        }

        /* Partial pivoting: find the largest magnitude element in rows >= j */
        {
            size_t pivot_row = j;
            lmmc_real_t max_val = lmmc_abs(col_dense[j]);

            for (size_t i = j + 1; i < n; i++) {
                lmmc_real_t av = lmmc_abs(col_dense[i]);
                if (av > max_val) {
                    max_val = av;
                    pivot_row = i;
                }
            }

            /* Check for singular matrix (Requirement 8.7) */
            if (max_val == 0.0) {
                status = LMMC_STATUS_SINGULAR_MATRIX;
                goto cleanup;
            }

            /* Swap rows j and pivot_row in the dense column */
            if (pivot_row != j) {
                lmmc_real_t tmp = col_dense[j];
                col_dense[j] = col_dense[pivot_row];
                col_dense[pivot_row] = tmp;

                /* Update row permutation:
                 * We need to track which original row is now at position j */
                /* Find original rows at positions j and pivot_row */
                size_t orig_j = lu->row_perm[j];
                size_t orig_p = lu->row_perm[pivot_row];
                lu->row_perm[j] = orig_p;
                lu->row_perm[pivot_row] = orig_j;
                piv_inv[orig_p] = j;
                piv_inv[orig_j] = pivot_row;

                /* Also swap L entries in rows j and pivot_row for previous columns */
                for (size_t k = 0; k < j; k++) {
                    size_t L_start = lu->L_col_ptr[k];
                    size_t L_end = lu->L_col_ptr[k + 1];
                    /* Find entries for rows j and pivot_row */
                    size_t idx_j = (size_t)-1;
                    size_t idx_p = (size_t)-1;
                    for (size_t p = L_start; p < L_end; p++) {
                        if (lu->L_row_idx[p] == j) idx_j = p;
                        if (lu->L_row_idx[p] == pivot_row) idx_p = p;
                    }
                    if (idx_j != (size_t)-1 && idx_p != (size_t)-1) {
                        lmmc_real_t tmp_v = lu->L_values[idx_j];
                        lu->L_values[idx_j] = lu->L_values[idx_p];
                        lu->L_values[idx_p] = tmp_v;
                    } else if (idx_j != (size_t)-1) {
                        lu->L_row_idx[idx_j] = pivot_row;
                    } else if (idx_p != (size_t)-1) {
                        lu->L_row_idx[idx_p] = j;
                    }
                }
            }
        }

        /* Store U entries (rows 0..j) and L entries (rows j+1..n-1) for column j */
        lu->U_col_ptr[j] = lu->U_nnz;
        for (size_t i = 0; i <= j; i++) {
            if (col_dense[i] != 0.0) {
                lu->U_row_idx[lu->U_nnz] = i;
                lu->U_values[lu->U_nnz] = col_dense[i];
                lu->U_nnz++;
            }
        }

        /* Store L entries (below diagonal), divided by diagonal U(j,j) */
        lu->L_col_ptr[j] = lu->L_nnz;
        {
            lmmc_real_t diag = col_dense[j]; /* U(j,j) - already checked non-zero */
            for (size_t i = j + 1; i < n; i++) {
                if (col_dense[i] != 0.0) {
                    lu->L_row_idx[lu->L_nnz] = i;
                    lu->L_values[lu->L_nnz] = col_dense[i] / diag;
                    lu->L_nnz++;
                }
            }
        }
    }

    /* Finalize column pointers */
    lu->L_col_ptr[n] = lu->L_nnz;
    lu->U_col_ptr[n] = lu->U_nnz;

cleanup:
    if (col_dense) lmmc_free(col_dense);
    if (nonzero_flag) lmmc_free(nonzero_flag);
    if (nonzero_list) lmmc_free(nonzero_list);
    if (piv_inv) lmmc_free(piv_inv);
    if (csc_needs_free) {
        lmmc_sparse_destroy(&csc);
    }

    return status;
}

/* ========================================================================
 * lmmc_sparse_lu_solve
 *
 * Solve Ax = b using the LU factorization:
 *   PA = LU  =>  Ax = b  =>  LUx = Pb
 *   1. Apply row permutation: y = Pb
 *   2. Forward substitution: Lz = y  (L has unit diagonal)
 *   3. Back substitution: Ux = z
 * ======================================================================== */
lmmc_status_t lmmc_sparse_lu_solve(
    const lmmc_sparse_lu_t* lu,
    const lmmc_vec_t* b,
    lmmc_vec_t* x)
{
    size_t n;
    lmmc_real_t* work = NULL;

    /* Parameter validation */
    if (lu == NULL || b == NULL || x == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (b->data == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = lu->n;

    if (b->size != n || x->size != n) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    /* Allocate workspace */
    work = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    if (work == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /* Step 1: Apply row permutation: work[i] = b[row_perm[i]]
     * row_perm[i] gives the original row that is now at position i */
    for (size_t i = 0; i < n; i++) {
        work[i] = b->data[lu->row_perm[i]];
    }

    /* Step 2: Forward substitution with L (unit lower triangular stored in CSC)
     * L has unit diagonal (not stored), entries below diagonal stored.
     * For each column j: for each entry L(i,j) with i > j:
     *   work[i] -= L(i,j) * work[j]
     */
    for (size_t j = 0; j < n; j++) {
        /* work[j] is already final for forward sub (unit diagonal) */
        size_t L_start = lu->L_col_ptr[j];
        size_t L_end = lu->L_col_ptr[j + 1];
        for (size_t p = L_start; p < L_end; p++) {
            size_t i = lu->L_row_idx[p];
            work[i] -= lu->L_values[p] * work[j];
        }
    }

    /* Step 3: Back substitution with U (upper triangular stored in CSC)
     * Process columns from right to left.
     * For column j: U(j,j) is the last entry in column j (or we find it).
     * x[j] = work[j] / U(j,j)
     * Then for each entry U(i,j) with i < j:
     *   work[i] -= U(i,j) * x[j]
     */
    for (size_t jj = 0; jj < n; jj++) {
        size_t j = n - 1 - jj;
        size_t U_start = lu->U_col_ptr[j];
        size_t U_end = lu->U_col_ptr[j + 1];

        /* Find diagonal element U(j,j) - it should be the last entry in column j */
        lmmc_real_t diag = 0.0;
        size_t diag_idx = (size_t)-1;
        for (size_t p = U_start; p < U_end; p++) {
            if (lu->U_row_idx[p] == j) {
                diag = lu->U_values[p];
                diag_idx = p;
                break;
            }
        }

        if (diag == 0.0) {
            lmmc_free(work);
            return LMMC_STATUS_SINGULAR_MATRIX;
        }

        work[j] = work[j] / diag;

        /* Update work for rows above j */
        for (size_t p = U_start; p < U_end; p++) {
            size_t i = lu->U_row_idx[p];
            if (i != j) {
                work[i] -= lu->U_values[p] * work[j];
            }
        }
    }

    /* Copy result to x (no column permutation in this simplified version) */
    memcpy(x->data, work, n * sizeof(lmmc_real_t));

    lmmc_free(work);
    return LMMC_STATUS_OK;
}

/* ========================================================================
 * lmmc_sparse_lu_destroy
 * ======================================================================== */
void lmmc_sparse_lu_destroy(lmmc_sparse_lu_t* lu)
{
    if (lu == NULL) return;

    if (lu->col_perm) lmmc_free(lu->col_perm);
    if (lu->row_perm) lmmc_free(lu->row_perm);
    if (lu->L_col_ptr) lmmc_free(lu->L_col_ptr);
    if (lu->U_col_ptr) lmmc_free(lu->U_col_ptr);
    if (lu->L_row_idx) lmmc_free(lu->L_row_idx);
    if (lu->L_values) lmmc_free(lu->L_values);
    if (lu->U_row_idx) lmmc_free(lu->U_row_idx);
    if (lu->U_values) lmmc_free(lu->U_values);

    lmmc_free(lu);
}

/* ========================================================================
 * Sparse Cholesky factorization (up-looking / column-by-column algorithm)
 *
 * Implements:
 *   - lmmc_sparse_chol_symbolic: Symbolic analysis (allocate structures)
 *   - lmmc_sparse_chol_numeric:  Numerical factorization (column-by-column)
 *   - lmmc_sparse_chol_solve:    Forward/back substitution
 *   - lmmc_sparse_chol_destroy:  Free all allocated memory
 *
 * Storage: L is stored in CSC format. Within each column j, the diagonal
 * entry L(j,j) is stored first, followed by sub-diagonal entries L(i,j)
 * with i > j in increasing row order. Worst-case allocation n*(n+1)/2.
 *
 * Requirements: 9.1–9.7
 * ======================================================================== */

/* Internal struct definition for sparse Cholesky */
struct lmmc_sparse_chol_t {
    size_t n;
    size_t* perm;        /* identity permutation for now */
    size_t* L_col_ptr;
    size_t* L_row_idx;
    lmmc_real_t* L_values;
    size_t L_nnz;
};

/* ========================================================================
 * lmmc_sparse_chol_symbolic
 * ======================================================================== */
lmmc_status_t lmmc_sparse_chol_symbolic(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_chol_t** out_chol)
{
    lmmc_sparse_chol_t* chol = NULL;
    size_t n;
    size_t max_nnz;

    /* Parameter validation */
    if (a == NULL || out_chol == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows == 0 || a->cols == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    /* Square matrix check (Requirement 9.7) */
    if (a->rows != a->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = a->rows;

    /* Allocate the Cholesky structure */
    chol = (lmmc_sparse_chol_t*)lmmc_alloc(sizeof(lmmc_sparse_chol_t));
    if (chol == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(chol, 0, sizeof(lmmc_sparse_chol_t));
    chol->n = n;

    /* Allocate identity permutation */
    chol->perm = (size_t*)lmmc_alloc(n * sizeof(size_t));
    if (chol->perm == NULL) {
        lmmc_sparse_chol_destroy(chol);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    for (size_t i = 0; i < n; i++) {
        chol->perm[i] = i;
    }

    /* Allocate column pointers (n+1 entries) */
    chol->L_col_ptr = (size_t*)lmmc_alloc((n + 1) * sizeof(size_t));
    if (chol->L_col_ptr == NULL) {
        lmmc_sparse_chol_destroy(chol);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(chol->L_col_ptr, 0, (n + 1) * sizeof(size_t));

    /* Worst case: L has n*(n+1)/2 non-zero entries (fully dense lower triangle) */
    max_nnz = n * (n + 1) / 2;
    if (max_nnz == 0) max_nnz = 1; /* defensive: avoid zero-size allocation */

    chol->L_row_idx = (size_t*)lmmc_alloc(max_nnz * sizeof(size_t));
    chol->L_values = (lmmc_real_t*)lmmc_alloc(max_nnz * sizeof(lmmc_real_t));
    if (chol->L_row_idx == NULL || chol->L_values == NULL) {
        lmmc_sparse_chol_destroy(chol);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    chol->L_nnz = 0;
    *out_chol = chol;
    return LMMC_STATUS_OK;
}

/* ========================================================================
 * lmmc_sparse_chol_numeric
 *
 * Up-looking Cholesky (column-by-column):
 *   For column j = 0..n-1:
 *     diag = A(j,j) - sum_{k<j} L(j,k)^2
 *     L(j,j) = sqrt(diag)              (must be > 0 for SPD)
 *     For i > j:
 *       L(i,j) = (A(i,j) - sum_{k<j} L(i,k)*L(j,k)) / L(j,j)
 *
 * Implementation uses a dense column workspace `col_dense`. For each
 * previous column k with L(j,k) != 0, we update col_dense[i] for i >= j
 * using the stored entries of column k.
 * ======================================================================== */
lmmc_status_t lmmc_sparse_chol_numeric(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_chol_t* chol)
{
    lmmc_sparse_mat_t csc;
    int csc_needs_free = 0;
    lmmc_status_t status = LMMC_STATUS_OK;
    size_t n;
    lmmc_real_t* col_dense = NULL;

    /* Parameter validation */
    if (a == NULL || chol == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != a->cols) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a->rows != chol->n) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = chol->n;

    /* Ensure we have CSC format */
    status = ensure_csc(a, &csc, &csc_needs_free);
    if (status != LMMC_STATUS_OK) {
        return status;
    }

    /* Allocate dense column workspace */
    col_dense = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    if (col_dense == NULL) {
        if (csc_needs_free) lmmc_sparse_destroy(&csc);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /* Reset Cholesky nnz counter */
    chol->L_nnz = 0;

    /* Process each column j */
    for (size_t j = 0; j < n; j++) {
        /* Record start of column j in L */
        chol->L_col_ptr[j] = chol->L_nnz;

        /* Clear workspace */
        memset(col_dense, 0, n * sizeof(lmmc_real_t));

        /* Scatter the lower triangular part of column j of A (rows >= j).
         * In CSC format: row_ptr is col_ptr, col_idx is row_idx. */
        {
            size_t col_start = csc.row_ptr[j];
            size_t col_end = csc.row_ptr[j + 1];
            for (size_t p = col_start; p < col_end; p++) {
                size_t row = csc.col_idx[p];
                if (row >= j) {
                    col_dense[row] = csc.values[p];
                }
            }
        }

        /* Subtract contributions from previous columns:
         * For each k < j with L(j,k) != 0:
         *   col_dense[i] -= L(i,k) * L(j,k)  for all i >= j */
        for (size_t k = 0; k < j; k++) {
            size_t L_start = chol->L_col_ptr[k];
            size_t L_end = chol->L_col_ptr[k + 1];
            lmmc_real_t Ljk = 0.0;

            /* Find L(j, k) entry in column k of L.
             * Column k entries are stored in increasing row order, so we can
             * stop early if we pass row j. */
            for (size_t p = L_start; p < L_end; p++) {
                size_t row = chol->L_row_idx[p];
                if (row == j) {
                    Ljk = chol->L_values[p];
                    break;
                }
                if (row > j) {
                    break;
                }
            }

            if (Ljk == 0.0) continue;

            /* Update col_dense for rows i >= j */
            for (size_t p = L_start; p < L_end; p++) {
                size_t i = chol->L_row_idx[p];
                if (i >= j) {
                    col_dense[i] -= chol->L_values[p] * Ljk;
                }
            }
        }

        /* Diagonal element check (Requirement 9.6: positive definite) */
        {
            lmmc_real_t diag = col_dense[j];
            lmmc_real_t sqrt_diag;

            if (diag <= 0.0) {
                status = LMMC_STATUS_NOT_POSITIVE_DEFINITE;
                goto cleanup;
            }

            sqrt_diag = sqrt(diag);

            /* Store L(j, j) first */
            chol->L_row_idx[chol->L_nnz] = j;
            chol->L_values[chol->L_nnz] = sqrt_diag;
            chol->L_nnz++;

            /* Store L(i, j) = col_dense[i] / sqrt_diag for i > j (non-zero only) */
            for (size_t i = j + 1; i < n; i++) {
                if (col_dense[i] != 0.0) {
                    chol->L_row_idx[chol->L_nnz] = i;
                    chol->L_values[chol->L_nnz] = col_dense[i] / sqrt_diag;
                    chol->L_nnz++;
                }
            }
        }
    }

    /* Finalize column pointer */
    chol->L_col_ptr[n] = chol->L_nnz;

cleanup:
    if (col_dense) lmmc_free(col_dense);
    if (csc_needs_free) {
        lmmc_sparse_destroy(&csc);
    }
    return status;
}

/* ========================================================================
 * lmmc_sparse_chol_solve
 *
 * Solve A*x = b using the Cholesky factorization A = L * L^T:
 *   1. Apply permutation:    work = P*b   (identity for now)
 *   2. Forward substitution: solve L*z = work
 *   3. Back substitution:    solve L^T*x = z
 *   4. Apply inverse permutation: x[perm[i]] = work[i]
 * ======================================================================== */
lmmc_status_t lmmc_sparse_chol_solve(
    const lmmc_sparse_chol_t* chol,
    const lmmc_vec_t* b,
    lmmc_vec_t* x)
{
    size_t n;
    lmmc_real_t* work = NULL;

    /* Parameter validation */
    if (chol == NULL || b == NULL || x == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (b->data == NULL || x->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    n = chol->n;

    if (b->size != n || x->size != n) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    /* Allocate workspace */
    work = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    if (work == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /* Step 1: Apply permutation: work[i] = b[perm[i]] (identity here) */
    for (size_t i = 0; i < n; i++) {
        work[i] = b->data[chol->perm[i]];
    }

    /* Step 2: Forward substitution L * z = work
     * Column-oriented: for each column j:
     *   - L(j,j) is the first stored entry in column j.
     *   - z[j] = work[j] / L(j,j)
     *   - For each entry L(i,j) with i > j: work[i] -= L(i,j) * z[j]
     */
    for (size_t j = 0; j < n; j++) {
        size_t L_start = chol->L_col_ptr[j];
        size_t L_end = chol->L_col_ptr[j + 1];
        lmmc_real_t diag = 0.0;
        size_t diag_idx = (size_t)-1;

        /* Find diagonal entry (should be the first one stored) */
        for (size_t p = L_start; p < L_end; p++) {
            if (chol->L_row_idx[p] == j) {
                diag = chol->L_values[p];
                diag_idx = p;
                break;
            }
        }

        if (diag == 0.0) {
            lmmc_free(work);
            return LMMC_STATUS_NOT_POSITIVE_DEFINITE;
        }

        work[j] = work[j] / diag;

        /* Update remaining rows */
        for (size_t p = L_start; p < L_end; p++) {
            if (p == diag_idx) continue;
            size_t i = chol->L_row_idx[p];
            work[i] -= chol->L_values[p] * work[j];
        }
    }

    /* Step 3: Back substitution L^T * x = z (work currently holds z)
     * Process columns from j = n-1 down to 0:
     *   - For each entry L(i,j) with i > j: work[j] -= L(i,j) * x[i]  (x[i] already computed)
     *   - x[j] = work[j] / L(j,j)
     */
    for (size_t jj = 0; jj < n; jj++) {
        size_t j = n - 1 - jj;
        size_t L_start = chol->L_col_ptr[j];
        size_t L_end = chol->L_col_ptr[j + 1];
        lmmc_real_t diag = 0.0;
        size_t diag_idx = (size_t)-1;

        for (size_t p = L_start; p < L_end; p++) {
            if (chol->L_row_idx[p] == j) {
                diag = chol->L_values[p];
                diag_idx = p;
                break;
            }
        }

        if (diag == 0.0) {
            lmmc_free(work);
            return LMMC_STATUS_NOT_POSITIVE_DEFINITE;
        }

        for (size_t p = L_start; p < L_end; p++) {
            if (p == diag_idx) continue;
            size_t i = chol->L_row_idx[p];
            work[j] -= chol->L_values[p] * work[i];
        }
        work[j] = work[j] / diag;
    }

    /* Step 4: Apply inverse permutation: x[perm[i]] = work[i] (identity here) */
    for (size_t i = 0; i < n; i++) {
        x->data[chol->perm[i]] = work[i];
    }

    lmmc_free(work);
    return LMMC_STATUS_OK;
}

/* ========================================================================
 * lmmc_sparse_chol_destroy
 * ======================================================================== */
void lmmc_sparse_chol_destroy(lmmc_sparse_chol_t* chol)
{
    if (chol == NULL) return;

    if (chol->perm) lmmc_free(chol->perm);
    if (chol->L_col_ptr) lmmc_free(chol->L_col_ptr);
    if (chol->L_row_idx) lmmc_free(chol->L_row_idx);
    if (chol->L_values) lmmc_free(chol->L_values);

    lmmc_free(chol);
}
