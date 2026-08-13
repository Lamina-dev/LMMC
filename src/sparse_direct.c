/**
 * @file sparse_direct.c
 * 稀疏 LU / Cholesky 直接分解，含 AMD 重排序与 Gilbert-Peierls 数值分解。
 */
#include <string.h>
#include <math.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/sparse.h"


struct lmmc_sparse_lu_t {
    size_t n;
    size_t* col_perm;      /* AMD column permutation: new_col -> old_col */
    size_t* col_perm_inv;  /* inverse: old_col -> new_col */
    size_t* row_perm;      /* row permutation from partial pivoting */
    size_t* L_col_ptr;
    size_t* U_col_ptr;
    size_t* L_row_idx;
    lmmc_real_t* L_values;
    size_t* U_row_idx;
    lmmc_real_t* U_values;
    size_t L_nnz;
    size_t U_nnz;
    size_t L_capacity;
    size_t U_capacity;
    size_t* etree;         /* elimination tree parent[k] */
};


struct lmmc_sparse_chol_t {
    size_t n;
    size_t* perm;          /* AMD permutation: new -> old */
    size_t* perm_inv;      /* inverse: old -> new */
    size_t* L_col_ptr;
    size_t* L_row_idx;
    lmmc_real_t* L_values;
    size_t L_nnz;
    size_t L_capacity;
    size_t* etree;         /* elimination tree */
};
/* 确保 CSC 格式                                                            */

static lmmc_status_t ensure_csc(const lmmc_sparse_mat_t* a,
                                lmmc_sparse_mat_t* csc_out,
                                int* needs_free)
{
    *needs_free = 0;
    if (a->format == LMMC_SPARSE_CSC) {
        *csc_out = *a;
        return LMMC_STATUS_OK;
    }
    *needs_free = 1;
    return lmmc_sparse_to_csc(a, csc_out);
}
/* 动态容量缓冲区增长                                                       */

static lmmc_status_t sparse_ensure_capacity(size_t** idx, lmmc_real_t** vals,
                                            size_t* capacity, size_t needed)
{
    if (needed < *capacity) {
        return LMMC_STATUS_OK;
    }
    size_t new_cap = *capacity * 2;
    if (new_cap <= needed) {
        new_cap = needed + 1;
    }
    size_t* new_idx = (size_t*)lmmc_realloc(*idx, new_cap * sizeof(size_t));
    if (new_idx == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    *idx = new_idx;
    lmmc_real_t* new_vals = (lmmc_real_t*)lmmc_realloc(*vals, new_cap * sizeof(lmmc_real_t));
    if (new_vals == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    *vals = new_vals;
    *capacity = new_cap;
    return LMMC_STATUS_OK;
}
/* AMD (Approximate Minimum Degree) Reordering                              */

/**
 * @brief Compute AMD column permutation for a symmetric sparsity pattern.
 *
 * Simplified AMD using external degree with bucket sort. For each step,
 * selects the node with minimum degree, eliminates it, and decrements
 * degrees of non-eliminated neighbors (mass elimination approximation).
 *
 * Input: symmetric CSC pattern (col_ptr, row_idx) of dimension n.
 * Output: perm[k] = original column that becomes column k in new ordering.
 *         perm_inv[j] = new position of original column j.
 */
static lmmc_status_t amd_reorder(size_t n,
                                  const size_t* col_ptr,
                                  const size_t* row_idx,
                                  size_t* perm,
                                  size_t* perm_inv)
{
    if (n == 0) return LMMC_STATUS_OK;
    if (n == 1) {
        perm[0] = 0;
        perm_inv[0] = 0;
        return LMMC_STATUS_OK;
    }

    size_t* degree = (size_t*)lmmc_alloc(n * sizeof(size_t));
    int* eliminated = (int*)lmmc_alloc(n * sizeof(int));
    if (!degree || !eliminated) {
        if (degree) lmmc_free(degree);
        if (eliminated) lmmc_free(eliminated);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(eliminated, 0, n * sizeof(int));

    /* Compute initial external degrees (exclude self-loops) */
    for (size_t i = 0; i < n; i++) {
        size_t deg = 0;
        for (size_t p = col_ptr[i]; p < col_ptr[i + 1]; p++) {
            if (row_idx[p] != i) deg++;
        }
        degree[i] = deg;
    }

    /* Main loop: greedily select minimum degree node */
    for (size_t step = 0; step < n; step++) {
        /* Find non-eliminated node with minimum degree */
        size_t min_node = (size_t)-1;
        size_t min_deg = (size_t)-1;
        for (size_t i = 0; i < n; i++) {
            if (!eliminated[i] && degree[i] < min_deg) {
                min_deg = degree[i];
                min_node = i;
            }
        }

        perm[step] = min_node;
        perm_inv[min_node] = step;
        eliminated[min_node] = 1;

        /* Update degrees of non-eliminated neighbors */
        for (size_t p = col_ptr[min_node]; p < col_ptr[min_node + 1]; p++) {
            size_t nb = row_idx[p];
            if (nb == min_node || eliminated[nb]) continue;
            if (degree[nb] > 0) degree[nb]--;
        }
    }

    lmmc_free(degree);
    lmmc_free(eliminated);
    return LMMC_STATUS_OK;
}


/**
 * @brief Build symmetric pattern (A + A^T) in CSC for AMD input.
 *
 * For unsymmetric A, AMD needs a symmetric pattern. We form A+A^T.
 * Allocates sym_col_ptr (n+1) and sym_row_idx internally.
 * Caller must free sym_row_idx when done.
 */
static lmmc_status_t build_symmetric_csc(
    size_t n,
    const size_t* col_ptr,
    const size_t* row_idx,
    size_t** out_sym_col_ptr,
    size_t** out_sym_row_idx)
{
    /* First pass: count entries per column in A+A^T (excluding diagonal) */
    size_t* sym_col_ptr = (size_t*)lmmc_alloc((n + 1) * sizeof(size_t));
    int* marker = (int*)lmmc_alloc(n * sizeof(int));
    if (!sym_col_ptr || !marker) {
        if (sym_col_ptr) lmmc_free(sym_col_ptr);
        if (marker) lmmc_free(marker);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(sym_col_ptr, 0, (n + 1) * sizeof(size_t));

    for (size_t i = 0; i < n; i++) marker[i] = -1;

    /* Count: for each entry (i,j) in A with i!=j, ensure both (i,j) and (j,i) */
    for (size_t j = 0; j < n; j++) {
        marker[j] = (int)j; /* mark diagonal */
        for (size_t p = col_ptr[j]; p < col_ptr[j + 1]; p++) {
            size_t i = row_idx[p];
            if (i == j) continue;
            if (marker[i] != (int)j) {
                marker[i] = (int)j;
                sym_col_ptr[j + 1]++;  /* (i,j) in column j */
            }
        }
    }

    /* Also count transpose entries */
    for (size_t i = 0; i < n; i++) marker[i] = -1;
    for (size_t j = 0; j < n; j++) {
        marker[j] = (int)j;
        for (size_t p = col_ptr[j]; p < col_ptr[j + 1]; p++) {
            size_t i = row_idx[p];
            if (i == j) continue;
            /* Transpose: entry (j,i) goes into column i */
            if (marker[j] != (int)i) {
                /* We need to check if (j,i) already counted in column i */
            }
        }
    }

    /* Simpler: just count all off-diagonal entries in A, then for each (i,j),
       add to both col j and col i if not already present. */
    memset(sym_col_ptr, 0, (n + 1) * sizeof(size_t));
    for (size_t i = 0; i < n; i++) marker[i] = -1;

    for (size_t j = 0; j < n; j++) {
        marker[j] = (int)j;
        for (size_t p = col_ptr[j]; p < col_ptr[j + 1]; p++) {
            size_t i = row_idx[p];
            if (i == j) continue;
            if (marker[i] != (int)j) {
                marker[i] = (int)j;
                sym_col_ptr[j + 1]++;
                sym_col_ptr[i + 1]++;
            }
        }
    }

    /* Prefix sum */
    for (size_t j = 0; j < n; j++) {
        sym_col_ptr[j + 1] += sym_col_ptr[j];
    }
    size_t sym_nnz = sym_col_ptr[n];

    /* Allocate row indices */
    size_t* sym_row_idx = NULL;
    if (sym_nnz > 0) {
        sym_row_idx = (size_t*)lmmc_alloc(sym_nnz * sizeof(size_t));
        if (!sym_row_idx) {
            lmmc_free(sym_col_ptr);
            lmmc_free(marker);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
    }

    /* Second pass: fill row indices */
    size_t* work = (size_t*)lmmc_alloc(n * sizeof(size_t));
    if (!work) {
        if (sym_row_idx) lmmc_free(sym_row_idx);
        lmmc_free(sym_col_ptr);
        lmmc_free(marker);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    for (size_t j = 0; j < n; j++) work[j] = sym_col_ptr[j];
    for (size_t i = 0; i < n; i++) marker[i] = -1;

    for (size_t j = 0; j < n; j++) {
        marker[j] = (int)j;
        for (size_t p = col_ptr[j]; p < col_ptr[j + 1]; p++) {
            size_t i = row_idx[p];
            if (i == j) continue;
            if (marker[i] != (int)j) {
                marker[i] = (int)j;
                sym_row_idx[work[j]++] = i;
                sym_row_idx[work[i]++] = j;
            }
        }
    }

    lmmc_free(work);
    lmmc_free(marker);

    *out_sym_col_ptr = sym_col_ptr;
    *out_sym_row_idx = sym_row_idx;
    return LMMC_STATUS_OK;
}
/* Elimination Tree                                                          */

/**
 * @brief Build elimination tree from a CSC lower-triangular pattern.
 *
 * For column k, parent[k] = min { i > k : entry (i,k) exists in pattern }.
 * Uses union-find with path compression for efficiency.
 */
static void build_etree(size_t n,
                         const size_t* col_ptr,
                         const size_t* row_idx,
                         size_t* parent)
{
    size_t* ancestor = (size_t*)lmmc_alloc(n * sizeof(size_t));
    if (!ancestor) {
        for (size_t i = 0; i < n; i++)
            parent[i] = (i + 1 < n) ? i + 1 : (size_t)-1;
        return;
    }

    for (size_t i = 0; i < n; i++) {
        parent[i] = (size_t)-1;
        ancestor[i] = (size_t)-1;
    }

    for (size_t k = 0; k < n; k++) {
        for (size_t p = col_ptr[k]; p < col_ptr[k + 1]; p++) {
            size_t i = row_idx[p];
            if (i >= k) continue;
            /* Walk from i to root, set parent */
            size_t node = i;
            while (ancestor[node] != (size_t)-1 && ancestor[node] != k) {
                size_t t = ancestor[node];
                ancestor[node] = k;
                node = t;
            }
            if (ancestor[node] == (size_t)-1) {
                parent[node] = k;
                ancestor[node] = k;
            }
        }
    }
    lmmc_free(ancestor);
}
/* Sparse LU: Symbolic Phase                                                 */

lmmc_status_t lmmc_sparse_lu_symbolic(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_lu_t** out_lu)
{
    lmmc_sparse_lu_t* lu = NULL;
    lmmc_sparse_mat_t csc;
    int csc_needs_free = 0;
    size_t n;
    lmmc_status_t status;

    if (a == NULL || out_lu == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows == 0 || a->cols == 0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows != a->cols) return LMMC_STATUS_INVALID_ARGUMENT;

    n = a->rows;

    /* Convert to CSC for analysis */
    status = ensure_csc(a, &csc, &csc_needs_free);
    if (status != LMMC_STATUS_OK) return status;

    /* Allocate LU context */
    lu = (lmmc_sparse_lu_t*)lmmc_alloc(sizeof(lmmc_sparse_lu_t));
    if (lu == NULL) {
        if (csc_needs_free) lmmc_sparse_destroy(&csc);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(lu, 0, sizeof(lmmc_sparse_lu_t));
    lu->n = n;

    /* Allocate permutation arrays */
    lu->col_perm = (size_t*)lmmc_alloc(n * sizeof(size_t));
    lu->col_perm_inv = (size_t*)lmmc_alloc(n * sizeof(size_t));
    lu->row_perm = (size_t*)lmmc_alloc(n * sizeof(size_t));
    lu->etree = (size_t*)lmmc_alloc(n * sizeof(size_t));
    lu->L_col_ptr = (size_t*)lmmc_alloc((n + 1) * sizeof(size_t));
    lu->U_col_ptr = (size_t*)lmmc_alloc((n + 1) * sizeof(size_t));

    if (!lu->col_perm || !lu->col_perm_inv || !lu->row_perm ||
        !lu->etree || !lu->L_col_ptr || !lu->U_col_ptr) {
        lmmc_sparse_lu_destroy(lu);
        if (csc_needs_free) lmmc_sparse_destroy(&csc);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    memset(lu->L_col_ptr, 0, (n + 1) * sizeof(size_t));
    memset(lu->U_col_ptr, 0, (n + 1) * sizeof(size_t));

    /* Initialize row_perm to identity (will be modified by partial pivoting) */
    for (size_t i = 0; i < n; i++) lu->row_perm[i] = i;

    /* Compute AMD column permutation on symmetric pattern A+A^T */
    {
        size_t* sym_col_ptr = NULL;
        size_t* sym_row_idx = NULL;
        status = build_symmetric_csc(n, csc.row_ptr, csc.col_idx,
                                     &sym_col_ptr, &sym_row_idx);
        if (status != LMMC_STATUS_OK) {
            lmmc_sparse_lu_destroy(lu);
            if (csc_needs_free) lmmc_sparse_destroy(&csc);
            return status;
        }

        status = amd_reorder(n, sym_col_ptr, sym_row_idx,
                             lu->col_perm, lu->col_perm_inv);

        lmmc_free(sym_col_ptr);
        if (sym_row_idx) lmmc_free(sym_row_idx);

        if (status != LMMC_STATUS_OK) {
            lmmc_sparse_lu_destroy(lu);
            if (csc_needs_free) lmmc_sparse_destroy(&csc);
            return status;
        }
    }

    /* Build elimination tree from permuted structure */
    build_etree(n, csc.row_ptr, csc.col_idx, lu->etree);

    /* Allocate factor arrays with initial capacity */
    {
        size_t initial_capacity = (a->nnz > 64) ? a->nnz : 64;
        /* Heuristic: expect some fill-in */
        if (initial_capacity < n) initial_capacity = n;

        lu->L_row_idx = (size_t*)lmmc_alloc(initial_capacity * sizeof(size_t));
        lu->L_values = (lmmc_real_t*)lmmc_alloc(initial_capacity * sizeof(lmmc_real_t));
        lu->U_row_idx = (size_t*)lmmc_alloc(initial_capacity * sizeof(size_t));
        lu->U_values = (lmmc_real_t*)lmmc_alloc(initial_capacity * sizeof(lmmc_real_t));

        if (!lu->L_row_idx || !lu->L_values ||
            !lu->U_row_idx || !lu->U_values) {
            lmmc_sparse_lu_destroy(lu);
            if (csc_needs_free) lmmc_sparse_destroy(&csc);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }

        lu->L_capacity = initial_capacity;
        lu->U_capacity = initial_capacity;
    }

    lu->L_nnz = 0;
    lu->U_nnz = 0;

    if (csc_needs_free) lmmc_sparse_destroy(&csc);
    *out_lu = lu;
    return LMMC_STATUS_OK;
}
/* Sparse LU: Numeric Phase (Gilbert-Peierls)                                */

/**
 * @brief Gilbert-Peierls sparse LU factorization with partial pivoting.
 *
 * For each column k (in permuted order):
 * 1. Scatter permuted column of A into dense workspace.
 * 2. Use depth-first reach in the partially-built L to find the non-zero
 *    pattern of column k of L and U.
 * 3. Perform numeric update using the symbolic pattern.
 * 4. Apply partial pivoting.
 * 5. Store L and U entries.
 */
lmmc_status_t lmmc_sparse_lu_numeric(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_lu_t* lu)
{
    lmmc_sparse_mat_t csc;
    int csc_needs_free = 0;
    lmmc_status_t status = LMMC_STATUS_OK;
    size_t n;
    lmmc_real_t* col_dense = NULL;
    int* nonzero_flag = NULL;
    size_t* nonzero_list = NULL;
    size_t* piv_inv = NULL;

    if (a == NULL || lu == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows != a->cols) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows != lu->n) return LMMC_STATUS_INVALID_ARGUMENT;

    n = lu->n;

    status = ensure_csc(a, &csc, &csc_needs_free);
    if (status != LMMC_STATUS_OK) return status;

    /* Allocate workspace */
    col_dense = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    nonzero_flag = (int*)lmmc_alloc(n * sizeof(int));
    nonzero_list = (size_t*)lmmc_alloc(n * sizeof(size_t));
    piv_inv = (size_t*)lmmc_alloc(n * sizeof(size_t));

    if (!col_dense || !nonzero_flag || !nonzero_list || !piv_inv) {
        status = LMMC_STATUS_ALLOCATION_FAILED;
        goto cleanup;
    }

    /* Initialize row permutation inverse */
    for (size_t i = 0; i < n; i++) piv_inv[i] = i;

    /* Reset factor counts */
    lu->L_nnz = 0;
    lu->U_nnz = 0;

    /* Reset row_perm to identity */
    for (size_t i = 0; i < n; i++) lu->row_perm[i] = i;

    /* Gilbert-Peierls: process each column in AMD order */
    for (size_t j = 0; j < n; j++) {
        size_t nz_count = 0;
        size_t orig_col = lu->col_perm[j]; /* original column index */

        /* Set L_col_ptr[j] BEFORE the triangular solve so that
         * L_col_ptr[k+1] is valid when k = j-1. At this point L_nnz
         * equals the end of column j-1's entries. */
        lu->L_col_ptr[j] = lu->L_nnz;

        /* Clear workspace */
        memset(col_dense, 0, n * sizeof(lmmc_real_t));
        memset(nonzero_flag, 0, n * sizeof(int));

        /* Scatter column orig_col of A into dense workspace (permuted rows) */
        {
            size_t col_start = csc.row_ptr[orig_col];
            size_t col_end = csc.row_ptr[orig_col + 1];
            for (size_t p = col_start; p < col_end; p++) {
                size_t row = csc.col_idx[p];
                size_t prow = piv_inv[row]; /* permuted row */
                col_dense[prow] += csc.values[p];
                if (!nonzero_flag[prow]) {
                    nonzero_flag[prow] = 1;
                    nonzero_list[nz_count++] = prow;
                }
            }
        }

        /* Depth-first reach: solve L_{1:j-1, 1:j-1} * u = a_j for pattern.
         * For each previously factored column k < j where col_dense[k] != 0,
         * subtract L(:,k) * u_k from col_dense, discovering new fill-in. */
        for (size_t k = 0; k < j; k++) {
            if (col_dense[k] == 0.0) continue;

            lmmc_real_t ukj = col_dense[k];

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

        /* Partial pivoting: find largest magnitude in col_dense[j:n-1] */
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

            if (max_val == 0.0) {
                status = LMMC_STATUS_SINGULAR_MATRIX;
                goto cleanup;
            }

            /* Swap rows if needed */
            if (pivot_row != j) {
                lmmc_real_t tmp = col_dense[j];
                col_dense[j] = col_dense[pivot_row];
                col_dense[pivot_row] = tmp;

                /* Update row permutation */
                size_t orig_j = lu->row_perm[j];
                size_t orig_p = lu->row_perm[pivot_row];
                lu->row_perm[j] = orig_p;
                lu->row_perm[pivot_row] = orig_j;
                piv_inv[orig_p] = j;
                piv_inv[orig_j] = pivot_row;

                /* Swap L entries in previous columns */
                for (size_t k = 0; k < j; k++) {
                    size_t L_start = lu->L_col_ptr[k];
                    size_t L_end = lu->L_col_ptr[k + 1];
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

        /* Store U entries (rows 0..j) */
        lu->U_col_ptr[j] = lu->U_nnz;
        for (size_t i = 0; i <= j; i++) {
            if (col_dense[i] != 0.0) {
                status = sparse_ensure_capacity(&lu->U_row_idx, &lu->U_values,
                                                &lu->U_capacity, lu->U_nnz);
                if (status != LMMC_STATUS_OK) goto cleanup;
                lu->U_row_idx[lu->U_nnz] = i;
                lu->U_values[lu->U_nnz] = col_dense[i];
                lu->U_nnz++;
            }
        }

        /* Store L entries (rows j+1..n-1), divided by diagonal.
         * L_col_ptr[j] was already set at the top of this iteration. */
        {
            lmmc_real_t diag = col_dense[j];
            for (size_t i = j + 1; i < n; i++) {
                if (col_dense[i] != 0.0) {
                    status = sparse_ensure_capacity(&lu->L_row_idx, &lu->L_values,
                                                    &lu->L_capacity, lu->L_nnz);
                    if (status != LMMC_STATUS_OK) goto cleanup;
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
    if (csc_needs_free) lmmc_sparse_destroy(&csc);
    return status;
}
/* Sparse LU: Solve Phase                                                    */

/**
 * @brief Solve Ax = b using the factored LU with AMD permutation.
 *
 * The factorization is P*A*Q = L*U where:
 *   P = row permutation (partial pivoting)
 *   Q = column permutation (AMD)
 *
 * Solve: A*x = b  =>  P*A*Q * Q^{-1}*x = P*b
 *        L*U*y = P*b  where y = Q^{-1}*x
 *        x = Q*y
 */
lmmc_status_t lmmc_sparse_lu_solve(
    const lmmc_sparse_lu_t* lu,
    const lmmc_vec_t* b,
    lmmc_vec_t* x)
{
    size_t n;
    lmmc_real_t* work = NULL;

    if (lu == NULL || b == NULL || x == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    if (b->data == NULL || x->data == NULL) return LMMC_STATUS_INVALID_ARGUMENT;

    n = lu->n;
    if (b->size != n || x->size != n) return LMMC_STATUS_DIMENSION_MISMATCH;

    work = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    if (work == NULL) return LMMC_STATUS_ALLOCATION_FAILED;

    /* Apply row permutation: work = P * b */
    for (size_t i = 0; i < n; i++) {
        work[i] = b->data[lu->row_perm[i]];
    }

    /* Forward substitution: solve L * z = P*b */
    for (size_t j = 0; j < n; j++) {
        size_t L_start = lu->L_col_ptr[j];
        size_t L_end = lu->L_col_ptr[j + 1];
        for (size_t p = L_start; p < L_end; p++) {
            size_t i = lu->L_row_idx[p];
            work[i] -= lu->L_values[p] * work[j];
        }
    }

    /* Back substitution: solve U * y = z */
    for (size_t jj = 0; jj < n; jj++) {
        size_t j = n - 1 - jj;
        size_t U_start = lu->U_col_ptr[j];
        size_t U_end = lu->U_col_ptr[j + 1];

        /* Find diagonal */
        lmmc_real_t diag = 0.0;
        for (size_t p = U_start; p < U_end; p++) {
            if (lu->U_row_idx[p] == j) {
                diag = lu->U_values[p];
                break;
            }
        }
        if (diag == 0.0) {
            lmmc_free(work);
            return LMMC_STATUS_SINGULAR_MATRIX;
        }

        work[j] = work[j] / diag;

        /* Eliminate */
        for (size_t p = U_start; p < U_end; p++) {
            size_t i = lu->U_row_idx[p];
            if (i != j) {
                work[i] -= lu->U_values[p] * work[j];
            }
        }
    }

    /* Apply inverse column permutation: x = Q * y */
    /* col_perm[k] = original column that is now column k */
    /* So x[col_perm[k]] = y[k] */
    for (size_t k = 0; k < n; k++) {
        x->data[lu->col_perm[k]] = work[k];
    }

    lmmc_free(work);
    return LMMC_STATUS_OK;
}
/* Sparse LU: Destroy                                                        */

void lmmc_sparse_lu_destroy(lmmc_sparse_lu_t* lu)
{
    if (lu == NULL) return;
    if (lu->col_perm) lmmc_free(lu->col_perm);
    if (lu->col_perm_inv) lmmc_free(lu->col_perm_inv);
    if (lu->row_perm) lmmc_free(lu->row_perm);
    if (lu->etree) lmmc_free(lu->etree);
    if (lu->L_col_ptr) lmmc_free(lu->L_col_ptr);
    if (lu->U_col_ptr) lmmc_free(lu->U_col_ptr);
    if (lu->L_row_idx) lmmc_free(lu->L_row_idx);
    if (lu->L_values) lmmc_free(lu->L_values);
    if (lu->U_row_idx) lmmc_free(lu->U_row_idx);
    if (lu->U_values) lmmc_free(lu->U_values);
    lmmc_free(lu);
}
/* Sparse Cholesky: Symbolic Phase                                           */

lmmc_status_t lmmc_sparse_chol_symbolic(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_chol_t** out_chol)
{
    lmmc_sparse_chol_t* chol = NULL;
    lmmc_sparse_mat_t csc;
    int csc_needs_free = 0;
    size_t n;
    lmmc_status_t status;

    if (a == NULL || out_chol == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows == 0 || a->cols == 0) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows != a->cols) return LMMC_STATUS_INVALID_ARGUMENT;

    n = a->rows;

    status = ensure_csc(a, &csc, &csc_needs_free);
    if (status != LMMC_STATUS_OK) return status;

    /* Allocate Cholesky context */
    chol = (lmmc_sparse_chol_t*)lmmc_alloc(sizeof(lmmc_sparse_chol_t));
    if (chol == NULL) {
        if (csc_needs_free) lmmc_sparse_destroy(&csc);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(chol, 0, sizeof(lmmc_sparse_chol_t));
    chol->n = n;

    /* Allocate permutation and tree arrays */
    chol->perm = (size_t*)lmmc_alloc(n * sizeof(size_t));
    chol->perm_inv = (size_t*)lmmc_alloc(n * sizeof(size_t));
    chol->etree = (size_t*)lmmc_alloc(n * sizeof(size_t));
    chol->L_col_ptr = (size_t*)lmmc_alloc((n + 1) * sizeof(size_t));

    if (!chol->perm || !chol->perm_inv || !chol->etree || !chol->L_col_ptr) {
        lmmc_sparse_chol_destroy(chol);
        if (csc_needs_free) lmmc_sparse_destroy(&csc);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(chol->L_col_ptr, 0, (n + 1) * sizeof(size_t));

    /* Compute AMD permutation on the symmetric pattern */
    /* For Cholesky, the matrix is already symmetric, so use it directly */
    {
        size_t* sym_col_ptr = NULL;
        size_t* sym_row_idx = NULL;
        status = build_symmetric_csc(n, csc.row_ptr, csc.col_idx,
                                     &sym_col_ptr, &sym_row_idx);
        if (status != LMMC_STATUS_OK) {
            lmmc_sparse_chol_destroy(chol);
            if (csc_needs_free) lmmc_sparse_destroy(&csc);
            return status;
        }

        status = amd_reorder(n, sym_col_ptr, sym_row_idx,
                             chol->perm, chol->perm_inv);

        lmmc_free(sym_col_ptr);
        if (sym_row_idx) lmmc_free(sym_row_idx);

        if (status != LMMC_STATUS_OK) {
            lmmc_sparse_chol_destroy(chol);
            if (csc_needs_free) lmmc_sparse_destroy(&csc);
            return status;
        }
    }

    /* Build elimination tree */
    build_etree(n, csc.row_ptr, csc.col_idx, chol->etree);

    /* Allocate factor arrays with initial capacity */
    {
        size_t initial_capacity = (a->nnz > 64) ? a->nnz : 64;
        if (initial_capacity < n) initial_capacity = n;

        chol->L_row_idx = (size_t*)lmmc_alloc(initial_capacity * sizeof(size_t));
        chol->L_values = (lmmc_real_t*)lmmc_alloc(initial_capacity * sizeof(lmmc_real_t));

        if (!chol->L_row_idx || !chol->L_values) {
            lmmc_sparse_chol_destroy(chol);
            if (csc_needs_free) lmmc_sparse_destroy(&csc);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
        chol->L_capacity = initial_capacity;
    }

    chol->L_nnz = 0;

    if (csc_needs_free) lmmc_sparse_destroy(&csc);
    *out_chol = chol;
    return LMMC_STATUS_OK;
}
/* Sparse Cholesky: Numeric Phase                                            */

lmmc_status_t lmmc_sparse_chol_numeric(
    const lmmc_sparse_mat_t* a,
    lmmc_sparse_chol_t* chol)
{
    lmmc_sparse_mat_t csc;
    int csc_needs_free = 0;
    lmmc_status_t status = LMMC_STATUS_OK;
    size_t n;
    lmmc_real_t* col_dense = NULL;

    if (a == NULL || chol == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows != a->cols) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows != chol->n) return LMMC_STATUS_INVALID_ARGUMENT;

    n = chol->n;

    status = ensure_csc(a, &csc, &csc_needs_free);
    if (status != LMMC_STATUS_OK) return status;

    col_dense = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    if (col_dense == NULL) {
        if (csc_needs_free) lmmc_sparse_destroy(&csc);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    chol->L_nnz = 0;

    /* Left-looking Cholesky with AMD permutation */
    for (size_t j = 0; j < n; j++) {
        size_t orig_col = chol->perm[j]; /* original column */

        chol->L_col_ptr[j] = chol->L_nnz;

        /* Clear workspace */
        memset(col_dense, 0, n * sizeof(lmmc_real_t));

        /* Scatter permuted column: gather entries from original column orig_col,
         * but only the lower triangle in permuted ordering */
        {
            size_t col_start = csc.row_ptr[orig_col];
            size_t col_end = csc.row_ptr[orig_col + 1];
            for (size_t p = col_start; p < col_end; p++) {
                size_t orig_row = csc.col_idx[p];
                size_t perm_row = chol->perm_inv[orig_row];
                if (perm_row >= j) {
                    col_dense[perm_row] += csc.values[p];
                }
            }
        }

        /* Subtract contributions from previous columns of L */
        for (size_t k = 0; k < j; k++) {
            size_t L_start = chol->L_col_ptr[k];
            size_t L_end = chol->L_col_ptr[k + 1];
            lmmc_real_t Ljk = 0.0;

            /* Find L(j,k) */
            for (size_t p = L_start; p < L_end; p++) {
                size_t row = chol->L_row_idx[p];
                if (row == j) {
                    Ljk = chol->L_values[p];
                    break;
                }
                if (row > j) break;
            }
            if (Ljk == 0.0) continue;

            /* col_dense[i] -= L(i,k) * L(j,k) for i >= j */
            for (size_t p = L_start; p < L_end; p++) {
                size_t i = chol->L_row_idx[p];
                if (i >= j) {
                    col_dense[i] -= chol->L_values[p] * Ljk;
                }
            }
        }

        /* Compute L(:,j) */
        {
            lmmc_real_t diag = col_dense[j];
            lmmc_real_t sqrt_diag;

            if (diag <= 0.0) {
                status = LMMC_STATUS_NOT_POSITIVE_DEFINITE;
                goto chol_cleanup;
            }

            sqrt_diag = sqrt(diag);

            /* Store diagonal */
            status = sparse_ensure_capacity(&chol->L_row_idx, &chol->L_values,
                                            &chol->L_capacity, chol->L_nnz);
            if (status != LMMC_STATUS_OK) goto chol_cleanup;
            chol->L_row_idx[chol->L_nnz] = j;
            chol->L_values[chol->L_nnz] = sqrt_diag;
            chol->L_nnz++;

            /* Store sub-diagonal entries */
            for (size_t i = j + 1; i < n; i++) {
                if (col_dense[i] != 0.0) {
                    status = sparse_ensure_capacity(&chol->L_row_idx, &chol->L_values,
                                                    &chol->L_capacity, chol->L_nnz);
                    if (status != LMMC_STATUS_OK) goto chol_cleanup;
                    chol->L_row_idx[chol->L_nnz] = i;
                    chol->L_values[chol->L_nnz] = col_dense[i] / sqrt_diag;
                    chol->L_nnz++;
                }
            }
        }
    }

    chol->L_col_ptr[n] = chol->L_nnz;

chol_cleanup:
    if (col_dense) lmmc_free(col_dense);
    if (csc_needs_free) lmmc_sparse_destroy(&csc);
    return status;
}
/* Sparse Cholesky: Solve Phase                                              */

/**
 * @brief Solve A*x = b using Cholesky factorization with AMD permutation.
 *
 * P*A*P^T = L*L^T where P is the AMD permutation.
 * Solve: L*L^T * (P*x) = P*b
 */
lmmc_status_t lmmc_sparse_chol_solve(
    const lmmc_sparse_chol_t* chol,
    const lmmc_vec_t* b,
    lmmc_vec_t* x)
{
    size_t n;
    lmmc_real_t* work = NULL;

    if (chol == NULL || b == NULL || x == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    if (b->data == NULL || x->data == NULL) return LMMC_STATUS_INVALID_ARGUMENT;

    n = chol->n;
    if (b->size != n || x->size != n) return LMMC_STATUS_DIMENSION_MISMATCH;

    work = (lmmc_real_t*)lmmc_alloc(n * sizeof(lmmc_real_t));
    if (work == NULL) return LMMC_STATUS_ALLOCATION_FAILED;

    /* Apply permutation: work = P * b */
    for (size_t i = 0; i < n; i++) {
        work[i] = b->data[chol->perm[i]];
    }

    /* Forward substitution: solve L * z = P*b */
    for (size_t j = 0; j < n; j++) {
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

        work[j] = work[j] / diag;

        for (size_t p = L_start; p < L_end; p++) {
            if (p == diag_idx) continue;
            size_t i = chol->L_row_idx[p];
            work[i] -= chol->L_values[p] * work[j];
        }
    }

    /* Back substitution: solve L^T * y = z */
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

    /* Apply inverse permutation: x = P^T * y */
    /* perm[k] = original index that maps to position k */
    /* So x[perm[k]] = work[k] */
    for (size_t k = 0; k < n; k++) {
        x->data[chol->perm[k]] = work[k];
    }

    lmmc_free(work);
    return LMMC_STATUS_OK;
}
/* Sparse Cholesky: Destroy                                                  */

void lmmc_sparse_chol_destroy(lmmc_sparse_chol_t* chol)
{
    if (chol == NULL) return;
    if (chol->perm) lmmc_free(chol->perm);
    if (chol->perm_inv) lmmc_free(chol->perm_inv);
    if (chol->etree) lmmc_free(chol->etree);
    if (chol->L_col_ptr) lmmc_free(chol->L_col_ptr);
    if (chol->L_row_idx) lmmc_free(chol->L_row_idx);
    if (chol->L_values) lmmc_free(chol->L_values);
    lmmc_free(chol);
}
