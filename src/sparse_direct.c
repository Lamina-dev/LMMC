/**
 * @file sparse_direct.c
 * 稀疏 LU / Cholesky 直接分解，含残余度重排序和稠密工作区左看数值分解。
 */
#include <string.h>
#include <math.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/sparse.h"


struct lmmc_sparse_lu_t {
    size_t n;
    size_t* col_perm;      /* 残余度列置换:新索引 -> 原索引 */
    size_t* row_perm;      /* 部分主元法产生的行置换 */
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
};


struct lmmc_sparse_chol_t {
    size_t n;
    size_t* perm;          /* 残余度置换:新索引 -> 原索引 */
    size_t* perm_inv;      /* 逆置换:原索引 -> 新索引 */
    size_t* L_col_ptr;
    size_t* L_row_idx;
    lmmc_real_t* L_values;
    size_t L_nnz;
    size_t L_capacity;
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
    size_t new_cap;
    size_t idx_bytes;
    size_t value_bytes;

    if (needed < *capacity) {
        return LMMC_STATUS_OK;
    }
    if (!lmmc_safe_mul_size(*capacity, 2, &new_cap) || new_cap <= needed) {
        if (!lmmc_safe_add_size(needed, 1, &new_cap)) {
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
    }
    if (!lmmc_safe_mul_size(new_cap, sizeof(size_t), &idx_bytes) ||
        !lmmc_safe_mul_size(new_cap, sizeof(lmmc_real_t), &value_bytes)) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    size_t* new_idx = (size_t*)lmmc_realloc(*idx, idx_bytes);
    if (new_idx == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    *idx = new_idx;
    lmmc_real_t* new_vals = (lmmc_real_t*)lmmc_realloc(*vals, value_bytes);
    if (new_vals == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    *vals = new_vals;
    *capacity = new_cap;
    return LMMC_STATUS_OK;
}
/* 基于桶的贪心残余度重排序                                               */

static void degree_bucket_remove(size_t node, size_t degree,
                                 size_t* heads, size_t* next, size_t* prev)
{
    if (prev[node] != SIZE_MAX) {
        next[prev[node]] = next[node];
    } else {
        heads[degree] = next[node];
    }
    if (next[node] != SIZE_MAX) prev[next[node]] = prev[node];
    next[node] = SIZE_MAX;
    prev[node] = SIZE_MAX;
}

static void degree_bucket_insert(size_t node, size_t degree,
                                 size_t* heads, size_t* next, size_t* prev)
{
    next[node] = heads[degree];
    prev[node] = SIZE_MAX;
    if (heads[degree] != SIZE_MAX) prev[heads[degree]] = node;
    heads[degree] = node;
}

/**
 * 按当前残余度贪心选择顶点.度数桶支持常数时间更新;扫描桶时以原始索引
 * 最小者打破平局,保证结果确定.该算法不是近似最小度算法:消去顶点时仅
 * 删除其残余关联边,不模拟填充边.
 */
static lmmc_status_t bucketed_greedy_residual_degree_reorder(
    size_t n,
    const size_t* col_ptr,
    const size_t* row_idx,
    size_t* perm)
{
    size_t bytes;
    size_t* degree;
    size_t* heads;
    size_t* next;
    size_t* prev;
    unsigned char* eliminated;
    size_t current_min = SIZE_MAX;

    if (n == 0) return LMMC_STATUS_OK;
    if (!col_ptr || !perm) return LMMC_STATUS_INVALID_ARGUMENT;
    if (col_ptr[n] == 0) {
        for (size_t i = 0; i < n; ++i) perm[i] = i;
        return LMMC_STATUS_OK;
    }
    if (!row_idx) return LMMC_STATUS_INVALID_ARGUMENT;
    if (!lmmc_safe_mul_size(n, sizeof(size_t), &bytes)) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    degree = (size_t*)lmmc_alloc(bytes);
    heads = (size_t*)lmmc_alloc(bytes);
    next = (size_t*)lmmc_alloc(bytes);
    prev = (size_t*)lmmc_alloc(bytes);
    eliminated = (unsigned char*)lmmc_alloc(n);
    if (!degree || !heads || !next || !prev || !eliminated) {
        if (degree) lmmc_free(degree);
        if (heads) lmmc_free(heads);
        if (next) lmmc_free(next);
        if (prev) lmmc_free(prev);
        if (eliminated) lmmc_free(eliminated);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    memset(eliminated, 0, n);
    for (size_t i = 0; i < n; ++i) {
        heads[i] = SIZE_MAX;
        next[i] = SIZE_MAX;
        prev[i] = SIZE_MAX;
    }
    for (size_t i = 0; i < n; ++i) {
        degree[i] = col_ptr[i + 1] - col_ptr[i];
        degree_bucket_insert(i, degree[i], heads, next, prev);
        if (current_min == SIZE_MAX || degree[i] < current_min) {
            current_min = degree[i];
        }
    }

    for (size_t step = 0; step < n; ++step) {
        size_t node;
        size_t candidate;
        while (current_min < n && heads[current_min] == SIZE_MAX) ++current_min;
        if (current_min == n) {
            lmmc_free(degree);
            lmmc_free(heads);
            lmmc_free(next);
            lmmc_free(prev);
            lmmc_free(eliminated);
            return LMMC_STATUS_INVALID_ARGUMENT;
        }

        node = heads[current_min];
        for (candidate = next[node]; candidate != SIZE_MAX; candidate = next[candidate]) {
            if (candidate < node) node = candidate;
        }
        degree_bucket_remove(node, degree[node], heads, next, prev);
        eliminated[node] = 1;
        perm[step] = node;

        for (size_t p = col_ptr[node]; p < col_ptr[node + 1]; ++p) {
            const size_t neighbor = row_idx[p];
            size_t old_degree;
            if (eliminated[neighbor]) continue;
            old_degree = degree[neighbor];
            degree_bucket_remove(neighbor, old_degree, heads, next, prev);
            degree[neighbor] = old_degree - 1;
            degree_bucket_insert(neighbor, degree[neighbor], heads, next, prev);
            if (degree[neighbor] < current_min) current_min = degree[neighbor];
        }
    }

    lmmc_free(degree);
    lmmc_free(heads);
    lmmc_free(next);
    lmmc_free(prev);
    lmmc_free(eliminated);
    return LMMC_STATUS_OK;
}

/**
 * 以 CSC 形式构造 A + A^T 的去重对称简单图。
 * 丢弃自环，并用 size_t 标记数组原地压缩重复有向边；
 * 当 A(i,j) 与 A(j,i) 同时存在时产生的重复边也会被压缩。
 */
static lmmc_status_t build_symmetric_simple_graph(
    size_t n,
    const size_t* col_ptr,
    const size_t* row_idx,
    size_t** out_graph_col_ptr,
    size_t** out_graph_row_idx)
{
    size_t pointer_count;
    size_t pointer_bytes;
    size_t index_bytes;
    size_t* graph_col_ptr = NULL;
    size_t* graph_row_idx = NULL;
    size_t* cursor = NULL;
    size_t* stamp = NULL;
    size_t raw_nnz;
    size_t read_begin = 0;
    size_t write = 0;

    if (!lmmc_safe_add_size(n, 1, &pointer_count) ||
        !lmmc_safe_mul_size(pointer_count, sizeof(size_t), &pointer_bytes) ||
        !lmmc_safe_mul_size(n, sizeof(size_t), &index_bytes)) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    graph_col_ptr = (size_t*)lmmc_alloc(pointer_bytes);
    cursor = (size_t*)lmmc_alloc(index_bytes);
    stamp = (size_t*)lmmc_alloc(index_bytes);
    if (!graph_col_ptr || (n != 0 && (!cursor || !stamp))) {
        if (graph_col_ptr) lmmc_free(graph_col_ptr);
        if (cursor) lmmc_free(cursor);
        if (stamp) lmmc_free(stamp);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(graph_col_ptr, 0, pointer_bytes);

    for (size_t column = 0; column < n; ++column) {
        for (size_t p = col_ptr[column]; p < col_ptr[column + 1]; ++p) {
            const size_t row = row_idx[p];
            if (row >= n) {
                lmmc_free(graph_col_ptr);
                lmmc_free(cursor);
                lmmc_free(stamp);
                return LMMC_STATUS_INVALID_ARGUMENT;
            }
            if (row == column) continue;
            if (graph_col_ptr[column + 1] == SIZE_MAX ||
                graph_col_ptr[row + 1] == SIZE_MAX) {
                lmmc_free(graph_col_ptr);
                lmmc_free(cursor);
                lmmc_free(stamp);
                return LMMC_STATUS_ALLOCATION_FAILED;
            }
            ++graph_col_ptr[column + 1];
            ++graph_col_ptr[row + 1];
        }
    }
    for (size_t column = 0; column < n; ++column) {
        if (!lmmc_safe_add_size(graph_col_ptr[column],
                                graph_col_ptr[column + 1],
                                &graph_col_ptr[column + 1])) {
            lmmc_free(graph_col_ptr);
            lmmc_free(cursor);
            lmmc_free(stamp);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
    }
    raw_nnz = graph_col_ptr[n];
    if (raw_nnz == 0) {
        lmmc_free(cursor);
        lmmc_free(stamp);
        *out_graph_col_ptr = graph_col_ptr;
        *out_graph_row_idx = NULL;
        return LMMC_STATUS_OK;
    }
    if (!lmmc_safe_mul_size(raw_nnz, sizeof(size_t), &index_bytes)) {
        lmmc_free(graph_col_ptr);
        lmmc_free(cursor);
        lmmc_free(stamp);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    if (raw_nnz != 0) {
        graph_row_idx = (size_t*)lmmc_alloc(index_bytes);
        if (!graph_row_idx) {
            lmmc_free(graph_col_ptr);
            lmmc_free(cursor);
            lmmc_free(stamp);
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
    }

    memcpy(cursor, graph_col_ptr, n * sizeof(size_t));
    for (size_t column = 0; column < n; ++column) {
        for (size_t p = col_ptr[column]; p < col_ptr[column + 1]; ++p) {
            const size_t row = row_idx[p];
            if (row == column) continue;
            graph_row_idx[cursor[column]++] = row;
            graph_row_idx[cursor[row]++] = column;
        }
    }

    for (size_t i = 0; i < n; ++i) stamp[i] = SIZE_MAX;
    for (size_t column = 0; column < n; ++column) {
        const size_t read_end = graph_col_ptr[column + 1];
        graph_col_ptr[column] = write;
        for (size_t p = read_begin; p < read_end; ++p) {
            const size_t row = graph_row_idx[p];
            if (stamp[row] != column) {
                stamp[row] = column;
                graph_row_idx[write++] = row;
            }
        }
        read_begin = read_end;
    }
    graph_col_ptr[n] = write;

    lmmc_free(cursor);
    lmmc_free(stamp);
    *out_graph_col_ptr = graph_col_ptr;
    *out_graph_row_idx = graph_row_idx;
    return LMMC_STATUS_OK;
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

    /* 分配置换与因子索引数组。 */
    lu->col_perm = (size_t*)lmmc_alloc_array(n, sizeof(size_t));
    lu->row_perm = (size_t*)lmmc_alloc_array(n, sizeof(size_t));
    lu->L_col_ptr = (size_t*)lmmc_alloc_array_plus(n, 1, sizeof(size_t));
    lu->U_col_ptr = (size_t*)lmmc_alloc_array_plus(n, 1, sizeof(size_t));

    if (!lu->col_perm || !lu->row_perm ||
        !lu->L_col_ptr || !lu->U_col_ptr) {
        lmmc_sparse_lu_destroy(lu);
        if (csc_needs_free) lmmc_sparse_destroy(&csc);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    memset(lu->L_col_ptr, 0, (n + 1) * sizeof(size_t));
    memset(lu->U_col_ptr, 0, (n + 1) * sizeof(size_t));

    /* Initialize row_perm to identity (will be modified by partial pivoting) */
    for (size_t i = 0; i < n; i++) lu->row_perm[i] = i;

    /* 计算简单图 A+A^T 的残余度列置换. */
    {
        size_t* sym_col_ptr = NULL;
        size_t* sym_row_idx = NULL;
        status = build_symmetric_simple_graph(n, csc.row_ptr, csc.col_idx,
                                              &sym_col_ptr, &sym_row_idx);
        if (status != LMMC_STATUS_OK) {
            lmmc_sparse_lu_destroy(lu);
            if (csc_needs_free) lmmc_sparse_destroy(&csc);
            return status;
        }

        status = bucketed_greedy_residual_degree_reorder(
            n, sym_col_ptr, sym_row_idx, lu->col_perm);

        lmmc_free(sym_col_ptr);
        if (sym_row_idx) lmmc_free(sym_row_idx);

        if (status != LMMC_STATUS_OK) {
            lmmc_sparse_lu_destroy(lu);
            if (csc_needs_free) lmmc_sparse_destroy(&csc);
            return status;
        }
    }


    /* Allocate factor arrays with initial capacity */
    {
        size_t initial_capacity = (a->nnz > 64) ? a->nnz : 64;
        /* Heuristic: expect some fill-in */
        if (initial_capacity < n) initial_capacity = n;

        lu->L_row_idx = (size_t*)lmmc_alloc_array(
            initial_capacity, sizeof(size_t));
        lu->L_values = (lmmc_real_t*)lmmc_alloc_array(
            initial_capacity, sizeof(lmmc_real_t));
        lu->U_row_idx = (size_t*)lmmc_alloc_array(
            initial_capacity, sizeof(size_t));
        lu->U_values = (lmmc_real_t*)lmmc_alloc_array(
            initial_capacity, sizeof(lmmc_real_t));

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
/* 稀疏 LU：数值阶段                                                     */

/**
 * @brief 使用稠密列工作区执行左看稀疏 LU 分解.
 *
 * 每个置换后的输入列先散布到工作区,再由已存储的 L 列更新、选择主元,
 * 最后压缩为稀疏 L 和 U.
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
    size_t* piv_inv = NULL;

    if (a == NULL || lu == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows != a->cols) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows != lu->n) return LMMC_STATUS_INVALID_ARGUMENT;

    n = lu->n;

    status = ensure_csc(a, &csc, &csc_needs_free);
    if (status != LMMC_STATUS_OK) return status;

    /* 稠密数值工作区与逆行置换. */
    col_dense = (lmmc_real_t*)lmmc_alloc_array(n, sizeof(lmmc_real_t));
    piv_inv = (size_t*)lmmc_alloc_array(n, sizeof(size_t));

    if (!col_dense || !piv_inv) {
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

    /* 按残余度顺序执行左看分解. */
    for (size_t j = 0; j < n; j++) {
        size_t orig_col = lu->col_perm[j]; /* original column index */

        /* Set L_col_ptr[j] BEFORE the triangular solve so that
         * L_col_ptr[k+1] is valid when k = j-1. At this point L_nnz
         * equals the end of column j-1's entries. */
        lu->L_col_ptr[j] = lu->L_nnz;

        /* 清空稠密列工作区。 */
        memset(col_dense, 0, n * sizeof(lmmc_real_t));

        /* Scatter column orig_col of A into dense workspace (permuted rows) */
        {
            size_t col_start = csc.row_ptr[orig_col];
            size_t col_end = csc.row_ptr[orig_col + 1];
            for (size_t p = col_start; p < col_end; p++) {
                size_t row = csc.col_idx[p];
                size_t prow = piv_inv[row]; /* permuted row */
                col_dense[prow] += csc.values[p];
            }
        }

        /* 使用每个已分解且主元行非零的 L 列更新。 */
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
    if (piv_inv) lmmc_free(piv_inv);
    if (csc_needs_free) lmmc_sparse_destroy(&csc);
    return status;
}
/* Sparse LU: Solve Phase                                                    */

/**
 * @brief 使用残余度排序的 LU 因子求解 A*x = b.
 *
 * 分解满足 P*A*Q = L*U,其中 P 为部分主元行置换,Q 为残余度列置换.
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

    work = (lmmc_real_t*)lmmc_alloc_array(n, sizeof(lmmc_real_t));
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
    if (lu->row_perm) lmmc_free(lu->row_perm);
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

    /* 分配置换与因子索引数组. */
    chol->perm = (size_t*)lmmc_alloc_array(n, sizeof(size_t));
    chol->perm_inv = (size_t*)lmmc_alloc_array(n, sizeof(size_t));
    chol->L_col_ptr = (size_t*)lmmc_alloc_array_plus(n, 1, sizeof(size_t));

    if (!chol->perm || !chol->perm_inv || !chol->L_col_ptr) {
        lmmc_sparse_chol_destroy(chol);
        if (csc_needs_free) lmmc_sparse_destroy(&csc);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(chol->L_col_ptr, 0, (n + 1) * sizeof(size_t));

    /* 计算对称简单图的残余度置换. */
    {
        size_t* sym_col_ptr = NULL;
        size_t* sym_row_idx = NULL;
        status = build_symmetric_simple_graph(n, csc.row_ptr, csc.col_idx,
                                              &sym_col_ptr, &sym_row_idx);
        if (status != LMMC_STATUS_OK) {
            lmmc_sparse_chol_destroy(chol);
            if (csc_needs_free) lmmc_sparse_destroy(&csc);
            return status;
        }

        if (sym_row_idx) {
            status = bucketed_greedy_residual_degree_reorder(
                n, sym_col_ptr, sym_row_idx, chol->perm);
        } else {
            for (size_t i = 0; i < n; ++i) chol->perm[i] = i;
            status = LMMC_STATUS_OK;
        }

        lmmc_free(sym_col_ptr);
        if (sym_row_idx) lmmc_free(sym_row_idx);

        if (status != LMMC_STATUS_OK) {
            lmmc_sparse_chol_destroy(chol);
            if (csc_needs_free) lmmc_sparse_destroy(&csc);
            return status;
        }
    }
    for (size_t i = 0; i < n; ++i) chol->perm_inv[chol->perm[i]] = i;


    /* Allocate factor arrays with initial capacity */
    {
        size_t initial_capacity = (a->nnz > 64) ? a->nnz : 64;
        if (initial_capacity < n) initial_capacity = n;

        chol->L_row_idx = (size_t*)lmmc_alloc_array(
            initial_capacity, sizeof(size_t));
        chol->L_values = (lmmc_real_t*)lmmc_alloc_array(
            initial_capacity, sizeof(lmmc_real_t));

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

    col_dense = (lmmc_real_t*)lmmc_alloc_array(n, sizeof(lmmc_real_t));
    if (col_dense == NULL) {
        if (csc_needs_free) lmmc_sparse_destroy(&csc);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    /* 使用稠密列工作区执行左看 Cholesky 分解。 */
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
 * @brief 使用残余度排序的 Cholesky 因子求解 A*x = b.
 *
 * P*A*P^T = L*L^T,其中 P 为残余度置换.
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

    work = (lmmc_real_t*)lmmc_alloc_array(n, sizeof(lmmc_real_t));
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
    if (chol->L_col_ptr) lmmc_free(chol->L_col_ptr);
    if (chol->L_row_idx) lmmc_free(chol->L_row_idx);
    if (chol->L_values) lmmc_free(chol->L_values);
    lmmc_free(chol);
}
