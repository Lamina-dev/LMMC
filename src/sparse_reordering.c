#include <string.h>
#include <math.h>
#include "memory_bridge.h"
#include "internal.h"
#include "lmmc/sparse.h"
#include "sparse_direct_internal.h"


static void degree_bucket_remove(size_t node, size_t degree,
                                 size_t* heads, size_t* next, size_t* prev)
{
    if (prev[node] != SIZE_MAX) {
        next[prev[node]] = next[node];
    } else {
        heads[degree] = next[node];
    }
    if (next[node] != SIZE_MAX) {
        prev[next[node]] = prev[node];
    }
    next[node] = SIZE_MAX;
    prev[node] = SIZE_MAX;
}

static void degree_bucket_insert(size_t node, size_t degree,
                                 size_t* heads, size_t* next, size_t* prev)
{
    next[node] = heads[degree];
    prev[node] = SIZE_MAX;
    if (heads[degree] != SIZE_MAX) {
        prev[heads[degree]] = node;
    }
    heads[degree] = node;
}

/**
 * @brief 按当前残余度贪心选择顶点，度数桶支持常数时间更新。
 * 同度数时选择原始索引最小的顶点，保证结果确定。消去时仅删除残余关联边，
 * 不模拟填充边，区别于近似最小度算法。
 */
typedef struct {
    size_t* degree;
    size_t* heads;
    size_t* next;
    size_t* prev;
    unsigned char* eliminated;
} lmmc_sparse_degree_workspace_t;

static void degree_workspace_destroy(lmmc_sparse_degree_workspace_t* work)
{
    if (work->degree) {
        lmmc_memory_free(work->degree);
    }
    if (work->heads) {
        lmmc_memory_free(work->heads);
    }
    if (work->next) {
        lmmc_memory_free(work->next);
    }
    if (work->prev) {
        lmmc_memory_free(work->prev);
    }
    if (work->eliminated) {
        lmmc_memory_free(work->eliminated);
    }
}

static lmmc_status_t degree_workspace_create(size_t n, lmmc_sparse_degree_workspace_t* work)
{
    size_t bytes;
    if (!lmmc_safe_mul_size(n, sizeof(size_t), &bytes)) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    work->degree = (size_t*)lmmc_memory_alloc(bytes);
    work->heads = (size_t*)lmmc_memory_alloc(bytes);
    work->next = (size_t*)lmmc_memory_alloc(bytes);
    work->prev = (size_t*)lmmc_memory_alloc(bytes);
    work->eliminated = (unsigned char*)lmmc_memory_alloc(n);
    if (!work->degree || !work->heads || !work->next ||
        !work->prev || !work->eliminated) {
        degree_workspace_destroy(work);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    return LMMC_STATUS_OK;
}
static size_t eliminate_neighbors(size_t node, const size_t* col_ptr,
    const size_t* row_idx, lmmc_sparse_degree_workspace_t* work, size_t current_min)
{
    for (size_t p = col_ptr[node]; p < col_ptr[node + 1]; ++p) {
        const size_t neighbor = row_idx[p];
        if (work->eliminated[neighbor]) {
            continue;
        }
        size_t old_degree = work->degree[neighbor];
        degree_bucket_remove(neighbor, old_degree, work->heads, work->next, work->prev);
        work->degree[neighbor] = old_degree - 1;
        degree_bucket_insert(neighbor, work->degree[neighbor],
                             work->heads, work->next, work->prev);
        if (work->degree[neighbor] < current_min) {
            current_min = work->degree[neighbor];
        }
    }
    return current_min;
}

static lmmc_status_t reorder_buckets(size_t n, const size_t* col_ptr,
    const size_t* row_idx, size_t* perm, lmmc_sparse_degree_workspace_t* work)
{
    size_t current_min = SIZE_MAX;
    memset(work->eliminated, 0, n);
    for (size_t i = 0; i < n; ++i) {
        work->heads[i] = SIZE_MAX;
        work->next[i] = SIZE_MAX;
        work->prev[i] = SIZE_MAX;
    }
    for (size_t i = 0; i < n; ++i) {
        work->degree[i] = col_ptr[i + 1] - col_ptr[i];
        degree_bucket_insert(i, work->degree[i], work->heads, work->next, work->prev);
        if (current_min == SIZE_MAX || work->degree[i] < current_min) {
            current_min = work->degree[i];
        }
    }

    for (size_t step = 0; step < n; ++step) {
        size_t node;
        size_t candidate;
        while (current_min < n && work->heads[current_min] == SIZE_MAX) {
            ++current_min;
        }
        if (current_min == n) {
            return LMMC_STATUS_INVALID_ARGUMENT;
        }

        node = work->heads[current_min];
        for (candidate = work->next[node]; candidate != SIZE_MAX; candidate = work->next[candidate]) {
            if (candidate < node) {
                node = candidate;
            }
        }
        degree_bucket_remove(node, work->degree[node], work->heads, work->next, work->prev);
        work->eliminated[node] = 1;
        perm[step] = node;

        current_min = eliminate_neighbors(node, col_ptr, row_idx, work, current_min);
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t bucketed_greedy_residual_degree_reorder(
    size_t n,
    const size_t* col_ptr,
    const size_t* row_idx,
    size_t* perm)
{
    if (n == 0) {
        return LMMC_STATUS_OK;
    }
    if (!col_ptr || !perm) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (col_ptr[n] == 0) {
        for (size_t i = 0; i < n; ++i) perm[i] = i;
        return LMMC_STATUS_OK;
    }
    if (!row_idx) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_sparse_degree_workspace_t work = {0};
    lmmc_status_t status = degree_workspace_create(n, &work);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    status = reorder_buckets(n, col_ptr, row_idx, perm, &work);
    degree_workspace_destroy(&work);
    return status;
}
static lmmc_status_t count_graph_edges(size_t n, const size_t* col_ptr,
    const size_t* row_idx, size_t* graph_col_ptr)
{
    for (size_t column = 0; column < n; ++column) {
        for (size_t p = col_ptr[column]; p < col_ptr[column + 1]; ++p) {
            const size_t row = row_idx[p];
            if (row >= n) {
                return LMMC_STATUS_INVALID_ARGUMENT;
            }
            if (row == column) {
                continue;
            }
            if (graph_col_ptr[column + 1] == SIZE_MAX ||
                graph_col_ptr[row + 1] == SIZE_MAX) {
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
            return LMMC_STATUS_ALLOCATION_FAILED;
        }
    }
    return LMMC_STATUS_OK;
}

static void fill_graph_edges(size_t n, const size_t* col_ptr,
    const size_t* row_idx, const size_t* graph_col_ptr,
    size_t* graph_row_idx, size_t* cursor)
{
    memcpy(cursor, graph_col_ptr, n * sizeof(size_t));
    for (size_t column = 0; column < n; ++column) {
        for (size_t p = col_ptr[column]; p < col_ptr[column + 1]; ++p) {
            const size_t row = row_idx[p];
            if (row == column) {
                continue;
            }
            graph_row_idx[cursor[column]++] = row;
            graph_row_idx[cursor[row]++] = column;
        }
    }
}

static void compact_graph_edges(size_t n, size_t* graph_col_ptr,
    size_t* graph_row_idx, size_t* stamp)
{
    size_t read_begin = 0;
    size_t write = 0;
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
}

static lmmc_status_t allocate_graph_indices(size_t n, size_t** graph_col_ptr,
                                            size_t** cursor, size_t** stamp)
{
    size_t pointer_count, pointer_bytes, index_bytes;
    if (!lmmc_safe_add_size(n, 1, &pointer_count) ||
        !lmmc_safe_mul_size(pointer_count, sizeof(size_t), &pointer_bytes)) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    if (!lmmc_safe_mul_size(n, sizeof(size_t), &index_bytes)) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    *graph_col_ptr = (size_t*)lmmc_memory_alloc(pointer_bytes);
    *cursor = (size_t*)lmmc_memory_alloc(index_bytes);
    *stamp = (size_t*)lmmc_memory_alloc(index_bytes);
    if (!*graph_col_ptr || (n != 0 && (!*cursor || !*stamp))) {
        lmmc_memory_free(*graph_col_ptr);
        lmmc_memory_free(*cursor);
        lmmc_memory_free(*stamp);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(*graph_col_ptr, 0, pointer_bytes);
    return LMMC_STATUS_OK;
}

/**
 * @brief 以 CSC 构造 A + A^T 的去重对称简单图。
 * 丢弃自环，用 size_t 标记数组原地压缩重复有向边，包括 A(i,j) 与 A(j,i) 产生的重复边。
 */
static lmmc_status_t build_symmetric_simple_graph(
    size_t n,
    const size_t* col_ptr,
    const size_t* row_idx,
    size_t** out_graph_col_ptr,
    size_t** out_graph_row_idx)
{
    lmmc_status_t status;
    size_t index_bytes;
    size_t* graph_col_ptr = NULL;
    size_t* graph_row_idx = NULL;
    size_t* cursor = NULL;
    size_t* stamp = NULL;
    size_t raw_nnz;
    status = allocate_graph_indices(n, &graph_col_ptr, &cursor, &stamp);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
status = count_graph_edges(n, col_ptr, row_idx, graph_col_ptr);
    if (status != LMMC_STATUS_OK) {
        goto cleanup;
    }
    raw_nnz = graph_col_ptr[n];
    if (raw_nnz != 0) {
        status = LMMC_STATUS_ALLOCATION_FAILED;
        if (!lmmc_safe_mul_size(raw_nnz, sizeof(size_t), &index_bytes)) {
            goto cleanup;
        }
        graph_row_idx = (size_t*)lmmc_memory_alloc(index_bytes);
        if (!graph_row_idx) {
            goto cleanup;
        }
        fill_graph_edges(n, col_ptr, row_idx, graph_col_ptr, graph_row_idx, cursor);
        compact_graph_edges(n, graph_col_ptr, graph_row_idx, stamp);
    }
    *out_graph_col_ptr = graph_col_ptr;
    *out_graph_row_idx = graph_row_idx;
    status = LMMC_STATUS_OK;
cleanup:
    lmmc_memory_free(cursor);
    lmmc_memory_free(stamp);
    if (status != LMMC_STATUS_OK) {
        lmmc_memory_free(graph_col_ptr);
        if (graph_row_idx) {
            lmmc_memory_free(graph_row_idx);
        }
    }
    return status;
}

lmmc_status_t lmmc_sparse_residual_reorder(const lmmc_sparse_mat_t* csc,
                                         size_t* perm)
{
    size_t* sym_col_ptr = NULL;
    size_t* sym_row_idx = NULL;
    lmmc_status_t status = build_symmetric_simple_graph(
        csc->rows, csc->row_ptr, csc->col_idx, &sym_col_ptr, &sym_row_idx);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    status = bucketed_greedy_residual_degree_reorder(
        csc->rows, sym_col_ptr, sym_row_idx, perm);
    lmmc_memory_free(sym_col_ptr);
    if (sym_row_idx) {
        lmmc_memory_free(sym_row_idx);
    }
    return status;
}
