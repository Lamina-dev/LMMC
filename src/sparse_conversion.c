/**
 * @file sparse_conversion.c
 * @brief 稀疏矩阵格式转换：稠密互转、转置与 CSR/CSC/COO 转换。
 */
#include <stdlib.h>
#include <string.h>

#include "memory_bridge.h"
#include "internal.h"
#include "sparse_internal.h"
#include "lmmc/sparse.h"

lmmc_status_t lmmc_sparse_from_dense(const lmmc_mat_t* dense, lmmc_real_t eps, lmmc_sparse_mat_t* out_sparse) {
    size_t i = 0, j = 0, nz = 0;
    lmmc_status_t st = LMMC_STATUS_OK;
    if (dense == NULL || out_sparse == NULL || dense->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_real_t zero; LMMC_REAL_INIT(&zero);
    lmmc_real_t abs_v; LMMC_REAL_INIT(&abs_v);
    LMMC_REAL_SET_D(&zero, 0.0);

    if (LMMC_REAL_CMP(&eps, &zero) < 0) {
        LMMC_REAL_CLEAR(&zero);
        LMMC_REAL_CLEAR(&abs_v);
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    for (i = 0; i < dense->rows; ++i) {
        for (j = 0; j < dense->cols; ++j) {
            LMMC_REAL_ABS(&abs_v, &dense->data[i * dense->stride + j]);
            if (LMMC_REAL_CMP(&abs_v, &eps) > 0) ++nz;
        }
    }

    st = lmmc_sparse_create_csr(dense->rows, dense->cols, nz, out_sparse);
    if (st != LMMC_STATUS_OK) {
        LMMC_REAL_CLEAR(&zero);
        LMMC_REAL_CLEAR(&abs_v);
        return st;
    }

    nz = 0;
    out_sparse->row_ptr[0] = 0;
    for (i = 0; i < dense->rows; ++i) {
        for (j = 0; j < dense->cols; ++j) {
            lmmc_real_t* v_ptr = &dense->data[i * dense->stride + j];
            LMMC_REAL_ABS(&abs_v, v_ptr);
            if (LMMC_REAL_CMP(&abs_v, &eps) > 0) {
                out_sparse->col_idx[nz] = j;
                LMMC_REAL_SET(&out_sparse->values[nz], v_ptr);
                ++nz;
            }
        }
        out_sparse->row_ptr[i + 1] = nz;
    }

    LMMC_REAL_CLEAR(&zero);
    LMMC_REAL_CLEAR(&abs_v);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_to_dense(const lmmc_sparse_mat_t* sparse, lmmc_mat_t* out_dense) {
    size_t i = 0, p = 0;
    lmmc_status_t st = lmmc_sparse_validate(sparse);

    if (st != LMMC_STATUS_OK || out_dense == NULL || out_dense->data == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (out_dense->rows != sparse->rows || out_dense->cols != sparse->cols) {
        return LMMC_STATUS_DIMENSION_MISMATCH;
    }

    lmmc_real_t zero; LMMC_REAL_INIT(&zero);
    LMMC_REAL_SET_D(&zero, 0.0);
    st = lmmc_mat_fill(out_dense, zero);
    if (st != LMMC_STATUS_OK) {
        LMMC_REAL_CLEAR(&zero);
        return st;
    }

    if (sparse->format == LMMC_SPARSE_CSR) {
        for (i = 0; i < sparse->rows; ++i) {
            for (p = sparse->row_ptr[i]; p < sparse->row_ptr[i + 1]; ++p) {
                LMMC_REAL_SET(&out_dense->data[i * out_dense->stride + sparse->col_idx[p]], &sparse->values[p]);
            }
        }
    } else {
        for (i = 0; i < sparse->cols; ++i) {
            for (p = sparse->row_ptr[i]; p < sparse->row_ptr[i + 1]; ++p) {
                LMMC_REAL_SET(&out_dense->data[sparse->col_idx[p] * out_dense->stride + i], &sparse->values[p]);
            }
        }
    }

    LMMC_REAL_CLEAR(&zero);
    return LMMC_STATUS_OK;
}
lmmc_status_t lmmc_sparse_transpose(const lmmc_sparse_mat_t* sparse, lmmc_sparse_mat_t* out_transposed) {
    size_t i = 0, p = 0;
    size_t* next = NULL;
    lmmc_status_t st = lmmc_sparse_validate(sparse);

    if (st != LMMC_STATUS_OK || out_transposed == NULL) return LMMC_STATUS_INVALID_ARGUMENT;

    st = lmmc_sparse_create(sparse->cols, sparse->rows, sparse->nnz, sparse->format, out_transposed);
    if (st != LMMC_STATUS_OK) return st;

    size_t outer_size_src = (sparse->format == LMMC_SPARSE_CSR) ? sparse->rows : sparse->cols;
    size_t inner_size_src = (sparse->format == LMMC_SPARSE_CSR) ? sparse->cols : sparse->rows;

    for (p = 0; p < sparse->nnz; ++p) {
        ++out_transposed->row_ptr[sparse->col_idx[p] + 1];
    }

    for (i = 0; i < inner_size_src; ++i) {
        out_transposed->row_ptr[i + 1] += out_transposed->row_ptr[i];
    }

    if (out_transposed->nnz == 0) return LMMC_STATUS_OK;

    next = (size_t*)lmmc_alloc_array(inner_size_src, sizeof(size_t));
    if (next == NULL) {
        lmmc_sparse_destroy(out_transposed);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(next, out_transposed->row_ptr, inner_size_src * sizeof(size_t));

    for (i = 0; i < outer_size_src; ++i) {
        for (p = sparse->row_ptr[i]; p < sparse->row_ptr[i + 1]; ++p) {
            size_t inner = sparse->col_idx[p];
            size_t dst = next[inner]++;
            out_transposed->col_idx[dst] = i;
            LMMC_REAL_SET(&out_transposed->values[dst], &sparse->values[p]);
        }
    }

    lmmc_free(next);
    return LMMC_STATUS_OK;
}
lmmc_status_t lmmc_sparse_to_csc(const lmmc_sparse_mat_t* src, lmmc_sparse_mat_t* dst) {
    if (src == NULL || dst == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    lmmc_status_t st = lmmc_sparse_validate(src);
    if (st != LMMC_STATUS_OK) return st;

    if (src->format == LMMC_SPARSE_CSC) {
        st = lmmc_sparse_create_csc(src->rows, src->cols, src->nnz, dst);
        if (st != LMMC_STATUS_OK) return st;
        memcpy(dst->row_ptr, src->row_ptr, (src->cols + 1) * sizeof(size_t));
        if (src->nnz > 0) {
            memcpy(dst->col_idx, src->col_idx, src->nnz * sizeof(size_t));
            for (size_t k = 0; k < src->nnz; ++k) {
                LMMC_REAL_SET(&dst->values[k], &src->values[k]);
            }
        }
        return LMMC_STATUS_OK;
    }

    st = lmmc_sparse_create_csc(src->rows, src->cols, src->nnz, dst);
    if (st != LMMC_STATUS_OK) return st;

    for (size_t p = 0; p < src->nnz; ++p) {
        ++dst->row_ptr[src->col_idx[p] + 1];
    }
    for (size_t i = 0; i < src->cols; ++i) {
        dst->row_ptr[i + 1] += dst->row_ptr[i];
    }

    size_t* next = (size_t*)lmmc_alloc_array(src->cols, sizeof(size_t));
    if (next == NULL) {
        lmmc_sparse_destroy(dst);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(next, dst->row_ptr, src->cols * sizeof(size_t));

    for (size_t i = 0; i < src->rows; ++i) {
        for (size_t p = src->row_ptr[i]; p < src->row_ptr[i + 1]; ++p) {
            size_t col = src->col_idx[p];
            size_t dest_idx = next[col]++;
            dst->col_idx[dest_idx] = i;
            LMMC_REAL_SET(&dst->values[dest_idx], &src->values[p]);
        }
    }
    lmmc_free(next);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_to_csr(const lmmc_sparse_mat_t* src, lmmc_sparse_mat_t* dst) {
    if (src == NULL || dst == NULL) return LMMC_STATUS_INVALID_ARGUMENT;
    lmmc_status_t st = lmmc_sparse_validate(src);
    if (st != LMMC_STATUS_OK) return st;

    if (src->format == LMMC_SPARSE_CSR) {
        st = lmmc_sparse_create_csr(src->rows, src->cols, src->nnz, dst);
        if (st != LMMC_STATUS_OK) return st;
        memcpy(dst->row_ptr, src->row_ptr, (src->rows + 1) * sizeof(size_t));
        if (src->nnz > 0) {
            memcpy(dst->col_idx, src->col_idx, src->nnz * sizeof(size_t));
            for (size_t k = 0; k < src->nnz; ++k) {
                LMMC_REAL_SET(&dst->values[k], &src->values[k]);
            }
        }
        return LMMC_STATUS_OK;
    }

    st = lmmc_sparse_create_csr(src->rows, src->cols, src->nnz, dst);
    if (st != LMMC_STATUS_OK) return st;

    for (size_t p = 0; p < src->nnz; ++p) {
        ++dst->row_ptr[src->col_idx[p] + 1];
    }
    for (size_t i = 0; i < src->rows; ++i) {
        dst->row_ptr[i + 1] += dst->row_ptr[i];
    }

    size_t* next = (size_t*)lmmc_alloc_array(src->rows, sizeof(size_t));
    if (next == NULL) {
        lmmc_sparse_destroy(dst);
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memcpy(next, dst->row_ptr, src->rows * sizeof(size_t));

    for (size_t i = 0; i < src->cols; ++i) {
        for (size_t p = src->row_ptr[i]; p < src->row_ptr[i + 1]; ++p) {
            size_t row = src->col_idx[p];
            size_t dest_idx = next[row]++;
            dst->col_idx[dest_idx] = i;
            LMMC_REAL_SET(&dst->values[dest_idx], &src->values[p]);
        }
    }
    lmmc_free(next);
    return LMMC_STATUS_OK;
}
typedef struct {
    size_t row;
    size_t col;
    size_t orig_idx;
} lmmc_coo_sort_entry_t;

static int lmmc_coo_cmp_row_col(const void* a, const void* b) {
    const lmmc_coo_sort_entry_t* ea = (const lmmc_coo_sort_entry_t*)a;
    const lmmc_coo_sort_entry_t* eb = (const lmmc_coo_sort_entry_t*)b;
    if (ea->row != eb->row) return (ea->row < eb->row) ? -1 : 1;
    if (ea->col != eb->col) return (ea->col < eb->col) ? -1 : 1;
    return 0;
}


static int lmmc_coo_cmp_col_row(const void* a, const void* b) {
    const lmmc_coo_sort_entry_t* ea = (const lmmc_coo_sort_entry_t*)a;
    const lmmc_coo_sort_entry_t* eb = (const lmmc_coo_sort_entry_t*)b;
    if (ea->col != eb->col) return (ea->col < eb->col) ? -1 : 1;
    if (ea->row != eb->row) return (ea->row < eb->row) ? -1 : 1;
    return 0;
}

lmmc_status_t lmmc_sparse_coo_to_csr(
    const lmmc_sparse_coo_t* coo,
    lmmc_sparse_mat_t* out_csr
) {
    lmmc_coo_sort_entry_t* entries = NULL;
    size_t unique_nnz = 0;
    size_t i;
    lmmc_status_t st;

    if (coo == NULL || out_csr == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }


    if (coo->nnz == 0) {
        return lmmc_sparse_create_csr(coo->rows, coo->cols, 0, out_csr);
    }


    entries = (lmmc_coo_sort_entry_t*)lmmc_alloc_array(
        coo->nnz, sizeof(lmmc_coo_sort_entry_t));
    if (entries == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    for (i = 0; i < coo->nnz; ++i) {
        entries[i].row = coo->row_idx[i];
        entries[i].col = coo->col_idx[i];
        entries[i].orig_idx = i;
    }


    qsort(entries, coo->nnz, sizeof(lmmc_coo_sort_entry_t), lmmc_coo_cmp_row_col);


    unique_nnz = 1;
    for (i = 1; i < coo->nnz; ++i) {
        if (entries[i].row != entries[i - 1].row || entries[i].col != entries[i - 1].col) {
            unique_nnz++;
        }
    }


    st = lmmc_sparse_create_csr(coo->rows, coo->cols, unique_nnz, out_csr);
    if (st != LMMC_STATUS_OK) {
        lmmc_free(entries);
        return st;
    }


    {
        size_t csr_idx = 0;
        lmmc_real_t sum; LMMC_REAL_INIT(&sum);
        lmmc_real_t tmp; LMMC_REAL_INIT(&tmp);


        memset(out_csr->row_ptr, 0, (coo->rows + 1) * sizeof(size_t));


        {
            size_t prev_row = entries[0].row;
            size_t prev_col = entries[0].col;
            out_csr->row_ptr[prev_row + 1]++;
            for (i = 1; i < coo->nnz; ++i) {
                if (entries[i].row != prev_row || entries[i].col != prev_col) {
                    out_csr->row_ptr[entries[i].row + 1]++;
                    prev_row = entries[i].row;
                    prev_col = entries[i].col;
                }
            }
        }


        for (i = 0; i < coo->rows; ++i) {
            out_csr->row_ptr[i + 1] += out_csr->row_ptr[i];
        }


        LMMC_REAL_SET(&sum, &coo->values[entries[0].orig_idx]);
        {
            size_t cur_row = entries[0].row;
            size_t cur_col = entries[0].col;

            for (i = 1; i < coo->nnz; ++i) {
                if (entries[i].row == cur_row && entries[i].col == cur_col) {

                    LMMC_REAL_ADD(&tmp, &sum, &coo->values[entries[i].orig_idx]);
                    LMMC_REAL_SET(&sum, &tmp);
                } else {

                    out_csr->col_idx[csr_idx] = cur_col;
                    LMMC_REAL_SET(&out_csr->values[csr_idx], &sum);
                    csr_idx++;


                    cur_row = entries[i].row;
                    cur_col = entries[i].col;
                    LMMC_REAL_SET(&sum, &coo->values[entries[i].orig_idx]);
                }
            }

            out_csr->col_idx[csr_idx] = cur_col;
            LMMC_REAL_SET(&out_csr->values[csr_idx], &sum);
        }

        LMMC_REAL_CLEAR(&sum);
        LMMC_REAL_CLEAR(&tmp);
    }

    lmmc_free(entries);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_sparse_coo_to_csc(
    const lmmc_sparse_coo_t* coo,
    lmmc_sparse_mat_t* out_csc
) {
    lmmc_coo_sort_entry_t* entries = NULL;
    size_t unique_nnz = 0;
    size_t i;
    lmmc_status_t st;

    if (coo == NULL || out_csc == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }


    if (coo->nnz == 0) {
        return lmmc_sparse_create_csc(coo->rows, coo->cols, 0, out_csc);
    }


    entries = (lmmc_coo_sort_entry_t*)lmmc_alloc_array(
        coo->nnz, sizeof(lmmc_coo_sort_entry_t));
    if (entries == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    for (i = 0; i < coo->nnz; ++i) {
        entries[i].row = coo->row_idx[i];
        entries[i].col = coo->col_idx[i];
        entries[i].orig_idx = i;
    }


    qsort(entries, coo->nnz, sizeof(lmmc_coo_sort_entry_t), lmmc_coo_cmp_col_row);


    unique_nnz = 1;
    for (i = 1; i < coo->nnz; ++i) {
        if (entries[i].row != entries[i - 1].row || entries[i].col != entries[i - 1].col) {
            unique_nnz++;
        }
    }


    st = lmmc_sparse_create_csc(coo->rows, coo->cols, unique_nnz, out_csc);
    if (st != LMMC_STATUS_OK) {
        lmmc_free(entries);
        return st;
    }


    {
        size_t csc_idx = 0;
        lmmc_real_t sum; LMMC_REAL_INIT(&sum);
        lmmc_real_t tmp; LMMC_REAL_INIT(&tmp);

        LMMC_REAL_SET(&sum, &coo->values[entries[0].orig_idx]);
        size_t cur_row = entries[0].row;
        size_t cur_col = entries[0].col;

        for (i = 1; i < coo->nnz; ++i) {
            if (entries[i].row == cur_row && entries[i].col == cur_col) {

                LMMC_REAL_ADD(&tmp, &sum, &coo->values[entries[i].orig_idx]);
                LMMC_REAL_SET(&sum, &tmp);
            } else {

                out_csc->col_idx[csc_idx] = cur_row;
                LMMC_REAL_SET(&out_csc->values[csc_idx], &sum);
                csc_idx++;


                cur_row = entries[i].row;
                cur_col = entries[i].col;
                LMMC_REAL_SET(&sum, &coo->values[entries[i].orig_idx]);
            }
        }

        out_csc->col_idx[csc_idx] = cur_row;
        LMMC_REAL_SET(&out_csc->values[csc_idx], &sum);


        memset(out_csc->row_ptr, 0, (coo->cols + 1) * sizeof(size_t));


        {
            size_t prev_row = entries[0].row;
            size_t prev_col = entries[0].col;

            out_csc->row_ptr[entries[0].col + 1]++;
            for (i = 1; i < coo->nnz; ++i) {
                if (entries[i].row != prev_row || entries[i].col != prev_col) {
                    out_csc->row_ptr[entries[i].col + 1]++;
                    prev_row = entries[i].row;
                    prev_col = entries[i].col;
                }
            }


            for (i = 0; i < coo->cols; ++i) {
                out_csc->row_ptr[i + 1] += out_csc->row_ptr[i];
            }
        }

        LMMC_REAL_CLEAR(&sum);
        LMMC_REAL_CLEAR(&tmp);
    }

    lmmc_free(entries);
    return LMMC_STATUS_OK;
}

void lmmc_sparse_coo_destroy(lmmc_sparse_coo_t* coo) {
    if (coo == NULL) {
        return;
    }
    if (coo->values != NULL) {
        for (size_t k = 0; k < coo->capacity; ++k) {
            LMMC_REAL_CLEAR(&coo->values[k]);
        }
        lmmc_free(coo->values);
    }
    if (coo->row_idx != NULL) lmmc_free(coo->row_idx);
    if (coo->col_idx != NULL) lmmc_free(coo->col_idx);
    coo->rows = 0;
    coo->cols = 0;
    coo->nnz = 0;
    coo->capacity = 0;
    coo->row_idx = NULL;
    coo->col_idx = NULL;
    coo->values = NULL;
}
