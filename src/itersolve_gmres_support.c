#include "itersolve_gmres_internal.h"

static lmmc_vec_t lmmc_gmres_view(size_t n, lmmc_real_t** next) {
    lmmc_vec_t view = {n, *next, 0};
    *next += n;
    return view;
}

static lmmc_status_t lmmc_gmres_scalar_count(size_t n, size_t restart,
    size_t* h_count, size_t* scalar_count) {
    size_t basis_count, vector_count, rotations;
    if (!lmmc_safe_add_size(restart, 1, &basis_count) ||
        !lmmc_safe_mul_size(basis_count, restart, h_count)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_safe_add_size(basis_count, 6, &vector_count) ||
        !lmmc_safe_mul_size(vector_count, n, scalar_count)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_safe_mul_size(restart, 4, &rotations) ||
        !lmmc_safe_add_size(rotations, 1, &rotations)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (!lmmc_safe_add_size(*scalar_count, *h_count, scalar_count) ||
        !lmmc_safe_add_size(*scalar_count, rotations, scalar_count)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return LMMC_STATUS_OK;
}

static lmmc_status_t lmmc_gmres_allocation_bytes(size_t basis_count,
    size_t scalar_count, size_t* metadata, size_t* bytes) {
    if (!lmmc_safe_mul_size(basis_count, sizeof(lmmc_vec_t), metadata)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    const size_t alignment = _Alignof(lmmc_real_t);
    size_t padding = (alignment - *metadata % alignment) % alignment;
    if (!lmmc_safe_add_size(*metadata, padding, metadata) ||
        !lmmc_safe_mul_size(scalar_count, sizeof(lmmc_real_t), bytes) ||
        !lmmc_safe_add_size(*metadata, *bytes, bytes)) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_gmres_workspace(size_t n, size_t restart,
    lmmc_gmres_state_t* s) {
    size_t scalar_count, bytes, metadata;
    s->restart = restart == 0 ? (n < 30 ? n : 30) : restart;
    if (s->restart > n) {
        s->restart = n;
    }
    if (s->restart == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    lmmc_status_t st = lmmc_gmres_scalar_count(n, s->restart, &s->h_count, &scalar_count);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    size_t basis_count = s->restart + 1;
    st = lmmc_gmres_allocation_bytes(basis_count, scalar_count, &metadata, &bytes);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    s->basis = lmmc_memory_alloc(bytes);
    if (s->basis == NULL) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }
    memset(s->basis, 0, bytes);
    lmmc_real_t* next = (lmmc_real_t*)((unsigned char*)s->basis + metadata);
    s->r = lmmc_gmres_view(n, &next);
    s->z = lmmc_gmres_view(n, &next);
    s->w = lmmc_gmres_view(n, &next);
    s->ax = lmmc_gmres_view(n, &next);
    s->x_base = lmmc_gmres_view(n, &next);
    s->x_trial = lmmc_gmres_view(n, &next);
    for (size_t i = 0; i < basis_count; ++i) {
        s->basis[i] = lmmc_gmres_view(n, &next);
    }
    s->h = next;
    next += s->h_count;
    s->cs = next;
    next += s->restart;
    s->sn = next;
    next += s->restart;
    s->g = next;
    next += basis_count;
    s->y = next;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_gmres_arnoldi(const lmmc_sparse_mat_t* a,
    const lmmc_precond_t* precond, lmmc_gmres_state_t* s,
    size_t j, lmmc_real_t* h_next) {
    lmmc_status_t st = lmmc_sparse_mat_vec_mul(a, &s->basis[j], &s->ax);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    st = lmmc_apply_precond_or_identity(precond, &s->ax, &s->w);
    if (st != LMMC_STATUS_OK) {
        return st;
    }
    for (size_t i = 0; i <= j; ++i) {
        lmmc_real_t hij;
        st = lmmc_vec_dot_checked(&s->basis[i], &s->w, &hij);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
        s->h[i * s->restart + j] = hij;
        st = lmmc_vec_axpy(-hij, &s->basis[i], &s->w);
        if (st != LMMC_STATUS_OK) {
            return st;
        }
    }
    st = lmmc_vec_norm2_checked(&s->w, h_next);
    if (st == LMMC_STATUS_OK) {
        s->h[(j + 1) * s->restart + j] = *h_next;
    }
    return st;
}

static void lmmc_gmres_previous_rotations(lmmc_gmres_state_t* s, size_t j) {
    for (size_t i = 0; i < j; ++i) {
        lmmc_real_t h0 = s->h[i * s->restart + j];
        lmmc_real_t h1 = s->h[(i + 1) * s->restart + j];
        lmmc_real_t first = s->cs[i] * h0;
        lmmc_real_t second = s->sn[i] * h1;
        s->h[i * s->restart + j] = first + second;
        first = -s->sn[i] * h0;
        second = s->cs[i] * h1;
        s->h[(i + 1) * s->restart + j] = first + second;
    }
}

lmmc_status_t lmmc_gmres_rotate(lmmc_gmres_state_t* s, size_t j, int* breakdown) {
    *breakdown = 0;
    lmmc_gmres_previous_rotations(s, j);
    lmmc_real_t hj = s->h[j * s->restart + j];
    lmmc_real_t hsub = s->h[(j + 1) * s->restart + j];
    lmmc_real_t first = hj * hj;
    lmmc_real_t second = hsub * hsub;
    lmmc_real_t denom = sqrt(first + second);
    if (!isfinite(denom)) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    if (denom <= 1e-30) {
        *breakdown = 1;
        return LMMC_STATUS_OK;
    }
    s->cs[j] = hj / denom;
    s->sn[j] = hsub / denom;
    if (!isfinite(s->cs[j]) || !isfinite(s->sn[j])) {
        return LMMC_STATUS_NUMERICAL_FAILURE;
    }
    first = s->cs[j] * hj;
    second = s->sn[j] * hsub;
    s->h[j * s->restart + j] = first + second;
    s->h[(j + 1) * s->restart + j] = 0.0;
    lmmc_real_t gtmp = s->cs[j] * s->g[j];
    s->g[j + 1] = -s->sn[j] * s->g[j];
    s->g[j] = gtmp;
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_gmres_back_substitute(lmmc_gmres_state_t* s, size_t dim) {
    for (size_t ii = dim; ii > 0; --ii) {
        size_t i = ii - 1;
        lmmc_real_t sum = s->g[i];
        for (size_t j = i + 1; j < dim; ++j) {
            lmmc_real_t product = s->h[i * s->restart + j] * s->y[j];
            sum = sum - product;
        }
        lmmc_real_t diag = s->h[i * s->restart + i];
        if (!isfinite(sum) || !isfinite(diag) || fabs(diag) <= 1e-30) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        s->y[i] = sum / diag;
        if (!isfinite(s->y[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_gmres_trial(lmmc_gmres_state_t* s, size_t j) {
    for (size_t i = 0; i < s->x_base.size; ++i) {
        lmmc_real_t val = s->x_base.data[i];
        for (size_t k = 0; k <= j; ++k) {
            lmmc_real_t product = s->y[k] * s->basis[k].data[i];
            val = val + product;
        }
        if (!isfinite(val)) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
        s->x_trial.data[i] = val;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_gmres_normalize(lmmc_vec_t* out,
    const lmmc_vec_t* in, lmmc_real_t norm) {
    for (size_t i = 0; i < out->size; ++i) {
        out->data[i] = in->data[i] / norm;
        if (!isfinite(out->data[i])) {
            return LMMC_STATUS_NUMERICAL_FAILURE;
        }
    }
    return LMMC_STATUS_OK;
}
