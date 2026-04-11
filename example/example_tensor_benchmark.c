#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(_WIN32)
#include <windows.h>
#endif
#include "lmmc/lmmc.h"

typedef struct {
    size_t dim0;
    size_t dim1;
    size_t dim2;
    size_t iters;
    size_t iters_view;
    double alpha;
} bench_options_t;

typedef lmmc_status_t (*tensor_binary_op_t)(const lmmc_tensor_t*, const lmmc_tensor_t*, lmmc_tensor_t*);

static double now_seconds(void) {
#if defined(_WIN32)
    static LARGE_INTEGER freq = {0};
    LARGE_INTEGER counter;
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static int mul_overflow_size(size_t a, size_t b, size_t* out) {
    if (a == 0 || b == 0) {
        *out = 0;
        return 0;
    }
    if (a > ((size_t)-1) / b) {
        return 1;
    }
    *out = a * b;
    return 0;
}

static int tensor_numel(const lmmc_tensor_t* t, size_t* out_numel) {
    size_t n = 0;
    if (t == NULL || out_numel == NULL) {
        return 1;
    }
    if (mul_overflow_size(t->dim0, t->dim1, &n) || mul_overflow_size(n, t->dim2, &n)) {
        return 1;
    }
    *out_numel = n;
    return 0;
}

static void parse_args(int argc, char** argv, bench_options_t* opts) {
    int i = 0;
    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dim0") == 0 && i + 1 < argc) {
            opts->dim0 = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--dim1") == 0 && i + 1 < argc) {
            opts->dim1 = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--dim2") == 0 && i + 1 < argc) {
            opts->dim2 = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            opts->iters = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--iters-view") == 0 && i + 1 < argc) {
            opts->iters_view = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
            opts->alpha = strtod(argv[++i], NULL);
        }
    }
}

static void init_tensor_data(lmmc_tensor_t* a, lmmc_tensor_t* b) {
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
    size_t idx = 0;
    for (i = 0; i < a->dim0; ++i) {
        for (j = 0; j < a->dim1; ++j) {
            for (k = 0; k < a->dim2; ++k) {
                double va = 0.5 + sin(0.01 * (double)idx);
                double vb = 1.0 + cos(0.02 * (double)idx);
                a->data[i * a->stride0 + j * a->stride1 + k * a->stride2] = va;
                b->data[i * b->stride0 + j * b->stride1 + k * b->stride2] = vb;
                ++idx;
            }
        }
    }
}

static double checksum_tensor(const lmmc_tensor_t* t) {
    size_t i = 0;
    size_t j = 0;
    size_t k = 0;
    double s = 0.0;
    for (i = 0; i < t->dim0; ++i) {
        for (j = 0; j < t->dim1; ++j) {
            for (k = 0; k < t->dim2; ++k) {
                s += t->data[i * t->stride0 + j * t->stride1 + k * t->stride2];
            }
        }
    }
    return s;
}

static double checksum_mat(const lmmc_mat_t* m) {
    size_t i = 0;
    size_t j = 0;
    double s = 0.0;
    for (i = 0; i < m->rows; ++i) {
        for (j = 0; j < m->cols; ++j) {
            s += m->data[i * m->stride + j];
        }
    }
    return s;
}

static void print_bench_line(
    const char* op,
    const bench_options_t* opts,
    size_t iters,
    double total_ms,
    size_t elem_count,
    double checksum
) {
    double ns_per_elem = 0.0;
    if (iters > 0 && elem_count > 0) {
        ns_per_elem = (total_ms * 1e6) / ((double)iters * (double)elem_count);
    }
    printf(
        "%s,%zu,%zu,%zu,%zu,%.6f,%.6f,%.12e\n",
        op,
        opts->dim0,
        opts->dim1,
        opts->dim2,
        iters,
        total_ms,
        ns_per_elem,
        checksum
    );
}

static int bench_binary_op(
    const char* op_name,
    const bench_options_t* opts,
    const lmmc_tensor_t* a,
    const lmmc_tensor_t* b,
    lmmc_tensor_t* out,
    tensor_binary_op_t op
) {
    size_t elem_count = 0;
    size_t i = 0;
    double t0 = 0.0;
    double t1 = 0.0;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (tensor_numel(a, &elem_count) != 0) {
        fprintf(stderr, "numel overflow in %s\n", op_name);
        return 1;
    }

    t0 = now_seconds();
    for (i = 0; i < opts->iters; ++i) {
        st = op(a, b, out);
        if (st != LMMC_STATUS_OK) {
            fprintf(stderr, "%s failed: %s\n", op_name, lmmc_status_string(st));
            return 1;
        }
    }
    t1 = now_seconds();

    print_bench_line(
        op_name,
        opts,
        opts->iters,
        (t1 - t0) * 1000.0,
        elem_count,
        checksum_tensor(out)
    );

    return 0;
}

static int bench_scale_op(
    const bench_options_t* opts,
    const lmmc_tensor_t* a,
    lmmc_tensor_t* out
) {
    size_t elem_count = 0;
    size_t i = 0;
    double t0 = 0.0;
    double t1 = 0.0;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (tensor_numel(a, &elem_count) != 0) {
        fprintf(stderr, "numel overflow in scale\n");
        return 1;
    }

    t0 = now_seconds();
    for (i = 0; i < opts->iters; ++i) {
        st = lmmc_tensor_scale(a, opts->alpha, out);
        if (st != LMMC_STATUS_OK) {
            fprintf(stderr, "scale failed: %s\n", lmmc_status_string(st));
            return 1;
        }
    }
    t1 = now_seconds();

    print_bench_line(
        "scale",
        opts,
        opts->iters,
        (t1 - t0) * 1000.0,
        elem_count,
        checksum_tensor(out)
    );

    return 0;
}

static int bench_sum_axis_op(
    const bench_options_t* opts,
    const lmmc_tensor_t* a,
    size_t axis,
    lmmc_mat_t* out,
    const char* op_name
) {
    size_t elem_count = 0;
    size_t i = 0;
    double t0 = 0.0;
    double t1 = 0.0;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (tensor_numel(a, &elem_count) != 0) {
        fprintf(stderr, "numel overflow in %s\n", op_name);
        return 1;
    }

    t0 = now_seconds();
    for (i = 0; i < opts->iters; ++i) {
        st = lmmc_tensor_sum_axis(a, axis, out);
        if (st != LMMC_STATUS_OK) {
            fprintf(stderr, "%s failed: %s\n", op_name, lmmc_status_string(st));
            return 1;
        }
    }
    t1 = now_seconds();

    print_bench_line(
        op_name,
        opts,
        opts->iters,
        (t1 - t0) * 1000.0,
        elem_count,
        checksum_mat(out)
    );

    return 0;
}

static int bench_reshape_view_op(
    const bench_options_t* opts,
    const lmmc_tensor_t* a
) {
    lmmc_tensor_t view = {0};
    size_t elem_count = 0;
    size_t new_dim0 = 0;
    size_t new_dim1 = 0;
    size_t new_dim2 = 1;
    size_t i = 0;
    double t0 = 0.0;
    double t1 = 0.0;
    double checksum = 0.0;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (tensor_numel(a, &elem_count) != 0) {
        fprintf(stderr, "numel overflow in reshape_view\n");
        return 1;
    }

    new_dim0 = a->dim0;
    if (mul_overflow_size(a->dim1, a->dim2, &new_dim1)) {
        fprintf(stderr, "shape overflow in reshape_view\n");
        return 1;
    }

    t0 = now_seconds();
    for (i = 0; i < opts->iters_view; ++i) {
        st = lmmc_tensor_reshape_view(a, new_dim0, new_dim1, new_dim2, &view);
        if (st != LMMC_STATUS_OK) {
            fprintf(stderr, "reshape_view failed: %s\n", lmmc_status_string(st));
            return 1;
        }
        checksum += (double)view.stride0 + (double)view.stride1 + (double)view.dim1;
    }
    t1 = now_seconds();

    print_bench_line(
        "reshape_view",
        opts,
        opts->iters_view,
        (t1 - t0) * 1000.0,
        elem_count,
        checksum
    );

    return 0;
}

static int bench_slice_view_op(
    const bench_options_t* opts,
    const lmmc_tensor_t* a
) {
    lmmc_tensor_t view = {0};
    size_t elem_count = 0;
    size_t i = 0;
    size_t end1 = a->dim1 / 2;
    double t0 = 0.0;
    double t1 = 0.0;
    double checksum = 0.0;
    lmmc_status_t st = LMMC_STATUS_OK;

    if (tensor_numel(a, &elem_count) != 0) {
        fprintf(stderr, "numel overflow in slice_view\n");
        return 1;
    }

    if (end1 == 0) {
        end1 = 1;
    }

    t0 = now_seconds();
    for (i = 0; i < opts->iters_view; ++i) {
        st = lmmc_tensor_slice_view(a, 0, a->dim0, 0, end1, 0, a->dim2, &view);
        if (st != LMMC_STATUS_OK) {
            fprintf(stderr, "slice_view failed: %s\n", lmmc_status_string(st));
            return 1;
        }
        checksum += (double)view.dim0 + (double)view.dim1 + (double)view.dim2;
    }
    t1 = now_seconds();

    print_bench_line(
        "slice_view",
        opts,
        opts->iters_view,
        (t1 - t0) * 1000.0,
        elem_count,
        checksum
    );

    return 0;
}

int main(int argc, char** argv) {
    bench_options_t opts = {64, 64, 8, 200, 200000, 0.75};
    lmmc_tensor_t a = {0};
    lmmc_tensor_t b = {0};
    lmmc_tensor_t out = {0};
    lmmc_mat_t axis0 = {0};
    lmmc_mat_t axis1 = {0};
    lmmc_mat_t axis2 = {0};
    lmmc_status_t st = LMMC_STATUS_OK;
    int rc = 0;

    parse_args(argc, argv, &opts);

    if (opts.dim0 == 0 || opts.dim1 == 0 || opts.dim2 == 0 || opts.iters == 0 || opts.iters_view == 0) {
        fprintf(stderr, "invalid benchmark arguments\n");
        return 1;
    }

    st = lmmc_tensor3_create(opts.dim0, opts.dim1, opts.dim2, &a);
    if (st != LMMC_STATUS_OK) {
        fprintf(stderr, "tensor3_create(a) failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor3_create(opts.dim0, opts.dim1, opts.dim2, &b);
    if (st != LMMC_STATUS_OK) {
        fprintf(stderr, "tensor3_create(b) failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_tensor3_create(opts.dim0, opts.dim1, opts.dim2, &out);
    if (st != LMMC_STATUS_OK) {
        fprintf(stderr, "tensor3_create(out) failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(opts.dim1, opts.dim2, &axis0);
    if (st != LMMC_STATUS_OK) {
        fprintf(stderr, "mat_create(axis0) failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(opts.dim0, opts.dim2, &axis1);
    if (st != LMMC_STATUS_OK) {
        fprintf(stderr, "mat_create(axis1) failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    st = lmmc_mat_create(opts.dim0, opts.dim1, &axis2);
    if (st != LMMC_STATUS_OK) {
        fprintf(stderr, "mat_create(axis2) failed: %s\n", lmmc_status_string(st));
        rc = 1;
        goto cleanup;
    }

    init_tensor_data(&a, &b);

    printf("op,dim0,dim1,dim2,iters,total_ms,ns_per_elem,checksum\n");

    if (bench_binary_op("add", &opts, &a, &b, &out, lmmc_tensor_add) != 0) {
        rc = 1;
        goto cleanup;
    }
    if (bench_binary_op("mul", &opts, &a, &b, &out, lmmc_tensor_mul) != 0) {
        rc = 1;
        goto cleanup;
    }
    if (bench_scale_op(&opts, &a, &out) != 0) {
        rc = 1;
        goto cleanup;
    }

    if (bench_sum_axis_op(&opts, &a, 0, &axis0, "sum_axis0") != 0) {
        rc = 1;
        goto cleanup;
    }
    if (bench_sum_axis_op(&opts, &a, 1, &axis1, "sum_axis1") != 0) {
        rc = 1;
        goto cleanup;
    }
    if (bench_sum_axis_op(&opts, &a, 2, &axis2, "sum_axis2") != 0) {
        rc = 1;
        goto cleanup;
    }

    if (bench_reshape_view_op(&opts, &a) != 0) {
        rc = 1;
        goto cleanup;
    }
    if (bench_slice_view_op(&opts, &a) != 0) {
        rc = 1;
        goto cleanup;
    }

cleanup:
    lmmc_mat_destroy(&axis2);
    lmmc_mat_destroy(&axis1);
    lmmc_mat_destroy(&axis0);
    lmmc_tensor_destroy(&out);
    lmmc_tensor_destroy(&b);
    lmmc_tensor_destroy(&a);
    return rc;
}
