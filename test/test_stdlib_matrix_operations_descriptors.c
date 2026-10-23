#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include <float.h>

#include "test_stdlib_common.h"
#include "internal_test_hooks.h"

enum output_kind { OUT_VEC, OUT_MAT, OUT_BOOL_VEC, OUT_BOOL_MAT,
                   OUT_EIG, OUT_SVD, OUT_EIG_TABLE, OUT_SVD_TABLE };

union allocating_output {
    lmmc_vec_t vec;
    lmmc_mat_t mat;
    lmmc_std_bool_vec_t bool_vec;
    lmmc_std_bool_mat_t bool_mat;
    lmmc_eigen_gen_full_result_t eig;
    lmmc_svd_result_t svd;
    lmmc_std_eig_table_t eig_table;
    lmmc_std_svd_table_t svd_table;
};

struct output_descriptor {
    size_t rows, cols, stride;
    void* data;
    int owns_data;
};

struct allocating_operation {
    const char* name;
    enum output_kind kind;
    lmmc_status_t (*call)(const lmmc_vec_t*, const lmmc_vec_t*,
                         const lmmc_mat_t*, const lmmc_mat_t*,
                         union allocating_output*);
};

#define ALLOCATING_OPERATIONS(X) \
    X(shape_vec, OUT_VEC, (&m[0], &out->vec)) \
    X(cross, OUT_VEC, (a, b, &out->vec)) \
    X(matvec, OUT_VEC, (&m[0], a, &out->vec)) \
    X(vec_add, OUT_VEC, (a, b, &out->vec)) \
    X(vec_add_scalar, OUT_VEC, (a, 2, &out->vec)) \
    X(vec_sub, OUT_VEC, (a, b, &out->vec)) \
    X(vec_sub_scalar, OUT_VEC, (a, 2, &out->vec)) \
    X(scalar_sub_vec, OUT_VEC, (2, a, &out->vec)) \
    X(vec_mul, OUT_VEC, (a, b, &out->vec)) \
    X(vec_div, OUT_VEC, (a, b, &out->vec)) \
    X(vec_div_scalar, OUT_VEC, (a, 2, &out->vec)) \
    X(scalar_div_vec, OUT_VEC, (2, a, &out->vec)) \
    X(vec_pow, OUT_VEC, (a, b, &out->vec)) \
    X(vec_pow_scalar, OUT_VEC, (a, 2, &out->vec)) \
    X(vec_scale, OUT_VEC, (a, 2, &out->vec)) \
    X(vec_mul_scalar, OUT_VEC, (a, 2, &out->vec)) \
    X(eye, OUT_MAT, (3, &out->mat)) \
    X(diag, OUT_MAT, (a, &out->mat)) \
    X(mat_add, OUT_MAT, (m, n, &out->mat)) \
    X(mat_add_scalar, OUT_MAT, (&m[0], 2, &out->mat)) \
    X(mat_sub, OUT_MAT, (m, n, &out->mat)) \
    X(mat_sub_scalar, OUT_MAT, (&m[0], 2, &out->mat)) \
    X(scalar_sub_mat, OUT_MAT, (2, &m[0], &out->mat)) \
    X(mat_mul_elem, OUT_MAT, (m, n, &out->mat)) \
    X(mat_mul_scalar, OUT_MAT, (&m[0], 2, &out->mat)) \
    X(mat_div, OUT_MAT, (m, n, &out->mat)) \
    X(mat_div_scalar, OUT_MAT, (&m[0], 2, &out->mat)) \
    X(scalar_div_mat, OUT_MAT, (2, &m[0], &out->mat)) \
    X(mat_pow_elem, OUT_MAT, (m, n, &out->mat)) \
    X(mat_pow_scalar, OUT_MAT, (&m[0], 2, &out->mat)) \
    X(mat_pow_int, OUT_MAT, (&m[0], 3, &out->mat)) \
    X(mat_scale, OUT_MAT, (&m[0], 2, &out->mat)) \
    X(matmul, OUT_MAT, (m, n, &out->mat)) \
    X(transpose, OUT_MAT, (&m[0], &out->mat)) \
    X(adjoint, OUT_MAT, (&m[0], &out->mat)) \
    X(inv, OUT_MAT, (&m[0], &out->mat)) \
    X(solve_left, OUT_MAT, (m, n, &out->mat)) \
    X(solve_right, OUT_MAT, (n, m, &out->mat)) \
    X(vec_compare, OUT_BOOL_VEC, (a, b, LMMC_STD_COMPARE_LT, &out->bool_vec)) \
    X(vec_compare_scalar, OUT_BOOL_VEC, (a, LMMC_STD_COMPARE_LT, 2, &out->bool_vec)) \
    X(mat_compare, OUT_BOOL_MAT, (m, n, LMMC_STD_COMPARE_LT, &out->bool_mat)) \
    X(mat_compare_scalar, OUT_BOOL_MAT, (&m[0], LMMC_STD_COMPARE_LT, 2, &out->bool_mat)) \
    X(eig, OUT_EIG, (&m[0], &out->eig)) \
    X(svd, OUT_SVD, (&m[0], &out->svd)) \
    X(eig_table, OUT_EIG_TABLE, (&m[0], &out->eig_table)) \
    X(svd_table, OUT_SVD_TABLE, (&m[0], &out->svd_table))

#define DEFINE_OPERATION(name, kind, args) \
    static lmmc_status_t call_##name(const lmmc_vec_t* a, const lmmc_vec_t* b, \
                                    const lmmc_mat_t* m, const lmmc_mat_t* n, \
                                    union allocating_output* out) \
    { \
        (void)a; (void)b; (void)m; (void)n; \
        return lmmc_std_linalg_##name args; \
    } \
    static const struct allocating_operation operation_##name = {#name, kind, call_##name};
ALLOCATING_OPERATIONS(DEFINE_OPERATION)
#undef DEFINE_OPERATION

static size_t output_count(enum output_kind kind)
{
    if (kind == OUT_EIG || kind == OUT_EIG_TABLE) { return 4; }
    if (kind == OUT_SVD || kind == OUT_SVD_TABLE) { return 3; }
    return 1;
}

static lmmc_vec_t* output_vec(union allocating_output* out,
                              enum output_kind kind, size_t index)
{
    if (kind == OUT_VEC) { return &out->vec; }
    if (kind == OUT_EIG && index < 2) {
        return index == 0 ? &out->eig.real_parts : &out->eig.imag_parts;
    }
    if (kind == OUT_SVD && index == 1) { return &out->svd.sigma; }
    return NULL;
}

static lmmc_mat_t* output_mat(union allocating_output* out,
                              enum output_kind kind, size_t index)
{
    switch (kind) {
    case OUT_MAT: return &out->mat;
    case OUT_EIG:
        if (index < 2) { return NULL; }
        return index == 2 ? &out->eig.vectors_real : &out->eig.vectors_imag;
    case OUT_SVD:
        if (index == 1) { return NULL; }
        return index == 0 ? &out->svd.U : &out->svd.Vt;
    case OUT_EIG_TABLE:
        switch (index) {
        case 0: return &out->eig_table.values_real;
        case 1: return &out->eig_table.values_imag;
        case 2: return &out->eig_table.vectors_real;
        default: return &out->eig_table.vectors_imag;
        }
    case OUT_SVD_TABLE:
        if (index == 0) { return &out->svd_table.U; }
        return index == 1 ? &out->svd_table.S : &out->svd_table.Vt;
    default: return NULL;
    }
}

static struct output_descriptor output_get(union allocating_output* out,
                                           enum output_kind kind, size_t index)
{
    lmmc_vec_t* vec = output_vec(out, kind, index);
    lmmc_mat_t* mat = output_mat(out, kind, index);
    if (vec) {
        return (struct output_descriptor){vec->size, 0, 0, vec->data, vec->owns_data};
    }
    if (mat) {
        return (struct output_descriptor){mat->rows, mat->cols, mat->stride,
                                          mat->data, mat->owns_data};
    }
    if (kind == OUT_BOOL_VEC) {
        return (struct output_descriptor){out->bool_vec.size, 0, 0,
                                          out->bool_vec.data, out->bool_vec.owns_data};
    }
    return (struct output_descriptor){out->bool_mat.rows, out->bool_mat.cols,
                                      out->bool_mat.stride, out->bool_mat.data,
                                      out->bool_mat.owns_data};
}

static void output_set(union allocating_output* out, enum output_kind kind,
                       size_t index, struct output_descriptor value)
{
    lmmc_vec_t* vec = output_vec(out, kind, index);
    lmmc_mat_t* mat = output_mat(out, kind, index);
    if (vec) {
        *vec = (lmmc_vec_t){value.rows, value.data, value.owns_data};
    } else if (mat) {
        *mat = (lmmc_mat_t){value.rows, value.cols, value.stride,
                           value.data, value.owns_data};
    } else if (kind == OUT_BOOL_VEC) {
        out->bool_vec = (lmmc_std_bool_vec_t){value.rows, value.data, value.owns_data};
    } else {
        out->bool_mat = (lmmc_std_bool_mat_t){value.rows, value.cols, value.stride,
                                            value.data, value.owns_data};
    }
}

static void output_clear(union allocating_output* out, enum output_kind kind)
{
    for (size_t i = 0; i < output_count(kind); ++i) {
        output_set(out, kind, i, (struct output_descriptor){0});
    }
}

static int output_equal(union allocating_output* a, union allocating_output* b,
                         enum output_kind kind)
{
    for (size_t i = 0; i < output_count(kind); ++i) {
        struct output_descriptor x = output_get(a, kind, i);
        struct output_descriptor y = output_get(b, kind, i);
        if (x.rows != y.rows || x.cols != y.cols || x.stride != y.stride ||
            x.data != y.data || x.owns_data != y.owns_data) { return 0; }
    }
    return 1;
}

static void output_destroy(union allocating_output* out, enum output_kind kind)
{
    switch (kind) {
    case OUT_VEC: lmmc_vec_destroy(&out->vec); break;
    case OUT_MAT: lmmc_mat_destroy(&out->mat); break;
    case OUT_BOOL_VEC: lmmc_std_bool_vec_destroy(&out->bool_vec); break;
    case OUT_BOOL_MAT: lmmc_std_bool_mat_destroy(&out->bool_mat); break;
    case OUT_EIG: lmmc_eigen_gen_full_result_destroy(&out->eig); break;
    case OUT_SVD: lmmc_svd_result_destroy(&out->svd); break;
    case OUT_EIG_TABLE: lmmc_std_eig_table_destroy(&out->eig_table); break;
    case OUT_SVD_TABLE: lmmc_std_svd_table_destroy(&out->svd_table); break;
    }
}

/* Keep the original owners reachable even when a rejected call overwrites out. */
static void output_destroy_changes(union allocating_output* out,
                                   union allocating_output* before,
                                   enum output_kind kind)
{
    for (size_t i = 0; i < output_count(kind); ++i) {
        struct output_descriptor current = output_get(out, kind, i);
        for (size_t j = 0; j < output_count(kind); ++j) {
            if (current.data == output_get(before, kind, j).data) {
                output_set(out, kind, i, (struct output_descriptor){0});
                break;
            }
        }
    }
    output_destroy(out, kind);
}

static int output_rejected(const struct allocating_operation* op,
                            const lmmc_vec_t* a, const lmmc_vec_t* b,
                            const lmmc_mat_t* m, union allocating_output* out)
{
    union allocating_output before = *out;
    lmmc_status_t status = op->call(a, b, m, &m[1], out);
    int valid = status == LMMC_STATUS_INVALID_ARGUMENT &&
                output_equal(out, &before, op->kind);
    output_destroy_changes(out, &before, op->kind);
    *out = before;
    return valid;
}

static int allocating_output_contract(const struct allocating_operation* op)
{
    lmmc_real_t av[] = {1, 2, 3}, bv[] = {2, 3, 4};
    lmmc_real_t mv[] = {4, 1, 1, 1, 3, 1, 1, 1, 2};
    lmmc_vec_t a = {3, av, 0}, b = {3, bv, 0};
    lmmc_mat_t m[] = {{3, 3, 3, mv, 0}, {3, 3, 3, mv, 0}};
    union allocating_output out, empty;
    output_clear(&empty, op->kind);
    int valid = 1;
#ifdef LMMC_DEBUG_LEAKS
    const long long baseline = lmmc_debug_leaks_get_count();
#endif

    for (size_t i = 0; i < output_count(op->kind); ++i) {
        int matrix = output_mat(&empty, op->kind, i) != NULL ||
                     op->kind == OUT_BOOL_MAT;
        for (size_t field = 0; field < (matrix ? 5u : 3u); ++field) {
            struct output_descriptor dirty = {0};
            out = empty;
            if (field == 0) { dirty.rows = 1; }
            else if (matrix && field == 1) { dirty.cols = 1; }
            else if (matrix && field == 2) { dirty.stride = 1; }
            else if (field == (matrix ? 3u : 1u)) { dirty.data = av; }
            else { dirty.owns_data = 1; }
            output_set(&out, op->kind, i, dirty);
            valid &= output_rejected(op, &a, &b, m, &out);
        }
    }

    out = empty;
    lmmc_status_t status = op->call(&a, &b, m, &m[1], &out);
    valid &= status == LMMC_STATUS_OK;
    if (status == LMMC_STATUS_OK) {
        for (size_t i = 0; i < output_count(op->kind); ++i) {
            struct output_descriptor value = output_get(&out, op->kind, i);
            valid &= value.data != NULL && value.owns_data == 1 &&
                     value.data != av && value.data != bv && value.data != mv;
            for (size_t j = 0; j < i; ++j) {
                valid &= value.data != output_get(&out, op->kind, j).data;
            }
        }
        valid &= output_rejected(op, &a, &b, m, &out);
        union allocating_output owner = out;
        for (size_t i = 0; i < output_count(op->kind); ++i) {
            struct output_descriptor view = output_get(&out, op->kind, i);
            view.owns_data = 0;
            output_set(&out, op->kind, i, view);
        }
        valid &= output_rejected(op, &a, &b, m, &out);
        out = owner;
        /* Use the actual output member as the input descriptor, not a copy. */
        lmmc_vec_t* alias_vec = output_vec(&out, op->kind, 0);
        if (alias_vec) { valid &= output_rejected(op, alias_vec, &b, m, &out); }
        for (size_t i = 0; i < output_count(op->kind); ++i) {
            lmmc_mat_t* alias_mat = output_mat(&out, op->kind, i);
            if (alias_mat) {
                union allocating_output before = out;
                status = op->call(&a, &b, alias_mat, &m[1], &out);
                valid &= status == LMMC_STATUS_INVALID_ARGUMENT &&
                         output_equal(&out, &before, op->kind);
                output_destroy_changes(&out, &before, op->kind);
                out = before;
            }
        }
    }
    output_destroy(&out, op->kind);

    for (size_t fail_after = 0;; ++fail_after) {
        out = empty;
        lmmc_memory_fail_after_for_test(fail_after);
        status = op->call(&a, &b, m, &m[1], &out);
        lmmc_memory_fail_reset_for_test();
        if (status != LMMC_STATUS_OK) {
            valid &= status == LMMC_STATUS_ALLOCATION_FAILED &&
                     output_equal(&out, &empty, op->kind);
        }
        output_destroy(&out, op->kind);
#ifdef LMMC_DEBUG_LEAKS
        valid &= lmmc_debug_leaks_get_count() == baseline;
#endif
        if (status != LMMC_STATUS_ALLOCATION_FAILED) { break; }
    }
    valid &= status == LMMC_STATUS_OK;
    valid &= av[0] == 1 && av[1] == 2 && av[2] == 3 &&
             bv[0] == 2 && bv[1] == 3 && bv[2] == 4 &&
             mv[0] == 4 && mv[1] == 1 && mv[2] == 1 &&
             mv[3] == 1 && mv[4] == 3 && mv[5] == 1 &&
             mv[6] == 1 && mv[7] == 1 && mv[8] == 2;
    return valid;
}

#define DEFINE_CONTRACT_TEST(name, kind, args) \
    static void test_##name##_output_contract(void** state) \
    { \
        (void)state; \
        assert_true(allocating_output_contract(&operation_##name)); \
    }
ALLOCATING_OPERATIONS(DEFINE_CONTRACT_TEST)
#undef DEFINE_CONTRACT_TEST

static int error_outputs_preserved(void)
{
    lmmc_real_t av[] = {1, 2, 3}, zero[] = {0, 0, 0};
    lmmc_real_t huge[] = {DBL_MAX, DBL_MAX, DBL_MAX};
    lmmc_real_t singular[] = {1, 2, 3, 2, 4, 6, 3, 6, 9};
    lmmc_real_t matrix[] = {4, 1, 1, 1, 3, 1, 1, 1, 2};
    lmmc_vec_t a = {3, av, 0}, z = {3, zero, 0}, h = {3, huge, 0};
    lmmc_mat_t m[] = {{3, 3, 3, matrix, 0}, {3, 3, 3, matrix, 0}};
    lmmc_mat_t s[] = {{3, 3, 3, singular, 0}, {3, 3, 3, matrix, 0}};
    const struct allocating_operation* ops[] = {
        &operation_vec_div, &operation_scalar_div_vec, &operation_vec_mul,
        &operation_inv, &operation_solve_left, &operation_solve_right,
        &operation_mat_pow_int
    };
    int valid = 1;
    for (size_t i = 0; i < sizeof(ops) / sizeof(ops[0]); ++i) {
        const struct allocating_operation* op = ops[i];
        union allocating_output out, empty;
        output_clear(&empty, op->kind);
        out = empty;
        lmmc_status_t status;
        if (i < 2) { status = op->call(&z, &z, m, &m[1], &out); }
        else if (i == 2) { status = op->call(&h, &h, m, &m[1], &out); }
        else if (i == 6) {
            status = lmmc_std_linalg_mat_pow_int(&s[0], -1, &out.mat);
        } else { status = op->call(&a, &a, s, &s[1], &out); }
        valid &= status == (i < 3 ? LMMC_STATUS_NUMERICAL_FAILURE :
                                   LMMC_STATUS_SINGULAR_MATRIX) &&
                 output_equal(&out, &empty, op->kind);
        output_destroy(&out, op->kind);
    }
    return valid;
}

static void test_error_outputs_preserved(void** state)
{
    (void)state;
    assert_true(error_outputs_preserved());
}

struct test_fixture {
    lmmc_vec_t shape_result;
    lmmc_mat_t result;
    lmmc_vec_t matvec_result;
    lmmc_mat_t mat;
    lmmc_mat_t rhs;
    lmmc_std_bool_mat_t bool_mat_result;
    lmmc_mat_t rectangular;
    lmmc_mat_t singular;
};

static int setup(void **state) {
    *state = test_calloc(1, sizeof(struct test_fixture));
    return 0;
}

static int teardown(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_mat_destroy(&fixture->singular);
    lmmc_mat_destroy(&fixture->rectangular);
    lmmc_std_bool_mat_destroy(&fixture->bool_mat_result);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
    lmmc_vec_destroy(&fixture->matvec_result);
    lmmc_mat_destroy(&fixture->result);
    lmmc_vec_destroy(&fixture->shape_result);
    test_free(fixture);
    return 0;
}

static void test_shape_transforms_constructors_invalid_stride(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_mat_t invalid_stride = {0};
    size_t rows = 0;
    size_t cols = 0;
    invalid_stride.rows = 2;
    invalid_stride.cols = 2;
    invalid_stride.stride = 1;
    invalid_stride.data = matrix_values;
    invalid_stride.owns_data = 0;

    assert_true(lmmc_std_linalg_shape(&invalid_stride, &rows, &cols) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_shape_vec(&invalid_stride, &fixture->shape_result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_transpose(&invalid_stride, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_adjoint(&invalid_stride, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_trace(&invalid_stride, &out) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&invalid_stride);
    lmmc_vec_destroy(&fixture->shape_result);
}

static void test_products_invalid_stride(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    lmmc_real_t vec_short_values[] = {1, 2};
    lmmc_vec_t vec_short = {2, vec_short_values, 0};
    lmmc_mat_t invalid_stride = {0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);
    invalid_stride.rows = 2;
    invalid_stride.cols = 2;
    invalid_stride.stride = 1;
    invalid_stride.data = matrix_values;
    invalid_stride.owns_data = 0;

    assert_true(lmmc_std_linalg_matmul(&invalid_stride, &fixture->rhs, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_matmul(&fixture->mat, &invalid_stride, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_matvec(&invalid_stride, &vec_short, &fixture->matvec_result) ==
                LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&invalid_stride);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
    lmmc_vec_destroy(&fixture->matvec_result);
}

static void test_addition_scaling_invalid_stride(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    lmmc_mat_t invalid_stride = {0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);
    invalid_stride.rows = 2;
    invalid_stride.cols = 2;
    invalid_stride.stride = 1;
    invalid_stride.data = matrix_values;
    invalid_stride.owns_data = 0;

    assert_true(lmmc_std_linalg_mat_add(&invalid_stride, &fixture->rhs, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_add(&fixture->mat, &invalid_stride, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_add_scalar(&invalid_stride, 10, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_scale(&invalid_stride, 2, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&invalid_stride);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_subtraction_invalid_stride(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    lmmc_mat_t invalid_stride = {0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);
    invalid_stride.rows = 2;
    invalid_stride.cols = 2;
    invalid_stride.stride = 1;
    invalid_stride.data = matrix_values;
    invalid_stride.owns_data = 0;

    assert_true(lmmc_std_linalg_mat_sub(&invalid_stride, &fixture->rhs, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_sub(&fixture->mat, &invalid_stride, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_sub_scalar(&invalid_stride, 1, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&invalid_stride);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_multiplication_invalid_stride(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    lmmc_mat_t invalid_stride = {0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);
    invalid_stride.rows = 2;
    invalid_stride.cols = 2;
    invalid_stride.stride = 1;
    invalid_stride.data = matrix_values;
    invalid_stride.owns_data = 0;

    assert_true(lmmc_std_linalg_mat_mul_elem(&invalid_stride, &fixture->rhs, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_mul_elem(&fixture->mat, &invalid_stride, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&invalid_stride);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_division_invalid_stride(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    lmmc_mat_t invalid_stride = {0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);
    invalid_stride.rows = 2;
    invalid_stride.cols = 2;
    invalid_stride.stride = 1;
    invalid_stride.data = matrix_values;
    invalid_stride.owns_data = 0;

    assert_true(lmmc_std_linalg_mat_div(&invalid_stride, &fixture->rhs, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_div(&fixture->mat, &invalid_stride, &fixture->result) == LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_div_scalar(&invalid_stride, 2, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&invalid_stride);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_powers_invalid_stride(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    lmmc_mat_t invalid_stride = {0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);
    invalid_stride.rows = 2;
    invalid_stride.cols = 2;
    invalid_stride.stride = 1;
    invalid_stride.data = matrix_values;
    invalid_stride.owns_data = 0;

    assert_true(lmmc_std_linalg_mat_pow_elem(&invalid_stride, &fixture->rhs, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_pow_elem(&fixture->mat, &invalid_stride, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_pow_scalar(&invalid_stride, 2, &fixture->result) ==
                LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&invalid_stride);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_comparison_invalid_stride(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rhs_values[] = {5, 6, 7, 8};
    lmmc_mat_t invalid_stride = {0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(2, 2, &fixture->rhs) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rhs, rhs_values);
    invalid_stride.rows = 2;
    invalid_stride.cols = 2;
    invalid_stride.stride = 1;
    invalid_stride.data = matrix_values;
    invalid_stride.owns_data = 0;

    assert_true(lmmc_std_linalg_mat_compare(
                    &invalid_stride,
                    &fixture->rhs,
                    LMMC_STD_COMPARE_EQ,
                    &fixture->bool_mat_result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_compare(
                    &fixture->mat,
                    &invalid_stride,
                    LMMC_STD_COMPARE_EQ,
                    &fixture->bool_mat_result) ==
                LMMC_STATUS_INVALID_ARGUMENT);
    assert_true(lmmc_std_linalg_mat_compare_scalar(
                    &invalid_stride,
                    LMMC_STD_COMPARE_GT,
                    1,
                    &fixture->bool_mat_result) ==
                LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&invalid_stride);
    lmmc_mat_destroy(&fixture->rhs);
    lmmc_mat_destroy(&fixture->mat);
    lmmc_std_bool_mat_destroy(&fixture->bool_mat_result);
}

static void test_addition_scaling_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_mat_add(&fixture->mat, &fixture->rectangular, &fixture->result) == LMMC_STATUS_DIMENSION_MISMATCH);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rectangular);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_subtraction_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_mat_sub(&fixture->mat, &fixture->rectangular, &fixture->result) == LMMC_STATUS_DIMENSION_MISMATCH);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rectangular);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_multiplication_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_mat_mul_elem(&fixture->mat, &fixture->rectangular, &fixture->result) ==
                LMMC_STATUS_DIMENSION_MISMATCH);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rectangular);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_division_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_mat_div(&fixture->mat, &fixture->rectangular, &fixture->result) == LMMC_STATUS_DIMENSION_MISMATCH);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rectangular);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_powers_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_mat_pow_elem(&fixture->mat, &fixture->rectangular, &fixture->result) ==
                LMMC_STATUS_DIMENSION_MISMATCH);
    assert_true(lmmc_std_linalg_mat_pow_int(&fixture->rectangular, 2, &fixture->result) == LMMC_STATUS_DIMENSION_MISMATCH);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rectangular);
    lmmc_mat_destroy(&fixture->mat);
}

static void test_comparison_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_mat_compare(&fixture->mat, &fixture->rectangular, LMMC_STD_COMPARE_EQ, &fixture->bool_mat_result) ==
                LMMC_STATUS_DIMENSION_MISMATCH);

    lmmc_mat_destroy(&fixture->rectangular);
    lmmc_mat_destroy(&fixture->mat);
    lmmc_std_bool_mat_destroy(&fixture->bool_mat_result);
}

static void test_products_dimension_mismatch(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t matrix_values[] = {1, 2, 3, 4};
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    lmmc_real_t vec_a_values[] = {1, 2, 3};
    lmmc_vec_t vec_a = {3, vec_a_values, 0};
    assert_true(lmmc_mat_create(2, 2, &fixture->mat) == LMMC_STATUS_OK);
    set_mat_values(&fixture->mat, matrix_values);
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_matmul(&fixture->mat, &fixture->rectangular, &fixture->result) == LMMC_STATUS_DIMENSION_MISMATCH);
    assert_true(lmmc_std_linalg_matvec(&fixture->mat, &vec_a, &fixture->matvec_result) == LMMC_STATUS_DIMENSION_MISMATCH);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->rectangular);
    lmmc_mat_destroy(&fixture->mat);
    lmmc_vec_destroy(&fixture->matvec_result);
}

static void test_powers_singularity(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t singular_values[] = {1, 2, 2, 4};
    assert_true(lmmc_mat_create(2, 2, &fixture->singular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->singular, singular_values);

    assert_true(lmmc_std_linalg_mat_pow_int(&fixture->singular, -1, &fixture->result) == LMMC_STATUS_SINGULAR_MATRIX);

    lmmc_mat_destroy(&fixture->result);
    lmmc_mat_destroy(&fixture->singular);
}

static void test_shape_transforms_constructors_rectangular_errors(void **state) {
    struct test_fixture *fixture = *state;
    lmmc_real_t out = 0;
    lmmc_real_t rectangular_values[] = {1, 2, 2, 4, 0, 0};
    assert_true(lmmc_mat_create(3, 2, &fixture->rectangular) == LMMC_STATUS_OK);
    set_mat_values(&fixture->rectangular, rectangular_values);

    assert_true(lmmc_std_linalg_trace(&fixture->rectangular, &out) == LMMC_STATUS_INVALID_ARGUMENT);

    lmmc_mat_destroy(&fixture->rectangular);
}

int main(void) {
    const struct CMUnitTest tests[] = {
#define REGISTER_CONTRACT_TEST(name, kind, args) cmocka_unit_test(test_##name##_output_contract),
        ALLOCATING_OPERATIONS(REGISTER_CONTRACT_TEST)
#undef REGISTER_CONTRACT_TEST
        cmocka_unit_test(test_error_outputs_preserved),
        cmocka_unit_test_setup_teardown(test_shape_transforms_constructors_invalid_stride, setup, teardown),
        cmocka_unit_test_setup_teardown(test_products_invalid_stride, setup, teardown),
        cmocka_unit_test_setup_teardown(test_addition_scaling_invalid_stride, setup, teardown),
        cmocka_unit_test_setup_teardown(test_subtraction_invalid_stride, setup, teardown),
        cmocka_unit_test_setup_teardown(test_multiplication_invalid_stride, setup, teardown),
        cmocka_unit_test_setup_teardown(test_division_invalid_stride, setup, teardown),
        cmocka_unit_test_setup_teardown(test_powers_invalid_stride, setup, teardown),
        cmocka_unit_test_setup_teardown(test_comparison_invalid_stride, setup, teardown),
        cmocka_unit_test_setup_teardown(test_addition_scaling_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_subtraction_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_multiplication_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_division_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_powers_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_comparison_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_products_dimension_mismatch, setup, teardown),
        cmocka_unit_test_setup_teardown(test_powers_singularity, setup, teardown),
        cmocka_unit_test_setup_teardown(test_shape_transforms_constructors_rectangular_errors, setup, teardown),
    };
    return cmocka_run_group_tests(tests, NULL, NULL);
}
