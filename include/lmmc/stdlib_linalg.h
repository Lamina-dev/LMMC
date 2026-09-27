/** @file stdlib_linalg.h */
#ifndef LMMC_STDLIB_LINALG_H
#define LMMC_STDLIB_LINALG_H

#include "lmmc/stdlib_types.h"
#include "lmmc/eigen.h"

#ifdef __cplusplus
extern "C" {
#endif

lmmc_status_t lmmc_std_linalg_shape(const lmmc_mat_t* a,
                                    size_t* out_rows,
                                    size_t* out_cols);
lmmc_status_t lmmc_std_linalg_shape_vec(const lmmc_mat_t* a,
                                        lmmc_vec_t* out);
lmmc_status_t lmmc_std_linalg_eye(size_t n, lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_diag(const lmmc_vec_t* diagonal,
                                   lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_dot(const lmmc_vec_t* a,
                                  const lmmc_vec_t* b,
                                  lmmc_real_t* out);
lmmc_status_t lmmc_std_linalg_cross(const lmmc_vec_t* a,
                                    const lmmc_vec_t* b,
                                    lmmc_vec_t* out);
lmmc_status_t lmmc_std_linalg_norm(const lmmc_vec_t* x,
                                   lmmc_real_t* out);
lmmc_status_t lmmc_std_linalg_vec_add(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out);
lmmc_status_t lmmc_std_linalg_vec_add_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out);
lmmc_status_t lmmc_std_linalg_vec_sub(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out);
lmmc_status_t lmmc_std_linalg_vec_sub_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out);
lmmc_status_t lmmc_std_linalg_scalar_sub_vec(lmmc_real_t scalar,
                                             const lmmc_vec_t* x,
                                             lmmc_vec_t* out);
lmmc_status_t lmmc_std_linalg_vec_mul(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out);
lmmc_status_t lmmc_std_linalg_vec_mul_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out);
lmmc_status_t lmmc_std_linalg_vec_div(const lmmc_vec_t* a,
                                      const lmmc_vec_t* b,
                                      lmmc_vec_t* out);
lmmc_status_t lmmc_std_linalg_vec_div_scalar(const lmmc_vec_t* x,
                                             lmmc_real_t scalar,
                                             lmmc_vec_t* out);
lmmc_status_t lmmc_std_linalg_scalar_div_vec(lmmc_real_t scalar,
                                             const lmmc_vec_t* x,
                                             lmmc_vec_t* out);
lmmc_status_t lmmc_std_linalg_vec_pow(const lmmc_vec_t* base,
                                      const lmmc_vec_t* exponent,
                                      lmmc_vec_t* out);
lmmc_status_t lmmc_std_linalg_vec_pow_scalar(const lmmc_vec_t* base,
                                             lmmc_real_t exponent,
                                             lmmc_vec_t* out);
lmmc_status_t lmmc_std_linalg_vec_scale(const lmmc_vec_t* x,
                                        lmmc_real_t alpha,
                                        lmmc_vec_t* out);
lmmc_status_t lmmc_std_linalg_mat_add(const lmmc_mat_t* a,
                                      const lmmc_mat_t* b,
                                      lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_mat_add_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_mat_sub(const lmmc_mat_t* a,
                                      const lmmc_mat_t* b,
                                      lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_mat_sub_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_scalar_sub_mat(lmmc_real_t scalar,
                                             const lmmc_mat_t* a,
                                             lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_mat_mul_elem(const lmmc_mat_t* a,
                                           const lmmc_mat_t* b,
                                           lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_mat_mul_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_mat_div(const lmmc_mat_t* a,
                                      const lmmc_mat_t* b,
                                      lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_mat_div_scalar(const lmmc_mat_t* a,
                                             lmmc_real_t scalar,
                                             lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_scalar_div_mat(lmmc_real_t scalar,
                                             const lmmc_mat_t* a,
                                             lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_mat_pow_elem(const lmmc_mat_t* base,
                                           const lmmc_mat_t* exponent,
                                           lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_mat_pow_scalar(const lmmc_mat_t* base,
                                             lmmc_real_t exponent,
                                             lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_mat_pow_int(const lmmc_mat_t* base,
                                          int64_t exponent,
                                          lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_mat_scale(const lmmc_mat_t* a,
                                        lmmc_real_t alpha,
                                        lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_vec_compare(const lmmc_vec_t* a,
                                          const lmmc_vec_t* b,
                                          lmmc_std_compare_op_t op,
                                          lmmc_std_bool_vec_t* out);
lmmc_status_t lmmc_std_linalg_vec_compare_scalar(const lmmc_vec_t* a,
                                                 lmmc_std_compare_op_t op,
                                                 lmmc_real_t scalar,
                                                 lmmc_std_bool_vec_t* out);
lmmc_status_t lmmc_std_linalg_mat_compare(const lmmc_mat_t* a,
                                          const lmmc_mat_t* b,
                                          lmmc_std_compare_op_t op,
                                          lmmc_std_bool_mat_t* out);
lmmc_status_t lmmc_std_linalg_mat_compare_scalar(const lmmc_mat_t* a,
                                                 lmmc_std_compare_op_t op,
                                                 lmmc_real_t scalar,
                                                 lmmc_std_bool_mat_t* out);
void lmmc_std_bool_vec_destroy(lmmc_std_bool_vec_t* vec);
void lmmc_std_bool_mat_destroy(lmmc_std_bool_mat_t* mat);
lmmc_status_t lmmc_std_linalg_matmul(const lmmc_mat_t* a,
                                     const lmmc_mat_t* b,
                                     lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_matvec(const lmmc_mat_t* a,
                                     const lmmc_vec_t* x,
                                     lmmc_vec_t* out);
lmmc_status_t lmmc_std_linalg_transpose(const lmmc_mat_t* a,
                                        lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_adjoint(const lmmc_mat_t* a,
                                      lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_det(const lmmc_mat_t* a,
                                  lmmc_real_t* out);
lmmc_status_t lmmc_std_linalg_inv(const lmmc_mat_t* a,
                                  lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_rank(const lmmc_mat_t* a,
                                   size_t* out_rank);
lmmc_status_t lmmc_std_linalg_trace(const lmmc_mat_t* a,
                                    lmmc_real_t* out);
lmmc_status_t lmmc_std_linalg_solve_left(const lmmc_mat_t* a,
                                         const lmmc_mat_t* b,
                                         lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_solve_right(const lmmc_mat_t* b,
                                          const lmmc_mat_t* a,
                                          lmmc_mat_t* out);
lmmc_status_t lmmc_std_linalg_eig(const lmmc_mat_t* a,
                                  lmmc_eigen_gen_full_result_t* out);
lmmc_status_t lmmc_std_linalg_svd(const lmmc_mat_t* a,
                                  lmmc_svd_result_t* out);

lmmc_status_t lmmc_std_linalg_eig_table(const lmmc_mat_t* a,
                                        lmmc_std_eig_table_t* out);
size_t lmmc_std_eig_table_count(const lmmc_std_eig_table_t* table);
const char* lmmc_std_eig_table_key(const lmmc_std_eig_table_t* table,
                                   size_t index);
const lmmc_mat_t* lmmc_std_eig_table_get(const lmmc_std_eig_table_t* table,
                                         const char* key);
void lmmc_std_eig_table_destroy(lmmc_std_eig_table_t* table);

lmmc_status_t lmmc_std_linalg_svd_table(const lmmc_mat_t* a,
                                        lmmc_std_svd_table_t* out);
size_t lmmc_std_svd_table_count(const lmmc_std_svd_table_t* table);
const char* lmmc_std_svd_table_key(const lmmc_std_svd_table_t* table,
                                   size_t index);
const lmmc_mat_t* lmmc_std_svd_table_get(const lmmc_std_svd_table_t* table,
                                         const char* key);
void lmmc_std_svd_table_destroy(lmmc_std_svd_table_t* table);

#ifdef __cplusplus
}
#endif

#endif
