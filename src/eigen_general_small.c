#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "internal.h"
#include "eigen_internal.h"
#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/linear_algebra.h"
#include "eigen_general_internal.h"

static void real_null_vector(lmmc_real_t a00, lmmc_real_t a01,
    lmmc_real_t a10, lmmc_real_t a11, lmmc_real_t lam,
    lmmc_real_t* v0, lmmc_real_t* v1)
{
    lmmc_real_t r0_diag = lmmc_abs(a00 - lam);
    lmmc_real_t r1_diag = lmmc_abs(a11 - lam);
    lmmc_real_t r0_off = lmmc_abs(a01);
    lmmc_real_t r1_off = lmmc_abs(a10);
    if (r0_off > 1e-15 || r0_diag > 1e-15) {
        if (r0_off >= r0_diag || r0_off >= 1e-15) {
            *v0 = -a01;
            *v1 = a00 - lam;
        } else {
            *v0 = (r0_diag > 1e-15) ? 0.0 : 1.0;
            *v1 = (r0_diag > 1e-15) ? 1.0 : 0.0;
        }
        return;
    }
    if (r1_off > 1e-15 || r1_diag > 1e-15) {
        if (r1_diag > 1e-15) {
            *v0 = 1.0;
            *v1 = -a10 / (a11 - lam);
        } else {
            *v0 = 0.0;
            *v1 = 1.0;
        }
        return;
    }
    *v0 = 1.0;
    *v1 = 0.0;
}

static void real_vectors_2x2(lmmc_real_t a00, lmmc_real_t a01,
    lmmc_real_t a10, lmmc_real_t a11, lmmc_real_t matrix_scale,
    lmmc_eigen_gen_full_result_t* out_result)
{
    for (size_t i = 0; i < 2; i++) {
        lmmc_real_t lam = out_result->real_parts.data[i] / matrix_scale;
        lmmc_real_t v0, v1, nrm;
        real_null_vector(a00, a01, a10, a11, lam, &v0, &v1);
                nrm = hypot(v0, v1);
                if (nrm == 0.0) { v0 = 1.0; v1 = 0.0; nrm = 1.0; }
                MAT_ELEM(&out_result->vectors_real, 0, i) = v0 / nrm;
                MAT_ELEM(&out_result->vectors_real, 1, i) = v1 / nrm;
                MAT_ELEM(&out_result->vectors_imag, 0, i) = 0.0;
                MAT_ELEM(&out_result->vectors_imag, 1, i) = 0.0;
            }
}

static void complex_vectors_2x2(lmmc_real_t a00, lmmc_real_t a01,
    lmmc_real_t a10, lmmc_real_t a11, lmmc_real_t matrix_scale,
    lmmc_eigen_gen_full_result_t* out_result)
{
    lmmc_real_t tr = a00 + a11;
            lmmc_real_t sq =
                out_result->imag_parts.data[0] / matrix_scale;
            out_result->real_parts.data[0] = tr * 0.5 * matrix_scale;
            out_result->imag_parts.data[0] = sq * matrix_scale;
            out_result->real_parts.data[1] = tr * 0.5 * matrix_scale;
            out_result->imag_parts.data[1] = -sq * matrix_scale;

            /**
             * @brief 求共轭特征值对中 lam = tr/2 + i*sq 的特征向量。
             * 第 0 行满足 (a00 - tr/2 - i*sq)*v0 + a01*v1 = 0；
             * 第 1 行满足 a10*v0 + (a11 - tr/2 - i*sq)*v1 = 0。
             */
            lmmc_real_t re_part = tr * 0.5 - a00;
            if (lmmc_abs(a01) > lmmc_abs(a10)) {
                /** @brief 取特征向量 v = (a01, re_part + i*sq)。 */
                lmmc_real_t vr0 = a01;
                lmmc_real_t vr1 = re_part;
                lmmc_real_t vi0 = 0.0;
                lmmc_real_t vi1 = sq;
                lmmc_real_t nrm = hypot(hypot(vr0, vr1), hypot(vi0, vi1));
                if (nrm == 0.0) {
                    nrm = 1.0;
                }
                MAT_ELEM(&out_result->vectors_real, 0, 0) = vr0 / nrm;
                MAT_ELEM(&out_result->vectors_real, 1, 0) = vr1 / nrm;
                MAT_ELEM(&out_result->vectors_imag, 0, 0) = vi0 / nrm;
                MAT_ELEM(&out_result->vectors_imag, 1, 0) = vi1 / nrm;
            } else {
                /** @brief 取特征向量 v = (tr/2 - a11 + i*sq, a10)。 */
                lmmc_real_t vr0 = tr * 0.5 - a11;
                lmmc_real_t vr1 = a10;
                lmmc_real_t vi0 = sq;
                lmmc_real_t vi1 = 0.0;
                lmmc_real_t nrm = hypot(hypot(vr0, vr1), hypot(vi0, vi1));
                if (nrm == 0.0) {
                    nrm = 1.0;
                }
                MAT_ELEM(&out_result->vectors_real, 0, 0) = vr0 / nrm;
                MAT_ELEM(&out_result->vectors_real, 1, 0) = vr1 / nrm;
                MAT_ELEM(&out_result->vectors_imag, 0, 0) = vi0 / nrm;
                MAT_ELEM(&out_result->vectors_imag, 1, 0) = vi1 / nrm;
            }
            /** @brief 第二列为第一列的复共轭。 */
            MAT_ELEM(&out_result->vectors_real, 0, 1) = MAT_ELEM(&out_result->vectors_real, 0, 0);
            MAT_ELEM(&out_result->vectors_real, 1, 1) = MAT_ELEM(&out_result->vectors_real, 1, 0);
            MAT_ELEM(&out_result->vectors_imag, 0, 1) = -MAT_ELEM(&out_result->vectors_imag, 0, 0);
            MAT_ELEM(&out_result->vectors_imag, 1, 1) = -MAT_ELEM(&out_result->vectors_imag, 1, 0);
}

void lmmc_eigen_vectors_2x2(const lmmc_mat_t* a, lmmc_real_t matrix_scale,
                           lmmc_eigen_gen_full_result_t* out_result)
{
        lmmc_real_t a00 = MAT_ELEM(a, 0, 0) / matrix_scale;
        lmmc_real_t a01 = MAT_ELEM(a, 0, 1) / matrix_scale;
        lmmc_real_t a10 = MAT_ELEM(a, 1, 0) / matrix_scale;
        lmmc_real_t a11 = MAT_ELEM(a, 1, 1) / matrix_scale;
        lmmc_eigen_values_2x2(
            a00, a01, a10, a11,
            &out_result->real_parts.data[0],
            &out_result->imag_parts.data[0],
            &out_result->real_parts.data[1],
            &out_result->imag_parts.data[1]);
        out_result->real_parts.data[0] *= matrix_scale;
        out_result->imag_parts.data[0] *= matrix_scale;
        out_result->real_parts.data[1] *= matrix_scale;
        out_result->imag_parts.data[1] *= matrix_scale;
if (out_result->imag_parts.data[0] == 0.0) {
        real_vectors_2x2(a00, a01, a10, a11, matrix_scale, out_result);
    } else {
        complex_vectors_2x2(a00, a01, a10, a11, matrix_scale, out_result);
    }
}
