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

static void hessenberg_column(lmmc_mat_t* H, lmmc_mat_t* Q,
                               size_t k, lmmc_real_t* vbuf)
{
    size_t n = H->rows;
    size_t i;
        size_t len = n - k - 1;
        lmmc_real_t tau, beta;

        for (i = 0; i < len; i++) {
            vbuf[i] = MAT_ELEM(H, k + 1 + i, k);
        }

        householder_make(vbuf, len, &tau, &beta);

        MAT_ELEM(H, k + 1, k) = beta;
        for (i = 1; i < len; i++) {
            MAT_ELEM(H, k + 1 + i, k) = 0.0;
        }

        /** @brief 左乘反射：H <- (I - tau*v*v^T) * H。 */
        householder_apply_left(H, k + 1, len, k + 1, n, vbuf, tau);

        /** @brief 右乘反射：H <- H * (I - tau*v*v^T)。 */
        householder_apply_right(H, 0, n, k + 1, len, vbuf, tau);

        /** @brief 累积正交变换：Q <- Q * (I - tau*v*v^T)。 */
        householder_apply_right(Q, 0, n, k + 1, len, vbuf, tau);
}

/**
 * @brief 使用 Householder 反射将一般方阵约化为上 Hessenberg 形.
 *
 * 输入时 H 保存 A;返回时 H 保存上 Hessenberg 形,Q 累积正交相似变换,
 * 满足 A = Q * H * Q^T.
 */
static lmmc_status_t hessenberg_reduce(lmmc_mat_t *H, lmmc_mat_t *Q) {
    size_t n = H->rows;
    size_t k, i, j;
    lmmc_real_t *vbuf;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            MAT_ELEM(Q, i, j) = (i == j) ? 1.0 : 0.0;
        }
    }

    if (n <= 2) {
        return LMMC_STATUS_OK;
    }

    vbuf = (lmmc_real_t *)lmmc_memory_alloc_array(n, sizeof(lmmc_real_t));
    if (!vbuf) {
        return LMMC_STATUS_ALLOCATION_FAILED;
    }

    for (k = 0; k < n - 2; k++) {
        hessenberg_column(H, Q, k, vbuf);
    }

    lmmc_memory_free(vbuf);
    return LMMC_STATUS_OK;
}
static lmmc_real_t schur_matrix_norm(const lmmc_mat_t* H)
{
    lmmc_real_t norm = 0.0;
    for (size_t i = 0; i < H->rows; i++) {
        for (size_t j = 0; j < H->rows; j++) {
            norm += lmmc_abs(MAT_ELEM(H, i, j));
        }
    }
    return norm;
}

static int schur_find_split(lmmc_mat_t* H, int en, lmmc_real_t norm)
{
    const lmmc_real_t eps = 2.2204460492503131e-16;
    int l;
    lmmc_real_t s;
            /** @brief 以可忽略的次对角元素划分子块。 */
            for (l = en; l > 0; l--) {
                s = lmmc_abs(MAT_ELEM(H, l - 1, l - 1)) +
                    lmmc_abs(MAT_ELEM(H, l, l));
                if (s == 0.0) {
                    s = norm;
                }
                if (lmmc_abs(MAT_ELEM(H, l, l - 1)) <= eps * s) {
                    MAT_ELEM(H, l, l - 1) = 0.0;
                    break;
                }
            }
    return l;
}

static void francis_initial_column(const lmmc_mat_t* H, int en, int l,
                                    int its, lmmc_real_t* v)
{
    lmmc_real_t x = MAT_ELEM(H, en, en);
    lmmc_real_t y = MAT_ELEM(H, en - 1, en - 1);
    lmmc_real_t w = MAT_ELEM(H, en, en - 1) * MAT_ELEM(H, en - 1, en);
    lmmc_real_t s, t, p, q, r;
            if (its == 10 || its == 20) {
                /** @brief 使用例外位移促进收敛。 */
                t = lmmc_abs(MAT_ELEM(H, en, en - 1)) +
                    lmmc_abs(MAT_ELEM(H, en - 1, en - 2));
                x = MAT_ELEM(H, en, en) + 1.5 * t;
                y = x;
                w = -0.4375 * t * t;
            }
            /** @brief 从末尾 2x2 块构造位移。 */
            s = x + y;
            t = x * y - w;

            /** @brief 构造 (H - s1*I)(H - s2*I) 的首列。 */
            p = MAT_ELEM(H, l, l) * (MAT_ELEM(H, l, l) - s) +
                MAT_ELEM(H, l, l + 1) * MAT_ELEM(H, l + 1, l) + t;
            q = MAT_ELEM(H, l + 1, l) *
                (MAT_ELEM(H, l, l) + MAT_ELEM(H, l + 1, l + 1) - s);
            r = MAT_ELEM(H, l + 1, l) * MAT_ELEM(H, l + 2, l + 1);
v[0] = p;
    v[1] = q;
    v[2] = r;
}

static void francis_reflector(lmmc_mat_t* H, lmmc_mat_t* Q,
    int k, int l, int en, lmmc_real_t* v)
{
    int n = (int)H->rows;
    int notlast;
                lmmc_real_t tau, beta_val;
                notlast = (k != en - 1);
                size_t len = notlast ? 3 : 2;

                if (k != l) {
                    v[0] = MAT_ELEM(H, k, k - 1);
                    v[1] = MAT_ELEM(H, k + 1, k - 1);
                    if (notlast) {
                        v[2] = MAT_ELEM(H, k + 2, k - 1);
                    }
    }
                householder_make(v, len, &tau, &beta_val);

                if (k != l) {
                    MAT_ELEM(H, k, k - 1) = beta_val;
                    MAT_ELEM(H, k + 1, k - 1) = 0.0;
                    if (notlast) {
                        MAT_ELEM(H, k + 2, k - 1) = 0.0;
                    }
                } else if (l != 0) {
                    /** @brief 反转已有次对角元素符号以应用隐式位移。 */
                    MAT_ELEM(H, k, k - 1) = -MAT_ELEM(H, k, k - 1);
                }

                /** @brief 左作用范围为 H[k:k+len, k:n]。 */
                householder_apply_left(H, (size_t)k, len, (size_t)k, (size_t)n, v, tau);

                /** @brief 右作用范围为 H[0:min(k+len+1,en+1), k:k+len]。 */
                {
                    size_t nr = (size_t)((int)(k + len + 1) <= en + 1 ?
                                         (int)(k + len + 1) : en + 1);
                    householder_apply_right(H, 0, nr, (size_t)k, len, v, tau);
                }

                householder_apply_right(Q, 0, (size_t)n, (size_t)k, len, v, tau);
}

static void francis_step(lmmc_mat_t* H, lmmc_mat_t* Q, int en, int l, int its)
{
    lmmc_real_t v[3];
    francis_initial_column(H, en, l, its, v);
    for (int k = l; k <= en - 1; k++) {
        francis_reflector(H, Q, k, l, en, v);
    }
}
/**
 * @brief 对上 Hessenberg 矩阵执行隐式 Francis 双位移 QR 迭代.
 *
 * 输入 H 为上 Hessenberg 形;返回时 H 为实 Schur 形,包含 1x1 与 2x2
 * 对角块,Q 累积全部相似变换.
 *
 * @param H 原地变换为实 Schur 形的上 Hessenberg 矩阵.
 * @param Q 累积变换的正交矩阵.
 * @param nn 矩阵阶数.
 * @return 收敛时返回 LMMC_STATUS_OK;超过迭代上限时返回
 *         LMMC_STATUS_CONVERGENCE_FAILED.
 *
 * @see J. G. F. Francis, "The QR Transformation: A Unitary Analogue
 *      to the LR Transformation," The Computer Journal 4, 1961-1962.
 * @see B. T. Smith et al., Matrix Eigensystem Routines-EISPACK Guide, 1976.
 */
static lmmc_status_t francis_qr_iteration(lmmc_mat_t* H, lmmc_mat_t* Q, size_t nn)
{
    const size_t max_iter = 30 * nn;
    size_t total_iter = 0;
    lmmc_real_t norm = schur_matrix_norm(H);
    if (norm == 0.0) {
        return LMMC_STATUS_OK;
    }
    int en = (int)nn - 1;
    while (en >= 2) {
        int its = 0;
        for (;;) {
            int l = schur_find_split(H, en, norm);
            if (l == en) {
                en--;
                break;
            }
            if (l == en - 1) {
                en -= 2;
                break;
            }
            if (total_iter >= max_iter) {
                return LMMC_STATUS_CONVERGENCE_FAILED;
            }
            francis_step(H, Q, en, l, its);
            its++;
            total_iter++;
        }
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_eigen_schur_reduce(lmmc_mat_t* h, lmmc_mat_t* q)
{
    lmmc_status_t status = hessenberg_reduce(h, q);
    if (status != LMMC_STATUS_OK) {
        return status;
    }
    return francis_qr_iteration(h, q, h->rows);
}
