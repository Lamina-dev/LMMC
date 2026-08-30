/**
 * @file eigen_general.c
 * @brief 一般矩阵特征值分解(Hessenberg + Francis QR 与逆迭代)实现.
 */

#include <math.h>
#include <string.h>
#include "memory_bridge.h"
#include "internal.h"
#include "eigen_internal.h"
#include "lmmc/config.h"
#include "lmmc/dense.h"
#include "lmmc/eigen.h"
#include "lmmc/linear_algebra.h"


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

    /** 初始化 Q = I. */
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            MAT_ELEM(Q, i, j) = (i == j) ? 1.0 : 0.0;

    if (n <= 2) return LMMC_STATUS_OK;

    vbuf = (lmmc_real_t *)lmmc_alloc(n * sizeof(lmmc_real_t));
    if (!vbuf) return LMMC_STATUS_ALLOCATION_FAILED;

    for (k = 0; k < n - 2; k++) {
        size_t len = n - k - 1;
        lmmc_real_t tau, beta;

        /** 提取对角线下方的列段. */
        for (i = 0; i < len; i++)
            vbuf[i] = MAT_ELEM(H, k + 1 + i, k);

        householder_make(vbuf, len, &tau, &beta);

        /** 写入次对角元素. */
        MAT_ELEM(H, k + 1, k) = beta;
        for (i = 1; i < len; i++)
            MAT_ELEM(H, k + 1 + i, k) = 0.0;

        /** 从左侧应用 H <- (I - tau*v*v^T) * H. */
        householder_apply_left(H, k + 1, len, k + 1, n, vbuf, tau);

        /** 从右侧应用 H <- H * (I - tau*v*v^T). */
        householder_apply_right(H, 0, n, k + 1, len, vbuf, tau);

        /** 累积 Q <- Q * (I - tau*v*v^T). */
        householder_apply_right(Q, 0, n, k + 1, len, vbuf, tau);
    }

    lmmc_free(vbuf);
    return LMMC_STATUS_OK;
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
static lmmc_status_t francis_qr_iteration(lmmc_mat_t *H, lmmc_mat_t *Q, size_t nn) {
    /**
     * 基于 EISPACK hqr2 的 Francis 隐式双位移 QR,
     * 将上 Hessenberg 矩阵约化为实 Schur 形.
     */
    const lmmc_real_t eps = 2.2204460492503131e-16;
    const size_t max_iter = 30 * nn;
    size_t total_iter = 0;
    int n = (int)nn;
    int i, j, k, l, its, en;
    lmmc_real_t p, q, r, s, t, w, x, y, norm;
    int notlast;
    lmmc_real_t v[3];

    /* Compute norm for convergence testing */
    norm = 0.0;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            norm += lmmc_abs(MAT_ELEM(H, i, j));
    if (norm == 0.0) return LMMC_STATUS_OK;

    en = n - 1;

    while (en >= 2) {
        its = 0;

        for (;;) {
            /* Look for single small sub-diagonal element */
            for (l = en; l > 0; l--) {
                s = lmmc_abs(MAT_ELEM(H, l - 1, l - 1)) +
                    lmmc_abs(MAT_ELEM(H, l, l));
                if (s == 0.0) s = norm;
                if (lmmc_abs(MAT_ELEM(H, l, l - 1)) <= eps * s) {
                    MAT_ELEM(H, l, l - 1) = 0.0;
                    break;
                }
            }

            x = MAT_ELEM(H, en, en);
            if (l == en) { en--; break; }

            y = MAT_ELEM(H, en - 1, en - 1);
            w = MAT_ELEM(H, en, en - 1) * MAT_ELEM(H, en - 1, en);
            if (l == en - 1) { en -= 2; break; }

            if (total_iter >= max_iter)
                return LMMC_STATUS_CONVERGENCE_FAILED;

            if (its == 10 || its == 20) {
                /* Exceptional shift */
                t = lmmc_abs(MAT_ELEM(H, en, en - 1)) +
                    lmmc_abs(MAT_ELEM(H, en - 1, en - 2));
                x = MAT_ELEM(H, en, en) + 1.5 * t;
                y = x;
                w = -0.4375 * t * t;
            }

            its++;
            total_iter++;

            /* Form shift from trailing 2x2 */
            s = x + y;
            t = x * y - w;

            /* First column of (H - s1*I)(H - s2*I) */
            p = MAT_ELEM(H, l, l) * (MAT_ELEM(H, l, l) - s) +
                MAT_ELEM(H, l, l + 1) * MAT_ELEM(H, l + 1, l) + t;
            q = MAT_ELEM(H, l + 1, l) *
                (MAT_ELEM(H, l, l) + MAT_ELEM(H, l + 1, l + 1) - s);
            r = MAT_ELEM(H, l + 1, l) * MAT_ELEM(H, l + 2, l + 1);

            /* Chase the bulge using Householder reflectors */
            for (k = l; k <= en - 1; k++) {
                lmmc_real_t tau, beta_val;
                notlast = (k != en - 1);
                size_t len = notlast ? 3 : 2;

                if (k != l) {
                    v[0] = MAT_ELEM(H, k, k - 1);
                    v[1] = MAT_ELEM(H, k + 1, k - 1);
                    if (notlast) v[2] = MAT_ELEM(H, k + 2, k - 1);
                } else {
                    v[0] = p; v[1] = q;
                    if (notlast) v[2] = r;
                }

                householder_make(v, len, &tau, &beta_val);

                if (k != l) {
                    MAT_ELEM(H, k, k - 1) = beta_val;
                    MAT_ELEM(H, k + 1, k - 1) = 0.0;
                    if (notlast) MAT_ELEM(H, k + 2, k - 1) = 0.0;
                } else if (l != 0) {
                    /* Flip sign of existing subdiagonal for implicit shift */
                    MAT_ELEM(H, k, k - 1) = -MAT_ELEM(H, k, k - 1);
                }

                /* Apply from left: H[k:k+len, k:n] */
                householder_apply_left(H, (size_t)k, len, (size_t)k, (size_t)n, v, tau);

                /* Apply from right: H[0:min(k+len+1,en+1), k:k+len] */
                {
                    size_t nr = (size_t)((int)(k + len + 1) <= en + 1 ?
                                         (int)(k + len + 1) : en + 1);
                    householder_apply_right(H, 0, nr, (size_t)k, len, v, tau);
                }

                /* Accumulate in Q */
                householder_apply_right(Q, 0, (size_t)n, (size_t)k, len, v, tau);
            }
        }
    }

    return LMMC_STATUS_OK;
}

/**
 * @brief 从实 Schur 形提取特征值.
 *
 * 1x1 对角块产生实特征值,2x2 对角块产生共轭复特征值对.
 */
static void extract_eigenvalues_from_schur(const lmmc_mat_t *H, size_t n,
                                           lmmc_real_t *re, lmmc_real_t *im) {
    size_t i = 0;
    while (i < n) {
        if (i + 1 == n || MAT_ELEM(H, i + 1, i) == 0.0) {
            /** 1x1 块产生实特征值. */
            re[i] = MAT_ELEM(H, i, i);
            im[i] = 0.0;
            i++;
        } else {
            /** 2x2 块产生共轭复特征值对. */
            lmmc_real_t a11 = MAT_ELEM(H, i, i);
            lmmc_real_t a12 = MAT_ELEM(H, i, i + 1);
            lmmc_real_t a21 = MAT_ELEM(H, i + 1, i);
            lmmc_real_t a22 = MAT_ELEM(H, i + 1, i + 1);
            lmmc_real_t tr = a11 + a22;
            lmmc_real_t det = a11 * a22 - a12 * a21;
            lmmc_real_t disc = tr * tr - 4.0 * det;
            if (disc >= 0.0) {
                lmmc_real_t sq = sqrt(disc);
                lmmc_real_t lam1 = (tr + sq) * 0.5;
                lmmc_real_t lam2 = (tr - sq) * 0.5;
                re[i] = lam1; im[i] = 0.0;
                re[i + 1] = lam2; im[i + 1] = 0.0;
            } else {
                lmmc_real_t sq = sqrt(-disc) * 0.5;
                re[i] = tr * 0.5;     im[i] = sq;
                re[i + 1] = tr * 0.5; im[i + 1] = -sq;
            }
            i += 2;
        }
    }
}

lmmc_status_t lmmc_eigen_general(const lmmc_mat_t *a, lmmc_eigen_gen_result_t *out_result)
{
    lmmc_status_t status;
    if (!a || !out_result || !a->data) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows != a->cols || a->rows == 0) return LMMC_STATUS_INVALID_ARGUMENT;

    size_t n = a->rows;
    size_t i, j;

    /* Allocate output vectors */
    status = lmmc_vec_create(n, &out_result->real_parts);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_create(n, &out_result->imag_parts);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&out_result->real_parts);
        return status;
    }

    if (n == 1) {
        out_result->real_parts.data[0] = MAT_ELEM(a, 0, 0);
        out_result->imag_parts.data[0] = 0.0;
        return LMMC_STATUS_OK;
    }

    if (n == 2) {
        lmmc_real_t a00 = MAT_ELEM(a, 0, 0), a01 = MAT_ELEM(a, 0, 1);
        lmmc_real_t a10 = MAT_ELEM(a, 1, 0), a11 = MAT_ELEM(a, 1, 1);
        lmmc_real_t tr = a00 + a11;
        lmmc_real_t det = a00 * a11 - a01 * a10;
        lmmc_real_t disc = tr * tr - 4.0 * det;
        if (disc >= 0.0) {
            lmmc_real_t sq = sqrt(disc);
            out_result->real_parts.data[0] = (tr + sq) * 0.5;
            out_result->imag_parts.data[0] = 0.0;
            out_result->real_parts.data[1] = (tr - sq) * 0.5;
            out_result->imag_parts.data[1] = 0.0;
        } else {
            lmmc_real_t sq = sqrt(-disc) * 0.5;
            out_result->real_parts.data[0] = tr * 0.5;
            out_result->imag_parts.data[0] = sq;
            out_result->real_parts.data[1] = tr * 0.5;
            out_result->imag_parts.data[1] = -sq;
        }
        return LMMC_STATUS_OK;
    }

    /* Allocate workspace: H (copy of A) and Q (orthogonal accumulator) */
    lmmc_mat_t H, Q_mat;
    status = lmmc_mat_create(n, n, &H);
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&out_result->real_parts);
        lmmc_vec_destroy(&out_result->imag_parts);
        return status;
    }
    status = lmmc_mat_create(n, n, &Q_mat);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&H);
        lmmc_vec_destroy(&out_result->real_parts);
        lmmc_vec_destroy(&out_result->imag_parts);
        return status;
    }

    /* Copy A into H */
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            MAT_ELEM(&H, i, j) = MAT_ELEM(a, i, j);

    /* Step 1: Reduce to upper Hessenberg form */
    status = hessenberg_reduce(&H, &Q_mat);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&H);
        lmmc_mat_destroy(&Q_mat);
        lmmc_vec_destroy(&out_result->real_parts);
        lmmc_vec_destroy(&out_result->imag_parts);
        return status;
    }

    /* Step 2: Francis double-shift QR iteration */
    status = francis_qr_iteration(&H, &Q_mat, n);
    if (status != LMMC_STATUS_OK) {
        lmmc_mat_destroy(&H);
        lmmc_mat_destroy(&Q_mat);
        lmmc_vec_destroy(&out_result->real_parts);
        lmmc_vec_destroy(&out_result->imag_parts);
        return status;
    }

    /* Step 3: Extract eigenvalues from real Schur form */
    extract_eigenvalues_from_schur(&H, n,
                                   out_result->real_parts.data,
                                   out_result->imag_parts.data);

    lmmc_mat_destroy(&H);
    lmmc_mat_destroy(&Q_mat);
    return LMMC_STATUS_OK;
}


void lmmc_eigen_gen_result_destroy(lmmc_eigen_gen_result_t *result)
{
    if (!result) return;
    lmmc_vec_destroy(&result->real_parts);
    lmmc_vec_destroy(&result->imag_parts);
}

/**
 * @brief 释放通用特征分解结果持有的向量与矩阵缓冲区.
 */

void lmmc_eigen_gen_full_result_destroy(lmmc_eigen_gen_full_result_t *result) {
    if (!result) return;
    lmmc_vec_destroy(&result->real_parts);
    lmmc_vec_destroy(&result->imag_parts);
    lmmc_mat_destroy(&result->vectors_real);
    lmmc_mat_destroy(&result->vectors_imag);
}

/**
 * @brief 对实特征值执行逆迭代.
 *
 * 重复求解 (A - mu*I)x = b 并归一化,使向量收敛到对应特征向量.
 * 带部分主元的 LU 分解处理位移系统,小扰动使位移矩阵保持可分解.
 *
 * @see J. H. Wilkinson, The Algebraic Eigenvalue Problem, 1965.
 */
static lmmc_status_t inverse_iteration_real(
    const lmmc_mat_t *A, size_t n, lmmc_real_t mu,
    lmmc_real_t *vec_out)
{
    lmmc_status_t status;
    size_t i, j, iter;
    const size_t max_iter = 20;
    const lmmc_real_t eps = 2.2204460492503131e-16;

    /** 分配位移矩阵与工作区. */
    lmmc_mat_t shifted;
    status = lmmc_mat_create(n, n, &shifted);
    if (status != LMMC_STATUS_OK) return status;

    size_t *pivots = (size_t *)lmmc_alloc(n * sizeof(size_t));
    if (!pivots) { lmmc_mat_destroy(&shifted); return LMMC_STATUS_ALLOCATION_FAILED; }

    lmmc_vec_t b_vec, x_vec;
    status = lmmc_vec_create(n, &b_vec);
    if (status != LMMC_STATUS_OK) { lmmc_free(pivots); lmmc_mat_destroy(&shifted); return status; }
    status = lmmc_vec_create(n, &x_vec);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&b_vec); lmmc_free(pivots); lmmc_mat_destroy(&shifted); return status; }

    /** 计算 ||A||_F 作为扰动尺度. */
    lmmc_real_t norm_A = 0.0;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            norm_A += MAT_ELEM(A, i, j) * MAT_ELEM(A, i, j);
    norm_A = sqrt(norm_A);
    if (norm_A < 1.0) norm_A = 1.0;

    /** Wilkinson 风格扰动按 eps*||A||_F*n 设置,使位移矩阵保持可分解. */
    lmmc_real_t perturb = eps * norm_A * (lmmc_real_t)n;
    lmmc_real_t shift = mu + perturb;

    /** 构造 A - shift*I. */
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            MAT_ELEM(&shifted, i, j) = MAT_ELEM(A, i, j) - ((i == j) ? shift : 0.0);

    /** 对位移矩阵执行 LU 分解. */
    status = lmmc_lu_decompose_inplace(&shifted, pivots, NULL);
    if (status == LMMC_STATUS_SINGULAR_MATRIX) {
        /** 首次分解奇异时使用 sqrt(eps)*||A||_F 的扩大扰动. */
        perturb = sqrt(eps) * norm_A;
        shift = mu + perturb;
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                MAT_ELEM(&shifted, i, j) = MAT_ELEM(A, i, j) - ((i == j) ? shift : 0.0);
        status = lmmc_lu_decompose_inplace(&shifted, pivots, NULL);
    }
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&x_vec);
        lmmc_vec_destroy(&b_vec);
        lmmc_free(pivots);
        lmmc_mat_destroy(&shifted);
        return status;
    }

    /** 初始向量使用交替符号分量,以覆盖各坐标方向. */
    for (i = 0; i < n; i++)
        b_vec.data[i] = ((i % 2 == 0) ? 1.0 : -1.0) / (lmmc_real_t)(i + 1);

    /** 归一化初始向量. */
    {
        lmmc_real_t nrm = 0.0;
        for (i = 0; i < n; i++) nrm += b_vec.data[i] * b_vec.data[i];
        nrm = sqrt(nrm);
        if (nrm > 0.0) for (i = 0; i < n; i++) b_vec.data[i] /= nrm;
    }

    /** 执行逆迭代. */
    for (iter = 0; iter < max_iter; iter++) {
        status = lmmc_lu_solve(&shifted, pivots, &b_vec, &x_vec);
        if (status != LMMC_STATUS_OK) {
            lmmc_vec_destroy(&x_vec);
            lmmc_vec_destroy(&b_vec);
            lmmc_free(pivots);
            lmmc_mat_destroy(&shifted);
            return status;
        }

        /** 归一化当前向量. */
        lmmc_real_t nrm = 0.0;
        for (i = 0; i < n; i++) nrm += x_vec.data[i] * x_vec.data[i];
        nrm = sqrt(nrm);
        if (nrm == 0.0) nrm = 1.0;
        for (i = 0; i < n; i++) x_vec.data[i] /= nrm;

        /** 将当前向量作为下一轮右端. */
        for (i = 0; i < n; i++) b_vec.data[i] = x_vec.data[i];
    }

    /** 复制收敛后的特征向量. */
    for (i = 0; i < n; i++) vec_out[i] = x_vec.data[i];

    lmmc_vec_destroy(&x_vec);
    lmmc_vec_destroy(&b_vec);
    lmmc_free(pivots);
    lmmc_mat_destroy(&shifted);
    return LMMC_STATUS_OK;
}

/**
 * @brief 对共轭复特征值对执行逆迭代.
 *
 * 对 mu = alpha + i*beta 求解实数 2nx2n 系统:
 *   [A - alpha*I,  beta*I ] [x_re]   [b_re]
 *   [-beta*I,  A - alpha*I] [x_im] = [b_im]
 * 得到特征向量的实部与虚部.
 */
static lmmc_status_t inverse_iteration_complex(
    const lmmc_mat_t *A, size_t n, lmmc_real_t alpha, lmmc_real_t beta,
    lmmc_real_t *vec_real_out, lmmc_real_t *vec_imag_out)
{
    lmmc_status_t status;
    size_t i, j, iter;
    const size_t max_iter = 20;
    const lmmc_real_t eps = 2.2204460492503131e-16;

    /* Compute norm of A */
    lmmc_real_t norm_A = 0.0;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            norm_A += MAT_ELEM(A, i, j) * MAT_ELEM(A, i, j);
    norm_A = sqrt(norm_A);
    if (norm_A < 1.0) norm_A = 1.0;

    /*
     * For complex eigenvalue mu = alpha + i*beta, we solve the 2n x 2n real system:
     *   [A - alpha*I,  beta*I ] [x_re]   [b_re]
     *   [-beta*I,  A - alpha*I] [x_im] = [b_im]
     *
     * Apply a small perturbation to alpha to avoid singularity.
     */
    lmmc_real_t perturb = eps * norm_A * (lmmc_real_t)n;
    lmmc_real_t alpha_p = alpha + perturb;

    lmmc_mat_t big;
    size_t n2 = 2 * n;
    status = lmmc_mat_create(n2, n2, &big);
    if (status != LMMC_STATUS_OK) return status;

    size_t *pivots = (size_t *)lmmc_alloc(n2 * sizeof(size_t));
    if (!pivots) { lmmc_mat_destroy(&big); return LMMC_STATUS_ALLOCATION_FAILED; }

    lmmc_vec_t b_vec, x_vec;
    status = lmmc_vec_create(n2, &b_vec);
    if (status != LMMC_STATUS_OK) { lmmc_free(pivots); lmmc_mat_destroy(&big); return status; }
    status = lmmc_vec_create(n2, &x_vec);
    if (status != LMMC_STATUS_OK) { lmmc_vec_destroy(&b_vec); lmmc_free(pivots); lmmc_mat_destroy(&big); return status; }

    /* Fill the 2n x 2n block matrix */
    for (i = 0; i < n2; i++)
        for (j = 0; j < n2; j++)
            MAT_ELEM(&big, i, j) = 0.0;

    /* Top-left block: A - alpha_p*I */
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            MAT_ELEM(&big, i, j) = MAT_ELEM(A, i, j) - ((i == j) ? alpha_p : 0.0);

    /* Top-right block: beta*I */
    for (i = 0; i < n; i++)
        MAT_ELEM(&big, i, n + i) = beta;

    /* Bottom-left block: -beta*I */
    for (i = 0; i < n; i++)
        MAT_ELEM(&big, n + i, i) = -beta;

    /* Bottom-right block: A - alpha_p*I */
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            MAT_ELEM(&big, n + i, n + j) = MAT_ELEM(A, i, j) - ((i == j) ? alpha_p : 0.0);

    /* LU decompose */
    status = lmmc_lu_decompose_inplace(&big, pivots, NULL);
    if (status == LMMC_STATUS_SINGULAR_MATRIX) {
        /* Try a larger perturbation */
        alpha_p = alpha + sqrt(eps) * norm_A;
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++) {
                lmmc_real_t aij = MAT_ELEM(A, i, j) - ((i == j) ? alpha_p : 0.0);
                MAT_ELEM(&big, i, j) = aij;
                MAT_ELEM(&big, n + i, n + j) = aij;
            }
        for (i = 0; i < n; i++) {
            MAT_ELEM(&big, i, n + i) = beta;
            MAT_ELEM(&big, n + i, i) = -beta;
        }
        status = lmmc_lu_decompose_inplace(&big, pivots, NULL);
    }
    if (status != LMMC_STATUS_OK) {
        lmmc_vec_destroy(&x_vec);
        lmmc_vec_destroy(&b_vec);
        lmmc_free(pivots);
        lmmc_mat_destroy(&big);
        return status;
    }

    /* Initialize starting vector */
    for (i = 0; i < n; i++) {
        b_vec.data[i] = ((i % 2 == 0) ? 1.0 : -1.0) / (lmmc_real_t)(i + 1);
        b_vec.data[n + i] = 1.0 / (lmmc_real_t)(n + i + 1);
    }

    /* Normalize */
    {
        lmmc_real_t nrm = 0.0;
        for (i = 0; i < n2; i++) nrm += b_vec.data[i] * b_vec.data[i];
        nrm = sqrt(nrm);
        if (nrm > 0.0) for (i = 0; i < n2; i++) b_vec.data[i] /= nrm;
    }

    /* Inverse iteration loop */
    for (iter = 0; iter < max_iter; iter++) {
        status = lmmc_lu_solve(&big, pivots, &b_vec, &x_vec);
        if (status != LMMC_STATUS_OK) {
            lmmc_vec_destroy(&x_vec);
            lmmc_vec_destroy(&b_vec);
            lmmc_free(pivots);
            lmmc_mat_destroy(&big);
            return status;
        }

        /* Normalize */
        lmmc_real_t nrm = 0.0;
        for (i = 0; i < n2; i++) nrm += x_vec.data[i] * x_vec.data[i];
        nrm = sqrt(nrm);
        if (nrm == 0.0) nrm = 1.0;
        for (i = 0; i < n2; i++) x_vec.data[i] /= nrm;

        /* Copy x to b for next iteration */
        for (i = 0; i < n2; i++) b_vec.data[i] = x_vec.data[i];
    }

    /* Extract real and imaginary parts */
    for (i = 0; i < n; i++) {
        vec_real_out[i] = x_vec.data[i];
        vec_imag_out[i] = x_vec.data[n + i];
    }

    lmmc_vec_destroy(&x_vec);
    lmmc_vec_destroy(&b_vec);
    lmmc_free(pivots);
    lmmc_mat_destroy(&big);
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_eigen_general_full(
    const lmmc_mat_t *a,
    lmmc_eigen_gen_full_result_t *out_result)
{
    lmmc_status_t status;
    if (!a || !out_result || !a->data) return LMMC_STATUS_INVALID_ARGUMENT;
    if (a->rows != a->cols || a->rows == 0) return LMMC_STATUS_INVALID_ARGUMENT;

    size_t n = a->rows;
    size_t i, j;

    /* Initialize output to zeros so cleanup is safe on partial failure */
    memset(out_result, 0, sizeof(*out_result));

    /* Allocate output vectors and matrices */
    status = lmmc_vec_create(n, &out_result->real_parts);
    if (status != LMMC_STATUS_OK) return status;
    status = lmmc_vec_create(n, &out_result->imag_parts);
    if (status != LMMC_STATUS_OK) goto fail;
    status = lmmc_mat_create(n, n, &out_result->vectors_real);
    if (status != LMMC_STATUS_OK) goto fail;
    status = lmmc_mat_create(n, n, &out_result->vectors_imag);
    if (status != LMMC_STATUS_OK) goto fail;

    /* Handle trivial cases */
    if (n == 1) {
        out_result->real_parts.data[0] = MAT_ELEM(a, 0, 0);
        out_result->imag_parts.data[0] = 0.0;
        MAT_ELEM(&out_result->vectors_real, 0, 0) = 1.0;
        MAT_ELEM(&out_result->vectors_imag, 0, 0) = 0.0;
        return LMMC_STATUS_OK;
    }

    if (n == 2) {
        lmmc_real_t a00 = MAT_ELEM(a, 0, 0), a01 = MAT_ELEM(a, 0, 1);
        lmmc_real_t a10 = MAT_ELEM(a, 1, 0), a11 = MAT_ELEM(a, 1, 1);
        lmmc_real_t tr = a00 + a11;
        lmmc_real_t det = a00 * a11 - a01 * a10;
        lmmc_real_t disc = tr * tr - 4.0 * det;

        if (disc >= 0.0) {
            lmmc_real_t sq = sqrt(disc);
            lmmc_real_t lam1 = (tr + sq) * 0.5;
            lmmc_real_t lam2 = (tr - sq) * 0.5;
            out_result->real_parts.data[0] = lam1;
            out_result->imag_parts.data[0] = 0.0;
            out_result->real_parts.data[1] = lam2;
            out_result->imag_parts.data[1] = 0.0;

            /* Compute eigenvectors for 2x2 real case.
             * For eigenvalue lam, solve (A - lam*I)*v = 0. */
            for (i = 0; i < 2; i++) {
                lmmc_real_t lam = out_result->real_parts.data[i];
                lmmc_real_t v0, v1, nrm;
                /* Row 0: (a00 - lam)*v0 + a01*v1 = 0 */
                /* Row 1: a10*v0 + (a11 - lam)*v1 = 0 */
                lmmc_real_t r0_diag = lmmc_abs(a00 - lam);
                lmmc_real_t r1_diag = lmmc_abs(a11 - lam);
                lmmc_real_t r0_off = lmmc_abs(a01);
                lmmc_real_t r1_off = lmmc_abs(a10);

                if (r0_off > 1e-15 || r0_diag > 1e-15) {
                    /* Use row 0: (a00 - lam)*v0 + a01*v1 = 0 */
                    if (r0_off >= r0_diag) {
                        /* v0 = -a01, v1 = a00 - lam */
                        v0 = -a01;
                        v1 = a00 - lam;
                    } else {
                        /* v0 = a01, v1 = lam - a00 ... but if a01=0, use row 1 */
                        if (r0_off < 1e-15) {
                            /* a01 = 0, so row 0 gives (a00-lam)*v0 = 0 */
                            /* If a00 != lam, then v0 = 0 */
                            if (r0_diag > 1e-15) {
                                v0 = 0.0; v1 = 1.0;
                            } else {
                                v0 = 1.0; v1 = 0.0;
                            }
                        } else {
                            v0 = -a01;
                            v1 = a00 - lam;
                        }
                    }
                } else if (r1_off > 1e-15 || r1_diag > 1e-15) {
                    /* Use row 1: a10*v0 + (a11 - lam)*v1 = 0 */
                    if (r1_diag > 1e-15) {
                        v0 = 1.0; v1 = -a10 / (a11 - lam);
                    } else {
                        v0 = 0.0; v1 = 1.0;
                    }
                } else {
                    /* Both rows are zero - any vector is an eigenvector */
                    v0 = 1.0; v1 = 0.0;
                }
                nrm = sqrt(v0 * v0 + v1 * v1);
                if (nrm < 1e-15) { v0 = 1.0; v1 = 0.0; nrm = 1.0; }
                MAT_ELEM(&out_result->vectors_real, 0, i) = v0 / nrm;
                MAT_ELEM(&out_result->vectors_real, 1, i) = v1 / nrm;
                MAT_ELEM(&out_result->vectors_imag, 0, i) = 0.0;
                MAT_ELEM(&out_result->vectors_imag, 1, i) = 0.0;
            }
        } else {
            lmmc_real_t sq = sqrt(-disc) * 0.5;
            out_result->real_parts.data[0] = tr * 0.5;
            out_result->imag_parts.data[0] = sq;
            out_result->real_parts.data[1] = tr * 0.5;
            out_result->imag_parts.data[1] = -sq;

            /* Complex-conjugate pair: eigenvector for lam = (tr/2) + i*sq
             * From row 0: (a00 - tr/2 - i*sq)*v0 + a01*v1 = 0
             * From row 1: a10*v0 + (a11 - tr/2 - i*sq)*v1 = 0 */
            lmmc_real_t re_part = tr * 0.5 - a00;
            if (lmmc_abs(a01) > lmmc_abs(a10)) {
                /* v = (a01, re_part + i*sq) */
                lmmc_real_t vr0 = a01;
                lmmc_real_t vr1 = re_part;
                lmmc_real_t vi0 = 0.0;
                lmmc_real_t vi1 = sq;
                lmmc_real_t nrm = sqrt(vr0*vr0 + vr1*vr1 + vi0*vi0 + vi1*vi1);
                if (nrm == 0.0) nrm = 1.0;
                MAT_ELEM(&out_result->vectors_real, 0, 0) = vr0 / nrm;
                MAT_ELEM(&out_result->vectors_real, 1, 0) = vr1 / nrm;
                MAT_ELEM(&out_result->vectors_imag, 0, 0) = vi0 / nrm;
                MAT_ELEM(&out_result->vectors_imag, 1, 0) = vi1 / nrm;
            } else {
                /* v = (tr/2 - a11 + i*sq, a10) */
                lmmc_real_t vr0 = tr * 0.5 - a11;
                lmmc_real_t vr1 = a10;
                lmmc_real_t vi0 = sq;
                lmmc_real_t vi1 = 0.0;
                lmmc_real_t nrm = sqrt(vr0*vr0 + vr1*vr1 + vi0*vi0 + vi1*vi1);
                if (nrm == 0.0) nrm = 1.0;
                MAT_ELEM(&out_result->vectors_real, 0, 0) = vr0 / nrm;
                MAT_ELEM(&out_result->vectors_real, 1, 0) = vr1 / nrm;
                MAT_ELEM(&out_result->vectors_imag, 0, 0) = vi0 / nrm;
                MAT_ELEM(&out_result->vectors_imag, 1, 0) = vi1 / nrm;
            }
            /* Conjugate column */
            MAT_ELEM(&out_result->vectors_real, 0, 1) = MAT_ELEM(&out_result->vectors_real, 0, 0);
            MAT_ELEM(&out_result->vectors_real, 1, 1) = MAT_ELEM(&out_result->vectors_real, 1, 0);
            MAT_ELEM(&out_result->vectors_imag, 0, 1) = -MAT_ELEM(&out_result->vectors_imag, 0, 0);
            MAT_ELEM(&out_result->vectors_imag, 1, 1) = -MAT_ELEM(&out_result->vectors_imag, 1, 0);
        }
        return LMMC_STATUS_OK;
    }

    /* General case: use Hessenberg + Francis QR to get eigenvalues,
     * then inverse iteration for eigenvectors */
    {
        lmmc_mat_t H, Q_mat;
        status = lmmc_mat_create(n, n, &H);
        if (status != LMMC_STATUS_OK) goto fail;
        status = lmmc_mat_create(n, n, &Q_mat);
        if (status != LMMC_STATUS_OK) { lmmc_mat_destroy(&H); goto fail; }

        /* Copy A into H */
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++)
                MAT_ELEM(&H, i, j) = MAT_ELEM(a, i, j);

        /* Step 1: Reduce to upper Hessenberg form */
        status = hessenberg_reduce(&H, &Q_mat);
        if (status != LMMC_STATUS_OK) {
            lmmc_mat_destroy(&H);
            lmmc_mat_destroy(&Q_mat);
            goto fail;
        }

        /* Step 2: Francis double-shift QR iteration */
        status = francis_qr_iteration(&H, &Q_mat, n);
        if (status != LMMC_STATUS_OK) {
            lmmc_mat_destroy(&H);
            lmmc_mat_destroy(&Q_mat);
            goto fail;
        }

        /* Step 3: Extract eigenvalues from real Schur form */
        extract_eigenvalues_from_schur(&H, n,
                                       out_result->real_parts.data,
                                       out_result->imag_parts.data);

        lmmc_mat_destroy(&H);
        lmmc_mat_destroy(&Q_mat);
    }

    /* Step 4: Compute eigenvectors via inverse iteration */
    /* Initialize vectors_imag to zero */
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            MAT_ELEM(&out_result->vectors_imag, i, j) = 0.0;

    i = 0;
    while (i < n) {
        if (out_result->imag_parts.data[i] == 0.0) {
            /* Real eigenvalue: inverse iteration for real eigenvector */
            lmmc_real_t *col = (lmmc_real_t *)lmmc_alloc(n * sizeof(lmmc_real_t));
            if (!col) { status = LMMC_STATUS_ALLOCATION_FAILED; goto fail; }

            status = inverse_iteration_real(a, n, out_result->real_parts.data[i], col);
            if (status != LMMC_STATUS_OK) {
                lmmc_free(col);
                goto fail;
            }

            /* Store in vectors_real column i, vectors_imag column i = 0 */
            for (j = 0; j < n; j++) {
                MAT_ELEM(&out_result->vectors_real, j, i) = col[j];
                MAT_ELEM(&out_result->vectors_imag, j, i) = 0.0;
            }
            lmmc_free(col);
            i++;
        } else {
            /* Complex-conjugate pair: eigenvalues at i and i+1 */
            lmmc_real_t alpha_val = out_result->real_parts.data[i];
            lmmc_real_t beta_val = out_result->imag_parts.data[i];

            lmmc_real_t *vr = (lmmc_real_t *)lmmc_alloc(n * sizeof(lmmc_real_t));
            lmmc_real_t *vi = (lmmc_real_t *)lmmc_alloc(n * sizeof(lmmc_real_t));
            if (!vr || !vi) {
                if (vr) lmmc_free(vr);
                if (vi) lmmc_free(vi);
                status = LMMC_STATUS_ALLOCATION_FAILED;
                goto fail;
            }

            status = inverse_iteration_complex(a, n, alpha_val, beta_val, vr, vi);
            if (status != LMMC_STATUS_OK) {
                lmmc_free(vr);
                lmmc_free(vi);
                goto fail;
            }

            /* Store: column i gets (vr, vi), column i+1 gets (vr, -vi) */
            for (j = 0; j < n; j++) {
                MAT_ELEM(&out_result->vectors_real, j, i) = vr[j];
                MAT_ELEM(&out_result->vectors_imag, j, i) = vi[j];
                MAT_ELEM(&out_result->vectors_real, j, i + 1) = vr[j];
                MAT_ELEM(&out_result->vectors_imag, j, i + 1) = -vi[j];
            }
            lmmc_free(vr);
            lmmc_free(vi);
            i += 2;
        }
    }

    return LMMC_STATUS_OK;

fail:
    lmmc_eigen_gen_full_result_destroy(out_result);
    memset(out_result, 0, sizeof(*out_result));
    return status;
}

