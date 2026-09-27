#ifndef LMMC_SPECIAL_FUNCTIONS_H
#define LMMC_SPECIAL_FUNCTIONS_H

#include <stddef.h>
#include "lmmc/config.h"
#include "lmmc/status.h"
#include "lmmc/numeric_scalar.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief 计算误差函数 @f$\mathrm{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt@f$ .
 *
 * @param[in]  x   输入值.
 * @param[out] out 输出 @f$\mathrm{erf}(x)@f$ .
 * @return LMMC_STATUS_OK 成功;LMMC_STATUS_INVALID_ARGUMENT 若 out 为 NULL.
 */
lmmc_status_t lmmc_erf(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算互补误差函数 @f$\mathrm{erfc}(x) = 1 - \mathrm{erf}(x)@f$ .
 *
 * 对大 @f$|x|@f$ 使用直接近似以保持有效精度.
 *
 * @param[in]  x   输入值.
 * @param[out] out 输出 @f$\mathrm{erfc}(x)@f$ .
 * @return LMMC_STATUS_OK 成功;LMMC_STATUS_INVALID_ARGUMENT 若 out 为 NULL.
 */
lmmc_status_t lmmc_erfc(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算对数伽马函数 @f$\ln\Gamma(x)@f$ ,要求 @f$x > 0@f$ .
 *
 * 使用 Lanczos 近似(g=7, n=9 系数),精度约 15 位有效数字.
 *
 * @param[in]  x   输入值,必须为正数.
 * @param[out] out 输出 @f$\ln\Gamma(x)@f$ .
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 x <= 0 或 out 为 NULL.
 *
 * @note 不分配内存，输入保持原值。
 */
lmmc_status_t lmmc_lgamma(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算伽马函数 @f$\Gamma(x)@f$ ,要求 @f$x > 0@f$ .
 *
 * @param[in]  x   输入值,必须为正数.
 * @param[out] out 输出 @f$\Gamma(x)@f$ .
 * @return LMMC_STATUS_OK 成功;LMMC_STATUS_INVALID_ARGUMENT 若 x <= 0 或 out 为 NULL;
 *         LMMC_STATUS_NUMERICAL_FAILURE 若结果溢出.
 */
lmmc_status_t lmmc_tgamma(lmmc_real_t x, lmmc_real_t* out);

/**
 * @brief 计算贝塔函数 @f$B(a,b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}@f$ ,要求 @f$a,b > 0@f$ .
 *
 * 使用 log-beta、Gamma 比值及 Stirling 公式稳定求值，支持 a+b 超出可表示范围。
 * @param[in] a 第一个有限正形状参数。
 * @param[in] b 第二个有限正形状参数。
 * @param[out] out B(a,b)；有效下溢返回零，仅成功时写入。
 * @return OK 表示成功；INVALID_ARGUMENT 表示形状参数无效或输出为空；
 *         NUMERICAL_FAILURE 表示最终结果上溢或数值求值失败。
 */
lmmc_status_t lmmc_beta(lmmc_real_t a, lmmc_real_t b, lmmc_real_t* out);

/**
 * @brief 计算双伽马函数 @f$\psi(x) = \frac{d}{dx}\ln\Gamma(x)@f$ ,要求 @f$x > 0@f$ .
 *
 * 使用递推关系将 x 提升到大值区域后应用渐近展开.
 *
 * @param[in]  x   输入值,必须为正数.
 * @param[out] out 输出 @f$\psi(x)@f$ .
 * @return LMMC_STATUS_OK 成功;LMMC_STATUS_INVALID_ARGUMENT 若 x <= 0 或 out 为 NULL.
 */
lmmc_status_t lmmc_digamma(lmmc_real_t x, lmmc_real_t* out);


/**
 * @brief 计算 Lambert W 函数主分支 @f$W_0(z)@f$ .
 *
 * 微小输入使用逆级数，分支点附近使用抗消减级数，其余使用带保护的
 * Halley/二分迭代求解单调残差。恰等于 -LMMC_INV_E 的 binary64 输入
 * 代表端点 -1/e，返回 -1；相邻输入按各自值处理。W0 保留零的符号。
 *
 * 前向误差工程目标为 64*DBL_EPSILON*abs(W)+2*minimum_subnormal。
 * 验收以残差误差半径包围根，计入 binary64 舍入，并假设所支持 libm 的
 * log/exp/expm1 误差至多为 2 ULP；此为工程误差模型，
 * 不构成精确算术证明或 C 标准对 libm 的保证。
 *
 * @param[in] z 大于或等于端点代表值的有限输入。
 * @param[out] out_res 有限结果，仅在成功时写入。
 * @return OK 表示成功；INVALID_ARGUMENT 表示有限输入超出定义域或输出为空；
 *         NUMERICAL_FAILURE 表示输入或中间量非有限；
 *         CONVERGENCE_FAILED 表示 100 次迭代仍无法确认前向误差目标。
 */
lmmc_status_t lmmc_lambertw(lmmc_real_t z, lmmc_real_t* out_res);

/**
 * @brief 计算 Lambert W 函数 @f$W_{-1}@f$ 分支,要求 @f$z \in [-1/e, 0)@f$ .
 *
 * 保护迭代、工程精度模型、端点约定和失败时保留输出的规则同 lmmc_lambertw。
 * @param[in] z [-LMMC_INV_E,0) 内的有限输入；正零和负零均无效。
 * @param[out] out_res <= -1 的有限结果，仅在成功时写入。
 * @return OK、INVALID_ARGUMENT、NUMERICAL_FAILURE 或 CONVERGENCE_FAILED，
 *         含义同 lmmc_lambertw。
 */
lmmc_status_t lmmc_lambertw_wm1(lmmc_real_t z, lmmc_real_t* out_res);

#ifdef __cplusplus
}
#endif

#endif
