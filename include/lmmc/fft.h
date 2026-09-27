#ifndef LMMC_FFT_H
#define LMMC_FFT_H

#include <stddef.h>
#include "lmmc/config.h"
#include "lmmc/status.h"

#ifdef __cplusplus
extern "C" {
#endif


/** @brief 计算不小于 @p n 的最小 4 的幂，用于 FFT 长度对齐。 */
lmmc_status_t lmmc_fft_radix4_next_size(size_t n, size_t* out_nfft);

/**
 * @brief 对长度为 4 的幂的复数序列执行就地 radix-4 FFT。
 *
 * @param[in,out] real    实部数组,长度 @p n ;变换后就地覆盖为频域实部.
 * @param[in,out] imag    虚部数组,长度 @p n ;变换后就地覆盖为频域虚部.
 * @param[in]     n       FFT 长度,必须为 4 的幂.
 * @param[in]     inverse 非 0 时执行逆变换并归一化(除以 n).
 *
 * @return ::LMMC_STATUS_OK 表示成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 表示 n 位于 4 的幂集合之外、指针为
 *         NULL、数组重叠或数组存储范围的地址计算溢出.
 *
 * @note 不分配堆内存；两数组各覆写 n 个元素，写入前检查并拒绝重叠。
 */
lmmc_status_t lmmc_fft_radix4(lmmc_real_t* real, lmmc_real_t* imag, size_t n, int inverse);

/** @brief 等价于 ::lmmc_fft_radix4(real, imag, n, 0) . */
lmmc_status_t lmmc_fft_radix4_forward(lmmc_real_t* real, lmmc_real_t* imag, size_t n);

/** @brief 等价于 ::lmmc_fft_radix4(real, imag, n, 1) . */
lmmc_status_t lmmc_fft_radix4_inverse(lmmc_real_t* real, lmmc_real_t* imag, size_t n);

/**
 * @brief 将长度 @p n 的输入零填充至不小于它的最小 4 的幂。
 *
 * @param[in]  real_in   输入实部数组,长度 @p n .
 * @param[in]  imag_in   输入虚部数组,长度 @p n .
 * @param[in]  n         输入长度.
 * @param[out] real_out  输出实部数组,调用方需预分配至少由
 *                       ::lmmc_fft_radix4_next_size 得到的元素数.
 * @param[out] imag_out  输出虚部数组,容量要求同 @p real_out.
 * @param[out] out_nfft  写入实际填充后的长度(4 的幂).
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若参数无效、存储范围计算溢出，
 *         或任一输出与输入、另一输出、@p out_nfft 重叠.
 *
 * 所有参数和存储关系均在写入前验证；两个只读输入可以相互重叠。
 */
lmmc_status_t lmmc_fft_radix4_pad_into(
    const lmmc_real_t* real_in, const lmmc_real_t* imag_in, size_t n,
    lmmc_real_t* real_out, lmmc_real_t* imag_out, size_t* out_nfft);


/**
 * @brief 计算任意长度 N 点 DFT(正变换或逆变换).
 *
 * N 为 4 的幂时使用 radix-4，其余 2 的幂使用 radix-2 Cooley-Tukey，
 * 其他长度使用 Bluestein chirp-z 算法。
 *
 * @param[in,out] real    实部数组,长度 @p n ;变换后就地覆盖.
 * @param[in,out] imag    虚部数组,长度 @p n ;变换后就地覆盖.
 * @param[in]     n       FFT 长度,任意正整数.
 * @param[in]     inverse 非 0 时执行逆变换并归一化(除以 n).
 *
 * @return ::LMMC_STATUS_OK 成功;
 *         ::LMMC_STATUS_INVALID_ARGUMENT 若 n == 0、指针为 NULL、数组重叠
 *         或数组存储范围的地址计算溢出;
 *         ::LMMC_STATUS_ALLOCATION_FAILED 若 Bluestein 路径内存分配失败.
 *
 * @note 两数组各覆写 n 个元素，写入前检查并拒绝重叠。
 *       Bluestein 路径分配临时缓冲区并在返回前释放。
 */
lmmc_status_t lmmc_fft(lmmc_real_t* real, lmmc_real_t* imag, size_t n, int inverse);

/** @brief 等价于 ::lmmc_fft(real, imag, n, 0) . */
lmmc_status_t lmmc_fft_forward(lmmc_real_t* real, lmmc_real_t* imag, size_t n);

/** @brief 等价于 ::lmmc_fft(real, imag, n, 1) . */
lmmc_status_t lmmc_fft_inverse(lmmc_real_t* real, lmmc_real_t* imag, size_t n);

#ifdef __cplusplus
}
#endif

#endif
