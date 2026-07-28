/**
 * @file lmmc.h
 * @brief LMMC 总入口头文件，聚合所有公共子模块。
 *
 * 包含此文件即可访问 LMMC 提供的全部公共 API：初始化、状态码、
 * 稠密 / 稀疏线性代数、迭代求解、特征值、插值、积分、ODE、随机数、
 * 统计、张量等。
 *
 * @copyright Copyright (c) LMMC contributors.
 */
#ifndef LMMC_H
#define LMMC_H

#include "lmmc/init.h"
#include "lmmc/status.h"
#include "lmmc/complex.h"
#include "lmmc/dense.h"
#include "lmmc/precond.h"
#include "lmmc/itersolve.h"
#include "lmmc/linear_algebra.h"
#include "lmmc/numeric.h"
#include "lmmc/nonlinear.h"
#include "lmmc/ode.h"
#include "lmmc/sparse.h"
#include "lmmc/stats.h"
#include "lmmc/tensor.h"
#include "lmmc/eigen.h"
#include "lmmc/interp.h"
#include "lmmc/quadrature.h"
#include "lmmc/random.h"
#include "lmmc/lsr_stdlib.h"

/** @brief LMMC 主版本号。 */
#define LMMC_VERSION_MAJOR 0
/** @brief LMMC 次版本号。 */
#define LMMC_VERSION_MINOR 1
/** @brief LMMC 补丁版本号。 */
#define LMMC_VERSION_PATCH 0

#endif
