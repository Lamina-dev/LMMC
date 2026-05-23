/**
 * @file quadrature.c
 * @brief Numerical integration (quadrature) module for LMMC.
 *
 * Implements:
 * - Composite trapezoidal rule (lmmc_quad_trapezoid)
 * - Composite Simpson's rule (lmmc_quad_simpson)
 * - Gauss-Legendre quadrature (lmmc_quad_gauss_legendre)
 * - Adaptive Gauss-Kronrod quadrature (lmmc_quad_adaptive) [future]
 */

#include "lmmc/quadrature.h"
#include "lmmc/config.h"
#include "lmmc/status.h"
#include "internal.h"

/* ========================================================================
 * Composite Trapezoidal Rule
 * ========================================================================
 *
 * Formula: integral ≈ h/2 * [f(a) + 2*f(a+h) + 2*f(a+2h) + ... + 2*f(b-h) + f(b)]
 * where h = (b - a) / n
 *
 * Exact for linear functions (polynomials of degree <= 1).
 * Error: O(h^2) for smooth functions.
 */
lmmc_status_t lmmc_quad_trapezoid(
    lmmc_quad_func_t func,
    void* user_data,
    lmmc_real_t a,
    lmmc_real_t b,
    size_t n,
    lmmc_real_t* out_result)
{
    /* Parameter validation */
    if (func == NULL || out_result == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (n == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Compute step size h = (b - a) / n */
    lmmc_real_t h = (b - a) / (lmmc_real_t)n;

    /* Sum = f(a) + f(b) */
    lmmc_real_t sum = func(a, user_data) + func(b, user_data);

    /* Add 2 * f(a + i*h) for i = 1, ..., n-1 */
    for (size_t i = 1; i < n; i++) {
        lmmc_real_t x_i = a + (lmmc_real_t)i * h;
        sum += 2.0 * func(x_i, user_data);
    }

    /* Result = h/2 * sum */
    *out_result = (h / 2.0) * sum;

    return LMMC_STATUS_OK;
}

/* ========================================================================
 * Composite Simpson's Rule
 * ========================================================================
 *
 * Formula: integral ≈ h/3 * [f(a) + 4*f(a+h) + 2*f(a+2h) + 4*f(a+3h) + ... + f(b)]
 * where h = (b - a) / n, and n must be even.
 *
 * Exact for polynomials of degree <= 3.
 * Error: O(h^4) for smooth functions.
 */
lmmc_status_t lmmc_quad_simpson(
    lmmc_quad_func_t func,
    void* user_data,
    lmmc_real_t a,
    lmmc_real_t b,
    size_t n,
    lmmc_real_t* out_result)
{
    /* Parameter validation */
    if (func == NULL || out_result == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (n == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (n % 2 != 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Compute step size h = (b - a) / n */
    lmmc_real_t h = (b - a) / (lmmc_real_t)n;

    /* Sum = f(a) + f(b) */
    lmmc_real_t sum = func(a, user_data) + func(b, user_data);

    /* Add weighted interior points:
     * - Odd indices (1, 3, 5, ...): coefficient 4
     * - Even indices (2, 4, 6, ...): coefficient 2
     */
    for (size_t i = 1; i < n; i++) {
        lmmc_real_t x_i = a + (lmmc_real_t)i * h;
        if (i % 2 == 1) {
            sum += 4.0 * func(x_i, user_data);
        } else {
            sum += 2.0 * func(x_i, user_data);
        }
    }

    /* Result = h/3 * sum */
    *out_result = (h / 3.0) * sum;

    return LMMC_STATUS_OK;
}

/* ========================================================================
 * Gauss-Legendre Quadrature
 * ========================================================================
 *
 * Uses precomputed nodes and weights on [-1, 1] for orders 2 through 20.
 * For an interval [a, b], the change of variable is:
 *   x = (b-a)/2 * t + (a+b)/2
 *   integral = (b-a)/2 * sum( w[i] * f(x[i]) )
 *
 * An n-point Gauss-Legendre rule is exact for polynomials of degree <= 2n-1.
 */

/* Maximum supported Gauss-Legendre order */
#define LMMC_GL_MAX_ORDER 20
#define LMMC_GL_MIN_ORDER 2

/* Static tables of Gauss-Legendre nodes and weights for orders 2-20.
 * Nodes are on [-1, 1]. Weights sum to 2.
 * Values are given to at least 16 significant digits.
 */

/* Order 2 */
static const lmmc_real_t gl_nodes_2[] = {
    -0.5773502691896257645,
     0.5773502691896257645
};
static const lmmc_real_t gl_weights_2[] = {
     1.0000000000000000000,
     1.0000000000000000000
};

/* Order 3 */
static const lmmc_real_t gl_nodes_3[] = {
    -0.7745966692414833771,
     0.0000000000000000000,
     0.7745966692414833771
};
static const lmmc_real_t gl_weights_3[] = {
     0.5555555555555555556,
     0.8888888888888888889,
     0.5555555555555555556
};

/* Order 4 */
static const lmmc_real_t gl_nodes_4[] = {
    -0.8611363115940525752,
    -0.3399810435848562648,
     0.3399810435848562648,
     0.8611363115940525752
};
static const lmmc_real_t gl_weights_4[] = {
     0.3478548451374538574,
     0.6521451548625461426,
     0.6521451548625461426,
     0.3478548451374538574
};

/* Order 5 */
static const lmmc_real_t gl_nodes_5[] = {
    -0.9061798459386639928,
    -0.5384693101056830910,
     0.0000000000000000000,
     0.5384693101056830910,
     0.9061798459386639928
};
static const lmmc_real_t gl_weights_5[] = {
     0.2369268850561890875,
     0.4786286704993664680,
     0.5688888888888888889,
     0.4786286704993664680,
     0.2369268850561890875
};

/* Order 6 */
static const lmmc_real_t gl_nodes_6[] = {
    -0.9324695142031520278,
    -0.6612093864662645137,
    -0.2386191860831969086,
     0.2386191860831969086,
     0.6612093864662645137,
     0.9324695142031520278
};
static const lmmc_real_t gl_weights_6[] = {
     0.1713244923791703450,
     0.3607615730481386076,
     0.4679139345726910474,
     0.4679139345726910474,
     0.3607615730481386076,
     0.1713244923791703450
};

/* Order 7 */
static const lmmc_real_t gl_nodes_7[] = {
    -0.9491079123427585245,
    -0.7415311855993944399,
    -0.4058451513773971669,
     0.0000000000000000000,
     0.4058451513773971669,
     0.7415311855993944399,
     0.9491079123427585245
};
static const lmmc_real_t gl_weights_7[] = {
     0.1294849661688696932,
     0.2797053914892766679,
     0.3818300505051189449,
     0.4179591836734693878,
     0.3818300505051189449,
     0.2797053914892766679,
     0.1294849661688696932
};

/* Order 8 */
static const lmmc_real_t gl_nodes_8[] = {
    -0.9602898564975362317,
    -0.7966664774136267396,
    -0.5255324099163289858,
    -0.1834346424956498049,
     0.1834346424956498049,
     0.5255324099163289858,
     0.7966664774136267396,
     0.9602898564975362317
};
static const lmmc_real_t gl_weights_8[] = {
     0.1012285362903762591,
     0.2223810344533744706,
     0.3137066458778872874,
     0.3626837833783619830,
     0.3626837833783619830,
     0.3137066458778872874,
     0.2223810344533744706,
     0.1012285362903762591
};

/* Order 9 */
static const lmmc_real_t gl_nodes_9[] = {
    -0.9681602395076260899,
    -0.8360311073266357943,
    -0.6133714327005903973,
    -0.3242534234038089290,
     0.0000000000000000000,
     0.3242534234038089290,
     0.6133714327005903973,
     0.8360311073266357943,
     0.9681602395076260899
};
static const lmmc_real_t gl_weights_9[] = {
     0.0812743883615744120,
     0.1806481606948574041,
     0.2606106964029354624,
     0.3123470770400028401,
     0.3302393550012597632,
     0.3123470770400028401,
     0.2606106964029354624,
     0.1806481606948574041,
     0.0812743883615744120
};

/* Order 10 */
static const lmmc_real_t gl_nodes_10[] = {
    -0.9739065285171717200,
    -0.8650633666889845108,
    -0.6794095682990244063,
    -0.4333953941292471908,
    -0.1488743389816312108,
     0.1488743389816312108,
     0.4333953941292471908,
     0.6794095682990244063,
     0.8650633666889845108,
     0.9739065285171717200
};
static const lmmc_real_t gl_weights_10[] = {
     0.0666713443086881376,
     0.1494513491505805931,
     0.2190863625159820440,
     0.2692667193099963550,
     0.2955242247147528702,
     0.2955242247147528702,
     0.2692667193099963550,
     0.2190863625159820440,
     0.1494513491505805931,
     0.0666713443086881376
};

/* Order 11 */
static const lmmc_real_t gl_nodes_11[] = {
    -0.9782286581460569928,
    -0.8870625997680952990,
    -0.7301520055740493240,
    -0.5190961292068118160,
    -0.2695431559523449723,
     0.0000000000000000000,
     0.2695431559523449723,
     0.5190961292068118160,
     0.7301520055740493240,
     0.8870625997680952990,
     0.9782286581460569928
};
static const lmmc_real_t gl_weights_11[] = {
     0.0556685671161736665,
     0.1255803694649046246,
     0.1862902109277342514,
     0.2331937645919904800,
     0.2628045445102466622,
     0.2729250867779006307,
     0.2628045445102466622,
     0.2331937645919904800,
     0.1862902109277342514,
     0.1255803694649046246,
     0.0556685671161736665
};

/* Order 12 */
static const lmmc_real_t gl_nodes_12[] = {
    -0.9815606342467192507,
    -0.9041172563704748567,
    -0.7699026741943046870,
    -0.5873179542866174473,
    -0.3678314989981801938,
    -0.1252334085114689155,
     0.1252334085114689155,
     0.3678314989981801938,
     0.5873179542866174473,
     0.7699026741943046870,
     0.9041172563704748567,
     0.9815606342467192507
};
static const lmmc_real_t gl_weights_12[] = {
     0.0471753363865118272,
     0.1069393259953184310,
     0.1600783285433462264,
     0.2031674267230659217,
     0.2334925365383548088,
     0.2491470458134027850,
     0.2491470458134027850,
     0.2334925365383548088,
     0.2031674267230659217,
     0.1600783285433462264,
     0.1069393259953184310,
     0.0471753363865118272
};

/* Order 13 */
static const lmmc_real_t gl_nodes_13[] = {
    -0.9841830547185881494,
    -0.9175983992229779653,
    -0.8015780907333099128,
    -0.6423493394403402206,
    -0.4484927510364468528,
    -0.2304583159551347940,
     0.0000000000000000000,
     0.2304583159551347940,
     0.4484927510364468528,
     0.6423493394403402206,
     0.8015780907333099128,
     0.9175983992229779653,
     0.9841830547185881494
};
static const lmmc_real_t gl_weights_13[] = {
     0.0404840047653158796,
     0.0921214998377284580,
     0.1388735102197872385,
     0.1781459807619457383,
     0.2078160475368885023,
     0.2262831802628972384,
     0.2325515532308739102,
     0.2262831802628972384,
     0.2078160475368885023,
     0.1781459807619457383,
     0.1388735102197872385,
     0.0921214998377284580,
     0.0404840047653158796
};

/* Order 14 */
static const lmmc_real_t gl_nodes_14[] = {
    -0.9862838086968123388,
    -0.9284348836635735173,
    -0.8272013150697649931,
    -0.6872929048116854701,
    -0.5152486363581540919,
    -0.3191123689278897604,
    -0.1080549487073436621,
     0.1080549487073436621,
     0.3191123689278897604,
     0.5152486363581540919,
     0.6872929048116854701,
     0.8272013150697649931,
     0.9284348836635735173,
     0.9862838086968123388
};
static const lmmc_real_t gl_weights_14[] = {
     0.0351194603317518630,
     0.0801580871597602098,
     0.1215185706879031847,
     0.1572031671581935345,
     0.1855383974779378138,
     0.2051984637212956040,
     0.2152638534631577902,
     0.2152638534631577902,
     0.2051984637212956040,
     0.1855383974779378138,
     0.1572031671581935345,
     0.1215185706879031847,
     0.0801580871597602098,
     0.0351194603317518630
};

/* Order 15 */
static const lmmc_real_t gl_nodes_15[] = {
    -0.9879925180204854285,
    -0.9372733924007059044,
    -0.8482065834104272162,
    -0.7244177313601700474,
    -0.5709721726085388475,
    -0.3941513470775633699,
    -0.2011940939974345223,
     0.0000000000000000000,
     0.2011940939974345223,
     0.3941513470775633699,
     0.5709721726085388475,
     0.7244177313601700474,
     0.8482065834104272162,
     0.9372733924007059044,
     0.9879925180204854285
};
static const lmmc_real_t gl_weights_15[] = {
     0.0307532419961172684,
     0.0703660474881081247,
     0.1071592204671719350,
     0.1395706779261543144,
     0.1662692058169939336,
     0.1861610000155622110,
     0.1984314853271115765,
     0.2025782419255612729,
     0.1984314853271115765,
     0.1861610000155622110,
     0.1662692058169939336,
     0.1395706779261543144,
     0.1071592204671719350,
     0.0703660474881081247,
     0.0307532419961172684
};

/* Order 16 */
static const lmmc_real_t gl_nodes_16[] = {
    -0.9894009349916499326,
    -0.9445750230732326000,
    -0.8656312023878317439,
    -0.7554044083550030339,
    -0.6178762444026437485,
    -0.4580167776572273863,
    -0.2816035507792589132,
    -0.0950125098376374402,
     0.0950125098376374402,
     0.2816035507792589132,
     0.4580167776572273863,
     0.6178762444026437485,
     0.7554044083550030339,
     0.8656312023878317439,
     0.9445750230732326000,
     0.9894009349916499326
};
static const lmmc_real_t gl_weights_16[] = {
     0.0271524594117540949,
     0.0622535239386478929,
     0.0951585116824927848,
     0.1246289712555338721,
     0.1495959888165767320,
     0.1691565193950025381,
     0.1826034150449235717,
     0.1894506104550684962,
     0.1894506104550684962,
     0.1826034150449235717,
     0.1691565193950025381,
     0.1495959888165767320,
     0.1246289712555338721,
     0.0951585116824927848,
     0.0622535239386478929,
     0.0271524594117540949
};

/* Order 17 */
static const lmmc_real_t gl_nodes_17[] = {
    -0.9905754753144173356,
    -0.9506755217687677612,
    -0.8802391537269859021,
    -0.7815140038968014069,
    -0.6576711592166907658,
    -0.5126905370864769678,
    -0.3512317634538763153,
    -0.1784841814958478558,
     0.0000000000000000000,
     0.1784841814958478558,
     0.3512317634538763153,
     0.5126905370864769678,
     0.6576711592166907658,
     0.7815140038968014069,
     0.8802391537269859021,
     0.9506755217687677612,
     0.9905754753144173356
};
static const lmmc_real_t gl_weights_17[] = {
     0.0241483028685479319,
     0.0554595293739872012,
     0.0850361483171791809,
     0.1118838471934039711,
     0.1351363684685254732,
     0.1540457610768102880,
     0.1680041021564500271,
     0.1765627053669926463,
     0.1794464703562065254,
     0.1765627053669926463,
     0.1680041021564500271,
     0.1540457610768102880,
     0.1351363684685254732,
     0.1118838471934039711,
     0.0850361483171791809,
     0.0554595293739872012,
     0.0241483028685479319
};

/* Order 18 */
static const lmmc_real_t gl_nodes_18[] = {
    -0.9915651684209309160,
    -0.9558239495713977551,
    -0.8926024664975557393,
    -0.8037049589725231156,
    -0.6916870430603532079,
    -0.5597708310739475347,
    -0.4117511614628426460,
    -0.2518862256915055095,
    -0.0847750130417353013,
     0.0847750130417353013,
     0.2518862256915055095,
     0.4117511614628426460,
     0.5597708310739475347,
     0.6916870430603532079,
     0.8037049589725231156,
     0.8926024664975557393,
     0.9558239495713977551,
     0.9915651684209309160
};
static const lmmc_real_t gl_weights_18[] = {
     0.0216160135264833103,
     0.0497145488949697964,
     0.0764257302548890565,
     0.1009420441062871655,
     0.1225552067114784602,
     0.1406429146706506512,
     0.1546846751262652449,
     0.1642764837458327229,
     0.1691423829631435918,
     0.1691423829631435918,
     0.1642764837458327229,
     0.1546846751262652449,
     0.1406429146706506512,
     0.1225552067114784602,
     0.1009420441062871655,
     0.0764257302548890565,
     0.0497145488949697964,
     0.0216160135264833103
};

/* Order 19 */
static const lmmc_real_t gl_nodes_19[] = {
    -0.9924068438435844032,
    -0.9602081521348300308,
    -0.9031559036148179016,
    -0.8227146565371428250,
    -0.7209661773352293786,
    -0.6005453046616810235,
    -0.4645707413759609457,
    -0.3165640999636298320,
    -0.1603586456402253758,
     0.0000000000000000000,
     0.1603586456402253758,
     0.3165640999636298320,
     0.4645707413759609457,
     0.6005453046616810235,
     0.7209661773352293786,
     0.8227146565371428250,
     0.9031559036148179016,
     0.9602081521348300308,
     0.9924068438435844032
};
static const lmmc_real_t gl_weights_19[] = {
     0.0194617882297264771,
     0.0448142267656996003,
     0.0690445427376412265,
     0.0914900216224499995,
     0.1115666455473339947,
     0.1287539625393362276,
     0.1426067021736066117,
     0.1527660420658596667,
     0.1589688433939543476,
     0.1610544498487836959,
     0.1589688433939543476,
     0.1527660420658596667,
     0.1426067021736066117,
     0.1287539625393362276,
     0.1115666455473339947,
     0.0914900216224499995,
     0.0690445427376412265,
     0.0448142267656996003,
     0.0194617882297264771
};

/* Order 20 */
static const lmmc_real_t gl_nodes_20[] = {
    -0.9931285991850949247,
    -0.9639719272779137912,
    -0.9122344282513259059,
    -0.8391169718222188234,
    -0.7463319064601507926,
    -0.6360536807265150254,
    -0.5108670019508270980,
    -0.3737060887154195607,
    -0.2277858511416450681,
    -0.0765265211334973338,
     0.0765265211334973338,
     0.2277858511416450681,
     0.3737060887154195607,
     0.5108670019508270980,
     0.6360536807265150254,
     0.7463319064601507926,
     0.8391169718222188234,
     0.9122344282513259059,
     0.9639719272779137912,
     0.9931285991850949247
};
static const lmmc_real_t gl_weights_20[] = {
     0.0176140071391521183,
     0.0406014298003869413,
     0.0626720483341090636,
     0.0832767415767047487,
     0.1019301198172404351,
     0.1181945319615184174,
     0.1316886384491766269,
     0.1420961093183820514,
     0.1491729864726037467,
     0.1527533871307258507,
     0.1527533871307258507,
     0.1491729864726037467,
     0.1420961093183820514,
     0.1316886384491766269,
     0.1181945319615184174,
     0.1019301198172404351,
     0.0832767415767047487,
     0.0626720483341090636,
     0.0406014298003869413,
     0.0176140071391521183
};

/* Lookup tables for nodes and weights by order (index = order - 2) */
static const lmmc_real_t* const gl_nodes_table[] = {
    gl_nodes_2,  gl_nodes_3,  gl_nodes_4,  gl_nodes_5,
    gl_nodes_6,  gl_nodes_7,  gl_nodes_8,  gl_nodes_9,
    gl_nodes_10, gl_nodes_11, gl_nodes_12, gl_nodes_13,
    gl_nodes_14, gl_nodes_15, gl_nodes_16, gl_nodes_17,
    gl_nodes_18, gl_nodes_19, gl_nodes_20
};

static const lmmc_real_t* const gl_weights_table[] = {
    gl_weights_2,  gl_weights_3,  gl_weights_4,  gl_weights_5,
    gl_weights_6,  gl_weights_7,  gl_weights_8,  gl_weights_9,
    gl_weights_10, gl_weights_11, gl_weights_12, gl_weights_13,
    gl_weights_14, gl_weights_15, gl_weights_16, gl_weights_17,
    gl_weights_18, gl_weights_19, gl_weights_20
};

lmmc_status_t lmmc_quad_gauss_legendre(
    lmmc_quad_func_t func,
    void* user_data,
    lmmc_real_t a,
    lmmc_real_t b,
    size_t order,
    lmmc_real_t* out_result)
{
    /* Parameter validation */
    if (func == NULL || out_result == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (order < LMMC_GL_MIN_ORDER || order > LMMC_GL_MAX_ORDER) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Get nodes and weights for the requested order */
    const lmmc_real_t* nodes   = gl_nodes_table[order - LMMC_GL_MIN_ORDER];
    const lmmc_real_t* weights = gl_weights_table[order - LMMC_GL_MIN_ORDER];

    /* Interval transformation coefficients:
     * x = half_len * t + midpoint
     * where half_len = (b - a) / 2, midpoint = (a + b) / 2
     */
    lmmc_real_t half_len = (b - a) / 2.0;
    lmmc_real_t midpoint = (a + b) / 2.0;

    /* Compute weighted sum: sum( w[i] * f(x[i]) ) */
    lmmc_real_t sum = 0.0;
    for (size_t i = 0; i < order; i++) {
        lmmc_real_t x_i = half_len * nodes[i] + midpoint;
        sum += weights[i] * func(x_i, user_data);
    }

    /* Result = (b - a) / 2 * sum */
    *out_result = half_len * sum;

    return LMMC_STATUS_OK;
}

/* ========================================================================
 * Adaptive Gauss-Kronrod Quadrature (G7/K15)
 * ========================================================================
 *
 * Uses a 7-point Gauss rule embedded in a 15-point Kronrod rule.
 * The Kronrod rule reuses all 7 Gauss points plus 8 additional points,
 * so only 15 function evaluations are needed per subinterval to get both
 * a 7-point and a 15-point estimate.
 *
 * Error estimate: |K15 - G7| for each subinterval.
 * Acceptance criterion: error <= max(abs_tol, rel_tol * |K15_total|)
 * If max_depth is reached, returns LMMC_STATUS_WARNING_MAX_DEPTH with
 * the current best estimate.
 */

/* 15-point Kronrod nodes on [-1, 1] (sorted in increasing order).
 * The 7 Gauss nodes are at indices 1, 3, 5, 7, 9, 11, 13 (odd indices). */
static const lmmc_real_t gk15_nodes[15] = {
    -0.9914553711208126392,
    -0.9491079123427585245,
    -0.8648644233597690728,
    -0.7415311855993944399,
    -0.5860872354676911303,
    -0.4058451513773971669,
    -0.2077849550078984676,
     0.0000000000000000000,
     0.2077849550078984676,
     0.4058451513773971669,
     0.5860872354676911303,
     0.7415311855993944399,
     0.8648644233597690728,
     0.9491079123427585245,
     0.9914553711208126392
};

/* 15-point Kronrod weights on [-1, 1] */
static const lmmc_real_t gk15_weights[15] = {
     0.0229353220105292250,
     0.0630920926299785533,
     0.1047900103222501838,
     0.1406532597155259187,
     0.1690047266392679028,
     0.1903505780647854099,
     0.2044329400752988924,
     0.2094821410847278280,
     0.2044329400752988924,
     0.1903505780647854099,
     0.1690047266392679028,
     0.1406532597155259187,
     0.1047900103222501838,
     0.0630920926299785533,
     0.0229353220105292250
};

/* 7-point Gauss weights on [-1, 1].
 * These correspond to the Gauss nodes which are at Kronrod indices
 * 1, 3, 5, 7, 9, 11, 13. */
static const lmmc_real_t g7_weights[7] = {
     0.1294849661688696932,
     0.2797053914892766679,
     0.3818300505051189449,
     0.4179591836734693878,
     0.3818300505051189449,
     0.2797053914892766679,
     0.1294849661688696932
};

/* Indices into gk15_nodes that correspond to the 7 Gauss points */
static const int g7_kronrod_idx[7] = { 1, 3, 5, 7, 9, 11, 13 };

/**
 * @brief Internal recursive helper for adaptive Gauss-Kronrod quadrature.
 *
 * Computes the G7 and K15 estimates on [a, b]. If the error is within
 * tolerance, accepts the K15 estimate. Otherwise, subdivides at the
 * midpoint and recurses on each half.
 *
 * @param func        Integrand function pointer.
 * @param user_data   User data passed to func.
 * @param a           Left endpoint of the interval.
 * @param b           Right endpoint of the interval.
 * @param abs_tol     Absolute tolerance for this subinterval.
 * @param rel_tol     Relative tolerance.
 * @param depth       Current recursion depth.
 * @param max_depth   Maximum allowed recursion depth.
 * @param out_value   Output: integral estimate on this interval.
 * @param out_error   Output: error estimate on this interval.
 * @param out_evals   Output: number of function evaluations used.
 * @return LMMC_STATUS_OK or LMMC_STATUS_WARNING_MAX_DEPTH.
 */
static lmmc_status_t gk15_adaptive_recursive(
    lmmc_quad_func_t func,
    void* user_data,
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t abs_tol,
    lmmc_real_t rel_tol,
    size_t depth,
    size_t max_depth,
    lmmc_real_t* out_value,
    lmmc_real_t* out_error,
    size_t* out_evals)
{
    /* Transform from [-1,1] to [a,b]: x = half_len * t + midpoint */
    lmmc_real_t half_len = (b - a) / 2.0;
    lmmc_real_t midpoint = (a + b) / 2.0;

    /* Evaluate function at all 15 Kronrod points */
    lmmc_real_t fvals[15];
    for (int i = 0; i < 15; i++) {
        lmmc_real_t x_i = half_len * gk15_nodes[i] + midpoint;
        fvals[i] = func(x_i, user_data);
    }
    *out_evals = 15;

    /* Compute K15 estimate */
    lmmc_real_t k15_sum = 0.0;
    for (int i = 0; i < 15; i++) {
        k15_sum += gk15_weights[i] * fvals[i];
    }
    lmmc_real_t k15_result = half_len * k15_sum;

    /* Compute G7 estimate using only the 7 Gauss points */
    lmmc_real_t g7_sum = 0.0;
    for (int i = 0; i < 7; i++) {
        g7_sum += g7_weights[i] * fvals[g7_kronrod_idx[i]];
    }
    lmmc_real_t g7_result = half_len * g7_sum;

    /* Error estimate = |K15 - G7| */
    lmmc_real_t error = lmmc_abs(k15_result - g7_result);

    /* Acceptance criterion: error <= max(abs_tol, rel_tol * |K15|) */
    lmmc_real_t tolerance = lmmc_max(abs_tol, rel_tol * lmmc_abs(k15_result));

    if (error <= tolerance) {
        /* Accept this interval */
        *out_value = k15_result;
        *out_error = error;
        return LMMC_STATUS_OK;
    }

    /* Check if we've reached max depth */
    if (depth >= max_depth) {
        /* Return current best estimate with warning */
        *out_value = k15_result;
        *out_error = error;
        return LMMC_STATUS_WARNING_MAX_DEPTH;
    }

    /* Subdivide at midpoint and recurse */
    lmmc_real_t mid = (a + b) / 2.0;
    lmmc_real_t left_value, left_error;
    size_t left_evals;
    lmmc_real_t right_value, right_error;
    size_t right_evals;

    /* Each half gets half the absolute tolerance (to maintain global tolerance) */
    lmmc_real_t sub_abs_tol = abs_tol / 2.0;

    lmmc_status_t left_status = gk15_adaptive_recursive(
        func, user_data, a, mid,
        sub_abs_tol, rel_tol,
        depth + 1, max_depth,
        &left_value, &left_error, &left_evals);

    lmmc_status_t right_status = gk15_adaptive_recursive(
        func, user_data, mid, b,
        sub_abs_tol, rel_tol,
        depth + 1, max_depth,
        &right_value, &right_error, &right_evals);

    /* Combine results */
    *out_value = left_value + right_value;
    *out_error = left_error + right_error;
    *out_evals += left_evals + right_evals;

    /* If either half hit max depth, propagate the warning */
    if (left_status == LMMC_STATUS_WARNING_MAX_DEPTH ||
        right_status == LMMC_STATUS_WARNING_MAX_DEPTH) {
        return LMMC_STATUS_WARNING_MAX_DEPTH;
    }

    return LMMC_STATUS_OK;
}

/* ========================================================================
 * Public API: lmmc_quad_adaptive
 * ======================================================================== */
lmmc_status_t lmmc_quad_adaptive(
    lmmc_quad_func_t func,
    void* user_data,
    lmmc_real_t a,
    lmmc_real_t b,
    lmmc_real_t abs_tol,
    lmmc_real_t rel_tol,
    size_t max_depth,
    lmmc_quad_result_t* out_result)
{
    /* Parameter validation */
    if (func == NULL || out_result == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (abs_tol < 0.0 || rel_tol < 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Initialize output */
    out_result->value = 0.0;
    out_result->error = 0.0;
    out_result->num_evals = 0;

    /* Run recursive adaptive integration */
    lmmc_real_t value, error;
    size_t evals;

    lmmc_status_t status = gk15_adaptive_recursive(
        func, user_data, a, b,
        abs_tol, rel_tol,
        0, max_depth,
        &value, &error, &evals);

    /* Store results */
    out_result->value = value;
    out_result->error = error;
    out_result->num_evals = evals;

    return status;
}
