/**
 * @file quadrature.c
 * @brief 一元数值积分实现：梯形、Simpson、Gauss、自适应、Romberg、Tanh-Sinh、Gauss-Hermite、Gauss-Laguerre。
 */
#include "lmmc/quadrature.h"
#include "lmmc/config.h"
#include "lmmc/status.h"
#include "internal.h"


lmmc_status_t lmmc_quad_trapezoid(
    lmmc_quad_func_t func,
    void* user_data,
    lmmc_real_t a,
    lmmc_real_t b,
    size_t n,
    lmmc_real_t* out_result)
{
    if (func == NULL || out_result == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (n == 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_real_t h = (b - a) / (lmmc_real_t)n;
    lmmc_real_t sum = func(a, user_data) + func(b, user_data);

    for (size_t i = 1; i < n; i++) {
        lmmc_real_t x_i = a + (lmmc_real_t)i * h;
        sum += 2.0 * func(x_i, user_data);
    }

    *out_result = (h / 2.0) * sum;
    return LMMC_STATUS_OK;
}


lmmc_status_t lmmc_quad_simpson(
    lmmc_quad_func_t func,
    void* user_data,
    lmmc_real_t a,
    lmmc_real_t b,
    size_t n,
    lmmc_real_t* out_result)
{
    if (func == NULL || out_result == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (n == 0 || n % 2 != 0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_real_t h = (b - a) / (lmmc_real_t)n;
    lmmc_real_t sum = func(a, user_data) + func(b, user_data);

    for (size_t i = 1; i < n; i++) {
        lmmc_real_t x_i = a + (lmmc_real_t)i * h;
        if (i % 2 == 1) {
            sum += 4.0 * func(x_i, user_data);
        } else {
            sum += 2.0 * func(x_i, user_data);
        }
    }

    *out_result = (h / 3.0) * sum;
    return LMMC_STATUS_OK;
}

#define LMMC_GL_MAX_ORDER 20
#define LMMC_GL_MIN_ORDER 2

static const lmmc_real_t gl_nodes_2[] = {
    -0.5773502691896257645, 0.5773502691896257645
};
static const lmmc_real_t gl_weights_2[] = {
    1.0000000000000000000, 1.0000000000000000000
};

static const lmmc_real_t gl_nodes_3[] = {
    -0.7745966692414833771, 0.0000000000000000000, 0.7745966692414833771
};
static const lmmc_real_t gl_weights_3[] = {
    0.5555555555555555556, 0.8888888888888888889, 0.5555555555555555556
};

static const lmmc_real_t gl_nodes_4[] = {
    -0.8611363115940525752, -0.3399810435848562648,
     0.3399810435848562648,  0.8611363115940525752
};
static const lmmc_real_t gl_weights_4[] = {
    0.3478548451374538574, 0.6521451548625461426,
    0.6521451548625461426, 0.3478548451374538574
};

static const lmmc_real_t gl_nodes_5[] = {
    -0.9061798459386639928, -0.5384693101056830910, 0.0000000000000000000,
     0.5384693101056830910,  0.9061798459386639928
};
static const lmmc_real_t gl_weights_5[] = {
    0.2369268850561890875, 0.4786286704993664680, 0.5688888888888888889,
    0.4786286704993664680, 0.2369268850561890875
};

static const lmmc_real_t gl_nodes_6[] = {
    -0.9324695142031520278, -0.6612093864662645137, -0.2386191860831969086,
     0.2386191860831969086,  0.6612093864662645137,  0.9324695142031520278
};
static const lmmc_real_t gl_weights_6[] = {
    0.1713244923791703450, 0.3607615730481386076, 0.4679139345726910474,
    0.4679139345726910474, 0.3607615730481386076, 0.1713244923791703450
};

static const lmmc_real_t gl_nodes_7[] = {
    -0.9491079123427585245, -0.7415311855993944399, -0.4058451513773971669,
     0.0000000000000000000,  0.4058451513773971669,  0.7415311855993944399,
     0.9491079123427585245
};
static const lmmc_real_t gl_weights_7[] = {
    0.1294849661688696932, 0.2797053914892766679, 0.3818300505051189449,
    0.4179591836734693878, 0.3818300505051189449, 0.2797053914892766679,
    0.1294849661688696932
};

static const lmmc_real_t gl_nodes_8[] = {
    -0.9602898564975362317, -0.7966664774136267396,
    -0.5255324099163289858, -0.1834346424956498049,
     0.1834346424956498049,  0.5255324099163289858,
     0.7966664774136267396,  0.9602898564975362317
};
static const lmmc_real_t gl_weights_8[] = {
    0.1012285362903762591, 0.2223810344533744706,
    0.3137066458778872874, 0.3626837833783619830,
    0.3626837833783619830, 0.3137066458778872874,
    0.2223810344533744706, 0.1012285362903762591
};

static const lmmc_real_t gl_nodes_9[] = {
    -0.9681602395076260899, -0.8360311073266357943, -0.6133714327005903973,
    -0.3242534234038089290,  0.0000000000000000000,  0.3242534234038089290,
     0.6133714327005903973,  0.8360311073266357943,  0.9681602395076260899
};
static const lmmc_real_t gl_weights_9[] = {
    0.0812743883615744120, 0.1806481606948574041, 0.2606106964029354624,
    0.3123470770400028401, 0.3302393550012597632, 0.3123470770400028401,
    0.2606106964029354624, 0.1806481606948574041, 0.0812743883615744120
};

static const lmmc_real_t gl_nodes_10[] = {
    -0.9739065285171717200, -0.8650633666889845108, -0.6794095682990244063,
    -0.4333953941292471908, -0.1488743389816312108,  0.1488743389816312108,
     0.4333953941292471908,  0.6794095682990244063,  0.8650633666889845108,
     0.9739065285171717200
};
static const lmmc_real_t gl_weights_10[] = {
    0.0666713443086881376, 0.1494513491505805931, 0.2190863625159820440,
    0.2692667193099963550, 0.2955242247147528702, 0.2955242247147528702,
    0.2692667193099963550, 0.2190863625159820440, 0.1494513491505805931,
    0.0666713443086881376
};

static const lmmc_real_t gl_nodes_11[] = {
    -0.9782286581460569928, -0.8870625997680952990, -0.7301520055740493240,
    -0.5190961292068118160, -0.2695431559523449723,  0.0000000000000000000,
     0.2695431559523449723,  0.5190961292068118160,  0.7301520055740493240,
     0.8870625997680952990,  0.9782286581460569928
};
static const lmmc_real_t gl_weights_11[] = {
    0.0556685671161736665, 0.1255803694649046246, 0.1862902109277342514,
    0.2331937645919904800, 0.2628045445102466622, 0.2729250867779006307,
    0.2628045445102466622, 0.2331937645919904800, 0.1862902109277342514,
    0.1255803694649046246, 0.0556685671161736665
};

static const lmmc_real_t gl_nodes_12[] = {
    -0.9815606342467192507, -0.9041172563704748567, -0.7699026741943046870,
    -0.5873179542866174473, -0.3678314989981801938, -0.1252334085114689155,
     0.1252334085114689155,  0.3678314989981801938,  0.5873179542866174473,
     0.7699026741943046870,  0.9041172563704748567,  0.9815606342467192507
};
static const lmmc_real_t gl_weights_12[] = {
    0.0471753363865118272, 0.1069393259953184310, 0.1600783285433462264,
    0.2031674267230659217, 0.2334925365383548088, 0.2491470458134027850,
    0.2491470458134027850, 0.2334925365383548088, 0.2031674267230659217,
    0.1600783285433462264, 0.1069393259953184310, 0.0471753363865118272
};

static const lmmc_real_t gl_nodes_13[] = {
    -0.9841830547185881494, -0.9175983992229779653, -0.8015780907333099128,
    -0.6423493394403402206, -0.4484927510364468528, -0.2304583159551347940,
     0.0000000000000000000,  0.2304583159551347940,  0.4484927510364468528,
     0.6423493394403402206,  0.8015780907333099128,  0.9175983992229779653,
     0.9841830547185881494
};
static const lmmc_real_t gl_weights_13[] = {
    0.0404840047653158796, 0.0921214998377284580, 0.1388735102197872385,
    0.1781459807619457383, 0.2078160475368885023, 0.2262831802628972384,
    0.2325515532308739102, 0.2262831802628972384, 0.2078160475368885023,
    0.1781459807619457383, 0.1388735102197872385, 0.0921214998377284580,
    0.0404840047653158796
};

static const lmmc_real_t gl_nodes_14[] = {
    -0.9862838086968123388, -0.9284348836635735173, -0.8272013150697649931,
    -0.6872929048116854701, -0.5152486363581540919, -0.3191123689278897604,
    -0.1080549487073436621,  0.1080549487073436621,  0.3191123689278897604,
     0.5152486363581540919,  0.6872929048116854701,  0.8272013150697649931,
     0.9284348836635735173,  0.9862838086968123388
};
static const lmmc_real_t gl_weights_14[] = {
    0.0351194603317518630, 0.0801580871597602098, 0.1215185706879031847,
    0.1572031671581935345, 0.1855383974779378138, 0.2051984637212956040,
    0.2152638534631577902, 0.2152638534631577902, 0.2051984637212956040,
    0.1855383974779378138, 0.1572031671581935345, 0.1215185706879031847,
    0.0801580871597602098, 0.0351194603317518630
};

static const lmmc_real_t gl_nodes_15[] = {
    -0.9879925180204854285, -0.9372733924007059044, -0.8482065834104272162,
    -0.7244177313601700474, -0.5709721726085388475, -0.3941513470775633699,
    -0.2011940939974345223,  0.0000000000000000000,  0.2011940939974345223,
     0.3941513470775633699,  0.5709721726085388475,  0.7244177313601700474,
     0.8482065834104272162,  0.9372733924007059044,  0.9879925180204854285
};
static const lmmc_real_t gl_weights_15[] = {
    0.0307532419961172684, 0.0703660474881081247, 0.1071592204671719350,
    0.1395706779261543144, 0.1662692058169939336, 0.1861610000155622110,
    0.1984314853271115765, 0.2025782419255612729, 0.1984314853271115765,
    0.1861610000155622110, 0.1662692058169939336, 0.1395706779261543144,
    0.1071592204671719350, 0.0703660474881081247, 0.0307532419961172684
};

static const lmmc_real_t gl_nodes_16[] = {
    -0.9894009349916499326, -0.9445750230732326000, -0.8656312023878317439,
    -0.7554044083550030339, -0.6178762444026437485, -0.4580167776572273863,
    -0.2816035507792589132, -0.0950125098376374402,  0.0950125098376374402,
     0.2816035507792589132,  0.4580167776572273863,  0.6178762444026437485,
     0.7554044083550030339,  0.8656312023878317439,  0.9445750230732326000,
     0.9894009349916499326
};
static const lmmc_real_t gl_weights_16[] = {
    0.0271524594117540949, 0.0622535239386478929, 0.0951585116824927848,
    0.1246289712555338721, 0.1495959888165767320, 0.1691565193950025381,
    0.1826034150449235717, 0.1894506104550684962, 0.1894506104550684962,
    0.1826034150449235717, 0.1691565193950025381, 0.1495959888165767320,
    0.1246289712555338721, 0.0951585116824927848, 0.0622535239386478929,
    0.0271524594117540949
};

static const lmmc_real_t gl_nodes_17[] = {
    -0.9905754753144173356, -0.9506755217687677612, -0.8802391537269859021,
    -0.7815140038968014069, -0.6576711592166907658, -0.5126905370864769678,
    -0.3512317634538763153, -0.1784841814958478558,  0.0000000000000000000,
     0.1784841814958478558,  0.3512317634538763153,  0.5126905370864769678,
     0.6576711592166907658,  0.7815140038968014069,  0.8802391537269859021,
     0.9506755217687677612,  0.9905754753144173356
};
static const lmmc_real_t gl_weights_17[] = {
    0.0241483028685479319, 0.0554595293739872012, 0.0850361483171791809,
    0.1118838471934039711, 0.1351363684685254732, 0.1540457610768102880,
    0.1680041021564500271, 0.1765627053669926463, 0.1794464703562065254,
    0.1765627053669926463, 0.1680041021564500271, 0.1540457610768102880,
    0.1351363684685254732, 0.1118838471934039711, 0.0850361483171791809,
    0.0554595293739872012, 0.0241483028685479319
};

static const lmmc_real_t gl_nodes_18[] = {
    -0.9915651684209309160, -0.9558239495713977551, -0.8926024664975557393,
    -0.8037049589725231156, -0.6916870430603532079, -0.5597708310739475347,
    -0.4117511614628426460, -0.2518862256915055095, -0.0847750130417353013,
     0.0847750130417353013,  0.2518862256915055095,  0.4117511614628426460,
     0.5597708310739475347,  0.6916870430603532079,  0.8037049589725231156,
     0.8926024664975557393,  0.9558239495713977551,  0.9915651684209309160
};
static const lmmc_real_t gl_weights_18[] = {
    0.0216160135264833103, 0.0497145488949697964, 0.0764257302548890565,
    0.1009420441062871655, 0.1225552067114784602, 0.1406429146706506512,
    0.1546846751262652449, 0.1642764837458327229, 0.1691423829631435918,
    0.1691423829631435918, 0.1642764837458327229, 0.1546846751262652449,
    0.1406429146706506512, 0.1225552067114784602, 0.1009420441062871655,
    0.0764257302548890565, 0.0497145488949697964, 0.0216160135264833103
};

static const lmmc_real_t gl_nodes_19[] = {
    -0.9924068438435844032, -0.9602081521348300308, -0.9031559036148179016,
    -0.8227146565371428250, -0.7209661773352293786, -0.6005453046616810235,
    -0.4645707413759609457, -0.3165640999636298320, -0.1603586456402253758,
     0.0000000000000000000,  0.1603586456402253758,  0.3165640999636298320,
     0.4645707413759609457,  0.6005453046616810235,  0.7209661773352293786,
     0.8227146565371428250,  0.9031559036148179016,  0.9602081521348300308,
     0.9924068438435844032
};
static const lmmc_real_t gl_weights_19[] = {
    0.0194617882297264771, 0.0448142267656996003, 0.0690445427376412265,
    0.0914900216224499995, 0.1115666455473339947, 0.1287539625393362276,
    0.1426067021736066117, 0.1527660420658596667, 0.1589688433939543476,
    0.1610544498487836959, 0.1589688433939543476, 0.1527660420658596667,
    0.1426067021736066117, 0.1287539625393362276, 0.1115666455473339947,
    0.0914900216224499995, 0.0690445427376412265, 0.0448142267656996003,
    0.0194617882297264771
};

static const lmmc_real_t gl_nodes_20[] = {
    -0.9931285991850949247, -0.9639719272779137912, -0.9122344282513259059,
    -0.8391169718222188234, -0.7463319064601507926, -0.6360536807265150254,
    -0.5108670019508270980, -0.3737060887154195607, -0.2277858511416450681,
    -0.0765265211334973338,  0.0765265211334973338,  0.2277858511416450681,
     0.3737060887154195607,  0.5108670019508270980,  0.6360536807265150254,
     0.7463319064601507926,  0.8391169718222188234,  0.9122344282513259059,
     0.9639719272779137912,  0.9931285991850949247
};
static const lmmc_real_t gl_weights_20[] = {
    0.0176140071391521183, 0.0406014298003869413, 0.0626720483341090636,
    0.0832767415767047487, 0.1019301198172404351, 0.1181945319615184174,
    0.1316886384491766269, 0.1420961093183820514, 0.1491729864726037467,
    0.1527533871307258507, 0.1527533871307258507, 0.1491729864726037467,
    0.1420961093183820514, 0.1316886384491766269, 0.1181945319615184174,
    0.1019301198172404351, 0.0832767415767047487, 0.0626720483341090636,
    0.0406014298003869413, 0.0176140071391521183
};

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
    if (func == NULL || out_result == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (order < LMMC_GL_MIN_ORDER || order > LMMC_GL_MAX_ORDER) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    const lmmc_real_t* nodes   = gl_nodes_table[order - LMMC_GL_MIN_ORDER];
    const lmmc_real_t* weights = gl_weights_table[order - LMMC_GL_MIN_ORDER];

    lmmc_real_t half_len = (b - a) / 2.0;
    lmmc_real_t midpoint = (a + b) / 2.0;

    lmmc_real_t sum = 0.0;
    for (size_t i = 0; i < order; i++) {
        lmmc_real_t x_i = half_len * nodes[i] + midpoint;
        sum += weights[i] * func(x_i, user_data);
    }

    *out_result = half_len * sum;
    return LMMC_STATUS_OK;
}

static const lmmc_real_t gk15_nodes[15] = {
    -0.9914553711208126392, -0.9491079123427585245, -0.8648644233597690728,
    -0.7415311855993944399, -0.5860872354676911303, -0.4058451513773971669,
    -0.2077849550078984676,  0.0000000000000000000,  0.2077849550078984676,
     0.4058451513773971669,  0.5860872354676911303,  0.7415311855993944399,
     0.8648644233597690728,  0.9491079123427585245,  0.9914553711208126392
};

static const lmmc_real_t gk15_weights[15] = {
    0.0229353220105292250, 0.0630920926299785533, 0.1047900103222501838,
    0.1406532597155259187, 0.1690047266392679028, 0.1903505780647854099,
    0.2044329400752988924, 0.2094821410847278280, 0.2044329400752988924,
    0.1903505780647854099, 0.1690047266392679028, 0.1406532597155259187,
    0.1047900103222501838, 0.0630920926299785533, 0.0229353220105292250
};

static const lmmc_real_t g7_weights[7] = {
    0.1294849661688696932, 0.2797053914892766679, 0.3818300505051189449,
    0.4179591836734693878, 0.3818300505051189449, 0.2797053914892766679,
    0.1294849661688696932
};

static const int g7_kronrod_idx[7] = { 1, 3, 5, 7, 9, 11, 13 };

static lmmc_status_t gk15_adaptive_recursive(
    lmmc_quad_func_t func, void* user_data,
    lmmc_real_t a, lmmc_real_t b,
    lmmc_real_t abs_tol, lmmc_real_t rel_tol,
    size_t depth, size_t max_depth,
    lmmc_real_t* out_value, lmmc_real_t* out_error, size_t* out_evals)
{
    lmmc_real_t half_len = (b - a) / 2.0;
    lmmc_real_t midpoint = (a + b) / 2.0;

    lmmc_real_t fvals[15];
    for (int i = 0; i < 15; i++) {
        lmmc_real_t x_i = half_len * gk15_nodes[i] + midpoint;
        fvals[i] = func(x_i, user_data);
    }
    *out_evals = 15;

    lmmc_real_t k15_sum = 0.0;
    for (int i = 0; i < 15; i++) {
        k15_sum += gk15_weights[i] * fvals[i];
    }
    lmmc_real_t k15_result = half_len * k15_sum;

    lmmc_real_t g7_sum = 0.0;
    for (int i = 0; i < 7; i++) {
        g7_sum += g7_weights[i] * fvals[g7_kronrod_idx[i]];
    }
    lmmc_real_t g7_result = half_len * g7_sum;

    lmmc_real_t error = lmmc_abs(k15_result - g7_result);
    lmmc_real_t tolerance = lmmc_max(abs_tol, rel_tol * lmmc_abs(k15_result));

    if (error <= tolerance) {
        *out_value = k15_result;
        *out_error = error;
        return LMMC_STATUS_OK;
    }

    if (depth >= max_depth) {
        *out_value = k15_result;
        *out_error = error;
        return LMMC_STATUS_WARNING_MAX_DEPTH;
    }

    lmmc_real_t mid = (a + b) / 2.0;
    lmmc_real_t left_value, left_error, right_value, right_error;
    size_t left_evals, right_evals;

    /* Adaptive tolerance distribution: split tolerance proportionally */
    lmmc_real_t sub_abs_tol = abs_tol / 2.0;

    lmmc_status_t left_status = gk15_adaptive_recursive(
        func, user_data, a, mid,
        sub_abs_tol, rel_tol, depth + 1, max_depth,
        &left_value, &left_error, &left_evals);

    lmmc_status_t right_status = gk15_adaptive_recursive(
        func, user_data, mid, b,
        sub_abs_tol, rel_tol, depth + 1, max_depth,
        &right_value, &right_error, &right_evals);

    *out_value = left_value + right_value;
    *out_error = left_error + right_error;
    *out_evals += left_evals + right_evals;

    if (left_status == LMMC_STATUS_WARNING_MAX_DEPTH ||
        right_status == LMMC_STATUS_WARNING_MAX_DEPTH) {
        return LMMC_STATUS_WARNING_MAX_DEPTH;
    }
    return LMMC_STATUS_OK;
}

lmmc_status_t lmmc_quad_adaptive(
    lmmc_quad_func_t func, void* user_data,
    lmmc_real_t a, lmmc_real_t b,
    lmmc_real_t abs_tol, lmmc_real_t rel_tol,
    size_t max_depth, lmmc_quad_result_t* out_result)
{
    if (func == NULL || out_result == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (abs_tol < 0.0 || rel_tol < 0.0) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    out_result->value = 0.0;
    out_result->error = 0.0;
    out_result->num_evals = 0;

    lmmc_real_t value, error;
    size_t evals;

    lmmc_status_t status = gk15_adaptive_recursive(
        func, user_data, a, b,
        abs_tol, rel_tol, 0, max_depth,
        &value, &error, &evals);

    out_result->value = value;
    out_result->error = error;
    out_result->num_evals = evals;
    return status;
}

lmmc_status_t lmmc_quad_romberg(
    lmmc_quad_func_t f, void* ud,
    lmmc_real_t a, lmmc_real_t b,
    lmmc_real_t abs_tol, size_t max_iter,
    lmmc_quad_result_t* out)
{
    if (f == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (abs_tol < 1e-15 || abs_tol > 1e-1) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (max_iter < 1 || max_iter > 1000000) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Cap Romberg table rows to avoid huge stack allocation */
    size_t max_rows = max_iter;
    if (max_rows > 30) max_rows = 30;

    /* Two-row Romberg table */
    lmmc_real_t prev[30];
    lmmc_real_t curr[30];
    size_t num_evals = 0;

    /* R[0][0] = trapezoidal rule with 1 interval */
    lmmc_real_t h = b - a;
    prev[0] = h * (f(a, ud) + f(b, ud)) / 2.0;
    num_evals = 2;

    out->value = prev[0];
    out->error = lmmc_abs(prev[0]);
    out->num_evals = num_evals;

    if (max_rows == 1) {
        return LMMC_STATUS_OK;
    }

    for (size_t i = 1; i < max_rows; i++) {
        /* Trapezoidal rule with 2^i intervals */
        size_t n = (size_t)1 << i;
        h = (b - a) / (lmmc_real_t)n;

        /* Add new midpoints (only the odd-indexed points are new) */
        lmmc_real_t sum_new = 0.0;
        for (size_t k = 1; k <= n / 2; k++) {
            lmmc_real_t x = a + (2.0 * (lmmc_real_t)k - 1.0) * h;
            sum_new += f(x, ud);
            num_evals++;
        }
        curr[0] = prev[0] / 2.0 + h * sum_new;

        /* Richardson extrapolation */
        lmmc_real_t pow4 = 1.0;
        for (size_t j = 1; j <= i; j++) {
            pow4 *= 4.0;
            curr[j] = (pow4 * curr[j - 1] - prev[j - 1]) / (pow4 - 1.0);
        }

        /* Error estimate */
        lmmc_real_t err = lmmc_abs(curr[i] - prev[i - 1]);
        out->value = curr[i];
        out->error = err;
        out->num_evals = num_evals;

        if (err <= abs_tol) {
            return LMMC_STATUS_OK;
        }

        /* Copy curr to prev */
        for (size_t j = 0; j <= i; j++) {
            prev[j] = curr[j];
        }
    }

    /* Reached max iterations - return best estimate */
    return LMMC_STATUS_CONVERGENCE_FAILED;
}


/**
 * Tanh-Sinh (Double Exponential) 积分
 *
 * The double-exponential transformation maps [a,b] to (-inf, inf):
 *   x(t) = mid + half_len * tanh(pi/2 * sinh(t))
 *   dx/dt = half_len * (pi/2) * cosh(t) / cosh^2(pi/2 * sinh(t))
 *
 * We compute the integral as h * sum_j w_j * f(x_j) where
 *   w_j = (pi/2) * cosh(j*h) / cosh^2(pi/2 * sinh(j*h))
 *
 * Convergence is checked by comparing successive levels (halving h).
 */

lmmc_status_t lmmc_quad_tanh_sinh(
    lmmc_quad_func_t f, void* ud,
    lmmc_real_t a, lmmc_real_t b,
    lmmc_real_t abs_tol, size_t max_nodes,
    lmmc_quad_result_t* out)
{
    if (f == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (a >= b) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (abs_tol < 1e-15 || abs_tol > 1e-1) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (max_nodes < 1 || max_nodes > 1000000) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    lmmc_real_t mid = (a + b) / 2.0;
    lmmc_real_t half_len = (b - a) / 2.0;
    lmmc_real_t pi_half = LMMC_CONST_PI / 2.0;

    size_t num_evals = 0;
    int converged = 0;

    /*
     * Tanh-Sinh quadrature: compute the integral using progressively
     * smaller step sizes. At each level, compute the full trapezoidal
     * sum from scratch (simple but correct approach).
     */
    lmmc_real_t prev_integral = 0.0;
    lmmc_real_t integral = 0.0;
    size_t max_levels = 10;

    /* Step sizes: start small for good coverage near singularities */
    lmmc_real_t h = 0.03125;  /* 1/32 */

    for (size_t level = 0; level < max_levels && num_evals < max_nodes; level++) {
        lmmc_real_t sum = 0.0;

        /* t = 0: midpoint contribution */
        {
            lmmc_real_t fmid = f(mid, ud);
            num_evals++;
            if (!isfinite(fmid)) fmid = 0.0;
            sum += fmid * pi_half;
        }

        /* Symmetric pairs t = +/- j*h for j = 1, 2, 3, ... */
        for (size_t j = 1; num_evals < max_nodes; j++) {
            lmmc_real_t t = (lmmc_real_t)j * h;
            lmmc_real_t sinh_t = sinh(t);
            lmmc_real_t cosh_t = cosh(t);
            lmmc_real_t u = pi_half * sinh_t;

            if (u > 20.0) break;

            /* For large u, cosh(u) ≈ exp(u)/2, so
             * w = pi/2 * cosh_t / cosh^2(u) ≈ pi/2 * cosh_t * 4 * exp(-2u)
             * Use this to avoid overflow in cosh(u) for large u.
             */
            lmmc_real_t w;
            if (u > 6.0) {
                /* Use exponential form to avoid overflow */
                lmmc_real_t exp_neg_2u = exp(-2.0 * u);
                w = pi_half * cosh_t * 4.0 * exp_neg_2u;
            } else {
                lmmc_real_t cosh_u = cosh(u);
                w = pi_half * cosh_t / (cosh_u * cosh_u);
            }

            if (w < 1e-50) break;

            lmmc_real_t tanh_u;
            if (u > 6.0) {
                tanh_u = 1.0 - 2.0 * exp(-2.0 * u);
            } else {
                tanh_u = tanh(u);
            }
            lmmc_real_t xp = mid + half_len * tanh_u;
            lmmc_real_t xn = mid - half_len * tanh_u;

            lmmc_real_t fp = f(xp, ud);
            lmmc_real_t fn_val = f(xn, ud);
            num_evals += 2;

            if (!isfinite(fp)) fp = 0.0;
            if (!isfinite(fn_val)) fn_val = 0.0;

            sum += (fp + fn_val) * w;
        }

        integral = half_len * h * sum;

        /* Check convergence (need at least 2 levels) */
        if (level >= 1) {
            lmmc_real_t error = lmmc_abs(integral - prev_integral);
            out->value = integral;
            out->error = error;
            out->num_evals = num_evals;
            if (error <= abs_tol) {
                converged = 1;
                break;
            }
        }

        prev_integral = integral;
        h /= 2.0;
    }

    if (!converged) {
        out->value = integral;
        out->error = lmmc_abs(integral - prev_integral);
        out->num_evals = num_evals;
        return LMMC_STATUS_CONVERGENCE_FAILED;
    }
    return LMMC_STATUS_OK;
}


/**
 * Gauss-Hermite 求积 (weight: exp(-x^2), domain: (-inf, +inf))
 *
 * Uses Golub-Welsch algorithm: the nodes are eigenvalues of the symmetric
 * tridiagonal Jacobi matrix for Hermite polynomials, and weights are
 * derived from the first component of each eigenvector.
 *
 * For physicist's Hermite: H_n(x), weight exp(-x^2)
 *   Recurrence: x H_n = H_{n+1}/2 + n H_{n-1}
 *   Jacobi matrix: a_i = 0, b_i = sqrt(i/2) for i=1..n-1
 */

#define LMMC_GH_MAX_ORDER 20

/**
 * @internal
 * @brief Compute eigenvalues and first-component-squared of eigenvectors
 *        of a symmetric tridiagonal matrix using implicit QR (QL) algorithm.
 *
 * @param[in,out] diag   Main diagonal (n entries). On exit: eigenvalues.
 * @param[in,out] subdiag Sub-diagonal (n-1 entries). Destroyed on exit.
 * @param[out]    weights First-component-squared of eigenvectors * mu_0.
 * @param[in]     n      Matrix size.
 * @param[in]     mu0    Integral of weight function (sqrt(pi) for Hermite).
 */
static void tridiag_eigvals_weights(
    lmmc_real_t* diag, lmmc_real_t* subdiag,
    lmmc_real_t* weights, size_t n, lmmc_real_t mu0)
{
    /* Store first row of eigenvector matrix (initially identity) */
    lmmc_real_t z[LMMC_GH_MAX_ORDER];
    for (size_t i = 0; i < n; i++) z[i] = (i == 0) ? 1.0 : 0.0;

    /* Implicit QL algorithm with shifts */
    for (size_t l = 0; l < n; l++) {
        size_t iter = 0;
        size_t m;
        while (1) {
            /* Find small subdiagonal element */
            for (m = l; m < n - 1; m++) {
                lmmc_real_t dd = lmmc_abs(diag[m]) + lmmc_abs(diag[m + 1]);
                if (lmmc_abs(subdiag[m]) + dd == dd) break;
            }
            if (m == l) break;

            if (iter++ >= 100) break;  /* safety */

            /* Wilkinson shift */
            lmmc_real_t g = (diag[l + 1] - diag[l]) / (2.0 * subdiag[l]);
            lmmc_real_t r = sqrt(g * g + 1.0);
            lmmc_real_t sign_g = (g >= 0.0) ? 1.0 : -1.0;
            g = diag[m] - diag[l] + subdiag[l] / (g + sign_g * r);

            lmmc_real_t s = 1.0, c = 1.0, p = 0.0;
            size_t i;
            for (i = m; i > l; i--) {
                lmmc_real_t fi = s * subdiag[i - 1];
                lmmc_real_t bi = c * subdiag[i - 1];
                if (lmmc_abs(fi) >= lmmc_abs(g)) {
                    c = g / fi;
                    r = sqrt(c * c + 1.0);
                    subdiag[i] = fi * r;
                    s = 1.0 / r;
                    c *= s;
                } else {
                    s = fi / g;
                    r = sqrt(s * s + 1.0);
                    subdiag[i] = g * r;
                    c = 1.0 / r;
                    s *= c;
                }

                g = diag[i] - p;
                r = (diag[i - 1] - g) * s + 2.0 * c * bi;
                p = s * r;
                diag[i] = g + p;
                g = c * r - bi;

                /* Track first row of eigenvector matrix */
                lmmc_real_t zi = z[i];
                z[i] = s * z[i - 1] + c * zi;
                z[i - 1] = c * z[i - 1] - s * zi;
            }
            diag[l] -= p;
            subdiag[l] = g;
            if (m < n - 1) subdiag[m] = 0.0;
        }
    }

    /* Weights = mu0 * z[i]^2 */
    for (size_t i = 0; i < n; i++) {
        weights[i] = mu0 * z[i] * z[i];
    }
}

lmmc_status_t lmmc_quad_gauss_hermite(
    lmmc_quad_func_t f, void* ud,
    size_t order, lmmc_real_t* out)
{
    if (f == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (order < 1 || order > LMMC_GH_MAX_ORDER) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Build Jacobi matrix for physicist's Hermite polynomials.
     * Three-term recurrence: x H_n(x) = H_{n+1}(x)/2 + n*H_{n-1}(x)
     * Normalized: p_n = H_n / ||H_n||, ||H_n||^2 = sqrt(pi) * 2^n * n!
     * Jacobi matrix: alpha_i = 0, beta_i = sqrt(i/2) for i = 1..n-1
     */
    lmmc_real_t diag[LMMC_GH_MAX_ORDER];
    lmmc_real_t subdiag[LMMC_GH_MAX_ORDER];
    lmmc_real_t weights[LMMC_GH_MAX_ORDER];

    for (size_t i = 0; i < order; i++) {
        diag[i] = 0.0;  /* alpha_i = 0 for Hermite */
    }
    for (size_t i = 0; i < order - 1; i++) {
        subdiag[i] = sqrt((lmmc_real_t)(i + 1) / 2.0);
    }

    /* mu_0 = integral of exp(-x^2) over (-inf, inf) = sqrt(pi) */
    lmmc_real_t mu0 = sqrt(LMMC_CONST_PI);

    tridiag_eigvals_weights(diag, subdiag, weights, order, mu0);

    /* Compute the quadrature sum: sum_i w_i * f(x_i) */
    lmmc_real_t sum = 0.0;
    for (size_t i = 0; i < order; i++) {
        sum += weights[i] * f(diag[i], ud);
    }

    *out = sum;
    return LMMC_STATUS_OK;
}


/**
 * Gauss-Laguerre 求积 (weight: exp(-x), domain: [0, +inf))
 *
 * Uses Golub-Welsch algorithm with Laguerre polynomial recurrence.
 * Three-term recurrence for monic Laguerre:
 *   x L_n(x) = L_{n+1}(x) + (2n+1) L_n(x) - n^2 L_{n-1}(x)
 * Jacobi matrix: alpha_i = 2*i + 1, beta_i = i (for i = 1..n-1)
 * mu_0 = integral of exp(-x) over [0, inf) = 1
 */

#define LMMC_GL_LAG_MAX_ORDER 20

lmmc_status_t lmmc_quad_gauss_laguerre(
    lmmc_quad_func_t f, void* ud,
    size_t order, lmmc_real_t* out)
{
    if (f == NULL || out == NULL) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }
    if (order < 1 || order > LMMC_GL_LAG_MAX_ORDER) {
        return LMMC_STATUS_INVALID_ARGUMENT;
    }

    /* Build Jacobi matrix for Laguerre polynomials.
     * alpha_i = 2*i + 1 (i = 0..n-1)
     * beta_i = i (i = 1..n-1), so sqrt(beta_i) = sqrt(i)
     */
    lmmc_real_t diag[LMMC_GL_LAG_MAX_ORDER];
    lmmc_real_t subdiag[LMMC_GL_LAG_MAX_ORDER];
    lmmc_real_t weights[LMMC_GL_LAG_MAX_ORDER];

    for (size_t i = 0; i < order; i++) {
        diag[i] = 2.0 * (lmmc_real_t)i + 1.0;
    }
    for (size_t i = 0; i < order - 1; i++) {
        subdiag[i] = (lmmc_real_t)(i + 1);  /* sqrt(beta_{i+1}) = sqrt((i+1)^2) = i+1 */
    }

    /* mu_0 = integral of exp(-x) over [0, inf) = 1 */
    lmmc_real_t mu0 = 1.0;

    tridiag_eigvals_weights(diag, subdiag, weights, order, mu0);

    /* Compute the quadrature sum: sum_i w_i * f(x_i) */
    lmmc_real_t sum = 0.0;
    for (size_t i = 0; i < order; i++) {
        sum += weights[i] * f(diag[i], ud);
    }

    *out = sum;
    return LMMC_STATUS_OK;
}
