#include <stdio.h>
#include <math.h>
#include "lmmc/lmmc.h"

int main(void) {
    printf("=== LMMC Math Utilities Example ===\n\n");

    // 1. Numerical Stability: expm1 and log1p
    // When x is very small, exp(x) - 1 loses precision due to cancellation.
    // lmmc_expm1 is designed to handle this accurately.
    double x_small = 1e-15;
    double direct_expm1 = exp(x_small) - 1.0;
    double stable_expm1 = 0.0;
    lmmc_expm1(x_small, &stable_expm1);

    printf("1. Numerical Stability (Small x = %.2e):\n", x_small);
    printf("   exp(x) - 1 (Direct): %.20e\n", direct_expm1);
    printf("   lmmc_expm1(x):       %.20e\n", stable_expm1);
    printf("   (Notice the precision difference)\n\n");

    // 2. Geometry: Cartesian to Polar Coordinates
    // lmmc_hypot handles overflow internally, and lmmc_atan2 handles all quadrants.
    double dx = -3.0;
    double dy = 4.0;
    double r = 0.0, theta = 0.0;
    
    lmmc_hypot(dx, dy, &r);
    lmmc_atan2(dy, dx, &theta); // Returns angle in radians

    printf("2. Cartesian to Polar Conversion:\n");
    printf("   Input: (x: %.2f, y: %.2f)\n", dx, dy);
    printf("   Radius (hypot): %.2f\n", r);
    printf("   Angle (atan2):  %.4f rad (%.2f deg)\n\n", theta, theta * 180.0 / LMMC_PI);

    // 3. Robustness: Floating Point Properties
    double my_nan, my_inf;
    lmmc_nan(&my_nan);
    lmmc_inf(&my_inf);

    int is_n, is_i, is_f;
    lmmc_isnan(my_nan, &is_n);
    lmmc_isinf(my_inf, &is_i);
    lmmc_isfinite(1.0, &is_f);

    printf("3. Floating Point Properties:\n");
    printf("   Value NaN:  %s\n", is_n ? "Yes" : "No");
    printf("   Value Inf:  %s\n", is_i ? "Yes" : "No");
    printf("   Value 1.0 is Finite: %s\n\n", is_f ? "Yes" : "No");

    // 4. Precision Control: nextafter and approx_eq
    double base = 1.0;
    double next = 0.0;
    lmmc_nextafter(base, 2.0, &next); // Smallest double greater than 1.0

    int equal = 0;
    lmmc_approx_eq(base, next, 1e-15, &equal);

    printf("4. Precision Control:\n");
    printf("   Value: %.20f\n", base);
    printf("   Next float towards 2.0: %.20f\n", next);
    printf("   Are they approximately equal (eps=1e-15)? %s\n", equal ? "Yes" : "No");

    // 5. High-performance Trig: sincos
    double angle = LMMC_PI / 4.0; // 45 degrees
    double s = 0.0, c = 0.0;
    lmmc_sincos(angle, &s, &c);

    printf("\n5. Combined Trig (sincos):\n");
    printf("   Angle: 45 degrees\n");
    printf("   Sin: %.4f, Cos: %.4f\n", s, c);

    return 0;
}
