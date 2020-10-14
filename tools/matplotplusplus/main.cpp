/**
 * @file    main.cpp
 *
 * @author  btran
 *
 */

#include <cmath>
#include <matplot/matplot.h>

namespace mlt = matplot;

int main()
{
    std::vector<double> x = mlt::linspace(0, 2 * M_PI);
    std::vector<double> y = mlt::transform(x, [](auto x) { return std::sin(x); });

    mlt::plot(x, y, "-o");
    mlt::hold(mlt::on);
    mlt::plot(x, mlt::transform(y, [](auto y) { return -y; }), "--xr");
    mlt::plot(x, mlt::transform(x, [](auto x) { return x / M_PI - 1.; }), "-:gs");
    mlt::plot({1.0, 0.7, 0.4, 0.0, -0.4, -0.7, -1}, "k");

    mlt::show();

    return 0;
}
