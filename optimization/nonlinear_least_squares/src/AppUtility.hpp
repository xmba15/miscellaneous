/**
 * @file    AppUtility.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <random>
#include <vector>

#include <matplot/matplot.h>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace
{
static constexpr int NUM_SAMPLES = 100;

void generateSamples(int numSamples, std::vector<double>& sampleXs, std::vector<double>& sampleYs)
{
    std::mt19937 rng(2021);
    std::normal_distribution<> nd{0, 1};
    double a = -0.5, b = 2.8, c = 1.;

    for (int i = 0; i < numSamples; ++i) {
        double x = 1.0 * i / numSamples;
        sampleXs.emplace_back(x);
        sampleYs.emplace_back(std::exp(a * x * x + b * x + c) + nd(rng));
    }
}

void visualize(const std::vector<double>& sampleXs, const std::vector<double>& sampleYs,
               const std::vector<double>& estimatedYs)
{
    auto s = matplot::scatter(sampleXs, sampleYs);
    s->display_name("sample data");
    matplot::hold(matplot::on);

    auto p = matplot::plot(sampleXs, estimatedYs);
    p->display_name("estimated data");

    matplot::legend();
    matplot::show();
}
}  // namespace
