/**
 * @file    CeresExample.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include <ceres/ceres.h>

#include "AppUtility.hpp"

namespace
{
struct CurveFittingResidual {
    CurveFittingResidual(double x, double y)
        : m_x(x)
        , m_y(y)
    {
    }

    template <typename T> bool operator()(const T* const coeffs, T* residual) const
    {
        residual[0] = m_y - ceres::exp(coeffs[0] * m_x * m_x + coeffs[1] * m_x + coeffs[2]);
        return true;
    }

 private:
    const double m_x, m_y;
};
}  // namespace

int main(int argc, char* argv[])
{
    std::vector<double> sampleXs, sampleYs;
    ::generateSamples(NUM_SAMPLES, sampleXs, sampleYs);

    Eigen::Vector3d coeffs(2, -1, 5);

    ceres::Problem prob;
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        prob.AddResidualBlock(new ceres::AutoDiffCostFunction<CurveFittingResidual, 1, 3>(
                                  new CurveFittingResidual(sampleXs[i], sampleYs[i])),
                              nullptr, coeffs.data());
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &prob, &summary);

    std::cout << coeffs << "\n";

    auto nonLinearFunc = [](const Eigen::Vector3d& coeffs, double x) {
        return std::exp(coeffs[0] * x * x + coeffs[1] * x + coeffs[2]);
    };
    std::vector<double> estimatedYs;
    estimatedYs.reserve(sampleXs.size());
    std::transform(sampleXs.begin(), sampleXs.end(), std::back_inserter(estimatedYs),
                   [&nonLinearFunc, &coeffs](double x) { return nonLinearFunc(coeffs, x); });
    ::visualize(sampleXs, sampleYs, estimatedYs);

    return EXIT_SUCCESS;
}
