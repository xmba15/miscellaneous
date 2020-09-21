/**
 * @file    GaussianNewtonExample.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include "AppUtility.hpp"

int main(int argc, char* argv[])
{
    std::vector<double> sampleXs, sampleYs;
    ::generateSamples(NUM_SAMPLES, sampleXs, sampleYs);

    auto nonLinearFunc = [](const Eigen::Vector3d& coeffs, double x) {
        return std::exp(coeffs[0] * x * x + coeffs[1] * x + coeffs[2]);
    };

    auto getFandJacobian = [&nonLinearFunc](const std::vector<double>& sampleXs, const std::vector<double>& sampleYs,
                                              const Eigen::Vector3d& coeffs, Eigen::VectorXd& f, Eigen::MatrixXd& J) {
        int numSamples = sampleXs.size();
        f.resize(numSamples);
        J.resize(numSamples, coeffs.size());

        for (int i = 0; i < numSamples; ++i) {
            double expPart = nonLinearFunc(coeffs, sampleXs[i]);
            f[i] = sampleYs[i] - expPart;
            J.row(i) << -sampleXs[i] * sampleXs[i] * expPart, -sampleXs[i] * expPart, -expPart;
        }
    };

    Eigen::VectorXd f;
    Eigen::MatrixXd J;
    Eigen::Vector3d coeffs(2, -1, 5);
    int numIteration = 100;
    double cost = 0., lastCost = std::numeric_limits<double>::max();

    bool isSuccess = true;
    for (int i = 0; i < numIteration; ++i) {
        getFandJacobian(sampleXs, sampleYs, coeffs, f, J);
        Eigen::MatrixXd H = J.transpose() * J;
        Eigen::VectorXd g = -J.transpose() * f;
        cost = f.dot(f);

        Eigen::Vector3d coeffsDelta = H.ldlt().solve(g);
        if (coeffsDelta.hasNaN()) {
            isSuccess = false;
            break;
        }

        if (cost >= lastCost) {
            std::cout << "iter: " << i << ", cost: " << cost << "> last cost: " << lastCost << "\n";
            break;
        }

        coeffs += coeffsDelta;
        lastCost = cost;
    }

    std::cout << "estimation finished successful? : " << isSuccess << "\n";
    if (isSuccess) {
        std::cout << coeffs << "\n";
    }

    std::vector<double> estimatedYs;
    estimatedYs.reserve(sampleXs.size());
    std::transform(sampleXs.begin(), sampleXs.end(), std::back_inserter(estimatedYs),
                   [&nonLinearFunc, &coeffs](double x) { return nonLinearFunc(coeffs, x); });
    ::visualize(sampleXs, sampleYs, estimatedYs);

    return EXIT_SUCCESS;
}
