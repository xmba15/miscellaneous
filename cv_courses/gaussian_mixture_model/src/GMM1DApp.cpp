/**
 * @file    GMM1DApp.cpp
 *
 * @author  btran
 *
 */

#include "KMeans.hpp"
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include <matplot/matplot.h>

namespace
{
double gaussian(double x, double myu, double delta);

/**
 * @brief q(x) = 0.4Phi(x; -2, 1.5***2) + 0.2Phi(x; 2, 2**2) + 0.4Phi(x; 3, 1**2)
 *
 */
void createData(int numSamples, std::vector<double>& sampleXs);

void solve(const std::vector<double>& data, int numG, std::vector<double>& ws, std::vector<double>& myus,
           std::vector<double>& deltas, int numIterations = 20);

void visualize(const std::vector<double>& sampleXs, const std::vector<double>& sampleYs);

constexpr int NUM_SAMPLES = 1000;
}  // namespace

int main(int argc, char* argv[])
{
    std::vector<double> sampleXs, sampleYs;

    createData(NUM_SAMPLES, sampleXs);

    std::vector<double> ws, myus, deltas;
    int numG = 3;
    int numIterations = 200;

    solve(sampleXs, numG, ws, myus, deltas, numIterations);

    for (int i = 0; i < numG; ++i) {
        std::cout << i << "-th, weight, myu, delta: " << ws[i] << " " << myus[i] << " " << deltas[i] << "\n";
    }

    for (int i = 0; i < NUM_SAMPLES; ++i) {
        double y = 0;
        for (int j = 0; j < numG; ++j) {
            y += ws[j] * gaussian(sampleXs[i], myus[j], deltas[j]);
        }
        sampleYs.emplace_back(y);
    }

    visualize(sampleXs, sampleYs);

    return EXIT_SUCCESS;
}

namespace
{
double gaussian(double x, double myu, double delta)
{
    return std::exp(-(x - myu) * (x - myu) / (2 * delta * delta)) / std::sqrt(2 * M_PI * delta * delta);
}

void createData(int numSamples, std::vector<double>& data)
{
    std::mt19937 rng(2021);
    std::uniform_real_distribution<> urd(-2, 2);
    std::vector<double> ws{0.4, 0.2, 0.4};
    std::vector<double> myus{-2, 2, 8};
    std::vector<double> deltas{1.5, 2, 1};

    int numG = 3;
    for (int i = 0; i < numG; ++i) {
        int curSample = ws[i] * numSamples;
        for (int j = 0; j < curSample; ++j) {
            double x = urd(rng) + myus[i];
            data.emplace_back(x);
        }
    }
}

void solve(const std::vector<double>& data, int numG, std::vector<double>& ws, std::vector<double>& myus,
           std::vector<double>& deltas, int numIterations)
{
    if (numG <= 0) {
        throw std::runtime_error("invalid number of Gaussian components");
    }

    int numData = data.size();

    if (numData < 0) {
        throw std::runtime_error("number of data must be more than 0");
    }

    ws.resize(numG);
    myus.resize(numG);
    deltas.resize(numG, 0.);

    _cv::KMeans::Clusters clusters = _cv::KMeans::run(data, numG, myus);

    for (int i = 0; i < numG; ++i) {
        ws[i] = static_cast<double>(clusters[i].size()) / numData;
        for (int idx : clusters[i]) {
            deltas[i] += std::pow(data[idx] - ws[i], 2);
        }
        deltas[i] /= clusters[i].size();
        deltas[i] = std::sqrt(deltas[i]);
    }

    for (int iter = 0; iter < numIterations; ++iter) {
        std::vector<double> etas(numData * numG, 0);

        // E step
        for (int i = 0; i < numData; ++i) {
            double sum = 0.;
            for (int j = 0; j < numG; ++j) {
                etas[i * numG + j] = ws[j] * gaussian(data[i], myus[j], deltas[j]);
                sum += etas[i * numG + j];
            }
            for (int j = 0; j < numG; ++j) {
                etas[i * numG + j] /= sum;
            }
        }

        // M step
        std::vector<double> etasSumEachCol(numG, 0.);
        for (int i = 0; i < numData; ++i) {
            for (int j = 0; j < numG; ++j) {
                etasSumEachCol[j] += etas[i * numG + j];
            }
        }

        // update weights
        for (int j = 0; j < numG; ++j) {
            ws[j] = etasSumEachCol[j] / numData;
        }

        // update myus
        std::fill(myus.begin(), myus.end(), 0);
        for (int i = 0; i < numData; ++i) {
            for (int j = 0; j < numG; ++j) {
                myus[j] += etas[i * numG + j] * data[i];
            }
        }
        for (int j = 0; j < numG; ++j) {
            myus[j] /= etasSumEachCol[j];
        }

        // update deltas
        std::fill(deltas.begin(), deltas.end(), 0);
        for (int i = 0; i < numData; ++i) {
            for (int j = 0; j < numG; ++j) {
                deltas[j] += etas[i * numG + j] * std::pow(data[i] - myus[j], 2);
            }
        }
        for (int j = 0; j < numG; ++j) {
            deltas[j] /= etasSumEachCol[j];
            deltas[j] = std::sqrt(deltas[j]);
        }
    }
}

void visualize(const std::vector<double>& sampleXs, const std::vector<double>& sampleYs)
{
    auto s = matplot::scatter(sampleXs, sampleYs);
    s->display_name("sample data");
    matplot::hold(matplot::on);

    matplot::legend();
    matplot::show();
}
}  // namespace
