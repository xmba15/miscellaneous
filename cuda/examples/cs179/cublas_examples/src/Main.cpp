/**
 * @file    Main.cpp
 *
 * @author  btran
 *
 */

#include <iostream>
#include <random>
#include <vector>

#include "CudaUtils.cuh"
#include "SgemmGPU.cuh"
#include "Timer.hpp"

namespace
{
void sgemmCPU(int n, float alpha, const float* A, const float* B, float beta, float* C);
void createData(int n, std::vector<float>& hA, std::vector<float>& hB, std::vector<float>& hC, int seed = 2021);

auto timer = TimerBase();
int NUM_TEST = 10;
}  // namespace

int main(int argc, char* argv[])
{
    int n = 277;
    float alpha = 1.;
    float beta = 0.;
    std::vector<float> hA, hB, hC;

    ::createData(n, hA, hB, hC);

    std::vector<float> hCtmp1;
    timer.start();
    for (int i = 0; i < NUM_TEST; ++i) {
        hCtmp1 = hC;
        ::sgemmCPU(n, alpha, hA.data(), hB.data(), beta, hCtmp1.data());
    }
    std::cout << "processing time (cpu): " << timer.getMs() / NUM_TEST << "[ms]\n";

    for (int i = 0; i < NUM_TEST; ++i) {
        cuda::utils::warmUpGPU();
    }

    std::vector<float> hCtmp2;
    timer.clear();
    timer.start();
    for (int i = 0; i < NUM_TEST; ++i) {
        hCtmp2 = hC;
        cuda::sgemmGPU(n, alpha, hA.data(), hB.data(), beta, hCtmp2.data());
    }
    std::cout << "processing time (cublas gpu): " << timer.getMs() / NUM_TEST << "[ms]\n";

    float diff = 0;
    for (int i = 0; i < n * n; ++i) {
        float curDiff = hCtmp1[i] - hCtmp2[i];
        diff += curDiff * curDiff;
    }
    std::cout << "diff: " << std::sqrt(diff) << "\n";

    return 0;
}

namespace
{
void sgemmCPU(int n, float alpha, const float* A, const float* B, float beta, float* C)
{
    int i;
    int j;
    int k;

    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            float prod = 0;

            for (k = 0; k < n; ++k) {
                prod += A[k * n + i] * B[j * n + k];
            }

            C[j * n + i] = alpha * prod + beta * C[j * n + i];
        }
    }
}

void createData(int n, std::vector<float>& hA, std::vector<float>& hB, std::vector<float>& hC, int seed)
{
    int n2 = n * n;
    hA.resize(n2);
    hB.resize(n2);
    hC.resize(n2);

    srand(seed);
    for (int i = 0; i < n2; ++i) {
        hA[i] = 1. * rand() / RAND_MAX;
        hB[i] = 1. * rand() / RAND_MAX;
        hC[i] = 1. * rand() / RAND_MAX;
    }
}
}  // namespace
