/**
 * @file    Main.cu
 *
 * @author  btran
 *
 */

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "WarpShuffle.cuh"

namespace
{
constexpr int BLOCK_NUM = 2;
constexpr int NUM_THREAD = 32;

std::vector<float> createData(int seed = 2021)
{
    std::srand(2021);
    std::vector<float> output(BLOCK_NUM * NUM_THREAD);
    for (int i = 0; i < BLOCK_NUM * NUM_THREAD; ++i) {
        output[i] = 1. * rand() / RAND_MAX;
    }

    return output;
}
}  // namespace

int main(int argc, char* argv[])
{
    auto vec = ::createData();
    float sum = std::accumulate(vec.begin(), vec.end(), 0.);

    std::cout << "sum: " << sum << "\n";

    float sumGPU;

    sumGPU = cuda::sum(vec.data(), vec.size());

    std::cout << "sum GPU: " << sumGPU << "\n";

    return 0;
}
