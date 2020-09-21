/**
 * @file    App.cpp
 *
 * @author  btran
 *
 */

#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

#include "IntegralImageCPU.hpp"
#include "IntegralImageGPU.hpp"
#include "Timer.hpp"

namespace
{
__global__ void warmUpGPUKernel()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    ++idx;
}

cudaError_t warmUpGPU()
{
    warmUpGPUKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return cudaGetLastError();
}

constexpr int H = 2000;
constexpr int W = 2100;

auto timer = TimerBase();

template <typename T> std::vector<T> createData();

template <typename T> T diff(T* lhs, T* rhs, int H, int W);

using use_type = int;
}  // namespace

int main(int argc, char* argv[])
{
    warmUpGPU();

    auto mat = createData<use_type>();
    std::vector<use_type> iiCPUNaive(mat.size());

    std::vector<use_type> iiCPUCumulativeRow(mat.size());
    timer.clear();
    timer.start();
    cpu::createIntegralImageCumulativeRow(mat.data(), iiCPUCumulativeRow.data(), H, W);
    std::cout << "cpu cumulative row: " << timer.getMs() << "[ms]\n";

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    timer.clear();
    timer.start();
    cpu::createIntegralImageNaive(mat.data(), iiCPUNaive.data(), H, W);
    std::cout << "cpu naive: " << timer.getMs() << "[ms]\n";

    std::cout << "cpu naive vs cumulative row: " << diff<use_type>(iiCPUNaive.data(), iiCPUCumulativeRow.data(), H, W)
              << "\n";

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::vector<use_type> iiGPU(mat.size());

    timer.clear();
    timer.start();
    gpu::createIntegralImage(mat.data(), iiGPU.data(), H, W);
    std::cout << "gpu: " << timer.getMs() << "[ms]\n";

    std::cout << "cpu naive vs gpu: " << diff<use_type>(iiCPUNaive.data(), iiGPU.data(), H, W) << "\n";

    return 0;
}

namespace
{
template <typename T> std::vector<T> createData()
{
    srand(2021);

    std::vector<T> output;
    output.resize(H * W);

    for (int i = 0; i < H * W; ++i) {
        output[i] = 10. * rand() / RAND_MAX;
    }

    return output;
}

template <typename T> T diff(T* lhs, T* rhs, int H, int W)
{
    T result = 0;
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            result += std::abs(lhs[i * W + j] - rhs[i * W + j]);
        }
    }

    return result;
}
}  // namespace
