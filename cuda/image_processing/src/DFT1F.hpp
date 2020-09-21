/**
 * @file    DFT1F.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <complex>
#include <vector>

#include <thrust/complex.h>

namespace _cv
{
template <typename DataType>
std::vector<std::complex<DataType>> dft1d(const std::vector<std::complex<DataType>>& src, bool isInverted = false)
{
    if (src.empty()) {
        throw std::runtime_error("empty samples");
    }

    int numSamples = src.size();
    std::vector<std::complex<DataType>> result;
    result.reserve(numSamples);

    static constexpr std::complex<DataType> C_I(0, 1);
    std::complex<DataType> wN = std::exp(C_I * (2 * M_PI / numSamples * (isInverted ? 1 : -1)));
    std::complex<DataType> wNi(1);

    for (int i = 0; i < numSamples; ++i) {
        std::complex<DataType> curF = 0;
        std::complex<DataType> wNij(1);
        for (int j = 0; j < numSamples; ++j) {
            curF += src[j] * wNij;
            wNij *= wNi;
        }
        wNi *= wN;
        if (isInverted) {
            curF /= numSamples;
        }
        result.emplace_back(curF);
    }

    return result;
}

namespace cuda
{
template <typename DataType>
__global__ void dft1dGPU(thrust::complex<DataType>* src, thrust::complex<DataType>* dst, int numSamples,
                         bool isInverted)

{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= numSamples) {
        return;
    }

    thrust::complex<DataType> sum = 0;

    thrust::complex<DataType> C_I(0, 1);
    thrust::complex<DataType> wN = thrust::exp(C_I * (2 * M_PI * tid / numSamples * (isInverted ? 1 : -1)));
    thrust::complex<DataType> wNi(1);

    for (int j = 0; j < numSamples; ++j) {
        sum += src[j] * wNi;
        wNi *= wN;
    }

    dst[tid] = sum;
    if (isInverted) {
        dst[tid] /= numSamples;
    }
}
}  // namespace cuda

template <typename DataType>
std::vector<std::complex<DataType>> dft1dGPU(const std::vector<std::complex<DataType>>& src, bool isInverted = false)
{
    if (src.empty()) {
        throw std::runtime_error("empty samples");
    }

    int numSamples = src.size();

    thrust::complex<DataType>* deviceSrc;
    thrust::complex<DataType>* deviceDst;
    cudaMalloc((void**)&deviceSrc, numSamples * sizeof(thrust::complex<DataType>));
    cudaMalloc((void**)&deviceDst, numSamples * sizeof(thrust::complex<DataType>));

    cudaMemcpy(deviceSrc, reinterpret_cast<const thrust::complex<DataType>*>(src.data()),
               numSamples * sizeof(thrust::complex<DataType>), cudaMemcpyHostToDevice);

    dim3 blockDim(256);
    dim3 gridDim((numSamples + blockDim.x - 1) / blockDim.x);
    cuda::dft1dGPU<DataType><<<gridDim, blockDim>>>(deviceSrc, deviceDst, numSamples, isInverted);

    std::vector<std::complex<DataType>> dst(numSamples);
    cudaMemcpy(reinterpret_cast<thrust::complex<DataType>*>(dst.data()), deviceDst,
               numSamples * sizeof(thrust::complex<DataType>), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaFree(deviceSrc);
    cudaFree(deviceDst);

    return dst;
}
}  // namespace _cv
