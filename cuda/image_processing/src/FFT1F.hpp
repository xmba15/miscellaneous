/**
 * @file    FFT1F.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <iostream>

#include <complex>
#include <numeric>
#include <vector>

#include <thrust/complex.h>

// Ref: https://cp-algorithms.com/algebra/fft.html
namespace _cv
{
namespace
{
unsigned int reverseBits(unsigned int num, unsigned int lgn)
{
    unsigned int reversed = 0;
    for (int i = 0; i < lgn; ++i) {
        reversed <<= 1;
        if (num & 1) {
            reversed ^= 1;
        }
        num >>= 1;
    }

    return reversed;
}

template <typename DataType> void fft1dUtils(std::vector<std::complex<DataType>>& src, bool isInverted)
{
    int numSamples = src.size();
    if (numSamples == 1) {
        return;
    }

    std::vector<std::complex<DataType>> srcEven(numSamples / 2), srcOdd(numSamples / 2);
    for (int i = 0; 2 * i < numSamples; ++i) {
        srcEven[i] = src[2 * i];
        srcOdd[i] = src[2 * i + 1];
    }
    fft1dUtils(srcEven, isInverted);
    fft1dUtils(srcOdd, isInverted);

    static constexpr std::complex<DataType> C_I(0, 1);
    std::complex<DataType> wN = std::exp(C_I * (2 * M_PI / numSamples * (isInverted ? 1 : -1)));
    std::complex<DataType> w(1);

    for (int i = 0; 2 * i < numSamples; ++i) {
        src[i] = srcEven[i] + w * srcOdd[i];
        src[i + numSamples / 2] = srcEven[i] - w * srcOdd[i];
        if (isInverted) {
            src[i] /= 2;
            src[i + numSamples / 2] /= 2;
        }
        w *= wN;
    }
}
}  // namespace

template <typename DataType>
std::vector<std::complex<DataType>> fft1d(const std::vector<std::complex<DataType>>& src, bool isInverted = false)
{
    if (src.empty()) {
        throw std::runtime_error("empty samples");
    }

    int numSamples = src.size();

    if (numSamples & (numSamples - 1)) {
        throw std::runtime_error("number of samples needs to be power of 2");
    }

    std::vector<std::complex<DataType>> dst(src);
    fft1dUtils(dst, isInverted);

    return dst;
}

template <typename DataType>
std::vector<std::complex<DataType>> fft1d2(const std::vector<std::complex<DataType>>& src, bool isInverted = false)
{
    if (src.empty()) {
        throw std::runtime_error("empty samples");
    }

    int numSamples = src.size();

    if (numSamples & (numSamples - 1)) {
        throw std::runtime_error("number of samples needs to be power of 2");
    }

    int logn = std::log2(numSamples);

    std::vector<std::complex<DataType>> dst(src);

    for (int i = 0; i < numSamples; ++i) {
        int reversed = reverseBits(i, logn);
        if (i < reversed) {
            std::swap(dst[i], dst[reversed]);
        }
    }

    static constexpr std::complex<DataType> C_I(0, 1);
    for (int len = 2; len <= numSamples; len <<= 1) {
        std::complex<DataType> wN = std::exp(C_I * (2 * M_PI / len * (isInverted ? 1 : -1)));
        for (int i = 0; i < numSamples; i += len) {
            std::complex<DataType> w(1);
            for (int j = 0; j < len / 2; ++j) {
                std::complex<DataType> u = dst[i + j], v = dst[i + j + len / 2] * w;
                dst[i + j] = u + v;
                dst[i + j + len / 2] = u - v;
                w *= wN;
            }
        }
    }

    if (isInverted) {
        for (auto& elem : dst) {
            elem /= numSamples;
        }
    }

    return dst;
}

template <typename DataType>
std::vector<std::complex<DataType>> fft1d3(const std::vector<std::complex<DataType>>& src, bool isInverted = false)
{
    if (src.empty()) {
        throw std::runtime_error("empty samples");
    }

    int numSamples = src.size();

    if (numSamples & (numSamples - 1)) {
        throw std::runtime_error("number of samples needs to be power of 2");
    }

    std::vector<std::complex<DataType>> dst(src);

    for (int i = 1, j = 0; i < numSamples; ++i) {
        int bit = numSamples >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j)
            swap(dst[i], dst[j]);
    }

    static constexpr std::complex<DataType> C_I(0, 1);
    for (int len = 2; len <= numSamples; len <<= 1) {
        std::complex<DataType> wN = std::exp(C_I * (2 * M_PI / len * (isInverted ? 1 : -1)));
        for (int i = 0; i < numSamples; i += len) {
            std::complex<DataType> w(1);
            for (int j = 0; j < len / 2; ++j) {
                std::complex<DataType> u = dst[i + j], v = dst[i + j + len / 2] * w;
                dst[i + j] = u + v;
                dst[i + j + len / 2] = u - v;
                w *= wN;
            }
        }
    }

    if (isInverted) {
        for (auto& elem : dst) {
            elem /= numSamples;
        }
    }

    return dst;
}

namespace cuda
{
template <typename DataType>
__global__ void reverseBits(thrust::complex<DataType>* dst, const thrust::complex<DataType>* src, int logn,
                            int numSamples)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < numSamples) {
        dst[__brevll(tid) >> (64 - logn)] = src[tid];
        tid += blockDim.x * gridDim.x;
    }
}

template <typename DataType>
__global__ void fft1dGPUUtils(thrust::complex<DataType>* dst, int len, int numSamples, bool isInverted)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    while (tid < numSamples / len) {
        thrust::complex<DataType> C_I(0, 1);
        thrust::complex<DataType> wN = thrust::exp(C_I * (2 * M_PI / len * (isInverted ? 1 : -1)));
        thrust::complex<DataType> w(1);

#pragma unroll
        for (int j = 0; j < len / 2; ++j) {
            thrust::complex<DataType> u = dst[tid * len + j], v = dst[tid * len + j + len / 2] * w;
            dst[tid * len + j] = u + v;
            dst[tid * len + j + len / 2] = u - v;
            w *= wN;
        }

        tid += blockDim.x * gridDim.x;
    }
}
}  // namespace cuda

template <typename DataType>
std::vector<std::complex<DataType>> fft1dGPU(const std::vector<std::complex<DataType>>& src, bool isInverted = false)
{
    if (src.empty()) {
        throw std::runtime_error("empty samples");
    }

    int numSamples = src.size();

    if (numSamples & (numSamples - 1)) {
        throw std::runtime_error("number of samples needs to be power of 2");
    }

    int logn = std::log2(numSamples);

    thrust::complex<DataType>* deviceSrc;
    thrust::complex<DataType>* deviceDst;
    cudaMalloc((void**)&deviceSrc, numSamples * sizeof(thrust::complex<DataType>));
    cudaMalloc((void**)&deviceDst, numSamples * sizeof(thrust::complex<DataType>));

    cudaMemcpy(deviceSrc, reinterpret_cast<const thrust::complex<DataType>*>(src.data()),
               numSamples * sizeof(thrust::complex<DataType>), cudaMemcpyHostToDevice);

    int MAX_BLOCKS = 65535;
    dim3 blockDim(256);
    dim3 gridDim(min(MAX_BLOCKS, (numSamples + blockDim.x - 1) / blockDim.x));
    cuda::reverseBits<<<gridDim, blockDim, blockDim.x * sizeof(thrust::complex<DataType>)>>>(deviceDst, deviceSrc, logn,
                                                                                             numSamples);

    for (int len = 2; len <= numSamples; len <<= 1) {
        cuda::fft1dGPUUtils<<<min(MAX_BLOCKS, numSamples / len + blockDim.x - 1), blockDim>>>(deviceDst, len,
                                                                                              numSamples, isInverted);
    }

    std::vector<std::complex<DataType>> dst(numSamples);
    cudaMemcpy(reinterpret_cast<thrust::complex<DataType>*>(dst.data()), deviceDst,
               numSamples * sizeof(thrust::complex<DataType>), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    cudaFree(deviceSrc);
    cudaFree(deviceDst);

    return dst;
}
}  // namespace _cv
