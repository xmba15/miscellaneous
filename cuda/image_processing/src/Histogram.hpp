/**
 * @file    Histogram.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <thrust/device_vector.h>

namespace _cv
{
namespace
{
template <typename T> __global__ void histoKernel(const T* input, int* hist, int size, int numBins)
{
    extern __shared__ int sharedHist[];
    int tid = threadIdx.x;

    if (tid < numBins) {
        sharedHist[tid] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < size) {
        atomicAdd(&(sharedHist[input[idx] % numBins]), 1);
        idx += blockDim.x * gridDim.x;
    }

    __syncthreads();

    if (tid < numBins) {
        atomicAdd(&(hist[tid]), sharedHist[tid]);
    }
}
}  // namespace

template <typename T> void histogramGPU(const T* input, int* hist, int size, int numBins)
{
    thrust::device_vector<T> dInput(input, input + size);
    thrust::device_vector<int> dHist(numBins);
    dim3 blockDim = numBins;
    dim3 gridDim = (size + numBins - 1) / numBins;
    histoKernel<<<gridDim, blockDim, numBins * sizeof(int)>>>(thrust::raw_pointer_cast(dInput.data()),
                                                              thrust::raw_pointer_cast(dHist.data()), size, numBins);
    thrust::copy(dHist.begin(), dHist.end(), hist);
}

template <typename T> void histogramCPU(const T* input, int* hist, int size, int numBins)
{
    for (int i = 0; i < size; ++i) {
        hist[input[i] % numBins]++;
    }
}
}  // namespace _cv
