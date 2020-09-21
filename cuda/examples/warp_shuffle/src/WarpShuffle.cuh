/**
 * @file    WarpShuffle.cuh
 *
 * @author  btran
 *
 */

#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

namespace cuda
{
constexpr int WARP_SIZE = 32;

template <typename DataType> __device__ DataType warpReduction(DataType val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

template <class DataType> __device__ DataType blockReduction(DataType val)
{
    static __shared__ DataType s[WARP_SIZE];

    int laneId = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warpReduction<DataType>(val);

    if (laneId == 0) {
        s[wid] = val;
    }

    __syncthreads();

    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? s[laneId] : 0;

    if (wid == 0) {
        val = warpReduction<DataType>(val);
    }

    return val;
}

template <class DataType> __global__ void reduce(const DataType* in, DataType* out, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    DataType sum = 0;
    while (tid < N) {
        sum += in[tid];
        tid += blockDim.x * gridDim.x;
    }

    sum = blockReduction<DataType>(sum);

    if (threadIdx.x == 0) {
        out[blockIdx.x] = sum;
    }
}

template <typename DataType> DataType sum(const DataType* data, int numElem)
{
    thrust::device_vector<DataType> dData(data, data + numElem);
    int threads = 256;
    int blocks = min((numElem + threads - 1) / threads, 1024);
    thrust::device_vector<DataType> dOutput(blocks);

    reduce<DataType><<<blocks, threads>>>(thrust::raw_pointer_cast(dData.data()),
                                          thrust::raw_pointer_cast(dOutput.data()), numElem);
    reduce<DataType>
        <<<1, WARP_SIZE>>>(thrust::raw_pointer_cast(dOutput.data()), thrust::raw_pointer_cast(dOutput.data()), blocks);

    float sum = dOutput[0];

    // or
    // float sum = thrust::reduce(dOutput.begin(), dOutput.begin() + blocks, 0.);

    return sum;
}
}  // namespace cuda
