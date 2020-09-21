/**
 * @file    VecDouble.cu
 *
 * @author  btran
 *
 * @date    2020-05-03
 *
 * Copyright (c) organization
 *
 */

#include "VecDouble.hpp"
#include <cuda_runtime.h>

__global__ void kernelVecDouble(int *in, int *out, const int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < n) {
        out[tid] = in[tid] * 2;
        tid += blockDim.x * gridDim.x;
    }
}

void vecDouble(int *hIn, int *hOut, const int n)
{
    int *dIn;
    int *dOut;
    cudaMallocHost((void **)&dIn, n * sizeof(int));
    cudaMallocHost((void **)&dOut, n * sizeof(int));
    cudaMemcpy(dIn, hIn, n * sizeof(int), cudaMemcpyHostToDevice);

    kernelVecDouble<<<1, n>>>(dIn, dOut, n);
    cudaDeviceSynchronize();

    cudaMemcpy(hOut, dOut, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dIn);
    cudaFree(dOut);
}
