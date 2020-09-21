/**
 * @file    SharedMemory.cu
 *
 * @author  btran
 *
 * @date    2020-05-03
 *
 * Copyright (c) organization
 *
 */

#include "SharedMemory.hpp"
#include <iostream>

__global__ void kernelAccumulateAverage(float *dA, const int n)
{
    int i, index = threadIdx.x;
    if (index > n) {
        return;
    }

    float average, sum = 0.0f;

    extern __shared__ float shArr[];
    shArr[index] = dA[index];

    // directive to ensure all the writes to shared memory have completed
    __syncthreads();

    for (i = 0; i <= index; i++) {
        sum += shArr[i];
    }

    average = sum / (index + 1.0f);
    dA[index] = average;
}

void accumulateAverage(float *dA, const int n)
{
    float *dIn;
    cudaMalloc((void **)&dIn, n * sizeof(float));
    cudaMemcpy(dIn, dA, n * sizeof(float), cudaMemcpyHostToDevice);

    kernelAccumulateAverage<<<1, n, n * sizeof(float)>>>(dIn, n);
    cudaDeviceSynchronize();

    cudaMemcpy(dA, dIn, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dIn);
}
