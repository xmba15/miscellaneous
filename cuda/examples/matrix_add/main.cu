/**
 * @file    main.cpp
 *
 * @author  btran
 *
 * @date    2020-05-03
 *
 * Copyright (c) organization
 *
 */

#include <cuda_runtime.h>
#include <iostream>

__global__ void addMat(float *matA, float *matB, float *matC,
                       const uint64_t row, const uint64_t col)
{
    uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < row && j < col) {
        uint64_t index = i * col + j;
        matC[index] = matA[index] + matB[index];
    }
}

int main(int argc, char *argv[])
{
    const uint64_t row = 3000, col = 3000;

    cudaEvent_t hostStart, hostStop, deviceStart, deviceStop;
    cudaEventCreate(&hostStart);
    cudaEventCreate(&hostStop);
    cudaEventCreate(&deviceStart);
    cudaEventCreate(&deviceStop);

    float timeDifferenceOnHost, timeDifferenceOnDevice;

    float *a = new float[row * col];
    float *b = new float[row * col];
    float *c = new float[row * col];

    for (uint64_t i = 0; i < row; ++i) {
        for (uint64_t j = 0; j < col; ++j) {
            a[i * col + j] = i + j;
            b[i * col + j] = i + j;
        }
    }

    printf("Adding matrices on CPU...\n");
    cudaEventRecord(hostStart, 0);
    for (uint64_t i = 0; i < row * col; ++i) {
        c[i] = a[i] + b[i];
    }
    cudaEventRecord(hostStop, 0);
    cudaEventElapsedTime(&timeDifferenceOnHost, hostStart, hostStop);
    printf("Matrix addition over. Time taken on CPU: %5.5f\n",
           timeDifferenceOnHost);

    float *matA, *matB, *matC;

    printf("Adding matrices on GPU...\n");
    cudaEventRecord(deviceStart, 0);
    cudaMalloc((void **)&matA, row * col * sizeof(float));
    cudaMalloc((void **)&matB, row * col * sizeof(float));
    cudaMalloc((void **)&matC, row * col * sizeof(float));

    cudaMemcpy(matA, a, row * col * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(matB, b, row * col * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((row + 31) / 32, (col + 31) / 32);

    addMat<<<numBlocks, threadsPerBlock>>>(matA, matB, matC, row, col);
    cudaDeviceSynchronize();

    cudaMemcpy(c, matC, row * col * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(deviceStop, 0);
    cudaEventElapsedTime(&timeDifferenceOnDevice, deviceStart, deviceStop);
    printf("Matrix addition over. Time taken on GPU: %5.5f\n",
           timeDifferenceOnDevice);

    cudaFree(matA);
    cudaFree(matB);
    cudaFree(matC);
    cudaEventDestroy(deviceStart);
    cudaEventDestroy(deviceStop);
    cudaEventDestroy(hostStart);
    cudaEventDestroy(hostStop);

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;
}
