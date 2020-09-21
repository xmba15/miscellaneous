/**
 * @file    HelloWorld.cu
 *
 * @author  btran
 *
 * @date    2020-05-02
 *
 * Copyright (c) organization
 *
 */

#include <iostream>

__global__ void helloworld1()
{
    // compute local thread ID
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    // compute local block ID
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    printf("Hello from thread (%d, %d, %d) in block (%d, %d, %d) \n", tx, ty,
           tz, bx, by, bz);
}

int main(int argc, char *argv[])
{
    helloworld1<<<1, 10>>>();
    cudaDeviceSynchronize();
    return 0;
}
