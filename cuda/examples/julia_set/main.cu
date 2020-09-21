/**
 * @file    main.cu
 *
 * @author  btran
 *
 * @date    2020-05-07
 *
 * Copyright (c) organization
 *
 */

#include "CPUBitmap.hpp"

// uncomment the following line will force the code to run on cpu only
// #undef __CUDAACC__

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_HOST __host__
#define CUDA_DEV __device__
#define CUDA_DEV_FORCE_INLINE __device__ __forceinline__
#define CUDA_HOST_FORCE_INLINE __host__ __forceinline__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOSTDEV_FORCE_INLINE __host__ __device__ __forceinline__
#define CUDA_GLOBAL __global__
#else
#define CUDA_HOST
#define CUDA_DEV
#define CUDA_DEV_FORCE_INLINE
#define CUDA_HOST_FORCE_INLINE inline
#define CUDA_HOSTDEV
#define CUDA_HOSTDEV_FORCE_INLINE inline
#define CUDA_GLOBAL
#endif

static constexpr int DIM = 1000;

namespace
{
struct cuComplex {
    float r;
    float i;

    CUDA_HOSTDEV cuComplex(float a, float b) : r(a), i(b)
    {
    }

    CUDA_HOSTDEV float magnitude2(void)
    {
        return r * r + i * i;
    }

    CUDA_HOSTDEV cuComplex operator*(const cuComplex &a)
    {
        return cuComplex(r * a.r - i * a.i, i * a.r + r * a.i);
    }

    CUDA_HOSTDEV cuComplex operator+(const cuComplex &a)
    {
        return cuComplex(r + a.r, i + a.i);
    }
};

CUDA_HOSTDEV int julia(int x, int y)
{
    const float scale = 1.5;
    float jx = scale * (DIM / 2. - x) / (DIM / 2.);
    float jy = scale * (DIM / 2. - y) / (DIM / 2.);

    cuComplex c(-0.8, 0.156);
    cuComplex a(jx, jy);

    int i = 0;
    for (i = 0; i < 200; i++) {
        a = a * a + c;
        if (a.magnitude2() > 1000)
            return 0;
    }

    return 1;
}

CUDA_GLOBAL void kernel(unsigned char *ptr)
{
#ifdef __CUDACC__
    uint64_t x = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < DIM && y < DIM) {
        int offset = x + y * DIM;

        int juliaValue = julia(x, y);
        ptr[offset * 4 + 0] = 255 * juliaValue;
        ptr[offset * 4 + 1] = 0;
        ptr[offset * 4 + 2] = 0;
        ptr[offset * 4 + 3] = 255;
    }
#else
    for (int y = 0; y < DIM; ++y) {
        for (int x = 0; x < DIM; ++x) {
            int offset = x + y * DIM;

            int juliaValue = julia(x, y);
            ptr[offset * 4 + 0] = 255 * juliaValue;
            ptr[offset * 4 + 1] = 0;
            ptr[offset * 4 + 2] = 0;
            ptr[offset * 4 + 3] = 255;
        }
    }
#endif
}

struct DataBlock {
    unsigned char *devBitmap;
};

}  // namespace

int main(int argc, char *argv[])
{
#ifdef __CUDACC__
    DataBlock data;
    CPUBitmap bitmap(DIM, DIM, &data);
    unsigned char *devBitmap;

    cudaMalloc((void **)&devBitmap, bitmap.image_size());
    data.devBitmap = devBitmap;

    dim3 numThreads(32, 32);
    dim3 numBlocks((DIM + 31) / 32, (DIM + 31) / 32);

    kernel<<<numBlocks, numThreads>>>(devBitmap);

    cudaMemcpy(bitmap.get_ptr(), devBitmap, bitmap.image_size(),
               cudaMemcpyDeviceToHost);

    cudaFree(devBitmap);

    bitmap.display_and_exit();
#else
    CPUBitmap bitmap(DIM, DIM);
    unsigned char *ptr = bitmap.get_ptr();

    kernel(ptr);

    bitmap.display_and_exit();
#endif
    return EXIT_SUCCESS;
}
