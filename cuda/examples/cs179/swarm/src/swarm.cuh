/**
 * @file    swarm.cuh
 *
 * @author  btran
 *
 */

#pragma once

#include <cuda_runtime.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

namespace gpu
{
struct GPUVec3d {
    double x;
    double y;
    double z;

    CUDA_CALLABLE GPUVec3d();
    CUDA_CALLABLE GPUVec3d(double x, double y, double z);
    CUDA_CALLABLE GPUVec3d(const GPUVec3d &v);
    CUDA_CALLABLE ~GPUVec3d();

    CUDA_CALLABLE double sqNorm();
};
}  // namespace gpu
