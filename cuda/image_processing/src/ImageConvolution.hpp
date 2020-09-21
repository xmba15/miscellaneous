/**
 * @file    ImageConvolution.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <memory>

namespace _cv
{
template <typename DataType>
inline std::unique_ptr<DataType[]> padZero(const DataType* src, int srcWidth, int srcHeight, int padWidth,
                                           int padHeight)
{
    int dstWidth = srcWidth + 2 * padWidth;
    int dstHeight = srcHeight + 2 * padHeight;

    DataType* dst = new DataType[dstWidth * dstHeight];
    std::fill_n(dst, dstWidth * dstHeight, 0);

    for (int i = 0; i < srcHeight; ++i) {
        for (int j = 0; j < srcWidth; ++j) {
            dst[(i + padHeight) * dstWidth + (j + padWidth)] = src[i * srcWidth + j];
        }
    }

    return std::unique_ptr<DataType[]>(dst);
}

template <typename DataType>
inline std::unique_ptr<DataType[]> padReplication(const DataType* src, int srcWidth, int srcHeight, int padWidth,
                                                  int padHeight)
{
    int dstWidth = srcWidth + 2 * padWidth;
    int dstHeight = srcHeight + 2 * padHeight;

    DataType* dst = new DataType[dstWidth * dstHeight];
    std::fill_n(dst, dstWidth * dstHeight, 0);

    for (int i = 0; i < dstHeight; ++i) {
        for (int j = 0; j < dstWidth; ++j) {
            if (i < padHeight) {
                if (j < padWidth || j >= srcWidth + padWidth) {
                    continue;
                }
                dst[i * dstWidth + j] = src[j - padWidth];
                continue;
            }

            if (i >= srcHeight + padHeight) {
                if (j < padWidth || j >= srcWidth + padWidth) {
                    continue;
                }
                dst[i * dstWidth + j] = src[(srcHeight - 1) * srcWidth + j - padWidth];
                continue;
            }

            if (j < padWidth) {
                dst[i * dstWidth + j] = src[(i - padHeight) * srcWidth];
                continue;
            }

            if (j >= srcWidth + padWidth) {
                dst[i * dstWidth + j] = src[(i - padHeight) * srcWidth + srcWidth - 1];
                continue;
            }

            dst[i * dstWidth + j] = src[(i - padHeight) * srcWidth + j - padWidth];
        }
    }

    return std::unique_ptr<DataType[]>(dst);
}

template <typename DataType, typename KernelDataType>
inline std::unique_ptr<KernelDataType[]> convolve(const DataType* src, const KernelDataType* kernel, int srcWidth,
                                                  int srcHeight, int kernelHalfSize)
{
    auto paddedData = padReplication(src, srcWidth, srcHeight, kernelHalfSize, kernelHalfSize);
    int paddedWidth = srcWidth + 2 * kernelHalfSize;
    int filterSize = 2 * kernelHalfSize + 1;

    KernelDataType* convolved = new KernelDataType[srcWidth * srcHeight];

    for (int i = 0; i < srcHeight; ++i) {
        for (int j = 0; j < srcWidth; ++j) {
            KernelDataType value = 0;
            for (int k = -kernelHalfSize; k <= kernelHalfSize; ++k) {
                for (int l = -kernelHalfSize; l <= kernelHalfSize; ++l) {
                    value += paddedData[(i + kernelHalfSize + k) * paddedWidth + (j + kernelHalfSize + l)] *
                             kernel[(kernelHalfSize + k) * filterSize + (kernelHalfSize + l)];
                }
                convolved[i * srcWidth + j] = value;
            }
        }
    }

    return std::unique_ptr<KernelDataType[]>(convolved);
}

namespace cuda
{
const unsigned int MAX_FILTER_SIZE = 19;
__device__ __constant__ float DEVICE_CONSTANT_FILTER[MAX_FILTER_SIZE * MAX_FILTER_SIZE];

/**
 *  \brief use only global memory
 *
 */
template <typename DataType, typename KernelDataType>
void __global__ convolveGPU1(const DataType* padded, KernelDataType* convolved, int srcWidth, int srcHeight,
                             int kernelHalfSize)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int filterSize = 2 * kernelHalfSize + 1;
    int paddedWidth = srcWidth + 2 * kernelHalfSize;

    if (x < srcWidth && y < srcHeight) {
        KernelDataType value = 0;
#pragma unroll
        for (int k = -kernelHalfSize; k <= kernelHalfSize; ++k) {
#pragma unroll
            for (int l = -kernelHalfSize; l <= kernelHalfSize; ++l) {
                value += padded[(y + kernelHalfSize + k) * paddedWidth + (x + kernelHalfSize + l)] *
                         DEVICE_CONSTANT_FILTER[(kernelHalfSize + k) * filterSize + (kernelHalfSize + l)];
            }
            convolved[y * srcWidth + x] = value;
        }
    }
}

template <typename DataType, typename KernelDataType>
void __global__ convolveGPU2(const DataType* __restrict__ padded, KernelDataType* __restrict__ convolved, int srcWidth,
                             int srcHeight, int kernelHalfSize, int blockSize)
{
    extern __shared__ DataType pixels[];

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int lx = threadIdx.x;
    int ly = threadIdx.y;

    int filterSize = 2 * kernelHalfSize + 1;
    int paddedWidth = srcWidth + 2 * kernelHalfSize;
    int tileWidth = blockSize + 2 * kernelHalfSize;

    int blockSizeHat = blockSize + kernelHalfSize;

    pixels[(ly + kernelHalfSize) * tileWidth + lx + kernelHalfSize] =
        __ldg(&padded[(y + kernelHalfSize) * paddedWidth + x + kernelHalfSize]);

    if (threadIdx.x < kernelHalfSize && threadIdx.y < kernelHalfSize) {
        pixels[ly * tileWidth + lx] = __ldg(&padded[y * paddedWidth + x]);

        pixels[(ly + blockSizeHat) * tileWidth + lx + blockSizeHat] =
            __ldg(&padded[(y + blockSizeHat) * paddedWidth + x + blockSizeHat]);

        pixels[(ly + blockSizeHat) * tileWidth + lx] = __ldg(&padded[(y + blockSizeHat) * paddedWidth + x]);

        pixels[ly * tileWidth + lx + blockSizeHat] = __ldg(&padded[y * paddedWidth + x + blockSizeHat]);
    }

    if (threadIdx.x < kernelHalfSize) {
        pixels[(ly + kernelHalfSize) * tileWidth + lx] = __ldg(&padded[(y + kernelHalfSize) * paddedWidth + x]);
        pixels[(ly + kernelHalfSize) * tileWidth + lx + blockSizeHat] =
            __ldg(&padded[(y + kernelHalfSize) * paddedWidth + x + blockSizeHat]);
    }

    if (threadIdx.y < kernelHalfSize) {
        pixels[ly * tileWidth + lx + kernelHalfSize] = __ldg(&padded[y * paddedWidth + x + kernelHalfSize]);
        pixels[(ly + blockSizeHat) * tileWidth + lx + kernelHalfSize] =
            __ldg(&padded[(y + blockSizeHat) * paddedWidth + x + kernelHalfSize]);
    }

    __syncthreads();

    if (x < srcWidth && y < srcHeight) {
        KernelDataType value = 0;
#pragma unroll
        for (int k = -kernelHalfSize; k <= kernelHalfSize; ++k) {
#pragma unroll
            for (int l = -kernelHalfSize; l <= kernelHalfSize; ++l) {
                value = __fadd_rn(
                    value,
                    __fmul_rn(
                        pixels[(ly + kernelHalfSize + k) * tileWidth + lx + kernelHalfSize + l],
                        __ldg(&DEVICE_CONSTANT_FILTER[(kernelHalfSize + k) * filterSize + (kernelHalfSize + l)])));
            }
            convolved[y * srcWidth + x] = value;
        }
    }
}

template <typename DataType, typename KernelDataType>
void __global__ convolveGPU3(const DataType* __restrict__ padded, KernelDataType* __restrict__ convolved, int srcWidth,
                             int srcHeight, int kernelHalfSize, int blockSize, int paddedPitch, int convolvedPitch)
{
    extern __shared__ DataType pixels[];

    int paddedPitchInPixel = paddedPitch / sizeof(DataType);
    int convolvedPitchInPixel = convolvedPitch / sizeof(KernelDataType);

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int lx = threadIdx.x;
    int ly = threadIdx.y;

    int filterSize = 2 * kernelHalfSize + 1;
    int tileWidth = blockSize + 2 * kernelHalfSize;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
#pragma unroll
        for (int i = 0; i < tileWidth; ++i) {
            const DataType* paddedRowPtr = padded + (y + i) * paddedPitchInPixel;
#pragma unroll
            for (int j = 0; j < tileWidth; ++j) {
                pixels[i * tileWidth + j] = __ldg(&paddedRowPtr[x + j]);
            }
        }
    }

    __syncthreads();

    if (x < srcWidth && y < srcHeight) {
        KernelDataType value = 0;
#pragma unroll
        for (int k = -kernelHalfSize; k <= kernelHalfSize; ++k) {
#pragma unroll
            for (int l = -kernelHalfSize; l <= kernelHalfSize; ++l) {
                value += pixels[(ly + kernelHalfSize + k) * tileWidth + lx + kernelHalfSize + l] *
                         DEVICE_CONSTANT_FILTER[(kernelHalfSize + k) * filterSize + (kernelHalfSize + l)];
            }
            convolved[y * convolvedPitchInPixel + x] = value;
        }
    }
}
}  // namespace cuda

int divRoundUp(int value, int radix)
{
    return (value + radix) / radix;
}

template <typename DataType, typename KernelDataType>
inline std::unique_ptr<KernelDataType[]> convolveGPU1(const DataType* src, const KernelDataType* kernel, int srcWidth,
                                                      int srcHeight, int kernelHalfSize)
{
    std::unique_ptr<DataType[]> paddedData = padReplication(src, srcWidth, srcHeight, kernelHalfSize, kernelHalfSize);
    int paddedWidth = srcWidth + 2 * kernelHalfSize;
    int paddedHeight = srcHeight + 2 * kernelHalfSize;
    int filterSize = 2 * kernelHalfSize + 1;

    DataType* devicePaddedData;
    KernelDataType* deviceConvolved;

    cudaMalloc((void**)&devicePaddedData, paddedWidth * paddedHeight * sizeof(DataType));
    cudaMalloc((void**)&deviceConvolved, srcWidth * srcHeight * sizeof(KernelDataType));

    cudaMemcpy(devicePaddedData, paddedData.get(), paddedWidth * paddedHeight * sizeof(DataType),
               cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(cuda::DEVICE_CONSTANT_FILTER, reinterpret_cast<const float*>(kernel),
                       filterSize * filterSize * sizeof(float), 0, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim(divRoundUp(srcWidth, blockDim.x), divRoundUp(srcHeight, blockDim.y));

    cuda::convolveGPU1<DataType, KernelDataType>
        <<<gridDim, blockDim>>>(devicePaddedData, deviceConvolved, srcWidth, srcHeight, kernelHalfSize);

    KernelDataType* convolved = new KernelDataType[srcWidth * srcHeight];
    cudaMemcpy(convolved, deviceConvolved, srcWidth * srcHeight * sizeof(KernelDataType), cudaMemcpyDeviceToHost);

    cudaFree(devicePaddedData);
    cudaFree(deviceConvolved);
    return std::unique_ptr<KernelDataType[]>(convolved);
}

template <typename DataType, typename KernelDataType>
inline std::unique_ptr<KernelDataType[]> convolveGPU2(const DataType* src, const KernelDataType* kernel, int srcWidth,
                                                      int srcHeight, int kernelHalfSize)
{
    std::unique_ptr<DataType[]> paddedData = padReplication(src, srcWidth, srcHeight, kernelHalfSize, kernelHalfSize);
    int paddedWidth = srcWidth + 2 * kernelHalfSize;
    int paddedHeight = srcHeight + 2 * kernelHalfSize;
    int filterSize = 2 * kernelHalfSize + 1;

    DataType* devicePaddedData;
    KernelDataType* deviceConvolved;

    cudaMalloc((void**)&devicePaddedData, paddedWidth * paddedHeight * sizeof(DataType));
    cudaMalloc((void**)&deviceConvolved, srcWidth * srcHeight * sizeof(KernelDataType));

    cudaMemcpy(devicePaddedData, paddedData.get(), paddedWidth * paddedHeight * sizeof(DataType),
               cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(cuda::DEVICE_CONSTANT_FILTER, reinterpret_cast<const float*>(kernel),
                       filterSize * filterSize * sizeof(float), 0, cudaMemcpyHostToDevice);

    int BLOCK_SIZE = 32;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(divRoundUp(srcWidth, BLOCK_SIZE), divRoundUp(srcHeight, BLOCK_SIZE));
    int sharedMemorySizeByte = (BLOCK_SIZE + 2 * kernelHalfSize) * (BLOCK_SIZE + 2 * kernelHalfSize) * sizeof(DataType);

    cuda::convolveGPU2<DataType, KernelDataType><<<gridDim, blockDim, sharedMemorySizeByte>>>(
        devicePaddedData, deviceConvolved, srcWidth, srcHeight, kernelHalfSize, BLOCK_SIZE);

    KernelDataType* convolved = new KernelDataType[srcWidth * srcHeight];
    cudaMemcpy(convolved, deviceConvolved, srcWidth * srcHeight * sizeof(KernelDataType), cudaMemcpyDeviceToHost);

    cudaFree(devicePaddedData);
    cudaFree(deviceConvolved);
    return std::unique_ptr<KernelDataType[]>(convolved);
}

template <typename DataType, typename KernelDataType>
inline std::unique_ptr<KernelDataType[]> convolveGPU3(const DataType* src, const KernelDataType* kernel, int srcWidth,
                                                      int srcHeight, int kernelHalfSize)
{
    std::unique_ptr<DataType[]> paddedData = padReplication(src, srcWidth, srcHeight, kernelHalfSize, kernelHalfSize);
    int paddedWidth = srcWidth + 2 * kernelHalfSize;
    int paddedHeight = srcHeight + 2 * kernelHalfSize;
    int filterSize = 2 * kernelHalfSize + 1;

    DataType* devicePaddedData;
    std::size_t devicePaddedDataPitch;
    KernelDataType* deviceConvolved;
    std::size_t deviceConvolvedPitch;

    cudaMallocPitch((void**)&devicePaddedData, &devicePaddedDataPitch, paddedWidth * sizeof(DataType), paddedHeight);
    cudaMallocPitch((void**)&deviceConvolved, &deviceConvolvedPitch, srcWidth * sizeof(KernelDataType), srcHeight);

    cudaMemcpy2D(devicePaddedData, devicePaddedDataPitch, paddedData.get(), paddedWidth * sizeof(DataType),
                 paddedWidth * sizeof(DataType), paddedHeight, cudaMemcpyDefault);

    cudaMemcpyToSymbol(cuda::DEVICE_CONSTANT_FILTER, reinterpret_cast<const float*>(kernel),
                       filterSize * filterSize * sizeof(float), 0, cudaMemcpyHostToDevice);

    int BLOCK_SIZE = 32;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(divRoundUp(srcWidth, BLOCK_SIZE), divRoundUp(srcHeight, BLOCK_SIZE));
    int sharedMemorySizeByte = (BLOCK_SIZE + 2 * kernelHalfSize) * (BLOCK_SIZE + 2 * kernelHalfSize) * sizeof(DataType);

    cuda::convolveGPU3<DataType, KernelDataType><<<gridDim, blockDim, sharedMemorySizeByte>>>(
        devicePaddedData, deviceConvolved, srcWidth, srcHeight, kernelHalfSize, BLOCK_SIZE, devicePaddedDataPitch,
        deviceConvolvedPitch);

    KernelDataType* convolved = new KernelDataType[srcWidth * srcHeight];

    cudaMemcpy2D(convolved, srcWidth * sizeof(KernelDataType), deviceConvolved, deviceConvolvedPitch,
                 srcWidth * sizeof(KernelDataType), srcHeight, cudaMemcpyDeviceToHost);

    cudaFree(devicePaddedData);
    cudaFree(deviceConvolved);
    return std::unique_ptr<KernelDataType[]>(convolved);
}
}  // namespace _cv
