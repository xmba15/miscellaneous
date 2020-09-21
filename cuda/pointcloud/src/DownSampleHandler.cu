/**
 * @file    DownSampleHandler.cu
 *
 * @author  btran
 *
 */

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include "CudaUtils.cuh"
#include "DownSampleHandler.cuh"

namespace _pcl
{
namespace cuda
{
namespace
{
constexpr int MAX_BLOCKS = 65535;

struct HashElement {
    int pointIdx = -1;
    int bucketIdx = -1;
};

struct Bucket {
    int indexBegin = -1;
    int indexEnd = -1;
    int numPoints = 0;
};

struct CompareAXisFunctor {
    CompareAXisFunctor(int axis)
        : axis(axis)
    {
    }

    __host__ __device__ bool operator()(const Eigen::Vector3f& lhs, const Eigen::Vector3f& rhs)
    {
        return lhs[axis] < rhs[axis];
    }

    int axis;
};

__host__ __device__ int hashPoint(const Eigen::Vector3f& point, float leafSize,
                                  const DownSampleHandler::GridParam* gridParam)
{
    int x = (point[0] - gridParam->minPoint[0]) / leafSize;
    int y = (point[1] - gridParam->minPoint[1]) / leafSize;
    int z = (point[2] - gridParam->minPoint[2]) / leafSize;

    return x * gridParam->bucketNums[1] * gridParam->bucketNums[2] + y * gridParam->bucketNums[2] + z;
}

__global__ void updateHashTable(const __restrict__ Eigen::Vector3f* points,
                                const DownSampleHandler::GridParam* gridParam, HashElement* hashTable, float leafSize,
                                int numPoints)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < numPoints) {
        std::uint64_t curBucketIdx = hashPoint(points[tid], leafSize, gridParam);
        hashTable[tid].bucketIdx = curBucketIdx;
        hashTable[tid].pointIdx = tid;

        tid += blockDim.x * gridDim.x;
    }
}

struct CompareHashFunctor {
    __host__ __device__ bool operator()(const HashElement& lhs, const HashElement& rhs)
    {
        return lhs.bucketIdx < rhs.bucketIdx;
    }
};

__global__ void updateBucketsKernel(const HashElement* hashTable, Bucket* buckets, int numPoints)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < numPoints) {
        std::uint64_t curBucketIdx = hashTable[tid].bucketIdx;

        if (tid == numPoints - 1) {
            buckets[curBucketIdx].indexEnd = tid + 1;
            return;
        }

        std::uint64_t nextBucketIdx = hashTable[tid + 1].bucketIdx;

        if (tid == 0) {
            buckets[curBucketIdx].indexBegin = tid;
        }

        if (curBucketIdx != nextBucketIdx) {
            buckets[curBucketIdx].indexEnd = tid + 1;
            buckets[nextBucketIdx].indexBegin = tid + 1;
        }

        tid += blockDim.x * gridDim.x;
    }
}

struct UpdateBucketNumPointsFunctor {
    __host__ __device__ void operator()(Bucket& bucket)
    {
        bucket.numPoints = bucket.indexEnd - bucket.indexBegin;
    }
};

__global__ void markPoints(const Bucket* buckets, const HashElement* hashTable, bool* isMarked, int numBuckets)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (tid < numBuckets) {
        if (buckets[tid].numPoints > 0) {
            isMarked[hashTable[buckets[tid].indexBegin].pointIdx] = true;
        }

        tid += blockDim.x * gridDim.x;
    }
}
}  // namespace

std::vector<bool> DownSampleHandler::filter(const Eigen::Vector3f* hPoints, int numPoints)
{
    this->clear();
    HANDLE_ERROR(cudaMalloc((void**)&m_dPoints, numPoints * sizeof(Eigen::Vector3f)));
    HANDLE_ERROR(cudaMemcpy(m_dPoints, hPoints, numPoints * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));
    m_numPoints = numPoints;

    GridParam gridParam = this->estimateGridParam();
    GridParam* dGridParam;
    cudaMalloc((void**)&dGridParam, sizeof(GridParam));
    cudaMemcpy(dGridParam, &gridParam, sizeof(GridParam), cudaMemcpyHostToDevice);

    thrust::device_vector<Bucket> buckets(gridParam.bucketTotalNum);
    thrust::device_vector<HashElement> hashTable(numPoints);

    dim3 blockDim(16 * 16);
    dim3 gridDim(min(MAX_BLOCKS, (numPoints + blockDim.x - 1) / blockDim.x));

    updateHashTable<<<gridDim, blockDim>>>(m_dPoints, dGridParam, thrust::raw_pointer_cast(hashTable.data()),
                                           m_param.leafSize, numPoints);

    thrust::sort(hashTable.begin(), hashTable.end(), CompareHashFunctor());

    updateBucketsKernel<<<gridDim, blockDim>>>(thrust::raw_pointer_cast(hashTable.data()),
                                               thrust::raw_pointer_cast(buckets.data()), numPoints);

    thrust::for_each(thrust::device, buckets.begin(), buckets.end(), UpdateBucketNumPointsFunctor());

    thrust::device_vector<bool> isMarked(numPoints, false);

    markPoints<<<dim3(min(static_cast<std::uint64_t>(MAX_BLOCKS),
                          (gridParam.bucketTotalNum + blockDim.x - 1) / blockDim.x)),
                 blockDim>>>(thrust::raw_pointer_cast(buckets.data()), thrust::raw_pointer_cast(hashTable.data()),
                             thrust::raw_pointer_cast(isMarked.data()), gridParam.bucketTotalNum);

    std::vector<bool> hIsMarked(numPoints);
    thrust::copy(isMarked.begin(), isMarked.end(), hIsMarked.begin());

    cudaDeviceSynchronize();

    cudaFree(dGridParam);

    return hIsMarked;
}

DownSampleHandler::GridParam DownSampleHandler::estimateGridParam() const
{
    std::vector<cudaStream_t> streams(3);
    for (auto& s : streams)
        cudaStreamCreate(&s);

    using MinMax = thrust::pair<Eigen::Vector3f*, Eigen::Vector3f*>;
    std::vector<MinMax> minMaxList(3);
    for (int i = 0; i < 3; ++i) {
        minMaxList[i] = thrust::minmax_element(thrust::cuda::par.on(streams[i]), m_dPoints, m_dPoints + m_numPoints,
                                               CompareAXisFunctor(i));
    }
    for (auto& s : streams) {
        cudaStreamSynchronize(s);
    }
    for (auto& s : streams) {
        cudaStreamDestroy(s);
    }

    GridParam gridParam;

    Eigen::Vector3f bPoint;
    for (int i = 0; i < 3; ++i) {
        cudaMemcpy(&bPoint, minMaxList[i].first, sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost);
        gridParam.minPoint[i] = bPoint[i];
    }
    for (int i = 0; i < 3; ++i) {
        cudaMemcpy(&bPoint, minMaxList[i].second, sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost);
        gridParam.maxPoint[i] = bPoint[i];
    }

    for (int i = 0; i < 3; ++i) {
        gridParam.bucketNums[i] = std::ceil((gridParam.maxPoint[i] - gridParam.minPoint[i]) / m_param.leafSize);
    }
    gridParam.bucketTotalNum = gridParam.bucketNums.prod();

    return gridParam;
}

void DownSampleHandler::clear()
{
    if (m_dPoints) {
        cudaFree(m_dPoints);
        m_dPoints = nullptr;
    }

    m_numPoints = -1;
}
}  // namespace cuda
}  // namespace _pcl
