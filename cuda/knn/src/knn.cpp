/**
 * @file    knn.cpp
 *
 * @author  btran
 *
 */

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <knn_cuda/knn.hpp>

namespace
{
namespace cuda
{
constexpr int BLOCK_SIZE = 16;

__global__ void computeDistances(const float* ref, int refNum, const float* query, int queryNum, int dimension,
                                 float* dist)
{
    __shared__ float refShared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float queryShared[BLOCK_SIZE][BLOCK_SIZE];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= queryNum || y >= refNum) {
        return;
    }

    float ssd = 0.;

    for (int m = 0; m < (dimension + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        if (threadIdx.x == 0 || threadIdx.y == 0) {
#pragma unroll
            for (int i = 0; i < BLOCK_SIZE; ++i) {
                refShared[i][threadIdx.y] = ref[(m * BLOCK_SIZE + i) * refNum + y];
                queryShared[i][threadIdx.x] = query[(m * BLOCK_SIZE + i) * queryNum + x];
            }
        }

        __syncthreads();

#pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            float dif = refShared[i][threadIdx.y] - queryShared[i][threadIdx.x];
            ssd += dif * dif;
        }

        __syncthreads();
    }

    dist[y * queryNum + x] = sqrt(ssd);
}

__global__ void sortKElements(float* dist, int* indices, int refNum, int queryNum, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= queryNum) {
        return;
    }

    float* pDist = dist + idx;
    int* pIndices = indices + idx;
    pIndices[0] = 0;

    for (int curIndex = 0; curIndex < refNum; ++curIndex) {
        float curDist = pDist[curIndex * queryNum];

        if (curIndex >= k && curDist >= pDist[(k - 1) * queryNum]) {
            continue;
        }

        int j = min(curIndex, k - 1);
        while (j > 0 && pDist[(j - 1) * queryNum] > curDist) {
            pDist[j * queryNum] = pDist[(j - 1) * queryNum];
            pIndices[j * queryNum] = pIndices[(j - 1) * queryNum];
            --j;
        }

        pDist[j * queryNum] = curDist;
        pIndices[j * queryNum] = curIndex;
    }
}
}  // namespace cuda
}  // namespace

namespace knn
{
class KNNHandler::KNNHandlerImpl
{
 public:
    KNNHandlerImpl();
    ~KNNHandlerImpl();

    bool run(const float* ref, int refNum, const float* query, int queryNum, int dimension, int k);

    bool downloadToHost(float* knnDist, int* knnIndices);

 private:
    void clear()
    {
        m_dist.clear();
        m_indices.clear();

        m_dist.shrink_to_fit();
        m_indices.shrink_to_fit();
    }

 private:
    thrust::device_vector<float> m_dist;
    thrust::device_vector<int> m_indices;
};

KNNHandler::KNNHandler()
    : m_pimpl(std::make_unique<KNNHandlerImpl>())
{
}

KNNHandler::~KNNHandler() = default;

bool KNNHandler::run(const float* ref, int refNum, const float* query, int queryNum, int dimension, int k)
{
    if (k > refNum) {
        throw std::runtime_error("k can not be more than number of reference points");
    }

    return m_pimpl->run(ref, refNum, query, queryNum, dimension, k);
}

bool KNNHandler::downloadToHost(float* knnDist, int* knnIndices)
{
    return m_pimpl->downloadToHost(knnDist, knnIndices);
}

KNNHandler::KNNHandlerImpl::KNNHandlerImpl()
    : m_dist({})
    , m_indices({})
{
}

KNNHandler::KNNHandlerImpl::~KNNHandlerImpl() = default;

bool KNNHandler::KNNHandlerImpl::run(const float* ref, int refNum, const float* query, int queryNum, int dimension,
                                     int k)
{
    this->clear();

    thrust::device_vector<float> dRef(ref, ref + refNum * dimension);
    thrust::device_vector<float> dQuery(query, query + queryNum * dimension);

    m_dist = thrust::device_vector<float>(queryNum * refNum);
    m_indices = thrust::device_vector<int>(queryNum * k);

    // compute all distances
    {
        dim3 blockDim(::cuda::BLOCK_SIZE, ::cuda::BLOCK_SIZE, 1);
        dim3 gridDim((queryNum + blockDim.x - 1) / blockDim.x, (refNum + blockDim.y - 1) / blockDim.y, 1);

        ::cuda::computeDistances<<<gridDim, blockDim>>>(thrust::raw_pointer_cast(dRef.data()), refNum,
                                                        thrust::raw_pointer_cast(dQuery.data()), queryNum, dimension,
                                                        thrust::raw_pointer_cast(m_dist.data()));
    }

    // sort k elements
    {
        dim3 blockDim(::cuda::BLOCK_SIZE * ::cuda::BLOCK_SIZE);
        dim3 gridDim((queryNum + blockDim.x - 1) / blockDim.x);
        ::cuda::sortKElements<<<gridDim, blockDim>>>(thrust::raw_pointer_cast(m_dist.data()),
                                                     thrust::raw_pointer_cast(m_indices.data()), refNum, queryNum, k);
    }

    return true;
}

bool KNNHandler::KNNHandlerImpl::downloadToHost(float* knnDist, int* knnIndices)
{
    if (m_dist.empty() || m_indices.empty()) {
        return false;
    }

    thrust::copy(m_dist.begin(), m_dist.end(), knnDist);
    thrust::copy(m_indices.begin(), m_indices.end(), knnIndices);

    return true;
}
}  // namespace knn
