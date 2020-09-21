/**
 * @file    knn.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <memory>
#include <vector>

namespace knn
{
class KNNHandler
{
 public:
    KNNHandler();
    ~KNNHandler();

    bool run(const float* ref, int refNum, const float* query, int queryNum, int dimension, int k);
    bool downloadToHost(float* knnDist, int* knnIndices);

 private:
    class KNNHandlerImpl;
    std::unique_ptr<KNNHandlerImpl> m_pimpl;
};
}  // namespace knn
