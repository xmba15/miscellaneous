/**
 * @file    DownSampleHandler.cuh
 *
 * @author  btran
 *
 */

#pragma once

#include <cstdint>

#include <Eigen/Core>

namespace _pcl
{
namespace cuda
{
class DownSampleHandler
{
 public:
    struct Param {
        float leafSize = 0.1;
    };

    explicit DownSampleHandler(const Param& param)
        : m_param(param)
        , m_dPoints(nullptr)
    {
    }

    ~DownSampleHandler()
    {
        this->clear();
    }

    std::vector<bool> filter(const Eigen::Vector3f* hPoinst, int numPoints);

    struct GridParam {
        Eigen::Vector3f minPoint;
        Eigen::Vector3f maxPoint;
        Eigen::Vector3i bucketNums;
        std::uint64_t bucketTotalNum = 0;
    };

 private:
    GridParam estimateGridParam() const;

    void clear();

 private:
    Param m_param;

    Eigen::Vector3f* m_dPoints;
    int m_numPoints = -1;
};
}  // namespace cuda
}  // namespace _pcl
