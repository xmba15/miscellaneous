/**
 * @file    PerceptionUtils.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace _pcl
{
namespace utils
{
inline pcl::visualization::PCLVisualizer::Ptr initializeViewer()
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    pcl::PointXYZ o(0.1, 0, 0);
    viewer->setBackgroundColor(0.05, 0.05, 0.05, 0);
    viewer->addCoordinateSystem(0.5);
    viewer->setCameraPosition(-26, 0, 3, 10, -1, 0.5, 0, 0, 1);

    return viewer;
}

inline std::vector<std::array<double, 3>> generateColorCharts(const std::uint16_t numSources,
                                                              const std::uint16_t seed = 2021)
{
    std::srand(seed);
    std::vector<std::array<double, 3>> colors(numSources);
    for (std::uint16_t i = 0; i < numSources; ++i) {
        colors[i] =
            std::array<double, 3>{(std::rand() % 256) / 255., (std::rand() % 256) / 255., (std::rand() % 256) / 255.};
    }
    return colors;
}

template <typename PointType>
inline std::vector<Eigen::Vector3f> toSTLContainers(const typename pcl::PointCloud<PointType>::Ptr& inCloud)
{
    if (inCloud->empty()) {
        return {};
    }

    std::vector<Eigen::Vector3f> output;
    output.reserve(inCloud->size());
    std::transform(inCloud->points.begin(), inCloud->points.end(), std::back_inserter(output),
                   [](const auto& elem) { return elem.getVector3fMap(); });

    return output;
}

template <typename PointType = pcl::PointXYZ>
inline typename pcl::PointCloud<PointType>::Ptr toCloud(const std::vector<Eigen::Vector3f>& pointsVec)
{
    if (pointsVec.empty()) {
        return typename pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>);
    }

    typename pcl::PointCloud<PointType>::Ptr outCloud(new pcl::PointCloud<PointType>);
    outCloud->points.reserve(pointsVec.size());

    std::transform(pointsVec.begin(), pointsVec.end(), std::back_inserter(outCloud->points),
                   [](const auto& elem) { return PointType(elem[0], elem[1], elem[2]); });
    outCloud->height = 1;
    outCloud->width = pointsVec.size();

    return outCloud;
}
}  // namespace utils
}  // namespace _pcl
