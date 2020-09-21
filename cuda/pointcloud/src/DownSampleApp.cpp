/**
 * @file    DownSampleApp.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include <pcl/common/time.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include "CudaUtils.cuh"
#include "DownSampleHandler.cuh"
#include "PerceptionUtils.hpp"

namespace
{
using PointType = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointType>;
using PointCloudPtr = PointCloud::Ptr;

PointCloudPtr downSample(const PointCloudPtr& inCloud, float leafSize = 0.01f);

auto timer = pcl::StopWatch();
auto viewer = _pcl::utils::initializeViewer();

constexpr int NUM_TEST = 1;
constexpr float LEAF_SIZE = 0.1;
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "Usage: [app] [path/to/pcd]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string pclFilePath = argv[1];
    PointCloudPtr inCloud(new PointCloud);
    if (pcl::io::loadPCDFile(pclFilePath, *inCloud) == -1) {
        std::cerr << "Failed to load pcl file" << std::endl;
        return EXIT_FAILURE;
    }

    HANDLE_ERROR(cuda::utils::warmUpGPU());

    auto pointsVec = _pcl::utils::toSTLContainers<PointType>(inCloud);

    PointCloudPtr downSampledCloud(new PointCloud);
    timer.reset();
    for (int i = 0; i < NUM_TEST; ++i) {
        downSampledCloud = ::downSample(inCloud, LEAF_SIZE);
    }
    std::cout << "processing time (cpu): " << timer.getTime() / NUM_TEST << "[ms]\n";

    _pcl::cuda::DownSampleHandler::Param param;
    _pcl::cuda::DownSampleHandler downSampleHandler(param);

    std::vector<bool> isMarked;
    timer.reset();
    for (int i = 0; i < NUM_TEST; ++i) {
        isMarked = downSampleHandler.filter(pointsVec.data(), pointsVec.size());
    }
    std::cout << "processing time (gpu): " << timer.getTime() / NUM_TEST << "[ms]\n";

    std::vector<Eigen::Vector3f> downSampledPointsVec;
    downSampledPointsVec.reserve(inCloud->size());
    for (std::size_t i = 0; i < inCloud->size(); ++i) {
        if (isMarked[i]) {
            downSampledPointsVec.emplace_back(pointsVec[i]);
        }
    }
    auto downSampledCloudGPU = _pcl::utils::toCloud(downSampledPointsVec);

    viewer->addPointCloud<PointType>(downSampledCloudGPU, "down_sampled_cloud");
    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }

    return EXIT_SUCCESS;
}

namespace
{
PointCloudPtr downSample(const PointCloudPtr& inCloud, float leafSize)
{
    PointCloudPtr outCloud(new PointCloud);
    pcl::VoxelGrid<PointType> sor;
    sor.setInputCloud(inCloud);
    sor.setLeafSize(leafSize, leafSize, leafSize);
    sor.filter(*outCloud);

    return outCloud;
}
}  // namespace
