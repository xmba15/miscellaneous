/**
 * @file    App.cpp
 *
 * @author  btran
 *
 * @date    2021-11-10
 *
 */

#include <iostream>

#include <sophus/se3.hpp>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "ICPCeres.hpp"

namespace
{
using PointType = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<PointType>;
using PointCloudPtr = PointCloud::Ptr;

pcl::visualization::PCLVisualizer::Ptr initializeViewer();
Eigen::Affine3f getTransformMatrix();

auto viewer = initializeViewer();
auto T = getTransformMatrix();
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

    std::cout << "\noriginal cloud -> transformed cloud: \n" << T.matrix() << "\n\n";

    std::cout << "\ntransformed cloud -> original cloud (inverse matrix): \n" << T.matrix().inverse() << "\n\n";

    PointCloudPtr transformedCloud(new PointCloud);
    pcl::transformPointCloud(*inCloud, *transformedCloud, T);

    viewer->addPointCloud<PointType>(
        inCloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(inCloud, 0., 0., 255.),
        "original_cloud");
    viewer->addPointCloud<PointType>(
        transformedCloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(inCloud, 255., 0., 0.),
        "transformed_cloud");

    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "original_cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "transformed_cloud");

    Eigen::Affine3f invT = Eigen::Affine3f::Identity();
    _pcl::estimate3D3DPosePointToPlane<PointType>(transformedCloud, inCloud, invT);

    std::cout << "\ntransformed cloud -> original cloud (optimization): \n" << invT.matrix() << "\n\n";
    PointCloudPtr optimizedOriginalCloud(new PointCloud);
    pcl::transformPointCloud(*transformedCloud, *optimizedOriginalCloud, invT.matrix());
    viewer->addPointCloud<PointType>(
        optimizedOriginalCloud,
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(optimizedOriginalCloud, 0., 255., 0.),
        "optimized_original_cloud");

    double err = 0.;
    for (std::size_t i = 0; i < inCloud->size(); ++i) {
        err += (inCloud->points[i].getVector3fMap() - optimizedOriginalCloud->points[i].getVector3fMap()).norm();
    }

    std::cout << "reprojection err: " << err << "\n";

    while (!viewer->wasStopped()) {
        viewer->spinOnce();
    }

    return EXIT_SUCCESS;
}

namespace
{
pcl::visualization::PCLVisualizer::Ptr initializeViewer()
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->addCoordinateSystem(0.05);
    viewer->setCameraPosition(-0.0945661, 0.185679, 0.468724, -0.0337721, 0.0815661, 0.019675, -0.0272854, 0.972978,
                              -0.229281);

    return viewer;
}

Eigen::Affine3f getTransformMatrix()
{
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.rotate(Eigen::AngleAxisf(M_PI / 3, Eigen::Vector3f::UnitZ()));
    transform.rotate(Eigen::AngleAxisf(M_PI / 2.5, Eigen::Vector3f::UnitY()));
    transform.translation() << 0.3, 0.0, 0.1;

    return transform;
}
}  // namespace
