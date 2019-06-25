/**
 * @file    TestOpen3D.cpp
 *
 * @brief   test open3d code
 *
 * @author  BT
 *
 * @date    2019-03-04
 *
 * framework
 *
 * Copyright (c) organization
 *
 */

#include <Open3D/Open3D.h>
#include <memory>
#include <thread>

using PointCloud = open3d::geometry::PointCloud;
using PointCloudPtr = std::shared_ptr<PointCloud>;

int main(int argc, char *argv[])
{
    open3d::utility::PrintInfo("hello world open3d\n");
    PointCloudPtr cloud = std::make_shared<PointCloud>();

    // the following is of type std:vector<Eigen::Vector3d>
    // cloud->points_
    // cloud->normal_
    // cloud->colors_

    if (argc != 2) {
        open3d::utility::PrintError(
            "need to provid path to point cloud data\n");
        exit(1);
    }

    open3d::io::ReadPointCloud(argv[1], *cloud);

    cloud->NormalizeNormals();

    // function definition
    // bool DrawGeometries(const std::vector<std::shared_ptr<const geometry::Geometry>>
    //                             &geometry_ptrs,
    //                     const std::string &window_name /* = "Open3D"*/,
    //                     int width /* = 640*/,
    //                     int height /* = 480*/,
    //                     int left /* = 50*/,
    //                     int top /* = 50*/)

    // open3d::visualization::DrawGeometries({cloud}, "PointCloud", 1600, 900);
    std::cout << cloud->points_.size() << "\n";

    double color_index = 0.0;
    double color_index_step = 0.05;

    auto update_colors_func = [&cloud](double index) {
        auto color_map_ptr = open3d::visualization::GetGlobalColorMap();

        for (auto &c : cloud->colors_) {
            c = color_map_ptr->GetColor(index);
        }
    };

    update_colors_func(1.0);

    open3d::visualization::DrawGeometriesWithAnimationCallback(
        {cloud},
        [&](open3d::visualization::Visualizer *vis) {
            color_index += color_index_step;
            if (color_index > 2.0)
                color_index -= 2.0;
            update_colors_func(fabs(color_index - 1.0));
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return true;
        },
        "Rainbow", 1600, 900);

    return 0;
}
