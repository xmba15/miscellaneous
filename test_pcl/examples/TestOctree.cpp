/**
 * @file    TestOctree.cpp
 *
 * @brief   Test Octree Data
 *
 * @author  bt
 *
 * @date    2019-06-18
 *
 * Copyright (c) organization
 *
 */
#include <memory>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

using PointCloudXYZ = pcl::PointCloud<pcl::PointXYZ>;

int main(int argc, char *argv[])
{
    PointCloudXYZ::Ptr cloud(new PointCloudXYZ);

    return 0;
}
