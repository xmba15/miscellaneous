/**
 * @file    DisparityVis.cpp
 *
 * @author  btran
 *
 */

#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

using PointCloudType = pcl::PointXYZRGB;
using PointCloud = pcl::PointCloud<PointCloudType>;
using PointCloudPtr = PointCloud::Ptr;

double MAX_DEPTH = 40;
double BASE_LINE = -2.322887e-01;  // meter

pcl::visualization::PCLVisualizer::Ptr initializeViewer()
{
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    pcl::PointXYZ o(0.1, 0, 0);
    viewer->addSphere(o, 0.1, "sphere", 0);
    viewer->setBackgroundColor(0.05, 0.05, 0.05, 0);
    viewer->addCoordinateSystem(0.5);
    viewer->setCameraPosition(-26, 0, 3, 10, -1, 0.5, 0, 0, 1);

    return viewer;
}

int main(int argc, char* argv[])
{
    const std::string IMAGE_PATH = std::string(DATA_PATH) + "/";
    const std::string DISPARITY_PATH = IMAGE_PATH + "disparity.png";
    const std::string LEFT_PATH = IMAGE_PATH + "left.png";
    const double DISPARITY_MULTIPLIER = 256.0;
    const double MIN_DISP = 10;

    cv::Mat left = cv::imread(LEFT_PATH, 1);
    cv::Mat disp = cv::imread(DISPARITY_PATH, CV_16UC1);
    disp.convertTo(disp, CV_32FC1, 1.0 / DISPARITY_MULTIPLIER);

    double min, max;
    cv::minMaxLoc(disp, &min, &max);
    std::cout << "min: " << min << "\n";
    std::cout << "max: " << max << "\n";

    cv::Matx44d Q_;
    double fx = 4.184067e03;
    double fy = fx;
    double cx = 9.458483e02;
    double cy = 5.456414e02;

    Q_(0, 0) = fy * BASE_LINE;
    Q_(0, 3) = -fy * cx * BASE_LINE;
    Q_(1, 1) = fx * BASE_LINE;
    Q_(1, 3) = -fx * cy * BASE_LINE;
    Q_(2, 3) = fx * fy * BASE_LINE;
    Q_(3, 2) = -fy;
    Q_(3, 3) = 0.0;

    PointCloudPtr output(new PointCloud);
    for (int i = 0; i < disp.rows; ++i) {
        auto curRow = disp.ptr<float>(i);
        auto imgCurRow = left.ptr<cv::Vec3b>(i);
        for (int j = 0; j < disp.cols; ++j) {
            double disparity = curRow[j];
            if (disparity < MIN_DISP) {
                continue;
            }

            double u = j, v = i;

            cv::Point3d XYZ((Q_(0, 0) * u) + Q_(0, 3), (Q_(1, 1) * v) + Q_(1, 3), Q_(2, 3));
            double W = Q_(3, 2) * disparity + Q_(3, 3);
            cv::Point3d xyz = XYZ * (1.0 / W);

            PointCloudType point;

            point.x = xyz.z;
            point.y = -xyz.x;
            point.z = -xyz.y;

            point.r = imgCurRow[j][0];
            point.g = imgCurRow[j][1];
            point.b = imgCurRow[j][2];
            output->points.emplace_back(point);
        }
    }

    output->width = 1;
    output->height = output->points.size();
    pcl::io::savePCDFileASCII("output.pcd", *output);

    auto pclViewer = ::initializeViewer();
    pclViewer->addPointCloud<PointCloudType>(output, "output");

    while (!pclViewer->wasStopped()) {
        pclViewer->spinOnce();
    }

    return EXIT_SUCCESS;
}
