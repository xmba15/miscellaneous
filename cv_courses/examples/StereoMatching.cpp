/**
 * @file    StereoMatching.cpp
 *
 * @author  btran
 *
 */

#include <utility>

#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

// Ref: https://github.com/opencv/opencv/blob/master/samples/cpp/stereo_match.cpp

namespace
{
struct CalibrationParam {
    double fx = 1936;
    double fy = 1096;
    double cx = 972.332144;
    double cy = 522.410129;
};

std::pair<CalibrationParam, CalibrationParam> getCalibParams(const float scaleFactor = 1.0);

double BASE_LINE = -2.322887e-01 * 1000;  // mm

using PointCloudType = pcl::PointXYZRGB;
using PointCloud = pcl::PointCloud<PointCloudType>;
using PointCloudPtr = PointCloud::Ptr;

pcl::visualization::PCLVisualizer::Ptr initializeViewer();
}  // namespace

int main(int argc, char* argv[])
{
    const std::string IMAGE_PATH = std::string(DATA_PATH) + "/";
    const std::string LEFT_PATH = IMAGE_PATH + "left.jpg";
    const std::string RIGHT_PATH = IMAGE_PATH + "right.jpg";

    cv::Mat left = cv::imread(LEFT_PATH, 1);
    cv::Mat right = cv::imread(RIGHT_PATH, 1);

    const float scaleFactor = 0.5;
    cv::resize(left, left, cv::Size(), scaleFactor, scaleFactor);
    cv::resize(right, right, cv::Size(), scaleFactor, scaleFactor);

    int minDisparity = 0;
    // int rangeDisparity = ((left.cols / 8) + 15) & -16;
    int rangeDisparity = 16 * 10;
    int SADWindowSize = 5;
    const int disparityMultiplier = 16.0;

    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(minDisparity, rangeDisparity, SADWindowSize);
    // int cn = left.channels();
    // sgbm->setPreFilterCap(63);
    // sgbm->setP1(8 * cn * sgbmWinSize * sgbmWinSize);
    // sgbm->setP2(32 * cn * sgbmWinSize * sgbmWinSize);
    // sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

    cv::Mat disp;
    sgbm->compute(left, right, disp);
    // divide by 16 to get the true disparity value
    disp.convertTo(disp, CV_32F, 1.0 / disparityMultiplier);

    cv::Mat disp8U = cv::Mat(disp.rows, disp.cols, CV_8UC1);
    cv::normalize(disp, disp8U, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::imshow("results/BM.jpg", disp8U);
    cv::waitKey(0);
    cv::destroyAllWindows();

    const auto calibParams = ::getCalibParams(scaleFactor);
    cv::Mat Q = cv::Mat(4, 4, CV_64F, double(0));
    Q.at<double>(0, 0) = 1.0;
    Q.at<double>(0, 3) = -calibParams.first.cx;
    Q.at<double>(1, 1) = 1.0;
    Q.at<double>(1, 3) = -calibParams.first.cy;
    Q.at<double>(2, 3) = calibParams.first.fx;
    Q.at<double>(3, 2) = -1.0 / BASE_LINE * scaleFactor;
    Q.at<double>(3, 3) = (calibParams.first.cx - calibParams.second.cx) / BASE_LINE * scaleFactor;

    cv::Mat xyz;
    cv::reprojectImageTo3D(disp8U, xyz, Q, true);

    PointCloudPtr output(new PointCloud);

    for (int i = 0; i < xyz.rows; i++) {
        auto imgCurRow = left.ptr<cv::Vec3b>(i);
        auto curRow = disp8U.ptr<uchar>(i);
        for (int j = 0; j < xyz.cols; j++) {
            cv::Point3f p = xyz.at<cv::Point3f>(i, j);
            double radius = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
            if (radius > 20 * 1000) {
                continue;
            }

            float curDisp = curRow[j];
            double d = curDisp + minDisparity;
            pcl::PointXYZRGB point;
            point.x = p.x / 1000.;
            point.y = p.y / 1000.;
            point.z = p.z / 1000.;

            point.x = p.z / 1000.;
            point.y = -p.x / 1000.;
            point.z = -p.y / 1000.;

            if (point.x < 0 || point.x > 5) {
                continue;
            }

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

    return 0;
}

namespace
{
std::pair<CalibrationParam, CalibrationParam> getCalibParams(const float scaleFactor)
{
    CalibrationParam left, right;
    left.fx = 1936. * scaleFactor;
    left.fy = 1096.0 * scaleFactor;
    left.cx = 972.332144 * scaleFactor;
    left.cy = 522.410129 * scaleFactor;

    right.fx = 1936 * scaleFactor;
    right.fy = 1096 * scaleFactor;
    right.cx = 932.214570 * scaleFactor;
    right.cy = 525.280171 * scaleFactor;

    return std::make_pair(left, right);
}

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
}  // namespace
