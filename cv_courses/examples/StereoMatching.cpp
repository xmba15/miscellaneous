/**
 * @file    StereoMatching.cpp
 *
 * @author  btran
 *
 */

#include <utility>

#include <opencv2/opencv.hpp>
#include <opencv2/stereo.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

// Ref: https://github.com/opencv/opencv/blob/master/samples/cpp/stereo_match.cpp
// Ref: https://github.com/ros-perception/vision_opencv/blob/noetic/image_geometry/include/image_geometry/stereo_camera_model.h
// Ref: https://github.com/ros-perception/vision_opencv/blob/noetic/image_geometry/src/stereo_camera_model.cpp

namespace
{
struct CalibrationParam {
    double fx = 1936;
    double fy = 1096;
    double cx = 972.332144;
    double cy = 522.410129;
};

std::pair<CalibrationParam, CalibrationParam> getCalibParams(const float scaleFactorX = 1.0,
                                                             const float scaleFactorY = 1.0);

double BASE_LINE = 2.322887e-01;  // meter
double MAX_DEPTH = 10.0;           // meter

using PointCloudType = pcl::PointXYZRGB;
using PointCloud = pcl::PointCloud<PointCloudType>;
using PointCloudPtr = PointCloud::Ptr;

pcl::visualization::PCLVisualizer::Ptr initializeViewer();
}  // namespace

int main(int argc, char* argv[])
{
    const std::string IMAGE_PATH = std::string(DATA_PATH) + "/";
    const std::string LEFT_PATH = IMAGE_PATH + "left.png";
    const std::string RIGHT_PATH = IMAGE_PATH + "right.ppg";

    cv::Mat left = cv::imread(LEFT_PATH, 1);
    cv::Mat right = cv::imread(RIGHT_PATH, 1);

    const float scaleFactorX = 1.0;
    const float scaleFactorY = 1.0;
    auto calibParams = ::getCalibParams(scaleFactorX, scaleFactorY);
    PointCloudPtr output(new PointCloud);

    // cv::resize(left, left, cv::Size(), scaleFactorX, scaleFactorY);
    // cv::resize(right, right, cv::Size(), scaleFactorX, scaleFactorY);

    // int minDisparity = 0;
    // int rangeDisparity = 16 * 10;
    // int SADWindowSize = 5;
    // int channel = left.channels();
    // const int disparityMultiplier = 16.0;

    // cv::Ptr<cv::stereo::StereoBinarySGBM> sgbm =
    //     cv::stereo::StereoBinarySGBM::create(0,              //int minDisparity
    //                                          64,             //int numDisparities
    //                                          SADWindowSize,  //int SADWindowSize
    //                                          10,             //int P1 = 0
    //                                          100,            //int P2 = 0
    //                                          1,              //int disp12MaxDiff = 0
    //                                          0,              //int preFilterCap = 0
    //                                          5,              //int uniquenessRatio = 0
    //                                          400,            //int speckleWindowSize = 0
    //                                          0               //int speckleRange = 0
    //     );

    // cv::Mat disp;
    // sgbm->compute(left, right, disp);
    // // divide by 16 to get the true disparity value
    // disp.convertTo(disp, CV_32F, 1.0 / disparityMultiplier);

    // cv::Mat disp8U = cv::Mat(disp.rows, disp.cols, CV_8UC1);
    // cv::normalize(disp, disp8U, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    // cv::imshow("results/BM.jpg", disp8U);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    // const std::string IMAGE_PATH = std::string(DATA_PATH) + "/";
    const std::string DISPARITY_PATH = IMAGE_PATH + "outer_pred.png";

    cv::Mat disp8U = cv::imread(DISPARITY_PATH, 0);

    auto& leftCam = calibParams.first;
    leftCam.fx = 4.184067 * 1000;
    leftCam.fy = 4.184067 * 1000;
    leftCam.cx = 9.458483 * 100;
    leftCam.cy = 5.456414 * 100;

    cv::Matx44d Q_;
    Q_(0, 0) = calibParams.first.fy * BASE_LINE;
    Q_(0, 3) = -calibParams.first.fy * calibParams.first.cx * BASE_LINE;
    Q_(1, 1) = calibParams.first.fx * BASE_LINE;
    Q_(1, 3) = -calibParams.first.fx * calibParams.first.cy * BASE_LINE;
    Q_(2, 3) = calibParams.first.fx * calibParams.first.fy * BASE_LINE;
    Q_(3, 2) = -calibParams.first.fy;
    Q_(3, 3) = 0.0;

    for (int i = 0; i < disp8U.rows; ++i) {
        auto curRow = disp8U.ptr<uchar>(i);
        auto imgCurRow = left.ptr<cv::Vec3b>(i);

        for (int j = 0; j < disp8U.cols; ++j) {
            double z = calibParams.first.fx * BASE_LINE / static_cast<int>(curRow[j]);
            double x = (j - calibParams.first.cx) * z / calibParams.first.fx;
            double y = (i - calibParams.first.cy) * z / calibParams.first.fx;

            double disparity = static_cast<int>(curRow[j]);
            double u = j, v = i;
            cv::Point3d XYZ((Q_(0, 0) * u) + Q_(0, 3), (Q_(1, 1) * v) + Q_(1, 3), Q_(2, 3));
            double W = Q_(3, 2) * disparity + Q_(3, 3);
            XYZ = XYZ * (1.0 / W);

            PointCloudType point;
            // point.x = z;
            // point.y = -x;
            // point.z = -y;

            point.x = XYZ.z;
            point.y = -XYZ.x;
            point.z = -XYZ.y;

            if (point.x > MAX_DEPTH) {
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
std::pair<CalibrationParam, CalibrationParam> getCalibParams(const float scaleFactorX, const float scaleFactorY)
{
    CalibrationParam left, right;
    left.fx = 1936. * scaleFactorX;
    left.fy = 1096.0 * scaleFactorY;
    left.cx = 972.332144 * scaleFactorX;
    left.cy = 522.410129 * scaleFactorY;

    right.fx = 1936 * scaleFactorX;
    right.fy = 1096 * scaleFactorY;
    right.cx = 932.214570 * scaleFactorX;
    right.cy = 525.280171 * scaleFactorY;

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
