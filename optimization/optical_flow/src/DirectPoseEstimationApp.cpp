/**
 * @file    DirectPoseEstimationApp.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include <opencv2/opencv.hpp>

#include "DirectPoseEstimation.hpp"
#include "Utility.hpp"

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: [app] [first/image/path] [second/image/path] [disparity/image/[path]" << std::endl;
        return EXIT_FAILURE;
    }

    std::string imagePaths[2] = {argv[1], argv[2]};
    cv::Mat images[2];
    cv::Mat grayImages[2];
    cv::Mat disparityImage;

    for (int i = 0; i < 2; ++i) {
        images[i] = cv::imread(imagePaths[i]);
        if (images[i].empty()) {
            std::cerr << "failed to load: " << imagePaths[i] << std::endl;
            return EXIT_FAILURE;
        }
        cv::cvtColor(images[i], grayImages[i], cv::COLOR_BGR2GRAY);
    }

    disparityImage = cv::imread(argv[3], 0);
    if (disparityImage.empty()) {
        std::cerr << "failed to load: " << argv[3] << std::endl;
        return EXIT_FAILURE;
    }

    _cv::CameraMatrix K({.fx = 718.856, .fy = 718.856, .cx = 607.1928, .cy = 185.2157, .baseLine = 0.573});

    cv::RNG rng(2021);
    int nPoints = 2000;
    int borderMargin = 20;
    std::vector<cv::Point2d> refPoints;
    std::vector<double> refPointsDepth;

    for (int i = 0; i < nPoints; i++) {
        int x = rng.uniform(borderMargin, images[0].cols - 1 - borderMargin);
        int y = rng.uniform(borderMargin, images[0].rows - 1 - borderMargin);
        int disparity = disparityImage.at<uchar>(y, x);
        refPointsDepth.emplace_back(K.fx * K.baseLine / disparity);
        refPoints.emplace_back(cv::Point2d(x, y));
    }

    Sophus::SE3d T21;

    int numScale = 4;
    double scaleFactor = 0.5;

    std::vector<cv::Point2d> projecteds(refPoints.size(), cv::Point2d());
    calcDirectPoseEstimationMultiLayer(grayImages[0], grayImages[1], refPoints, refPointsDepth, K, T21, projecteds,
                                       numScale, scaleFactor);

    for (std::size_t i = 0; i < refPoints.size(); ++i) {
        const auto& refPoint = refPoints[i];
        const auto& curPoint = projecteds[i];
        if (curPoint.x > 0 && curPoint.y > 0) {
            cv::arrowedLine(images[1], refPoint, curPoint, cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0,
                            0.1 /* tip length*/);
        }
    }

    std::cout << "T21=\n" << T21.matrix() << "\n";

    cv::imshow("result", images[1]);
    cv::waitKey(0);

    return 0;
}
