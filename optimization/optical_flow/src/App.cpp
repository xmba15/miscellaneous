/**
 * @file    App.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include <opencv2/opencv.hpp>

#include "OpticalFlow.hpp"

namespace
{
constexpr int NUM_TEST = 100;
cv::TickMeter meter;
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: [app] [first/image/path] [second/image/path]" << std::endl;
        return EXIT_FAILURE;
    }

    std::string imagePaths[2] = {argv[1], argv[2]};
    cv::Mat images[2];
    cv::Mat grayImages[2];

    for (int i = 0; i < 2; ++i) {
        images[i] = cv::imread(imagePaths[i]);
        if (images[i].empty()) {
            std::cerr << "failed to load: " << imagePaths[i] << std::endl;
            return EXIT_FAILURE;
        }
        cv::cvtColor(images[i], grayImages[i], cv::COLOR_BGR2GRAY);
    }

    std::vector<cv::KeyPoint> kp0;
    // good feature to track
    cv::Ptr<cv::GFTTDetector> keyPointDetector =
        cv::GFTTDetector::create(500 /* max corners*/, 0.01 /* quality level*/, 20 /*min distance */);
    keyPointDetector->detect(grayImages[0], kp0);

    std::vector<cv::Point2f> pts[2];
    std::transform(kp0.begin(), kp0.end(), std::back_inserter(pts[0]), [](const auto elem) { return elem.pt; });
    std::vector<uchar> status;
    std::vector<float> error;

    // -------------------------------------------------------------------------
    // by opencv
    // -------------------------------------------------------------------------
    meter.reset();
    meter.start();
    for (int i = 0; i < NUM_TEST; ++i) {
        cv::calcOpticalFlowPyrLK(grayImages[0], grayImages[1], pts[0], pts[1], status, error);
    }
    meter.stop();
    std::cout << "by opencv: " << meter.getTimeMilli() / NUM_TEST << "[ms]" << std::endl;

    cv::Mat ofCV = images[1].clone();
    for (std::size_t i = 0; i < pts[1].size(); i++) {
        if (status[i]) {
            cv::arrowedLine(ofCV, pts[0][i], pts[1][i], cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.5 /* tip length*/);
        }
    }

    // -------------------------------------------------------------------------
    // optical flow single level
    // -------------------------------------------------------------------------

    std::vector<cv::KeyPoint> kp1Single;
    std::vector<bool> successSingle;

    meter.reset();
    meter.start();
    _cv::calcOpticalFlowSingleLevel(grayImages[0], grayImages[1], kp0, kp1Single, successSingle);
    meter.stop();
    std::cout << "single level: " << meter.getTimeMilli() / NUM_TEST << "[ms]" << std::endl;

    cv::Mat singleCV = images[1].clone();
    for (std::size_t i = 0; i < pts[1].size(); i++) {
        if (successSingle[i]) {
            cv::arrowedLine(singleCV, kp0[i].pt, kp1Single[i].pt, cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0,
                            0.5 /* tip length*/);
        }
    }

    // -------------------------------------------------------------------------
    // optical flow multi level
    // -------------------------------------------------------------------------

    std::vector<cv::KeyPoint> kp1Multi;
    std::vector<bool> successMulti;
    meter.reset();
    meter.start();
    _cv::calcOpticalFlowMultiLevel(grayImages[0], grayImages[1], kp0, kp1Multi, successMulti);
    meter.stop();
    std::cout << "multi level: " << meter.getTimeMilli() / NUM_TEST << "[ms]" << std::endl;

    cv::Mat multiCV = images[1].clone();
    for (std::size_t i = 0; i < pts[1].size(); i++) {
        if (successMulti[i]) {
            cv::arrowedLine(multiCV, kp0[i].pt, kp1Multi[i].pt, cv::Scalar(0, 0, 255), 2., cv::LINE_AA, 0,
                            0.5 /* tip length*/);
        }
    }

    cv::imshow("tracked by opencv", ofCV);
    cv::imshow("tracked by single level", singleCV);
    cv::imshow("tracked by multi level", multiCV);
    cv::waitKey(0);

    return EXIT_SUCCESS;
}
