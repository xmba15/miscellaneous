/**
 * @file    DenseOpticalFlowApp.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include <opencv2/opencv.hpp>

namespace
{
cv::Mat visualizeDenseFlow(const cv::Mat& image, const cv::Mat& denseFlow, int gridSize = 30)
{
    cv::Mat flows[2];
    cv::split(denseFlow, flows);

    cv::Mat result = image.clone();

    for (int i = 0; i < image.rows; i += gridSize) {
        for (int j = 0; j < image.cols; j += gridSize) {
            const auto& flow = denseFlow.at<cv::Point2f>(i, j);
            cv::arrowedLine(result, cv::Point(j, i), cv::Point(j + flow.x, i + flow.y), cv::Scalar(0, 0, 255), 2,
                            cv::LINE_AA, 0, 0.5 /* tip length*/);
        }
    }

    return result;
}

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

    cv::Mat denseFlow(images[0].size(), CV_32FC2);

    // -------------------------------------------------------------------------
    // by opencv
    // -------------------------------------------------------------------------
    meter.reset();
    meter.start();
    cv::calcOpticalFlowFarneback(grayImages[0], grayImages[1], denseFlow, 0.5, 4, 2, 10, 5, 1.1, 0);
    meter.stop();
    std::cout << "by opencv: " << meter.getTimeMilli() / NUM_TEST << "[ms]" << std::endl;

    cv::Mat ofCV;
    ofCV = visualizeDenseFlow(images[1], denseFlow);

    cv::imshow("dense optical flow by opencv", ofCV);
    cv::waitKey(0);

    return 0;
}
