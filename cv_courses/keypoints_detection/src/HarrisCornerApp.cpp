/**
 * @file    HarrisCornerApp.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include <opencv2/opencv.hpp>

namespace
{
cv::TickMeter meter;
constexpr int NUM_TEST = 10;
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cout << "[app] [path/to/image]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string imagePath = argv[1];

    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cout << "failed to read image: " << imagePath << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat imageGray;
    cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);

    const int blockSize = 3;
    const int apertureSize = 3;  // for sobel operator to calculate image derivatives
    const double k = 0.01;
    int thresh = 140;

    cv::Mat dst;

    meter.reset();
    meter.start();
    for (int i = 0; i < NUM_TEST; ++i) {
        cv::cornerHarris(imageGray, dst, blockSize, apertureSize, k);
    }
    meter.stop();
    std::cout << "opencv harris (cpu): " << meter.getTimeMilli() / NUM_TEST << "[ms]\n";



    cv::Mat normalizedDst;
    cv::normalize(dst, normalizedDst, 0, 255, cv::NORM_MINMAX, CV_32FC1);

    for (int i = 0; i < normalizedDst.rows; i++) {
        for (int j = 0; j < normalizedDst.cols; j++) {
            if ((int)normalizedDst.at<float>(i, j) > thresh) {
                cv::circle(image, cv::Point(j, i), 5, cv::Scalar(0), 2, 8, 0);
            }
        }
    }

    cv::imshow("corners", image);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}
