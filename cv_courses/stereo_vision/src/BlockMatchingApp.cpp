/**
 * @file    BlockMatchingApp.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include <opencv2/opencv.hpp>

#include "BlockMatching.hpp"
#include "Utility.hpp"

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: [app] [path/to/left/image] [path/to/right/image]\n";
        return EXIT_FAILURE;
    }

    const std::string leftImagePath = argv[1];
    const std::string rightImagePath = argv[2];

    cv::Mat leftImage = cv::imread(leftImagePath, 0);
    cv::Mat rightImage = cv::imread(rightImagePath, 0);

    std::cout << "height: " << leftImage.rows << " ,width: " << leftImage.cols << "\n";

    BlockMatchingParam params;
    std::unique_ptr<StereoEngine> stereoEngine(new BlockMatching(params));
    cv::Mat disparity = stereoEngine->match(leftImage, rightImage);

    return EXIT_SUCCESS;
}
