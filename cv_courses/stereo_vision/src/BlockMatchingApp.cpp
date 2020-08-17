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
    if (argc != 2) {
        std::cerr << "Usage: [app] [path/to/kitti/stereo/2015/training/dir]\n";
        return EXIT_FAILURE;
    }

    const std::string dataBasePath = argv[1];
    const std::string leftImagePath = dataBasePath + "/" + "image_2";
    const std::string rightImagePath = dataBasePath + "/" + "image_3";
    const auto leftImages = parseDirectory(leftImagePath);
    const auto rightImages = parseDirectory(rightImagePath);
    if (leftImages.size() != rightImages.size()) {
        throw std::runtime_error("two set must have same size\n");
    }

    for (std::size_t i = 0; i < leftImages.size(); ++i) {
        std::cout << leftImages[i] << std::endl;

        cv::Mat leftImage = cv::imread(leftImages[i]);
        cv::Mat rightImage = cv::imread(rightImages[i]);
        cv::Mat concatImage;
        cv::vconcat(leftImage, rightImage, concatImage);
        cv::imshow("Image", concatImage);
        if (static_cast<char>(cv::waitKey(200)) == 27) {
            break;
        }
    }

    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}
