/**
 * @file    Main.cpp
 *
 * @author  btran
 *
 */

#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cerr << "Usage: [app] [path/to/image/1] [path/to/image/2]" << std::endl;
        return EXIT_FAILURE;
    }
    std::vector<std::string> imagePaths = {argv[1], argv[2]};
    std::vector<cv::Mat> images, grays;
    std::transform(imagePaths.begin(), imagePaths.end(), std::back_inserter(images),
                   [](const std::string& imagePath) { return cv::imread(imagePath); });
    std::vector<std::vector<cv::KeyPoint>> keyPointsList(2);
    std::vector<cv::Mat> descriptorsList(2);
    cv::Ptr<cv::Feature2D> keyPointDetector = cv::BRISK::create();
    keyPointDetector->detect(images[0], keyPointsList[0]);
    std::cout << keyPointsList[0].size() << "\n";

    return EXIT_SUCCESS;
}
