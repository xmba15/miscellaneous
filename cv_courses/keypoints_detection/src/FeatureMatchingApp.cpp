/**
 * @file    FeatureMatchingApp.cpp
 *
 * @author  btran
 *
 */

#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
    if (argc != 3) {
        std::cout << "[app] [path/to/left/image] [path/to/right/image]" << std::endl;
        return EXIT_FAILURE;
    }

    std::string imagePaths[2] = {argv[1], argv[2]};
    cv::Mat images[2];

    for (int i = 0; i < 2; ++i) {
        images[i] = cv::imread(imagePaths[i]);
        if (images[i].empty()) {
            std::cerr << "failed to load: " + imagePaths[i] << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::vector<cv::KeyPoint> keyPoints[2];
    cv::Mat descriptors[2];

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    for (int i = 0; i < 2; ++i) {
        detector->detect(images[i], keyPoints[i]);
        descriptor->compute(images[i], keyPoints[i], descriptors[i]);
    }

    std::vector<cv::DMatch> matches, goodMatches;
    matcher->match(descriptors[0], descriptors[1], matches);
    auto minMax = std::minmax_element(matches.begin(), matches.end(),
                                      [](const auto& m1, const auto& m2) { return m1.distance < m2.distance; });

    for (int i = 0; i < descriptors[0].rows; ++i) {
        if (matches[i].distance <= std::max<double>(2 * minMax.first->distance, 30.0)) {
            goodMatches.emplace_back(matches[i]);
        }
    }

    cv::Mat outputs[2];
    cv::drawMatches(images[0], keyPoints[0], images[1], keyPoints[1], matches, outputs[0]);
    cv::drawMatches(images[0], keyPoints[0], images[1], keyPoints[1], goodMatches, outputs[1]);
    cv::imshow("all matches", outputs[0]);
    cv::imshow("good matches", outputs[1]);
    cv::waitKey(0);

    return EXIT_SUCCESS;
}
