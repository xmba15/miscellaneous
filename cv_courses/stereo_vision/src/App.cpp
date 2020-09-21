/**
 * @file    App.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include <opencv2/opencv.hpp>

#include "BlockMatching.hpp"
#include "SemiGlobalMatching.hpp"
#include "Utility.hpp"

namespace
{
cv::TickMeter meter;
}  // namespace

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

    cv::GaussianBlur(leftImage, leftImage, cv::Size(5, 5), 0.);
    cv::GaussianBlur(rightImage, rightImage, cv::Size(5, 5), 0.);

    // -------------------------------------------------------------------------
    // block matching
    // -------------------------------------------------------------------------
    _cv::BlockMatchingParam paramsBM;
    paramsBM.metric = _cv::MatchingMetric::SAD;
    paramsBM.halfWindowSize = 3;
    paramsBM.maxDisparity = 64;
    std::unique_ptr<_cv::StereoEngine> stereoEngineBM(new _cv::BlockMatching(paramsBM));

    /* meter.reset(); */
    /* meter.start(); */
    /* cv::Mat disparityBM = stereoEngineBM->match(leftImage, rightImage); */
    /* meter.stop(); */
    /* std::cout << "block matching: " << meter.getTimeMilli() << "[ms]" << std::endl; */

    /* cv::Mat visualizedBM = */
    /*     stereoEngineBM->visualizeDisparity(disparityBM, paramsBM.maxDisparity - paramsBM.minDisparity); */
    /* cv::imshow("disparityBM", visualizedBM); */
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // semi global matching
    // -------------------------------------------------------------------------
    _cv::SemiGlobalMatching::Param paramsSGBM;
    paramsSGBM.censusColHalfWindow = 3;
    paramsSGBM.censuswRowHalfWindow = 4;
    paramsSGBM.maxDisparity = 64;
    std::unique_ptr<_cv::StereoEngine> stereoEngineSGBM(new _cv::SemiGlobalMatching(paramsSGBM));
    meter.reset();
    meter.start();
    cv::Mat disparitySGBM = stereoEngineSGBM->match(leftImage, rightImage);
    meter.stop();

    std::cout << "semi global matching: " << meter.getTimeMilli() << "[ms]" << std::endl;

    cv::Mat visualizedSGBM =
        stereoEngineSGBM->visualizeDisparity(disparitySGBM, paramsBM.maxDisparity - paramsBM.minDisparity);
    cv::imshow("disparitySGBM", visualizedSGBM);
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------

    cv::imshow("leftImage", leftImage);
    cv::waitKey(0);

    return EXIT_SUCCESS;
}
