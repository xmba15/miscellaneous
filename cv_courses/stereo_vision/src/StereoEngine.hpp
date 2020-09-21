/**
 * @file    StereoEngine.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>

namespace _cv
{
struct StereoEngineParam {
    int maxDisparity = 50;
    int minDisparity = 0;
};

class StereoEngine
{
 protected:
    StereoEngine() = default;

 public:
    virtual ~StereoEngine() = default;

    static double calSAD(const cv::Mat& src, const cv::Mat& target);
    static double calSSD(const cv::Mat& src, const cv::Mat& target);

    cv::Mat visualizeDisparity(const cv::Mat& disparity, int disparitySize)
    {
        cv::Mat visualized;
        disparity.convertTo(visualized, CV_8U, 255. / disparitySize);
        cv::applyColorMap(visualized, visualized, cv::COLORMAP_JET);

        return visualized;
    }

 public:
    virtual cv::Mat match(const cv::Mat& leftImage, const cv::Mat& rightImage) const = 0;
};
}  // namespace _cv
