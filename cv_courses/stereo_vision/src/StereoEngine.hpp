/**
 * @file    StereoEngine.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>

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

 public:
    virtual cv::Mat match(const cv::Mat& leftImage, const cv::Mat& rightImage) const = 0;
};
