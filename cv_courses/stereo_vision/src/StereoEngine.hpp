/**
 * @file    StereoEngine.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>

struct StereoEngineParam {
    int maxDisparity;
};

class StereoEngine
{
 public:
    virtual ~StereoEngine() = default;

 public:
    virtual cv::Mat match(const cv::Mat& leftImage, const cv::Mat& rightImage) const = 0;
};
