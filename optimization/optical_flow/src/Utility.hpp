/**
 * @file    Utility.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>

namespace _cv
{
struct CameraMatrix {
    double fx;
    double fy;
    double cx;
    double cy;
    double baseLine;

    CameraMatrix scale(double scaleFactor) const
    {
        if (scaleFactor < 0) {
            throw std::runtime_error("scale factor must be > 0");
        }

        CameraMatrix result = *this;
        result.fx *= scaleFactor;
        result.fy *= scaleFactor;
        result.cx *= scaleFactor;
        result.cy *= scaleFactor;

        return result;
    }
};

inline std::vector<cv::Mat> createImagePyramid(const cv::Mat& img, int numScale = 4, double scaleFactor = 0.5)
{
    std::vector<cv::Mat> pyramid;
    pyramid.reserve(numScale);

    cv::Mat curScaleImg = img.clone();
    for (int i = 0; i < numScale; ++i) {
        if (i == 0) {
            pyramid.emplace_back(curScaleImg);
        } else {
            cv::GaussianBlur(pyramid.back(), curScaleImg, cv::Size(5, 5), 0);
            cv::resize(curScaleImg, curScaleImg,
                       cv::Size(curScaleImg.cols * scaleFactor, curScaleImg.rows * scaleFactor));
            pyramid.emplace_back(curScaleImg);
        }
    }
    return pyramid;
}

inline float getPixelValue(const cv::Mat& img, float x, float y)
{
    // boundary check
    if (x < 0)
        x = 0;
    if (y < 0)
        y = 0;
    if (x >= img.cols - 1)
        x = img.cols - 2;
    if (y >= img.rows - 1)
        y = img.rows - 2;

    float xx = x - floor(x);
    float yy = y - floor(y);
    int xCeil = std::min(img.cols - 1, int(x) + 1);
    int yCeil = std::min(img.rows - 1, int(y) + 1);

    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x) + xx * (1 - yy) * img.at<uchar>(y, xCeil) +
           (1 - xx) * yy * img.at<uchar>(yCeil, x) + xx * yy * img.at<uchar>(yCeil, xCeil);
}

inline float imageDerivativeX(const cv::Mat& img, float x, float y)
{
    return 0.5 * (getPixelValue(img, x + 1, y) - getPixelValue(img, x - 1, y));
}

inline float imageDerivativeY(const cv::Mat& img, float x, float y)
{
    return 0.5 * (getPixelValue(img, x, y + 1) - getPixelValue(img, x, y - 1));
}
}  // namespace _cv
