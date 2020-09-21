/**
 * @file    StereoEngine.cpp
 *
 * @author  btran
 *
 * Copyright (c) organization
 *
 */

#include "StereoEngine.hpp"

namespace _cv
{
namespace
{
void validateImagePatches(const cv::Mat& src, const cv::Mat& target)
{
    if (src.size() != target.size()) {
        throw std::runtime_error("src and target must be of the same size");
    }
    if (src.channels() != target.channels()) {
        throw std::runtime_error("src and target must be of the same channel");
    }
}
}  // namespace

double StereoEngine::calSAD(const cv::Mat& src, const cv::Mat& target)
{
    validateImagePatches(src, target);
    int height = src.rows;
    int width = src.cols;
    int channel = src.channels();

    int result = 0;
    for (int i = 0; i < height; ++i) {
        int step = i * width;
        for (int j = 0; j < width; ++j) {
            int elm = i * src.elemSize();
            for (int c = 0; c < channel; c++) {
                result += std::abs(src.data[step + elm + c] - target.data[step + elm + c]);
            }
        }
    }

    return result;
}

double StereoEngine::calSSD(const cv::Mat& src, const cv::Mat& target)
{
    validateImagePatches(src, target);
    int height = src.rows;
    int width = src.cols;
    int channel = src.channels();

    int result = 0;
    for (int i = 0; i < height; ++i) {
        int step = i * width;
        for (int j = 0; j < width; ++j) {
            int elm = i * src.elemSize();
            for (int c = 0; c < channel; c++) {
                result += std::pow(src.data[step + elm + c] - target.data[step + elm + c], 2);
            }
        }
    }

    return result;
}
}  // namespace _cv
