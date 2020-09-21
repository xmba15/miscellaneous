/**
 * @file    DirectPoseEstimation.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#include "Utility.hpp"

namespace _cv
{
void calcDirectPoseEstimationSingleLayer(const cv::Mat& img1, const cv::Mat& img2,
                                         const std::vector<cv::Point2d>& refPoints, const std::vector<double>& depth,
                                         const _cv::CameraMatrix& K, Sophus::SE3d& T21,
                                         std::vector<cv::Point2d>& projecteds, int numIterations = 20);

void calcDirectPoseEstimationMultiLayer(const cv::Mat& img1, const cv::Mat& img2,
                                        const std::vector<cv::Point2d>& refPoints, const std::vector<double>& depth,
                                        const _cv::CameraMatrix& K, Sophus::SE3d& T21,
                                        std::vector<cv::Point2d>& projecteds, int numScale, double scaleFactor,
                                        int numIterations = 20);
}  // namespace _cv
