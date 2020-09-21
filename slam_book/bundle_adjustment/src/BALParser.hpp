/**
 * @file    BALParser.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

namespace _cv
{
struct BALParser {
    explicit BALParser(const std::string& dataPath);

    // vector of size numObservations
    std::vector<int> camIndices;
    std::vector<int> pointIndices;
    std::vector<cv::Vec2d> observations;

    // vectors whose size equals numCams
    std::vector<cv::Affine3d> camPoses;
    std::vector<std::vector<float>> camIntrinsics;

    // vector of size numPoints
    std::vector<cv::Vec3d> point3Ds;

    int numCams;
    int numPoints;
    int numObservations;
};
}  // namespace _cv
