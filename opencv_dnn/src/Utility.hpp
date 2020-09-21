/**
 * @file    Utility.hpp
 *
 * @author  btran
 *
 * Copyright (c) organization
 *
 */

#pragma once

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace util
{
inline std::vector<std::array<int, 3>> generateColorCharts(const uint16_t numClasses = 1000, const uint16_t seed = 255)
{
    std::srand(seed);
    std::vector<std::array<int, 3>> colors;
    colors.reserve(numClasses);
    for (uint16_t i = 0; i < numClasses; ++i) {
        colors.emplace_back(std::array<int, 3>{std::rand() % 255, std::rand() % 255, std::rand() % 255});
    }

    return colors;
}

inline std::vector<cv::Scalar> toCvScalarColors(const std::vector<std::array<int, 3>>& colors)
{
    std::vector<cv::Scalar> result;
    result.reserve(colors.size());
    std::transform(std::begin(colors), std::end(colors), std::back_inserter(result),
                   [](const auto& elem) { return cv::Scalar(elem[0], elem[1], elem[2]); });

    return result;
}

inline std::vector<std::string> split(const std::string& s, const char delimiter)
{
    std::stringstream ss(s);
    std::string token;
    std::vector<std::string> tokens;
    while (std::getline(ss, token, delimiter)) {
        tokens.emplace_back(token);
    }
    return tokens;
}
}  // namespace util
