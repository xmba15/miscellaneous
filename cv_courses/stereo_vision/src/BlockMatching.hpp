/**
 * @file    BlockMatching.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <functional>

#include "StereoEngine.hpp"

namespace _cv
{
enum class MatchingMetric : int { SAD = 0, SSD = 1, NCC = 2, MAX = NCC };

struct BlockMatchingParam : public StereoEngineParam {
    int halfWindowSize = 5;
    MatchingMetric metric = MatchingMetric::SAD;
};

class BlockMatching : public StereoEngine
{
 public:
    explicit BlockMatching(const BlockMatchingParam& param);

    virtual ~BlockMatching() = default;

 public:
    cv::Mat match(const cv::Mat& leftImage, const cv::Mat& rightImage) const final;

 private:
    BlockMatchingParam m_param;
    std::function<double(const cv::Mat& src, const cv::Mat& target)> m_metricFunc;
};
}  // namespace _cv
