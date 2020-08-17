/**
 * @file    BlockMatching.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include "StereoEngine.hpp"

enum class MatchingMetric : int { SAD = 0, SSD = 1, NCC = 2 };

struct BlockMatchingParam : public StereoEngineParam {
    int windowSize = 9;
    MatchingMetric metric = MatchingMetric::SAD;
};

class BlockMatching : public StereoEngine
{
 public:
    explicit BlockMatching(const BlockMatchingParam& param)
        : m_param(param)
    {
    }

    virtual ~BlockMatching() = default;

 public:
    cv::Mat match(const cv::Mat& leftImage, const cv::Mat& rightImage) const final;

 private:
    BlockMatchingParam m_param;
};
