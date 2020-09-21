/**
 * @file    SemiGlobalMatching.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include "StereoEngine.hpp"

namespace _cv
{
class SemiGlobalMatching : public StereoEngine
{
 public:
    enum class PathType : int { SCAN_4PATH = 4, SCAN_8PATH = 8, SCAN_16PATH = 16 };

    struct Param : public StereoEngineParam {
        PathType pathType = PathType::SCAN_4PATH;
        int censusColHalfWindow = 1;
        int censuswRowHalfWindow = 1;
        int P1 = 7;
        int P2 = 86;
        float uniquenessRatio = 0.95;
        int max12Diff = 5;
    };

    explicit SemiGlobalMatching(const Param& param);

 public:
    cv::Mat match(const cv::Mat& leftImage, const cv::Mat& rightImage) const final;

    template <typename T> T static calcHammingDist(T val1, T val2)
    {
        T dist = 0;
        T d = val1 ^ val2;

        while (d) {
            d = d & (d - 1);
            dist++;
        }
        return dist;
    }

 private:
    std::vector<std::uint64_t> calcCensusTransform(const cv::Mat& img) const;

    std::vector<cv::Mat> calcPixelCost(int height, int width, const std::vector<std::uint64_t>& census1,
                                       std::vector<std::uint64_t>& census2) const;

    void aggregateCostSingleScanLine(int row, int col, int curDepth, int dRow, int dCol,
                                     const std::vector<cv::Mat>& pixelCost, std::vector<cv::Mat>& aggCost) const;

    void aggregateAllCost(const std::vector<cv::Mat>& pixelCost, std::vector<std::vector<cv::Mat>>& aggCost,
                          std::vector<cv::Mat>& sumCost) const;

    cv::Mat winnerTakesAll(const std::vector<cv::Mat>& sumCost) const;

 private:
    static constexpr int MAX_DIRECTIONS = 16;
    static const std::vector<int> DIRECTION_ROWS;
    static const std::vector<int> DIRECTION_COLS;
    static const std::vector<bool> DIRECTION_SIGNS;

    int m_numDisparities;
    Param m_param;
};
}  // namespace _cv
