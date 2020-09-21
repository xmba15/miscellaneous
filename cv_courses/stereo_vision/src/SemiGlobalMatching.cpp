/**
 * @file    SemiGlobalMatching.cpp
 *
 * @author  btran
 *
 */

#include "SemiGlobalMatching.hpp"

#include <opencv2/opencv.hpp>

namespace _cv
{
const std::vector<int> SemiGlobalMatching::DIRECTION_ROWS = {1, 0, -1, 0, 1, 1, -1, -1, 1, 2, 2, 1, -1, -2, -2, -1};
const std::vector<int> SemiGlobalMatching::DIRECTION_COLS = {0, -1, 0, 1, 1, -1, -1, 1, 2, 1, -1, -2, -2, -1, 1, -2};
const std::vector<bool> SemiGlobalMatching::DIRECTION_SIGNS = {1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0};

SemiGlobalMatching::SemiGlobalMatching(const Param& param)
    : m_param(param)
    , m_numDisparities(param.maxDisparity - param.minDisparity + 1)
{
}

std::vector<std::uint64_t> SemiGlobalMatching::calcCensusTransform(const cv::Mat& img) const
{
    int width = img.cols;
    int height = img.rows;
    std::vector<std::uint64_t> census(height * width);

#pragma omp parallel for
    for (int row = m_param.censuswRowHalfWindow; row < height - m_param.censuswRowHalfWindow; ++row) {
        const auto rowPtrImg = img.ptr<uchar>(row);
        const auto rowPtrCensus = census.data() + row * width;

        for (int col = m_param.censusColHalfWindow; col < width - m_param.censusColHalfWindow; ++col) {
            const uchar* centerPixelPtr = rowPtrImg + col;

            std::uint64_t val = 0;
            for (int u = -m_param.censuswRowHalfWindow; u <= m_param.censuswRowHalfWindow; ++u) {
                for (int v = -m_param.censusColHalfWindow; v <= m_param.censusColHalfWindow; ++v) {
                    if (u == 0 && v == 0) {
                        continue;
                    }
                    uchar curPixel = *(centerPixelPtr + v + u * width);
                    val = (val + (curPixel >= *centerPixelPtr)) << 1;
                }
            }
            rowPtrCensus[col] = val;
        }
    }

    return census;
}

std::vector<cv::Mat> SemiGlobalMatching::calcPixelCost(int height, int width, const std::vector<std::uint64_t>& census1,
                                                       std::vector<std::uint64_t>& census2) const
{
    std::vector<cv::Mat> pixelCost(m_numDisparities);
    std::fill(pixelCost.begin(), pixelCost.end(), cv::Mat::zeros(height, width, CV_32S));

#pragma omp parallel for
    for (int i = 0; i < height; ++i) {
        const auto rowPtr1 = census1.data() + i * width;
        const auto rowPtr2 = census2.data() + i * width;

        for (int j = 0; j < width; ++j) {
            auto leftVal = rowPtr1[j];
            for (int d = 0; d < m_numDisparities; ++d) {
                auto rightVal = j - d >= 0 ? rowPtr2[j - d] : 0;
                pixelCost[d].ptr<int>(i)[j] = calcHammingDist<int>(leftVal, rightVal);
            }
        }
    }

    return pixelCost;
}

void SemiGlobalMatching::aggregateCostSingleScanLine(int row, int col, int curDepth, int dRow, int dCol,
                                                     const std::vector<cv::Mat>& pixelCost,
                                                     std::vector<cv::Mat>& pathAggCost) const
{
    int val0 = std::numeric_limits<int>::max();
    int val1 = std::numeric_limits<int>::max();
    int val2 = std::numeric_limits<int>::max();
    int val3 = std::numeric_limits<int>::max();
    int minPrevD = std::numeric_limits<int>::max();

    int curPixelCost = pixelCost[curDepth].ptr<int>(row)[col];

    int height = pixelCost[0].rows;
    int width = pixelCost[1].cols;

    if (row - dRow < 0 || height <= row - dRow || col - dCol < 0 || width <= col - dCol) {
        pathAggCost[curDepth].ptr<int>(row)[col] = curPixelCost;
        return;
    }

    for (int d = 0; d < m_numDisparities; ++d) {
        int prev = pathAggCost[d].ptr<int>(row - dRow)[col - dCol];

        if (prev < minPrevD) {
            minPrevD = prev;
        }

        if (curDepth == d) {
            val0 = prev;
        } else if (curDepth == d + 1) {
            val1 = prev + m_param.P1;
        } else if (curDepth == d - 1) {
            val2 = prev + m_param.P1;
        } else {
            int otherDepth = prev + m_param.P2;
            if (otherDepth < val3) {
                val3 = otherDepth;
            }
        }
    }

    pathAggCost[curDepth].ptr<int>(row)[col] =
        std::min(std::min(std::min(val0, val1), val2), val3) + curPixelCost - minPrevD;
}

void SemiGlobalMatching::aggregateAllCost(const std::vector<cv::Mat>& pixelCost,
                                          std::vector<std::vector<cv::Mat>>& aggCost,
                                          std::vector<cv::Mat>& sumCost) const
{
    int width = pixelCost[0].cols;
    int height = pixelCost[0].rows;

    sumCost.resize(m_numDisparities);
    std::fill(sumCost.begin(), sumCost.end(), cv::Mat::zeros(height, width, CV_32S));

    int totalPaths = static_cast<int>(m_param.pathType);

    aggCost.resize(totalPaths);

    for (int i = 0; i < totalPaths; ++i) {
        aggCost[i].resize(m_numDisparities);
        std::fill(aggCost[i].begin(), aggCost[i].end(), cv::Mat::zeros(height, width, CV_32S));
    }

    std::vector<int> posPaths, negPaths;
    for (int i = 0; i < totalPaths; ++i) {
        if (DIRECTION_SIGNS[i]) {
            posPaths.emplace_back(i);
        } else {
            negPaths.emplace_back(i);
        }
    }

#pragma omp parallel for num_threads(totalPaths)
    for (int path = 0; path < totalPaths; ++path) {
        int bRow = 0, eRow = height - 1;
        int bCol = 0, eCol = width - 1;
        int sRow = 1, sCol = 1;

        if (!DIRECTION_SIGNS[path]) {
            std::swap(bRow, eRow);
            std::swap(bCol, eCol);
            sRow = -1;
            sCol = -1;
        }

        for (int row = bRow; row != eRow; row += sRow) {
            for (int col = bCol; col != eCol; col += sCol) {
                for (int depth = 0; depth < m_numDisparities; ++depth) {
                    this->aggregateCostSingleScanLine(row, col, depth, DIRECTION_ROWS[path], DIRECTION_COLS[path],
                                                      pixelCost, aggCost[path]);
                }
            }
        }
    }

#pragma omp parallel for
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            for (int depth = 0; depth < m_numDisparities; ++depth) {
                for (int path = 0; path < totalPaths; ++path) {
                    sumCost[depth].ptr<int>(row)[col] += aggCost[path][depth].ptr<int>(row)[col];
                }
            }
        }
    }
}

cv::Mat SemiGlobalMatching::winnerTakesAll(const std::vector<cv::Mat>& sumCost) const
{
    if (sumCost.empty()) {
        throw std::runtime_error("empty sum cost");
    }

    int width = sumCost[0].cols;
    int height = sumCost[0].rows;

    cv::Mat disp(height, width, CV_8UC1);

#pragma omp parallel for
    for (int i = 0; i < height; ++i) {
        auto rowPtrDisp = disp.ptr<uchar>(i);
        for (int j = 0; j < width; ++j) {
            uchar minD = 0;
            int minCost = sumCost[minD].ptr<int>(i)[j];
            for (int d = 1; d < m_numDisparities; ++d) {
                int curCost = sumCost[d].ptr<int>(i)[j];
                if (curCost < minCost) {
                    minCost = curCost;
                    minD = d;
                }
            }
            rowPtrDisp[j] = minD;
        }
    }

    return disp;
}

cv::Mat SemiGlobalMatching::match(const cv::Mat& leftImage, const cv::Mat& rightImage) const
{
    auto leftCensus = this->calcCensusTransform(leftImage);
    auto rightCensus = this->calcCensusTransform(rightImage);

    int height = leftImage.rows;
    int width = leftImage.cols;

    std::vector<cv::Mat> pixelCost = this->calcPixelCost(height, width, leftCensus, rightCensus);

    std::vector<std::vector<cv::Mat>> aggCost;
    std::vector<cv::Mat> sumCost;
    this->aggregateAllCost(pixelCost, aggCost, sumCost);
    cv::Mat disp = this->winnerTakesAll(sumCost);

    return disp;
}
}  // namespace _cv
