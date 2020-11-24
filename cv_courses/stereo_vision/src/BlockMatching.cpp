/**
 * @file    BlockMatching.cpp
 *
 * @author  btran
 *
 */

#include "BlockMatching.hpp"

BlockMatching::BlockMatching(const BlockMatchingParam& param)
    : m_param(param)
{
    switch (m_param.metric) {
        case MatchingMetric::SAD: {
            m_metricFunc = [](const cv::Mat& src, const cv::Mat& target) { return StereoEngine::calSAD(src, target); };
            break;
        }
        default:
            throw std::runtime_error("not supported metric");
    }
}

cv::Mat BlockMatching::match(const cv::Mat& leftImage, const cv::Mat& rightImage) const
{
    if (leftImage.channels() != 1 || rightImage.channels() != 1) {
        throw std::runtime_error("works with single-channel images");
    }

    int height = leftImage.rows;
    int width = leftImage.cols;

    cv::Mat disparity = cv::Mat::zeros(height, width, CV_8UC1);

    for (int i = 0; i < height; ++i) {
        int rowMin = std::max(0, i - m_param.halfWindowSize);
        int rowMax = std::min(height - 1, i + m_param.halfWindowSize);

        for (int j = 0; j < width; ++j) {
            int colMin = std::max(0, j - m_param.halfWindowSize);
            int colMax = std::min(width - 1, j + m_param.halfWindowSize);

            cv::Mat src = leftImage(cv::Rect(colMin, rowMin, colMax - colMin, rowMax - rowMin));

            int rangeMin = std::max(m_param.minDisparity, -colMin);
            int rangeMax = std::min(m_param.maxDisparity, width - 1 - colMax);

            double minCost = std::numeric_limits<double>::max();
            int curDisp = 0;

            for (int k = rangeMin; k <= rangeMax; ++k) {
                cv::Mat target = rightImage(cv::Rect(colMin + k, rowMin, colMax - colMin, rowMax - rowMin));
                double cost = m_metricFunc(src, target);
                if (cost < minCost) {
                    minCost = cost;
                    curDisp = rangeMin;
                }
            }
            disparity.ptr<uchar>(i)[j] = curDisp;
        }
    }

    return disparity;
}
