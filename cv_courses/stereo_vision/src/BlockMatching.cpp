/**
 * @file    BlockMatching.cpp
 *
 * @author  btran
 *
 */

#include "BlockMatching.hpp"

namespace _cv
{
BlockMatching::BlockMatching(const BlockMatchingParam& param) : m_param(param)
{
    switch (m_param.metric) {
        case MatchingMetric::SAD: {
            m_metricFunc = [](const cv::Mat& src, const cv::Mat& target) { return StereoEngine::calSAD(src, target); };
            break;
        }
        case MatchingMetric::SSD: {
            m_metricFunc = [](const cv::Mat& src, const cv::Mat& target) { return StereoEngine::calSSD(src, target); };
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

    int blockSize = 2 * m_param.halfWindowSize + 1;

    for (int i = m_param.halfWindowSize; i < height - m_param.halfWindowSize; ++i) {
        for (int j = m_param.halfWindowSize; j < width - m_param.halfWindowSize; ++j) {
            int minSSD = std::numeric_limits<int>::max();
            cv::Mat leftBlock =
                leftImage(cv::Rect(j - m_param.halfWindowSize, i - m_param.halfWindowSize, blockSize, blockSize));

            for (int range = m_param.minDisparity; range <= m_param.maxDisparity; ++range) {
                int rightBlockIdx = j - range - m_param.halfWindowSize;
                if (rightBlockIdx < 0) {
                    break;
                }

                cv::Mat rightBlock = rightImage(
                    cv::Rect(j - range - m_param.halfWindowSize, i - m_param.halfWindowSize, blockSize, blockSize));

                int SSD = m_metricFunc(leftBlock, rightBlock);

                if (SSD < minSSD) {
                    disparity.at<uchar>(i, j) = range;
                    minSSD = SSD;
                }
            }
        }
    }

    return disparity;
}
}  // namespace _cv
