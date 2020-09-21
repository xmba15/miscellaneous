/**
 * @file    OpticalFlow.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <opencv2/opencv.hpp>

#include "Utility.hpp"

namespace _cv
{
class OpticalFlowTracker : public cv::ParallelLoopBody
{
 public:
    OpticalFlowTracker(const cv::Mat& img1, const cv::Mat& img2, const std::vector<cv::KeyPoint>& kp1,
                       std::vector<cv::KeyPoint>& kp2, std::vector<bool>& success, bool inverse, bool hasInitialGuess,
                       int halfWindowSize = 4, float epsilon = 1e-3, int numIterations = 20)

        : m_img1(img1)
        , m_img2(img2)
        , m_kp1(kp1)
        , m_kp2(kp2)
        , m_success(success)
        , m_inverse(inverse)
        , m_hasInitialGuess(hasInitialGuess)
        , m_halfWindowSize(halfWindowSize)
        , m_epsilon(epsilon)
        , m_numIterations(numIterations)
    {
    }

    void operator()(const cv::Range& range) const;

 private:
    const cv::Mat& m_img1;
    const cv::Mat& m_img2;
    const std::vector<cv::KeyPoint>& m_kp1;
    std::vector<cv::KeyPoint>& m_kp2;
    std::vector<bool>& m_success;
    bool m_inverse;
    bool m_hasInitialGuess;

    int m_halfWindowSize;
    float m_epsilon;
    int m_numIterations;
};

inline void calcOpticalFlowSingleLevel(const cv::Mat& img1, const cv::Mat& img2, const std::vector<cv::KeyPoint>& kp1,
                                       std::vector<cv::KeyPoint>& kp2, std::vector<bool>& success, bool inverse = false,
                                       bool hasInitialGuess = false)
{
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, hasInitialGuess);
    cv::parallel_for_(cv::Range(0, kp1.size()), tracker);
}

inline void calcOpticalFlowMultiLevel(const cv::Mat& img1, const cv::Mat& img2, const std::vector<cv::KeyPoint>& kp1,
                                      std::vector<cv::KeyPoint>& kp2, std::vector<bool>& success, bool inverse = false,
                                      int numScale = 4, double scaleFactor = 0.5)
{
    std::vector<double> scales = {1.};
    for (int i = 1; i < numScale; ++i) {
        scales.emplace_back(scales.back() * scaleFactor);
    }

    auto pyr1 = createImagePyramid(img1, numScale, scaleFactor);
    auto pyr2 = createImagePyramid(img2, numScale, scaleFactor);

    std::vector<cv::KeyPoint> kp1Pyr;
    std::transform(kp1.begin(), kp1.end(), std::back_inserter(kp1Pyr), [&scales](const auto& elem) {
        auto newKp = elem;
        newKp.pt *= scales.back();
        return std::move(newKp);
    });

    for (int level = numScale - 1; level >= 0; --level) {
        calcOpticalFlowSingleLevel(pyr1[level], pyr2[level], level == 0 ? kp1 : kp1Pyr, kp2, success, inverse,
                                   level != numScale - 1);

        if (level > 0) {
            std::for_each(kp1Pyr.begin(), kp1Pyr.end(), [scaleFactor](auto& elem) { elem.pt /= scaleFactor; });
            std::for_each(kp2.begin(), kp2.end(), [scaleFactor](auto& elem) { elem.pt /= scaleFactor; });
        }
    }
}
}  // namespace _cv
