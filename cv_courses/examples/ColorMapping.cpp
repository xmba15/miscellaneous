/**
 * @file    ColorMapping.cpp
 *
 * @author  btran
 *
 * @date    2019-09-05
 *
 * Copyright (c) organization
 *
 */

#include <string>

#include <opencv2/opencv.hpp>

#include "Utility.hpp"

namespace npy
{
std::vector<int> linspace(const int min, const int max, const int num);
}  // namespace npy

int main(int argc, char *argv[])
{
#ifdef DATA_PATH
    const std::string IMAGE_PATH = std::string(DATA_PATH) + "/teddy/";
    const std::string FIRST = IMAGE_PATH + "im2.png";
    const std::string SECOND = IMAGE_PATH + "style_image_samesize.jpg";

    const cv::Mat first = cv::imread(FIRST);
    const cv::Mat second = cv::imread(SECOND);

    assert(first.cols == second.cols);
    assert(first.rows == second.rows);
    assert(first.channels() == 3);

    const size_t WIDTH = second.cols;
    const size_t HEIGHT = first.rows;

    const size_t numSampledXs = (WIDTH - 1) / 3;
    const size_t numSampledYs = (HEIGHT - 1) / 3;

    auto sampledXIndices = npy::linspace(0, WIDTH - 1, numSampledXs);
    auto sampledYIndices = npy::linspace(0, HEIGHT - 1, numSampledYs);
    const size_t numSamples = numSampledXs * numSampledYs;

    // M * Q = N
    cv::Mat M(numSamples, 3, CV_32FC1, cv::Scalar(0.0));
    cv::Mat N(numSamples, 3, CV_32FC1, cv::Scalar(0.0));

    int matrixRowIdx = 0;

    for (int ir : sampledYIndices) {
        // first.ptr<cv::Vec3b>(row)[col];
        const cv::Vec3b *firstRowPtr = first.ptr<cv::Vec3b>(ir);
        const cv::Vec3b *secondRowPtr = second.ptr<cv::Vec3b>(ir);

        for (int ic : sampledXIndices) {
            float *rowM = M.ptr<float>(matrixRowIdx);
            float *rowN = N.ptr<float>(matrixRowIdx);

            for (size_t k = 0; k < 3; ++k) {
                rowM[k] = firstRowPtr[ic][k];
                rowN[k] = secondRowPtr[ic][k];
            }

            ++matrixRowIdx;
        }
    }

    cv::Mat Q;
    Q.create(3, 3, CV_32FC1);
    cv::Mat Mt(3, numSamples, CV_32FC1);
    cv::transpose(M, Mt);

    Q = (Mt * M).inv() * Mt * N;

    cv::Mat result(first.rows, first.cols, CV_8UC3);

    for (size_t ir = 0; ir < first.rows; ++ir) {
        const cv::Vec3b *firstRowPtr = first.ptr<cv::Vec3b>(ir);
        cv::Vec3b *resultRowPtr = result.ptr<cv::Vec3b>(ir);

        for (size_t ic = 0; ic < first.cols; ++ic) {
            cv::Mat firstRowPtrM = (cv::Mat_<float>(1, 3) << firstRowPtr[ic][0],
                                    firstRowPtr[ic][1], firstRowPtr[ic][2]);

            cv::Mat multiplication = firstRowPtrM * Q;

            resultRowPtr[ic][0] = multiplication.at<float>(0, 0);
            resultRowPtr[ic][1] = multiplication.at<float>(0, 1);
            resultRowPtr[ic][2] = multiplication.at<float>(0, 2);
        }
    }

    cv::imshow("result", result);
    cv::waitKey(0);
    cv::destroyAllWindows();
#endif  // DATA_PATH

    return 0;
}

namespace npy
{
std::vector<int> linspace(const int min, const int max, const int num)
{
    assert(min < max);
    assert(num >= 2);

    std::vector<int> result;
    result.reserve(num);
    result.emplace_back(min);

    int delta = (max - min) / (num - 1);
    int curV = min;
    for (size_t i = 0; i < num - 2; ++i) {
        curV += delta;
        result.emplace_back(curV);
    }

    result.emplace_back(max);
    return result;
}

}  // namespace npy
