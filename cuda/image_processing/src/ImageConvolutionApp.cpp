/**
 * @file    ImageConvolutionApp.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include <opencv2/opencv.hpp>

#include "ImageConvolution.hpp"
#include "Timer.hpp"

namespace
{
template <typename DataType> DataType calcMSE(const DataType* data1, const DataType* data2, int width, int height)
{
    DataType mse = 0;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            DataType diff = data1[i * width + j] - data2[i * width + j];
            mse += diff * diff;
        }
    }
    return std::sqrt(mse);
}

basic::utils::Timer timer;
double processingTime = 0;
int NUM_TEST = 1;
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "[app] [path/to/image]\n";
        return EXIT_FAILURE;
    }
    const std::string PATH_TO_IMAGE = argv[1];
    cv::Mat image = cv::imread(PATH_TO_IMAGE, 0);

    if (image.empty()) {
        std::cerr << "failed to read image: " + PATH_TO_IMAGE << "\n";
        return EXIT_FAILURE;
    }

    int width = image.cols;
    int height = image.rows;

    int kernelHalfSize = 3;
    int filterSize = 2 * kernelHalfSize + 1;

    // clang-format off
    cv::Mat kernel = (cv::Mat_<float>(filterSize, filterSize) <<
                      1, 0, -1, 0, 1, 0, -1,
                      0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0.25, 0, 0.25, 0, 0,
                      0, 1, 0, 0, 0, -1, 0,
                      0, 0, 0.25, 0, 0.25, 0, 0,
                      0, 0, 0, 0, 0, 0, 0,
                      -1, 0, 1, 0, -1, 0, 1
                      );
    // clang-format on

    std::unique_ptr<float[]> convolvedDataCPU;
    timer.start();
    for (int i = 0; i < NUM_TEST; ++i) {
        convolvedDataCPU = _cv::convolve<uchar, float>(image.data, kernel.ptr<float>(), width, height, kernelHalfSize);
    }
    processingTime = timer.getMs();
    std::cout << "convolution (cpu): " << processingTime / NUM_TEST << "[ms]\n";

    std::unique_ptr<float[]> convolvedDataGPU1;
    timer.start();
    for (int i = 0; i < NUM_TEST; ++i) {
        convolvedDataGPU1 =
            _cv::convolveGPU1<uchar, float>(image.data, kernel.ptr<float>(), width, height, kernelHalfSize);
    }
    processingTime = timer.getMs();
    std::cout << "convolution (gpu 1): " << processingTime / NUM_TEST << "[ms]\n";

    std::unique_ptr<float[]> convolvedDataGPU2;
    timer.start();
    for (int i = 0; i < NUM_TEST; ++i) {
        convolvedDataGPU2 =
            _cv::convolveGPU2<uchar, float>(image.data, kernel.ptr<float>(), width, height, kernelHalfSize);
    }
    processingTime = timer.getMs();
    std::cout << "convolution (gpu 2): " << processingTime / NUM_TEST << "[ms]\n";

    std::unique_ptr<float[]> convolvedDataGPU3;
    timer.start();
    for (int i = 0; i < NUM_TEST; ++i) {
        convolvedDataGPU3 =
            _cv::convolveGPU3<uchar, float>(image.data, kernel.ptr<float>(), width, height, kernelHalfSize);
    }
    processingTime = timer.getMs();
    std::cout << "convolution (gpu 3): " << processingTime / NUM_TEST << "[ms]\n";

    std::cout << "mse of the convolved images from cpu and gpu1 processing: "
              << calcMSE(convolvedDataCPU.get(), convolvedDataGPU1.get(), width, height) << "\n";

    std::cout << "mse of the convolved images from cpu and gpu2 processing: "
              << calcMSE(convolvedDataCPU.get(), convolvedDataGPU2.get(), width, height) << "\n";

    std::cout << "mse of the convolved images from cpu and gpu3 processing: "
              << calcMSE(convolvedDataCPU.get(), convolvedDataGPU3.get(), width, height) << "\n";

    cv::Mat outImage(height, width, CV_32FC1, convolvedDataGPU3.get());
    outImage.convertTo(outImage, CV_8UC1);
    cv::imwrite("convolved.jpg", outImage);

    return 0;
}
