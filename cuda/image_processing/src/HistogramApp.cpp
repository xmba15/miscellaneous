/**
 * @file    HistogramApp.cpp
 *
 * @author  btran
 *
 */

#include <iostream>

#include <opencv2/opencv.hpp>

#include "Histogram.hpp"

namespace
{
__global__ void warmUpGPUKernel()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    ++idx;
}

cudaError_t warmUpGPU()
{
    warmUpGPUKernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return cudaGetLastError();
}

constexpr int NUM_BINS = 256;
cv::TickMeter meter;
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "[app] [path/to/image]\n";
        return EXIT_FAILURE;
    }
    const std::string PATH_TO_IMAGE = argv[1];
    cv::Mat image = cv::imread(PATH_TO_IMAGE, 1);
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    int height = image.rows;
    int width = image.cols;

    std::vector<int> histCPU(NUM_BINS);

    meter.reset();
    meter.start();
    _cv::histogramCPU(gray.data, histCPU.data(), height * width, NUM_BINS);
    meter.stop();
    std::cout << "histogram CPU: " << meter.getTimeMilli() << "[ms]" << std::endl;

    warmUpGPU();

    meter.reset();
    meter.start();
    std::vector<int> histGPU(NUM_BINS);
    _cv::histogramGPU(gray.data, histGPU.data(), height * width, NUM_BINS);
    meter.stop();
    std::cout << "histogram GPU: " << meter.getTimeMilli() << "[ms]" << std::endl;

    long err = 0;
    for (int i = 0; i < NUM_BINS; ++i) {
        err += std::abs(histCPU[i] - histGPU[i]);
    }
    std::cout << "error: " << err << "\n";

    return 0;
}
