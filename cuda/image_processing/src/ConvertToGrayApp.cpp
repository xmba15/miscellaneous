/**
 * @file    ConvertToGrayApp.cpp
 *
 * @author  btran
 *
 */

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include <cuda.h>

#include "Timer.hpp"

namespace
{
void convertToGray1(const cv::Mat& src, cv::Mat& dst);
void convertToGray2(const cv::Mat& src, cv::Mat& dst);

basic::utils::Timer timer;
double processingTime = 0;
int NUM_TEST = 10;
}  // namespace

int main(int argc, char* argv[])
{
    if (argc != 2) {
        std::cerr << "[app] [path/to/image]\n";
        return EXIT_FAILURE;
    }
    const std::string PATH_TO_IMAGE = argv[1];
    cv::Mat image = cv::imread(PATH_TO_IMAGE, 1);

    if (image.empty()) {
        std::cerr << "failed to read image: " + PATH_TO_IMAGE << "\n";
        return EXIT_FAILURE;
    }

    cv::Mat grayCpu;
    timer.start();
    for (int i = 0; i < NUM_TEST; ++i) {
        cv::cvtColor(image, grayCpu, cv::COLOR_BGR2GRAY);
    }
    processingTime = timer.getMs();
    std::cout << "to gray (cpu): " << processingTime / NUM_TEST << "[ms]\n";

    cv::Mat grayGpu;

    timer.start();
    for (int i = 0; i < NUM_TEST; ++i) {
        convertToGray1(image, grayGpu);
    }
    processingTime = timer.getMs();
    std::cout << "to gray (gpu 1): " << processingTime / NUM_TEST << "[ms]\n";

    timer.start();
    for (int i = 0; i < NUM_TEST; ++i) {
        convertToGray2(image, grayGpu);
    }
    processingTime = timer.getMs();
    std::cout << "to gray (gpu 2): " << processingTime / NUM_TEST << "[ms]\n";

    cv::Mat result;
    cv::hconcat(grayCpu, grayGpu, result);
    cv::imshow("result", result);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return EXIT_SUCCESS;
}

namespace
{
namespace cuda
{
__global__ void convertToGray1(const uchar3* rgb, uchar* gray)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    gray[idx] = static_cast<uchar>(0.299f * rgb[idx].z + 0.587f * rgb[idx].y + 0.114f * rgb[idx].x);
}
}  // namespace cuda

void convertToGray1(const cv::Mat& src, cv::Mat& dst)
{
    int width = src.cols;
    int height = src.rows;
    uchar3* deviceRGB;

    cudaMalloc(&deviceRGB, sizeof(uchar3) * width * height);
    cudaMemcpy(deviceRGB, src.data, sizeof(uchar3) * width * height, cudaMemcpyHostToDevice);

    uchar* deviceGray;
    cudaMalloc(&deviceGray, sizeof(uchar) * width * height);
    cuda::convertToGray1<<<width * height, 1>>>(deviceRGB, deviceGray);

    dst = cv::Mat(height, width, CV_8UC1);
    cudaMemcpy(dst.data, deviceGray, sizeof(uchar) * width * height, cudaMemcpyDeviceToHost);

    cudaFree(deviceRGB);
    cudaFree(deviceGray);
}

namespace cuda
{
__global__ void convertToGray2(const uchar3* rgb, std::size_t rgbPitch, uchar* gray, std::size_t grayPitch, int width,
                               int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int rgbPitchInPixel = rgbPitch / sizeof(uchar3);

    if (x < width && y < height) {
        int grayPos = y * grayPitch + x;
        int rgbPos = y * rgbPitchInPixel + x;
        gray[grayPos] = static_cast<uchar>(0.299f * rgb[rgbPos].z + 0.587f * rgb[rgbPos].y + 0.114f * rgb[rgbPos].x);
    }
}
}  // namespace cuda

int divRoundUp(int value, int radix)
{
    return (value + radix) / radix;
}

void convertToGray2(const cv::Mat& src, cv::Mat& dst)
{
    int width = src.cols;
    int height = src.rows;

    uchar3* deviceRGB;
    std::size_t deviceRGBPitch;
    uchar* deviceGray;
    std::size_t deviceGrayPitch;

    cudaMallocPitch(&deviceRGB, &deviceRGBPitch, width * sizeof(uchar3), height);
    cudaMallocPitch(&deviceGray, &deviceGrayPitch, width, height);
    cudaMemcpy2D(deviceRGB, deviceRGBPitch, src.data, src.step, width * sizeof(uchar3), height, cudaMemcpyDefault);

    dim3 blockDim(32, 32);
    dim3 gridDim(divRoundUp(width, blockDim.x), divRoundUp(height, blockDim.y));
    cuda::convertToGray2<<<gridDim, blockDim>>>(deviceRGB, deviceRGBPitch, deviceGray, deviceGrayPitch, width, height);

    dst = cv::Mat(height, width, CV_8UC1);
    cudaMemcpy2D(dst.data, dst.step, deviceGray, deviceGrayPitch, width, height, cudaMemcpyDefault);

    cudaFree(deviceRGB);
    cudaFree(deviceGray);
}
}  // namespace
