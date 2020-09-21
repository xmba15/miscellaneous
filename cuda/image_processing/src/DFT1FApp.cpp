/**
 * @file    DFT1FApp.cpp
 *
 * @author  btran
 *
 */

#include <complex>
#include <iostream>
#include <random>

#include <opencv2/opencv.hpp>

#include <cufft.h>

#include "DFT1F.hpp"
#include "FFT1F.hpp"
#include "Timer.hpp"

namespace
{
std::vector<std::complex<double>> createSampleData(int numSamples, int seed = 2021)
{
    std::vector<std::complex<double>> result;
    std::mt19937 mt(seed);
    std::normal_distribution<> d{0, 2};

    for (int i = 0; i < numSamples; ++i) {
        result.emplace_back(d(mt));
    }

    return result;
}

std::vector<std::complex<double>> fft1dByCuFFT(const std::vector<std::complex<double>>& src);

cv::TickMeter meter;
int NUM_TEST = 10;
}  // namespace

/**
 * @brief Overloading the << operator to quickly print out the content of the containers.
 */
template <typename T, template <typename, typename = std::allocator<T>> class Container>
std::ostream& operator<<(std::ostream& os, const Container<T>& container)
{
    using ContainerType = Container<T>;
    for (typename ContainerType::const_iterator it = container.begin(); it != container.end(); ++it) {
        os << *it << " ";
    }

    return os;
}

int main(int argc, char* argv[])
{
    int constexpr NUM_SAPLES = 10000;
    std::vector<std::complex<double>> src = ::createSampleData(NUM_SAPLES);

    cv::Mat cvDft1D;
    meter.start();
    for (int i = 0; i < NUM_TEST; ++i) {
        cv::dft(src, cvDft1D, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);
    }
    meter.stop();
    std::cout << "opencv dft: " << meter.getTimeMilli() / NUM_TEST << "[ms]\n";

    std::vector<std::complex<double>> srcDft;
    meter.reset();
    meter.start();
    for (int i = 0; i < 2; ++i) {
        srcDft = _cv::dft1d(src, false);
    }
    meter.stop();
    std::cout << "dft (cpu): " << meter.getTimeMilli() / 2 << "[ms]\n";

    std::vector<std::complex<double>> srcDftGPU;
    meter.reset();
    meter.start();
    for (int i = 0; i < NUM_TEST; ++i) {
        srcDftGPU = _cv::dft1dGPU(src, false);
    }
    meter.stop();
    std::cout << "dft (gpu): " << meter.getTimeMilli() / NUM_TEST << "[ms]\n";

    // -------------------------------------------------------------------------
    // cpu fft
    // -------------------------------------------------------------------------

    constexpr int NUM_FFT_SAMPLES = 32 * 32 * 32 * 32;
    std::vector<std::complex<double>> fftSamples = ::createSampleData(NUM_FFT_SAMPLES);

    cv::Mat cvfft1D;
    meter.start();
    for (int i = 0; i < NUM_TEST; ++i) {
        cv::dft(fftSamples, cvfft1D, cv::DFT_ROWS | cv::DFT_COMPLEX_OUTPUT);
    }
    meter.stop();
    std::cout << "opencv fft: " << meter.getTimeMilli() / NUM_TEST << "[ms]\n";

    std::vector<std::complex<double>> fftOutput;
    meter.reset();
    meter.start();
    for (int i = 0; i < NUM_TEST; ++i) {
        fftOutput = _cv::fft1d(fftSamples, false);
    }
    meter.stop();
    std::cout << "fft (cpu): " << meter.getTimeMilli() / NUM_TEST << "[ms]\n";

    std::vector<std::complex<double>> fftOutput2;
    meter.reset();
    meter.start();
    for (int i = 0; i < NUM_TEST; ++i) {
        fftOutput2 = _cv::fft1d2(fftSamples, false);
    }
    meter.stop();
    std::cout << "fft2 (cpu): " << meter.getTimeMilli() / NUM_TEST << "[ms]\n";

    std::vector<std::complex<double>> fftOutput3;
    meter.reset();
    meter.start();
    for (int i = 0; i < NUM_TEST; ++i) {
        fftOutput3 = _cv::fft1d3(fftSamples, false);
    }
    meter.stop();
    std::cout << "fft3 (cpu): " << meter.getTimeMilli() / NUM_TEST << "[ms]\n";

    // -------------------------------------------------------------------------
    // gpu fft
    // -------------------------------------------------------------------------

    std::vector<std::complex<double>> fftOutputGPU;
    meter.reset();
    meter.start();
    fftOutputGPU = _cv::fft1dGPU(fftSamples, false);
    meter.stop();
    std::cout << "fft (gpu): " << meter.getTimeMilli() / NUM_TEST << "[ms]\n";

    std::complex<double> sum = 0;
    for (int i = 0; i < NUM_FFT_SAMPLES; ++i) {
        sum += fftOutputGPU[i] - fftOutput3[i];
    }
    std::cout << "result discrepancy: " << sum << "\n";

    // -------------------------------------------------------------------------
    // cufft
    // -------------------------------------------------------------------------

    std::vector<std::complex<double>> fftOutputGPUByCuFFT;
    meter.reset();
    meter.start();
    for (int i = 0; i < NUM_TEST; ++i) {
        fftOutputGPUByCuFFT = ::fft1dByCuFFT(fftSamples);
    }
    meter.stop();
    std::cout << "fft by cufft (gpu): " << meter.getTimeMilli() / NUM_TEST << "[ms]\n";

    sum = 0;
    for (int i = 0; i < NUM_FFT_SAMPLES; ++i) {
        sum += fftOutputGPU[i] - fftOutputGPUByCuFFT[i];
    }
    std::cout << "result discrepancy (normal gpu vs cufft): " << sum << "\n";

    return 0;
}

namespace
{
std::vector<std::complex<double>> fft1dByCuFFT(const std::vector<std::complex<double>>& src)
{
    if (src.empty()) {
        throw std::runtime_error("empty samples");
    }

    int numSamples = src.size();

    cufftDoubleComplex* deviceSrc;
    cudaMalloc((void**)&deviceSrc, numSamples * sizeof(cufftDoubleComplex));
    cudaMemcpy(deviceSrc, reinterpret_cast<const cufftDoubleComplex*>(src.data()),
               numSamples * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, numSamples, CUFFT_Z2Z, 1);

    // CUFFT_INVERSE for ifft
    cufftExecZ2Z(plan, deviceSrc, deviceSrc, CUFFT_FORWARD);

    std::vector<std::complex<double>> dst(numSamples);
    cudaMemcpy(reinterpret_cast<cufftDoubleComplex*>(dst.data()), deviceSrc, numSamples * sizeof(cufftDoubleComplex),
               cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaFree(deviceSrc);
    cufftDestroy(plan);
    return dst;
}
}  // namespace
