/**
 * @file    App.cpp
 *
 * @author  btran
 *
 */

#include <iostream>
#include <memory>
#include <random>

#include <opencv2/opencv.hpp>

namespace
{
void createData(float* ref, int refNum, float* query, int queryNum, int dim);

cv::TickMeter meter;
// int NUM_TEST = 10;
int NUM_TEST = 1;
double processingTime = 0;
}  // namespace

#include <knn_cuda/knn.hpp>

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
    const int REF_NUMBER = 16384;
    int QUERY_NUMBER = 4096;
    int DIMENSION = 128;
    int K = 16;

    std::vector<float> ref(REF_NUMBER * DIMENSION);
    std::vector<float> query(QUERY_NUMBER * DIMENSION);
    ::createData(ref.data(), REF_NUMBER, query.data(), QUERY_NUMBER, DIMENSION);

    knn::KNNHandler knnHandler;

    meter.reset();
    meter.start();
    for (int i = 0; i < NUM_TEST; ++i) {
        knnHandler.run(ref.data(), REF_NUMBER, query.data(), QUERY_NUMBER, DIMENSION, K);
    }
    meter.stop();
    processingTime += meter.getTimeMilli() / NUM_TEST;

    meter.reset();
    meter.start();
    std::vector<float> knnDist(QUERY_NUMBER * REF_NUMBER);
    std::vector<int> knnIndices(QUERY_NUMBER * K);
    knnHandler.downloadToHost(knnDist.data(), knnIndices.data());
    meter.stop();
    processingTime += meter.getTimeMilli();

    std::cout << "total processing time: " << processingTime << "[ms]\n";

    return 0;
}

namespace
{
void createData(float* ref, int refNum, float* query, int queryNum, int dim)
{
    srand(2021);

    for (int i = 0; i < refNum * dim; ++i) {
        ref[i] = 10. * rand() / RAND_MAX;
    }

    for (int i = 0; i < queryNum * dim; ++i) {
        query[i] = 10. * rand() / RAND_MAX;
    }
}
}  // namespace
