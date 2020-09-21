/**
 * @file    KMeans.hpp
 *
 * @author  btran
 *
 */

#pragma once

#include <limits>
#include <stdexcept>
#include <vector>

namespace _cv
{
class KMeans
{
 public:
    using Cluster = std::vector<int>;
    using Clusters = std::vector<Cluster>;

    static Clusters run(const std::vector<double>& data, int k, std::vector<double>& centroids, int numIterations = 20,
                        double epsilon = 1e-3)
    {
        if (k <= 0) {
            throw std::runtime_error("k must be > 0");
        }
        Clusters clusters(k);
        centroids.resize(k);
        int numData = data.size();

        std::srand(2021);
        for (int i = 0; i < k; ++i) {
            centroids[i] = data[std::rand() % numData];
        }

        double prevCost = std::numeric_limits<double>::max();
        for (int iter = 0; iter < numIterations; ++iter) {
            for (auto& cluster : clusters) {
                cluster.clear();
            }

            double newCost = 0;
            for (int i = 0; i < numData; ++i) {
                int minClusterIdx = -1;
                double minDist = std::numeric_limits<double>::max();
                for (int idx = 0; idx < k; ++idx) {
                    double dist = std::abs(centroids[idx] - data[i]);
                    if (dist < minDist) {
                        minDist = dist;
                        minClusterIdx = idx;
                    }
                }
                clusters[minClusterIdx].emplace_back(i);
                newCost += minDist;
            }

            for (int i = 0; i < k; ++i) {
                centroids[i] = 0;
                for (int pointIdx : clusters[i]) {
                    centroids[i] += data[pointIdx];
                }
                centroids[i] /= clusters[i].size();
            }
            prevCost = newCost;
        }

        return clusters;
    }
};
}  // namespace _cv
