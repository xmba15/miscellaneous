/**
 * @file    BucketSort.hpp
 *
 * @brief   Template BucketSort
 *
 * @author  btran
 *
 * @date    2019-06-10
 *
 * Copyright (c) organization
 *
 */

#ifndef BUCKETSORT_HPP_
#define BUCKETSORT_HPP_

#include "InsertionSort.hpp"
#include "Sorting.hpp"
#include <algorithm>
#include <memory>
#include <vector>

/**
 *  Bucket Sort is useful when input is uniformly distributed over a range. like from 0.0 to 1.0
 *  Reference: https://www.geeksforgeeks.org/bucket-sort-2/
 *
 */
namespace algo
{
template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
class BucketSort : public Sorting<T, Container>
{
 public:
    using ContainerType = typename Sorting<T, Container>::ContainerType;
    using ContainerTypeIt = typename ContainerType::iterator;

    void sorting(ContainerType &container) override;
    void sorting(ContainerType &container, size_t numOfBucket);
};

template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
void BucketSort<T, Container>::sorting(ContainerType &container)
{
    this->sorting(container, container.size());
}

template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
void BucketSort<T, Container>::sorting(ContainerType &container,
                                       size_t numOfBucket)
{
    if (container.size() <= 1) {
        return;
    }

    if (numOfBucket <= 0) {
        throw std::invalid_argument("Number of buckets must be over 0");
    }

    // create n empty buckets
    std::vector<T> buckets[numOfBucket];

    T maxElem = *std::max_element(container.begin(), container.end());
    T minElem = *std::min_element(container.begin(), container.end());

    // range of one bucket
    double interval = static_cast<double>(maxElem - minElem + 1) / numOfBucket;

    // insert value into list
    for (T elem : container) {
        size_t bucketIdx = static_cast<size_t>((elem - minElem) / interval);
        buckets[bucketIdx].push_back(elem);
    }

    InsertionSort<T, std::vector> is;

    // sort individual bucket
    for (std::vector<T> &bucket : buckets) {
        is.sorting(bucket);
    }

    // concatenate all buckets into the original array
    ContainerTypeIt containerIt = container.begin();
    for (const std::vector<T> &bucket : buckets) {
        for (T elem : bucket) {
            *containerIt = elem;
            ++containerIt;
        }
    }
}

}  // namespace algo

#endif /* BUCKETSORT_HPP_ */
