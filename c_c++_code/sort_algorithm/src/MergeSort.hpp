/**
 * @file    MergeSort.hpp
 *
 * @brief   Template MergeSort
 *
 * @author  btran
 *
 * @date    2019-06-07
 *
 * Copyright (c) organization
 *
 */

#ifndef MERGESORT_HPP_
#define MERGESORT_HPP_

#include "Sorting.hpp"
#include <iostream>
#include <memory>

namespace algo
{
template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
class MergeSort : public Sorting<T, Container>
{
 public:
    using ContainerType = typename Sorting<T, Container>::ContainerType;
    using ContainerTypeIt = typename ContainerType::iterator;

    void sorting(ContainerType &container) override;
    void sorting(ContainerType &container, size_t low, size_t high);

 private:
    void merge(ContainerType &container, size_t low, size_t high);
};

template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
void MergeSort<T, Container>::sorting(ContainerType &container)
{
    this->sorting(container, 0, container.size() - 1);
}

template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
void MergeSort<T, Container>::sorting(ContainerType &container, size_t low,
                                      size_t high)
{
    if (high >= container.size() || high <= low) {
        return;
    }

    size_t length = high - low;
    size_t middle = low + length / 2;

    this->sorting(container, low, middle);
    this->sorting(container, middle + 1, high);

    this->merge(container, low, high);
}

template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
void MergeSort<T, Container>::merge(ContainerType &container, size_t low,
                                    size_t high)
{
    size_t length = high - low;
    size_t middle = low + length / 2;

    ContainerType first(container.begin() + low,
                        container.begin() + middle + 1);
    ContainerType second(container.begin() + middle + 1,
                         container.begin() + high + 1);

    ContainerTypeIt containerIt = container.begin() + low;
    ContainerTypeIt firstIt = first.begin();
    ContainerTypeIt secondIt = second.begin();

    while (firstIt != first.end() && secondIt != second.end()) {
        if (*firstIt < *secondIt) {
            *containerIt = *firstIt;
            ++firstIt;
        } else {
            *containerIt = *secondIt;
            ++secondIt;
        }
        ++containerIt;
    }

    if (firstIt != first.end()) {
        while (firstIt != first.end()) {
            *containerIt++ = *firstIt++;
        }
    }

    if (secondIt != second.end()) {
        while (secondIt != second.end()) {
            *containerIt++ = *secondIt++;
        }
    }
}

}  // namespace algo
#endif /* MERGESORT_HPP_ */
