/**
 * @file    QuickSort.hpp
 *
 * @brief   Template QuickSort
 *
 * @author  btran
 *
 * @date    2019-06-10
 *
 * Copyright (c) organization
 *
 */

#ifndef QUICKSORT_HPP_
#define QUICKSORT_HPP_

#include "Sorting.hpp"
#include <memory>

namespace algo
{
template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
class QuickSort : public Sorting<T, Container>
{
 public:
    using ContainerType = typename Sorting<T, Container>::ContainerType;
    using ContainerTypeIt = typename ContainerType::iterator;

    void sorting(ContainerType &container) override;
    void sorting(ContainerType &container, size_t low, size_t high);

 private:
    size_t partition(ContainerType &container, size_t low, size_t high);
    size_t partition(ContainerType &container, size_t low, size_t high,
                     bool isFirstElemAsPivot);

    void swapping(T &, T &);
};

template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
void QuickSort<T, Container>::sorting(ContainerType &container)
{
    this->sorting(container, 0, container.size() - 1);
}

template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
void QuickSort<T, Container>::sorting(ContainerType &container, size_t low,
                                      size_t high)
{
    if (high >= container.size() || high <= low) {
        return;
    }

    size_t pivotIdx = partition(container, low, high, true);
    this->sorting(container, low, pivotIdx - 1);
    this->sorting(container, pivotIdx + 1, high);
}

template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
size_t QuickSort<T, Container>::partition(ContainerType &container, size_t low,
                                          size_t high)
{
    T pivot = container.at(high);

    int pivotIdx = low - 1;

    for (int j = low; j <= high - 1; ++j) {
        if (container.at(j) <= pivot) {
            ++pivotIdx;
            this->swapping(container.at(j), container.at(pivotIdx));
        }
    }

    ++pivotIdx;
    this->swapping(container.at(high), container.at(pivotIdx));

    return static_cast<size_t>(pivotIdx);
}

template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
size_t QuickSort<T, Container>::partition(ContainerType &container, size_t low,
                                          size_t high, bool isFirstElemAsPivot)
{
    if (!isFirstElemAsPivot) {
        return partition(container, low, high);
    }

    T pivot = container.at(low);

    int pivotIdx = high + 1;
    for (int j = high; j > low; --j) {
        if (container.at(j) > pivot) {
            --pivotIdx;
            this->swapping(container.at(j), container.at(pivotIdx));
        }
    }

    --pivotIdx;
    this->swapping(container.at(low), container.at(pivotIdx));

    return static_cast<size_t>(pivotIdx);
}

template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
void QuickSort<T, Container>::swapping(T &x, T &y)
{
    if (x == y)
        return;

    x = x + y;
    y = x - y;
    x = x - y;
}

}  // namespace algo
#endif /* QUICKSORT_HPP_ */
