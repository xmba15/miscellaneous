/**
 * @file    InsertionSort.hpp
 *
 * @brief   Template InsertionSort
 *
 * @author  btran
 *
 * @date    2019-06-07
 *
 * Copyright (c) organization
 *
 */

#ifndef INSERTIONSORT_HPP_
#define INSERTIONSORT_HPP_

#include "Sorting.hpp"
#include <memory>

namespace algo
{
template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
class InsertionSort : public Sorting<T, Container>
{
 public:
    using ContainerType = typename Sorting<T, Container>::ContainerType;

    void sorting(ContainerType &container) override;
};

template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
void InsertionSort<T, Container>::sorting(ContainerType &container)
{
    if (container.size() <= 1) {
        return;
    }

    for (auto IIt = container.begin() + 1; IIt != container.end(); ++IIt) {
        T curV = *IIt;
        auto JIt = IIt - 1;
        while ((JIt != container.begin() - 1) && *JIt > *IIt) {
            *(JIt + 1) = *(JIt);
            --JIt;
        }
        ++JIt;
        *JIt = curV;
    }
}

}  // namespace algo

#endif /* INSERTIONSORT_HPP_ */
