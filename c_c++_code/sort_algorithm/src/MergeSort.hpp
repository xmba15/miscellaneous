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

#include "Sorting.hpp"
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
    void sorting(ContainerType &container, size_t begin, size_t end);

 private:
    void merge(ContainerType &container, size_t begin, size_t end);
};

template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
void MergeSort<T, Container>::sorting(ContainerType &container)
{
    this->sorting(container, 0, container.size());
}

template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
void MergeSort<T, Container>::sorting(ContainerType &container, size_t begin,
                                      size_t end)
{
    size_t length = end - begin;

    if (length <= 1 || end > container.size()) {
        return;
    }

    this->sorting(container, begin, begin + length / 2);
    this->sorting(container, begin + length / 2, end);

    this->merge(container, begin, end);
}

template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
void MergeSort<T, Container>::merge(ContainerType &container, size_t begin,
                                    size_t end)
{
    size_t length = end - begin;

    ContainerType first(container.begin() + begin,
                        container.begin() + begin + length / 2);
    ContainerType second(container.begin() + begin + length / 2,
                         container.begin() + end);

    ContainerTypeIt containerIt = container.begin() + begin;
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
