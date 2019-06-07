/**
 * @file    Sorting.hpp
 *
 * @brief   Template Sorting
 *
 * @author  btran
 *
 * @date    2019-06-07
 *
 * framework
 *
 * Copyright (c) organization
 *
 */

#ifndef SORTING_HPP_
#define SORTING_HPP_

#include <memory>
#include <ostream>
#include <vector>

template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
std::ostream &operator<<(std::ostream &os, const Container<T> &container)
{
    using ContainerType = Container<T>;
    for (typename ContainerType::const_iterator it = container.begin();
         it != container.end(); ++it) {
        os << *it << " ";
    }

    return os;
}

namespace algo
{
template <typename T,
          template <typename, typename = std::allocator<T>> class Container>
class Sorting
{
 public:
    using ContainerType = Container<T>;

    virtual void sorting(ContainerType &container)
    {
    }
};

}  // namespace algo
#endif /* SORTING_HPP_ */
